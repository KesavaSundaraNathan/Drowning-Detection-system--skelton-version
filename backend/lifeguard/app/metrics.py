"""[Phase 12] Prometheus metrics on a separate, loopback-only listener (audit C9: "expose Prometheus /metrics on an
internal port only").

    curl http://127.0.0.1:9464/metrics

Not a route of the FastAPI app: Caddy proxies only ``/api/*``, so nothing here is reachable through the operator
UI's port, and the reviewed route table (tests/api/test_auth_matrix.py) is unchanged. The listener binds
``settings.metrics_bind`` (default 127.0.0.1:9464, firewalled by install.ps1 step 12). A non-loopback bind is refused
by the settings unless ``metrics_token`` is set, and then every scrape must send ``Authorization: Bearer <token>``.

Values are read at scrape time from the same sources the operator UI uses (LiveService.status(), the delivery
service, the heartbeat files, the signed release gate), so there is no instrumentation spread through the code. Only
the HTTP request counter/histogram is updated per request, labelled by route TEMPLATE (bounded cardinality), never by
the raw path. Per-camera series are labelled by camera id; their number is bounded by the cameras the site has.
"""

from __future__ import annotations

import hmac
import logging
import threading
import time
from collections.abc import Callable, Iterable, Iterator, MutableMapping
from pathlib import Path
from typing import Any
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from prometheus_client import CollectorRegistry, Counter, Histogram, make_wsgi_app
from prometheus_client.core import GaugeMetricFamily, InfoMetricFamily, Metric
from prometheus_client.registry import Collector

from lifeguard import __version__
from lifeguard.core import heartbeat
from lifeguard.core.config import Settings

log = logging.getLogger(__name__)

MONITORING_STATES = ("MONITORED", "NOT_MONITORED")  # lifeguard.vision.faults
_GATE_REFRESH_S = 300.0  # the gate status hashes the detector file (~110 MB): not on every scrape
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    return None


class AppCollector(Collector):
    """Everything except the request counters, gathered at scrape time. A source that fails is reported as
    ``lifeguard_metrics_source_up{source=...} 0`` instead of failing the whole scrape."""

    def __init__(
        self,
        settings: Settings,
        live: Callable[[], Any] | None = None,
        delivery: Callable[[], Any] | None = None,
        gate: Callable[[], dict[str, Any]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.live = live
        self.delivery = delivery
        self.gate = gate
        self.clock = clock
        self._gate_cache: tuple[float, dict[str, Any]] | None = None

    def describe(self) -> Iterable[Metric]:
        return []  # collected on demand only (no registration-time collect)

    def collect(self) -> Iterator[Metric]:
        up = GaugeMetricFamily("lifeguard_metrics_source_up", "1 when the source could be read", labels=["source"])
        info = InfoMetricFamily("lifeguard_build", "Life Guard build")
        info.add_metric([], {"version": __version__, "env": self.settings.env})
        yield info
        for name, fn in (
            ("live", self._live),
            ("delivery", self._delivery),
            ("heartbeats", self._heartbeats),
            ("release_gate", self._gate),
        ):
            try:
                metrics = list(fn())
            except Exception:
                log.warning("metrics source %s failed", name, exc_info=True)
                up.add_metric([name], 0.0)
                continue
            up.add_metric([name], 1.0)
            yield from metrics
        yield up

    # ── live runtime ──────────────────────────────────────────────────────────
    def _live(self) -> Iterator[Metric]:
        service = self.live() if self.live is not None else None
        enabled = GaugeMetricFamily("lifeguard_live_enabled", "1 when the live runtime is configured to run")
        ready = GaugeMetricFamily("lifeguard_live_ready", "1 when /health/ready would answer 200 for the live part")
        if service is None:
            enabled.add_metric([], 0.0)
            yield enabled
            return
        st: dict[str, Any] = service.status()
        enabled.add_metric([], 1.0 if st.get("enabled") else 0.0)
        ready.add_metric([], 1.0 if service.ready() else 0.0)
        yield enabled
        yield ready
        sup = GaugeMetricFamily("lifeguard_supervisor_down", "1 while the API sees no fresh supervisor heartbeat")
        sup.add_metric([], 1.0 if st.get("supervisor_down") else 0.0)
        yield sup
        yield from _scheduler_metrics(st.get("scheduler"))
        yield from _capacity_metrics(st.get("capacity"))
        yield from _camera_metrics(st.get("cameras") or [])
        not_started = GaugeMetricFamily("lifeguard_cameras_not_started", "enabled cameras refused or not started")
        not_started.add_metric([], float(len(st.get("not_started") or {})))
        yield not_started
        startup = st.get("startup") or {}
        mon = _num(startup.get("all_monitored_s"))
        if mon is not None:
            g = GaugeMetricFamily(
                "lifeguard_startup_all_monitored_seconds", "process start -> every camera MONITORED (last start)"
            )
            g.add_metric([], mon)
            yield g

    # ── alert delivery ────────────────────────────────────────────────────────
    def _delivery(self) -> Iterator[Metric]:
        svc = self.delivery() if self.delivery is not None else None
        if svc is None:
            return
        d: dict[str, Any] = svc.status()
        alive = GaugeMetricFamily("lifeguard_delivery_worker_alive", "1 when the worker thread runs", labels=["worker"])
        alive.add_metric(["dispatcher"], 1.0 if d.get("dispatcher_alive") else 0.0)
        alive.add_metric(["escalator"], 1.0 if d.get("escalator_alive") else 0.0)
        yield alive
        pending = GaugeMetricFamily("lifeguard_outbox_pending", "undelivered outbox rows", labels=["channel"])
        oldest = GaugeMetricFamily(
            "lifeguard_outbox_oldest_pending_seconds", "age of the oldest undelivered row", labels=["channel"]
        )
        down = GaugeMetricFamily("lifeguard_channel_down", "1 while a channel counts as down", labels=["channel"])
        fails = GaugeMetricFamily(
            "lifeguard_channel_consecutive_failures", "failed deliveries in a row", labels=["channel"]
        )
        sent = GaugeMetricFamily("lifeguard_channel_sent", "deliveries sent since the API started", labels=["channel"])
        for ch, c in sorted((d.get("channels") or {}).items()):
            pending.add_metric([ch], float(c.get("pending") or 0))
            age = _num(c.get("oldest_pending_s"))
            oldest.add_metric([ch], age if age is not None else 0.0)
            down.add_metric([ch], 1.0 if c.get("down") else 0.0)
            fails.add_metric([ch], float(c.get("consecutive_failures") or 0))
            sent.add_metric([ch], float(c.get("sent") or 0))
        yield from (pending, oldest, down, fails, sent)

    # ── heartbeats of the other services ──────────────────────────────────────
    def _heartbeats(self) -> Iterator[Metric]:
        age = GaugeMetricFamily(
            "lifeguard_heartbeat_age_seconds", "age of a service's heartbeat file", labels=["component"]
        )
        present = GaugeMetricFamily(
            "lifeguard_heartbeat_present", "1 when the heartbeat file exists and is readable", labels=["component"]
        )
        for comp in heartbeat.COMPONENTS:
            r = heartbeat.read(self.settings.heartbeat_dir, comp)
            present.add_metric([comp], 0.0 if r.age_s is None else 1.0)
            if r.age_s is not None:
                age.add_metric([comp], round(r.age_s, 3))
        yield age
        yield present

    # ── release gate ──────────────────────────────────────────────────────────
    def _gate(self) -> Iterator[Metric]:
        if self.gate is None:
            return
        now = self.clock()
        if self._gate_cache is None or now - self._gate_cache[0] >= _GATE_REFRESH_S:
            self._gate_cache = (now, self.gate())
        st = self._gate_cache[1]
        g = GaugeMetricFamily(
            "lifeguard_assist_pilot",
            "1 = ASSIST / PILOT (no valid signed release gate); 0 = GATE_PASSED, bound to what runs",
        )
        g.add_metric([], 0.0 if st.get("mode") == "GATE_PASSED" else 1.0)
        yield g


def _scheduler_metrics(s: dict[str, Any] | None) -> Iterator[Metric]:
    if not s:
        return
    for key, name, doc in (
        ("rounds", "lifeguard_scheduler_rounds", "GPU batch rounds since start"),
        ("frames", "lifeguard_scheduler_frames", "frames analysed by the GPU scheduler since start"),
        ("mean_batch", "lifeguard_scheduler_mean_batch", "mean frames per GPU round"),
        ("last_round_ms", "lifeguard_scheduler_last_round_ms", "duration of the last GPU round"),
        ("last_round_age_s", "lifeguard_scheduler_last_round_age_seconds", "time since the last GPU round"),
        ("oldest_pending_s", "lifeguard_scheduler_oldest_pending_seconds", "oldest frame waiting for the GPU"),
        ("failed_requests", "lifeguard_scheduler_failed_requests", "GPU requests that failed since start"),
        ("stalled", "lifeguard_scheduler_stalled", "1 while the GPU scheduler is STALLED"),
    ):
        v = _num(s.get(key))
        if v is not None:
            g = GaugeMetricFamily(name, doc)
            g.add_metric([], v)
            yield g


def _capacity_metrics(c: dict[str, Any] | None) -> Iterator[Metric]:
    if not c:
        return
    for key, name, doc in (
        ("limit", "lifeguard_capacity_camera_limit", "cameras this GPU admits at the target fps"),
        ("overhead_ms", "lifeguard_capacity_overhead_ms", "measured per-frame CPU cost"),
        ("target_fps", "lifeguard_capacity_target_fps", "admission target analysis fps"),
    ):
        v = _num(c.get(key))
        if v is not None:
            g = GaugeMetricFamily(name, doc)
            g.add_metric([], v)
            yield g


def _camera_metrics(cams: list[dict[str, Any]]) -> Iterator[Metric]:
    lab = ["camera_id"]
    state = GaugeMetricFamily("lifeguard_camera_monitoring", "1 for the camera's current state", labels=[*lab, "state"])
    gauges = {
        "fps": GaugeMetricFamily("lifeguard_camera_capture_fps", "frames per second read from the camera", labels=lab),
        "last_frame_age_s": GaugeMetricFamily(
            "lifeguard_camera_last_frame_age_seconds", "time since the last frame", labels=lab
        ),
        "frames_read": GaugeMetricFamily("lifeguard_camera_frames_read", "frames decoded since start", labels=lab),
        "reconnects": GaugeMetricFamily("lifeguard_camera_reconnects", "stream reconnects since start", labels=lab),
        "analysed": GaugeMetricFamily("lifeguard_camera_frames_analysed", "frames analysed since start", labels=lab),
        "frame_errors": GaugeMetricFamily("lifeguard_camera_frame_errors", "frames that failed analysis", labels=lab),
        "proc_ms": GaugeMetricFamily("lifeguard_camera_processing_ms", "recent per-frame processing time", labels=lab),
        "lag_s": GaugeMetricFamily("lifeguard_camera_lag_seconds", "age of the oldest queued frame", labels=lab),
        "dropped": GaugeMetricFamily("lifeguard_camera_frames_dropped", "frames dropped from a full queue", labels=lab),
        "analysed_fraction": GaugeMetricFamily(
            "lifeguard_camera_analysed_fraction", "analysed / decoded frames", labels=lab
        ),
        "stalled": GaugeMetricFamily("lifeguard_camera_stalled", "1 while the camera is STALLED", labels=lab),
        "lagging": GaugeMetricFamily("lifeguard_camera_lagging", "1 while the camera is LAGGING", labels=lab),
    }
    for c in cams:
        cid = str(c.get("camera_id"))
        mon = str(c.get("monitoring"))
        for s in MONITORING_STATES:
            state.add_metric([cid, s], 1.0 if mon == s else 0.0)
        flat = {**(c.get("capture") or {}), **(c.get("analysis") or {}), **c}
        for key, g in gauges.items():
            v = _num(flat.get(key))
            if v is not None:
                g.add_metric([cid], v)
    yield state
    yield from gauges.values()


class AppMetrics:
    """The registry, the request instruments and the listener of one API process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry = CollectorRegistry(auto_describe=False)
        self.requests = Counter(
            "lifeguard_http_requests",
            "HTTP requests by route template and status class",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "lifeguard_http_request_duration_seconds",
            "HTTP request duration by route template",
            ["method", "route"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.collector: AppCollector | None = None
        self._server: WSGIServer | None = None
        self._thread: threading.Thread | None = None

    def observe_request(self, scope: MutableMapping[str, Any], status: int, seconds: float) -> None:
        route = scope.get("route")
        template = getattr(route, "path", None) or "unmatched"  # never the raw path: bounded label values
        method = str(scope.get("method") or "")
        self.requests.labels(method, template, f"{status // 100}xx").inc()
        self.latency.labels(method, template).observe(seconds)

    def attach(self, collector: AppCollector) -> None:
        if self.collector is not None:
            self.registry.unregister(self.collector)
        self.collector = collector
        self.registry.register(collector)

    # ── listener ──────────────────────────────────────────────────────────────
    def wsgi_app(self) -> Callable[..., Iterable[bytes]]:
        inner = make_wsgi_app(self.registry)
        token = self.settings.metrics_token.get_secret_value() if self.settings.metrics_token else None

        def app(environ: dict[str, Any], start_response: Callable[..., Any]) -> Iterable[bytes]:
            if environ.get("PATH_INFO") != "/metrics":
                start_response("404 Not Found", [("Content-Type", "text/plain")])
                return [b"not found\n"]
            if token is not None:
                sent = environ.get("HTTP_AUTHORIZATION", "")
                if not hmac.compare_digest(sent.encode(), f"Bearer {token}".encode()):
                    start_response("401 Unauthorized", [("Content-Type", "text/plain"), ("WWW-Authenticate", "Bearer")])
                    return [b"unauthorized\n"]
            result: Iterable[bytes] = inner(environ, start_response)
            return result

        return app

    def start(self) -> int:
        """Bind and serve in a daemon thread; returns the bound port (``:0`` picks a free one, used by tests)."""
        host, port = self.settings.metrics_address
        server = make_server(host, port, self.wsgi_app(), WSGIServer, _QuietHandler)
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, name="metrics-http", daemon=True)
        self._thread.start()
        bound = int(server.server_address[1])
        log.info("Prometheus metrics on http://%s:%d/metrics", host, bound)
        return bound

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


class _QuietHandler(WSGIRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return  # scrapes every few seconds would flood the log; the access log covers the API itself


def gate_status(settings: Settings) -> Callable[[], dict[str, Any]]:
    def read() -> dict[str, Any]:
        from lifeguard.eval import release_gate
        from lifeguard.vision import config as vision_config

        det = Path(settings.detector_model) if settings.detector_model else None
        det = det or vision_config.MODELS_DIR / vision_config.DEFAULT_DET_MODEL
        return release_gate.status(settings.data_dir, settings.keys_dir, det)  # as GET /system/release

    return read
