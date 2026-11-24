"""Glue between the API process and the live runtime (plan Phase 4).

`LiveService` owns:
  * the LiveRuntime (capture + batched GPU scheduler), with sinks that persist FAULT and behaviour
    alerts through the single DB writer;
  * camera specs built from the database (sources decrypted here and nowhere else; RTSP only, a legacy
    file row is refused as SOURCE_INVALID), zone geometry loaded fresh on every capture (re)connect;
  * the API heartbeat (`run/heartbeats/api.json`) read by the supervisor;
  * the API side of the mutual watchdog: a stale supervisor heartbeat raises SUPERVISOR_DOWN (delivered
    by this process's outbox - Phase 5), cleared when the heartbeat is back;
  * alert media: the alarm frame as a snapshot, and a pre/post-event recording for audible alarms;
  * FAULT clears: CAPACITY_EXCEEDED / SOURCE_INVALID when a camera starts, plus the runtime's clears.

At startup every enabled camera is started in name order. A camera that does not fit the measured
capacity is NOT started and gets a CAPACITY_EXCEEDED FAULT (NOT_MONITORED) - never a silent lag.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from sqlalchemy import select

from lifeguard.alerts.faults import clear_fault, record_fault
from lifeguard.alerts.media import save_snapshot
from lifeguard.alerts.writer import attach_media, close_idle_events, record_fire, record_ongoing
from lifeguard.core import events, heartbeat
from lifeguard.core.config import Settings
from lifeguard.core.security.keys import ServerKeys
from lifeguard.core.security.sources import SourceError, decrypt_source
from lifeguard.db.models import Camera, Schedule, Zone
from lifeguard.db.session import Database
from lifeguard.db.writer import DBWriter
from lifeguard.live.capacity import CapacityExceededError
from lifeguard.live.preopen import PreOpener, PreOpenSpec
from lifeguard.live.runtime import CameraSpec, LiveRuntime
from lifeguard.vision.faults import Fault
from lifeguard.vision.scene.hysteresis import AlertFire

logger = logging.getLogger(__name__)

CAPACITY_EXCEEDED = "CAPACITY_EXCEEDED"
SUPERVISOR_DOWN = "SUPERVISOR_DOWN"
SOURCE_INVALID = "SOURCE_INVALID"


class LiveService:
    def __init__(
        self,
        settings: Settings,
        keys: ServerKeys,
        db: Database,
        writer: DBWriter,
        det_model_path: str,
        *,
        runtime_enabled: bool | None = None,
        ready_fn: Callable[[], bool] | None = None,
        preopener: PreOpener | None = None,
    ) -> None:
        self.settings = settings
        self.keys = keys
        self.db = db
        self.writer = writer
        self.det_model_path = det_model_path
        self.enabled = settings.live_runtime_enabled if runtime_enabled is None else runtime_enabled
        self.runtime: LiveRuntime | None = None
        self._ready_fn = ready_fn or (lambda: True)
        self._hb: heartbeat.HeartbeatWriter | None = None
        self._watch_stop = threading.Event()
        self._watch: threading.Thread | None = None
        self.supervisor_down = False
        self.not_started: dict[str, str] = {}  # camera id -> reason (capacity / invalid source)
        self._supervisor_seen = False
        self._last_summary: str | None = None
        self._last_summary_at = 0.0
        self.delivery_status: Callable[[], dict[str, Any]] | None = None
        self.preopener = preopener  # [Phase 11 WP7] captures opened in parallel with the guard + capacity phase

    # ── sinks ─────────────────────────────────────────────────────────────────
    def fault_sink(self, fault: Fault, camera_id: str | None) -> None:
        record_fault(self.writer, fault, camera_id=camera_id)

    def clear_sink(self, code: str, source_id: str) -> None:
        clear_fault(self.writer, code, source_id)

    def fire_sink(
        self, camera_id: str, fire: AlertFire, audible: bool, image: Any = None, muted_reason: str | None = None
    ) -> None:
        # [2.19.0] an ONGOING update of an alert already raised: marked on that alert, no new row / snapshot / clip.
        # If that alert was acknowledged meanwhile, fall through and raise a new one.
        ongoing = not getattr(fire, "notify", True) and not fire.ghost
        if ongoing and record_ongoing(self.writer, fire, camera_id=camera_id) is not None:
            return
        alert_id = record_fire(self.writer, fire, camera_id=camera_id, audible=audible, muted_reason=muted_reason)
        if image is not None:
            name = save_snapshot(self.settings.snapshots_dir, alert_id, image, fire.bbox)
            if name is not None:
                attach_media(self.writer, alert_id, snapshot=name)
        rec = self.runtime.recorders.get(camera_id) if self.runtime is not None else None
        if audible and rec is not None:

            def attach(name: str | None) -> None:
                if name is not None:
                    attach_media(self.writer, alert_id, recording=name)

            rec.start_clip(alert_id, attach)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        if self.enabled:
            self.runtime = LiveRuntime(
                self.settings,
                self.det_model_path,
                self.settings.inference_provider,
                self.fault_sink,
                self.fire_sink,
                clear_sink=self.clear_sink,
            )
            # [Phase 11 WP5] on a capacity-cache hit, batch sizes up to the enabled cameras are warmed before admission
            self.runtime.start(expected_cameras=self.enabled_camera_count())
            try:
                self.start_enabled_cameras()
            finally:
                if self.preopener is not None:  # whatever was not admitted must not keep a stream open
                    self.preopener.close()
        else:
            logger.warning("live runtime DISABLED by configuration (live_enabled=false)")
        self._hb = heartbeat.HeartbeatWriter(
            self.settings.heartbeat_dir, heartbeat.API, self.heartbeat_payload, self.settings.heartbeat_interval_s
        )
        self._hb.start()
        if self.settings.supervisor_is_expected:
            self._watch = threading.Thread(target=self._watch_loop, name="supervisor-watch", daemon=True)
            self._watch.start()

    def stop(self) -> None:
        if self.preopener is not None:
            self.preopener.close()
        self._watch_stop.set()
        if self._watch is not None:
            self._watch.join(timeout=5)
        if self.runtime is not None:
            self.runtime.stop()
        if self._hb is not None:
            self._hb.stop()

    # ── cameras ───────────────────────────────────────────────────────────────
    def camera_spec(self, cam: Camera) -> CameraSpec:
        if cam.source_type != "rtsp":  # a legacy demo-profile file camera: Life Guard is RTSP-only
            raise SourceError(f"camera {cam.id}: file cameras were removed; give it an rtsp:// URL")
        source = decrypt_source(self.keys.fernet, cam.source_ciphertext)
        return CameraSpec(
            cam.id,
            cam.name,
            source,
            cam.source_type,
            cam.default_mode,
            self.geometry_loader(cam.id),
            scene_type=cam.scene_type,
        )

    def start_camera(self, camera_id: str) -> None:
        """Start one camera. Raises CapacityExceededError, KeyError (unknown/disabled), SourceError."""
        if self.runtime is None:
            raise RuntimeError("live runtime is disabled")
        with self.db.read_sessions() as rs:
            cam = rs.get(Camera, camera_id)
            if cam is None or not cam.is_enabled:
                raise KeyError(camera_id)
            spec = self.camera_spec(cam)
        self.runtime.start_camera(spec, adopt=self.preopener.adopt if self.preopener is not None else None)
        self.not_started.pop(camera_id, None)
        for code in (CAPACITY_EXCEEDED, SOURCE_INVALID):
            self.clear_sink(code, camera_id)

    def zones_changed(self, camera_id: str | None = None) -> None:
        """Called by the zones / schedules API after a change: running cameras reload their zones."""
        if self.runtime is not None:
            self.runtime.reload_geometry(camera_id)

    def scene_type_changed(self, camera_id: str, scene_type: str) -> None:
        """[v4] Called by the cameras API: a running camera switches pool / standard CCTV before its next frame."""
        if self.runtime is not None:
            self.runtime.set_scene_type(camera_id, scene_type)

    def stop_camera(self, camera_id: str) -> bool:
        return self.runtime is not None and self.runtime.stop_camera(camera_id)

    def forget_camera(self, camera_id: str) -> None:
        """A camera was removed: stop it and drop it from the not-started list."""
        self.stop_camera(camera_id)
        self.not_started.pop(camera_id, None)

    def enabled_camera_count(self) -> int:
        with self.db.read_sessions() as rs:
            n = len(list(rs.execute(select(Camera.id).where(Camera.is_enabled)).scalars()))
        limit = self.settings.startup_max_cameras
        return min(n, limit) if limit > 0 else n

    def start_enabled_cameras(self) -> None:
        with self.db.read_sessions() as rs:
            cams = list(rs.execute(select(Camera).where(Camera.is_enabled).order_by(Camera.name)).scalars())
            ids = [(c.id, c.name) for c in cams]
        limit = self.settings.startup_max_cameras
        deferred = ids[limit:] if limit > 0 else []
        ids = ids[:limit] if limit > 0 else ids
        for cid, name in deferred:  # not a fault: the operator starts these on purpose (Cameras page)
            self.not_started[cid] = "startup_limit"
            logger.warning("camera %s not started at start-up (startup_max_cameras=%d)", name, limit)
        admitted: list[str] = []
        for cid, name in ids:
            try:
                self.start_camera(cid)
                admitted.append(cid)
            except CapacityExceededError as exc:
                self.not_started[cid] = "capacity"
                self.fault_sink(Fault(CAPACITY_EXCEEDED, cid, f"{name}: NOT MONITORED - {exc}", exc.detail()), cid)
            except (SourceError, ValueError) as exc:
                self.not_started[cid] = "invalid_source"
                self.fault_sink(Fault(SOURCE_INVALID, cid, f"{name}: NOT MONITORED - source rejected: {exc}"), cid)
        # [Phase 11 WP5] startup timeline: the cameras the summary waits for
        if self.runtime is not None:
            self.runtime.timeline.expect(admitted)
            self.runtime.timeline.mark("cameras_admitted", cameras=len(admitted), refused=len(ids) - len(admitted))

    def geometry_loader(self, camera_id: str) -> Callable[[int, int], Any]:
        return make_geometry_loader(self.db, camera_id)

    # ── health / heartbeat ────────────────────────────────────────────────────
    def ready(self) -> bool:
        if not self._ready_fn():
            return False
        if self.runtime is None:
            return not self.enabled
        return self.runtime.running and not self.runtime.scheduler_stalled

    def publish_summary(self, force_after_s: float = 10.0) -> None:
        """live.summary on the WebSocket feed: on any monitoring-state change, else every 10 s."""
        if self.runtime is None:
            return
        cams = []
        for c in self.runtime.status()["cameras"]:
            cams.append(
                {
                    "camera_id": c["camera_id"],
                    "name": c["name"],
                    "monitoring": c["monitoring"],
                    "stalled": c["stalled"],
                    "lagging": c["lagging"],
                    "muted_zones": sorted(z["zone"] for z in c["zones"] if z["state"] == "MUTED"),
                    "scene_type": c["scene"]["type"],  # [v4] the suggestion is in /live/status (it changes often)
                }
            )
        not_started = sorted(self.not_started)
        key = repr((cams, not_started))
        now = time.monotonic()
        if key == self._last_summary and now - self._last_summary_at < force_after_s:
            return
        self._last_summary, self._last_summary_at = key, now
        events.publish({"type": "live.summary", "cameras": cams, "not_started": not_started})

    def heartbeat_payload(self) -> dict[str, Any]:
        try:
            self.publish_summary()
        except Exception:
            logger.exception("live summary failed")
        try:
            close_idle_events(self.writer)  # [Phase 5] events quiet for EVENT_IDLE_CLOSE_S close (event.closed)
        except Exception:
            logger.exception("closing idle events failed")
        payload: dict[str, Any] = {"ready": self.ready(), "live_enabled": self.enabled}
        if self.runtime is not None:
            payload.update(self.runtime.heartbeat())
        payload["not_started"] = dict(self.not_started)
        if self.delivery_status is not None:
            try:
                d = self.delivery_status()
                payload["delivery"] = {
                    "dispatcher_alive": d["dispatcher_alive"],
                    "escalator_alive": d["escalator_alive"],
                    "oldest_pending_s": d["oldest_pending_s"],
                    "channels_down": d["channels_down"],
                }
            except Exception as exc:  # noqa: BLE001 - reported to the supervisor instead
                payload["delivery"] = {"error": f"{type(exc).__name__}: {exc}"}
        return payload

    def status(self) -> dict[str, Any]:
        # [v3] `cameras` is always present: the UI maps it unconditionally (Phase 7 E2E found a blank screen
        # when the runtime was off and the key was missing)
        out: dict[str, Any] = {"enabled": self.enabled, "supervisor_down": self.supervisor_down, "cameras": []}
        if self.runtime is not None:
            out.update(self.runtime.status())
        out["not_started"] = dict(self.not_started)
        return out

    # ── mutual watchdog (API side) ────────────────────────────────────────────
    def check_supervisor(self, now: float | None = None) -> bool:
        """True when the supervisor heartbeat is fresh. Raises SUPERVISOR_DOWN once per outage."""
        reading = heartbeat.read(self.settings.heartbeat_dir, heartbeat.SUPERVISOR, now)
        fresh = reading.fresh(self.settings.supervisor_timeout_s)
        if not fresh and not self.supervisor_down:
            self.supervisor_down = True
            age = "missing" if reading.age_s is None else f"{reading.age_s:.0f} s old"
            fault = Fault(
                SUPERVISOR_DOWN,
                "system",
                f"supervisor service heartbeat {age} (limit {self.settings.supervisor_timeout_s:g} s): "
                "out-of-band SYSTEM FAULT delivery is not guaranteed",
            )
            # the supervisor cannot report its own death: this API's outbox delivers it (Phase 5)
            self.fault_sink(fault, None)
        elif fresh and (self.supervisor_down or not self._supervisor_seen):
            if self.supervisor_down:
                logger.warning("supervisor heartbeat is back")
            self.supervisor_down = False
            self.clear_sink(SUPERVISOR_DOWN, "system")
        self._supervisor_seen = self._supervisor_seen or fresh
        return fresh

    def _watch_loop(self) -> None:
        # first check only after one full timeout: the services start in any order
        if self._watch_stop.wait(self.settings.supervisor_timeout_s):
            return
        while True:
            try:
                self.check_supervisor()
            except Exception:
                logger.exception("supervisor watchdog check failed")
            if self._watch_stop.wait(max(1.0, self.settings.heartbeat_interval_s)):
                return


def preopen_specs(settings: Settings, keys: ServerKeys, db: Database) -> list[PreOpenSpec]:
    """[Phase 11 WP7] What can be opened before admission: enabled RTSP cameras, name order. A source that does not
    decrypt is skipped (admission reports it as SOURCE_INVALID exactly as before). Only RTSP sources exist."""
    del settings
    out: list[PreOpenSpec] = []
    with db.read_sessions() as rs:
        cams = list(rs.execute(select(Camera).where(Camera.is_enabled).order_by(Camera.name)).scalars())
        for cam in cams:
            if cam.source_type != "rtsp":
                continue
            try:
                out.append(PreOpenSpec(cam.id, decrypt_source(keys.fernet, cam.source_ciphertext), "rtsp"))
            except (SourceError, ValueError):
                continue
    return out


def make_geometry_loader(db: Database, camera_id: str, geometry_cls: Any = None) -> Callable[[int, int], Any]:
    """(w, h) -> the camera's ZoneGeometry (its own zones + projected pool zones + calibration) or None.
    [Phase 11 WP6] Module level so the site replay builds geometry with exactly this code (from a geometry
    snapshot database); `geometry_cls` lets the replay evaluate zone schedules at the RECORDING time."""

    def load(w: int, h: int) -> Any:
        from lifeguard.vision.constants import DEFAULT_ZONE_SENSITIVITY
        from lifeguard.vision.scene.geometry import (
            ScheduleSetting,
            ZoneGeometry,
            ZonePolygon,
            normalized_to_contour,
        )

        with db.read_sessions() as rs:
            zones = list(
                rs.execute(select(Zone).where(Zone.camera_id == camera_id, Zone.is_active).order_by(Zone.id))
                .scalars()
                .all()
            )
            cam = rs.get(Camera, camera_id)
            cal_data = cam.calibration if cam is not None else None
            if not zones and not cal_data:
                return None
            # [Phase 11 WP2] pool zones (pool plan, metres): only a calibrated camera can project them
            pool_zones = (
                list(
                    rs.execute(select(Zone).where(Zone.camera_id.is_(None), Zone.is_active).order_by(Zone.id))
                    .scalars()
                    .all()
                )
                if cal_data
                else []
            )
            sched_ids = {z.schedule_id for z in (*zones, *pool_zones) if z.schedule_id is not None}
            scheds = list(rs.execute(select(Schedule).where(Schedule.id.in_(sched_ids))).scalars().all())
            polys = [
                ZonePolygon(
                    z.id,
                    z.name,
                    z.zone_type,
                    normalized_to_contour(z.polygon or [], w, h),
                    float(z.sensitivity),
                    str(z.schedule_id) if z.schedule_id is not None else None,
                )
                for z in zones
            ]
            settings = {
                str(s.id): ScheduleSetting(list(s.days), s.start_time, s.end_time, s.monitoring_mode, s.status)
                for s in scheds
            }
        calibration = _load_calibration(camera_id, cal_data, w, h)
        if calibration is not None:
            polys += _project_pool_zones(camera_id, calibration, pool_zones, w, h)
        # [v3] v2 behaviour (zone_service.load_geometry): a calibrated camera with no WATER zone drawn
        # monitors the projected pool outline; people on the deck are ignored until they enter the pool.
        if calibration is not None and not any(p.monitored for p in polys):
            contour = calibration.water_contour(w, h)
            if contour is not None:
                polys.append(ZonePolygon(0, "Pool (calibrated)", "WATER", contour, DEFAULT_ZONE_SENSITIVITY, None))
        if not polys:
            return None
        geometry = (geometry_cls or ZoneGeometry)(camera_id, w, h, polys, settings)
        geometry.calibration = calibration
        # the camera's own drawn water polygon (not the pool-plan projection, not the auto outline above)
        geometry.drawn_water = any(z.zone_type in ("WATER", "SHALLOW") for z in zones)
        return geometry

    return load


def _project_pool_zones(camera_id: str, calibration: Any, pool_zones: list[Zone], w: int, h: int) -> list[Any]:
    """[Phase 11 WP2] The camera's pool's zones, projected from the pool plan into its view (out of view: skipped)."""
    from lifeguard.vision.scene.geometry import ZonePolygon
    from lifeguard.vision.scene.pool_zones import pool_key, project_polygon

    out = []
    pool = pool_key(calibration)
    for z in pool_zones:
        if z.pool_id != pool or not z.pool_polygon_m:
            continue
        contour = project_polygon(calibration, z.pool_polygon_m, w, h)
        if contour is None:
            logger.info("[live:%s] pool zone %r is not in this camera's view", camera_id, z.name)
            continue
        sched = str(z.schedule_id) if z.schedule_id is not None else None
        out.append(ZonePolygon(z.id, z.name, z.zone_type, contour, float(z.sensitivity), sched))
    return out


def _load_calibration(camera_id: str, data: dict[str, Any] | None, w: int, h: int) -> Any:
    """A stored calibration that no longer validates is logged and ignored (zones still apply)."""
    if not data:
        return None
    from lifeguard.vision.scene.pool_geometry import CalibrationError, PoolCalibration

    try:
        return PoolCalibration.from_json(data, w, h)
    except CalibrationError as exc:
        logger.warning("[live:%s] pool calibration unusable: %s", camera_id, exc)
        return None
