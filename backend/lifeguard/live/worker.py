"""Per-camera analysis worker (plan Phase 4).

One thread per camera takes frames from the camera's FrameQueue, oldest first, and runs the
unchanged Phase 3 pipeline (``SourcePipeline.process``); the detector call inside it is batched
across cameras by the scheduler. Behaviour alerts go to ``fire_sink``; FAULTs to ``fault_sink``.

Failure isolation and recovery:
  * ``DetectorUnavailableError`` -> the pipeline has already raised the camera's FAULT and marked it
    NOT_MONITORED. The worker keeps draining frames (counted as NOT analysed), and rebuilds the
    detector every ``recover_every_s``; a successful rebuild returns the camera to MONITORED.
  * Any other exception on a frame is logged and counted; ``MAX_FRAME_ERRORS`` in a row raise a
    PIPELINE_ERROR FAULT and the camera is NOT_MONITORED until a frame succeeds again.
  * A new capture epoch (reconnect) resets the tracker and behaviour state, then calls the optional ``on_reset``.
  * [Phase 5] Cameras are independent: no cross-camera identity is stamped on detections or alerts.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from lifeguard.live.frames import FrameQueue
from lifeguard.vision.faults import MONITORED, NOT_MONITORED, Fault, FaultSink
from lifeguard.vision.inference.detector import DetectorUnavailableError
from lifeguard.vision.scene.hysteresis import AlertFire

logger = logging.getLogger(__name__)

PIPELINE_ERROR = "PIPELINE_ERROR"
MAX_FRAME_ERRORS = 25

FireSink = Callable[[str, AlertFire, Any], None]  # (camera id, fire, frame image)
ClearSink = Callable[[str, str], None]  # (fault code, source id)
# every analysed frame (recorder ring): (image, detections, track ids whose drowning alert is active now)
FrameHook = Callable[[Any, Any, frozenset[int]], None]
# analysis faults this worker clears when the camera is analysed again (also after a restart)
ANALYSIS_FAULTS = ("DETECTOR_UNAVAILABLE", "PROVIDER_NOT_ACTIVE", PIPELINE_ERROR)
PipelineFactory = Callable[[float, tuple[int, int] | None], Any]  # (fps, frame_size) -> SourcePipeline


class CameraWorker:
    def __init__(
        self,
        camera_id: str,
        queue: FrameQueue,
        pipeline_factory: PipelineFactory,
        fault_sink: FaultSink,
        fire_sink: FireSink,
        *,
        fps: Callable[[], float],
        frame_size: Callable[[], tuple[int, int] | None],
        recover_every_s: float = 10.0,
        clear_sink: ClearSink | None = None,
        frame_hook: FrameHook | None = None,
        geometry_loader: Callable[[], Any] | None = None,
        on_reset: Callable[[], None] | None = None,
        on_frame_start: Callable[[Any], None] | None = None,
        on_frame_done: Callable[[Any], None] | None = None,
    ) -> None:
        self.on_reset = on_reset
        # [Phase 11 WP6] replay hooks (eval only): before / after every frame taken from the queue
        self.on_frame_start = on_frame_start
        self.on_frame_done = on_frame_done
        self.geometry_loader = geometry_loader
        self._reload_geometry = threading.Event()
        self._scene_type: str | None = None  # [v4] pending pool / standard CCTV switch (applied between frames)
        self.clear_sink = clear_sink
        self.frame_hook = frame_hook
        self._cleared_once = False
        self.camera_id = camera_id
        self.queue = queue
        self.pipeline_factory = pipeline_factory
        self.fault_sink = fault_sink
        self.fire_sink = fire_sink
        self._fps = fps
        self._frame_size = frame_size
        self.recover_every_s = recover_every_s
        self.pipeline: Any = None
        self.monitoring = MONITORED
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.epoch = 0
        self.analysed = 0
        self.not_analysed = 0  # frames taken while NOT_MONITORED
        self.frame_errors = 0
        self._errors_in_row = 0
        self.fires = 0
        self.last_analysed_at: float | None = None
        self.first_analysed_at: float | None = None  # [Phase 11 WP5] startup timeline (monitored)
        self.last_image: Any = None
        self.last_detections: Any = None
        self.alert_track_ids: frozenset[int] = frozenset()  # [Phase 5] tracks whose alert is raised (view)
        self.proc_ms_ewma = 0.0
        self.prefetched = 0  # [v4] next-frame detections started early
        self._next_recover = 0.0
        # [Phase 2] capture-to-screen latency: time.monotonic() at full analysis minus the frame's
        # captured_at (decode time), so it includes queue wait, not just pipeline.process(). A short
        # rolling window (not an EWMA) so p50/p95 can be read off it, matching the plan's latency report.
        self._latency_ms_window: deque[float] = deque(maxlen=200)
        self._stage_ms_ewma: dict[str, float] = {}  # [Phase 2] per res.stages_ms key, e.g. "detect_track_pose"

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"analyse-{self.camera_id}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def request_geometry_reload(self) -> None:
        """Zones or schedules changed: apply them before the next analysed frame (no restart)."""
        self._reload_geometry.set()

    def request_scene_type(self, scene_type: str) -> None:
        """[v4] Pool / standard CCTV changed: apply it before the next analysed frame (no restart)."""
        self._scene_type = scene_type

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── loop ──────────────────────────────────────────────────────────────────
    def _loop(self) -> None:
        while not self._stop.is_set():
            frame = self.queue.get(timeout=0.25)
            if frame is None:
                continue
            if self.on_frame_start is not None:
                try:
                    self.on_frame_start(frame)
                except Exception:
                    logger.exception("[live:%s] frame-start hook failed", self.camera_id)
            self._prefetch_next(frame)
            try:
                self._handle(frame)
            finally:
                # [Phase 11 WP6] every frame taken is reported done, whatever happened to it (replay lockstep)
                if self.on_frame_done is not None:
                    try:
                        self.on_frame_done(frame)
                    except Exception:
                        logger.exception("[live:%s] frame-done hook failed", self.camera_id)

    def _prefetch_next(self, frame: Any) -> None:
        """[v4] When the next frame is already waiting, start its detection now: its GPU round then runs while this
        frame is posed, tracked and scored. Only when the camera is behind (a frame is queued) - exactly when the
        overlap is needed - and never for a replayed recording (lockstep: one frame at a time) or a new epoch."""
        if self.on_frame_start is not None or self.pipeline is None or self.monitoring == NOT_MONITORED:
            return
        nxt = self.queue.peek()
        if nxt is None or nxt.epoch != frame.epoch:
            return
        det = getattr(self.pipeline, "detector", None)
        try:
            if det is not None and hasattr(det, "prefetch") and det.prefetch(nxt.image):
                self.prefetched += 1
        except Exception:  # an optimisation: the frame is detected normally when its turn comes
            logger.debug("[live:%s] prefetch failed", self.camera_id, exc_info=True)

    def _handle(self, frame: Any) -> None:
        self.last_image = frame.image  # operator view: newest decoded frame (reference, no copy)
        if self.pipeline is None or frame.epoch != self.epoch:
            self._new_epoch(frame.epoch)
        elif hasattr(self.pipeline, "retime"):  # [dynamic fps] the rate stepped on this stream: tracks are kept
            fps = self._fps()
            if fps and abs(fps - getattr(self.pipeline, "fps", fps)) > 1e-6:
                self.pipeline.retime(fps)
        if self._reload_geometry.is_set() and self.geometry_loader is not None:
            self._reload_geometry.clear()
            try:
                self.pipeline.set_geometry(self.geometry_loader())
                logger.info("[live:%s] zones / schedules reloaded", self.camera_id)
            except Exception:
                logger.exception("[live:%s] zone reload failed - previous zones kept", self.camera_id)
        scene_type, self._scene_type = self._scene_type, None
        if scene_type is not None and hasattr(self.pipeline, "set_scene_type"):
            self.pipeline.set_scene_type(scene_type)
            logger.info("[live:%s] scene type %s applied", self.camera_id, scene_type)
        if self.monitoring == NOT_MONITORED and self._detector_dead():
            self.not_analysed += 1
            self._maybe_recover()
            return
        t0 = time.perf_counter()
        try:
            res = self.pipeline.process(frame.image, frame.index, frame.t)
        except DetectorUnavailableError:
            self.monitoring = NOT_MONITORED  # FAULT already raised by the pipeline
            self.not_analysed += 1
            self._next_recover = time.monotonic() + self.recover_every_s
            return
        except Exception as exc:  # noqa: BLE001 - logged + counted in _frame_error; FAULT after MAX_FRAME_ERRORS
            self._frame_error(exc)
            return
        self._errors_in_row = 0
        recovered = self.monitoring == NOT_MONITORED
        if recovered:
            logger.warning("[live:%s] analysis recovered - camera MONITORED again", self.camera_id)
        self.monitoring = MONITORED
        if (recovered or not self._cleared_once) and self.clear_sink is not None:
            self._cleared_once = True
            for code in ANALYSIS_FAULTS:
                try:
                    self.clear_sink(code, self.camera_id)
                except Exception:
                    logger.exception("[live:%s] clearing %s failed", self.camera_id, code)
        # [Stage 4] the people whose drowning alert is active on THIS frame: drawn red on the live view and in the
        # recording (observed boxes only - a coasting box is never red)
        scene = getattr(self.pipeline, "scene", None)
        tracks = getattr(scene, "tracks", None) or {}
        alerting = frozenset(tid for tid, ts in tracks.items() if getattr(ts, "alert_active", False))
        if self.frame_hook is not None:
            try:
                self.frame_hook(frame.image, res.detections, alerting)
            except Exception:
                logger.exception("[live:%s] frame hook failed", self.camera_id)
        self.analysed += 1
        self.last_analysed_at = time.monotonic()
        if self.first_analysed_at is None:
            self.first_analysed_at = self.last_analysed_at
        self.last_detections = res.detections
        self.alert_track_ids = alerting
        ms = (time.perf_counter() - t0) * 1000.0
        self.proc_ms_ewma = ms if self.analysed == 1 else 0.95 * self.proc_ms_ewma + 0.05 * ms
        captured_at = getattr(frame, "captured_at", None)
        if captured_at is not None:
            self._latency_ms_window.append(max(0.0, (time.monotonic() - captured_at) * 1000.0))
        # [Phase 2] per-stage breakdown (vision/pipeline.py already times "detect_track_pose" and
        # "behaviour" per frame; this worker just keeps an EWMA per stage name so slow stages are visible
        # without digging into traces)
        for stage, stage_ms in (getattr(res, "stages_ms", None) or {}).items():
            prev = self._stage_ms_ewma.get(stage)
            self._stage_ms_ewma[stage] = stage_ms if prev is None else 0.95 * prev + 0.05 * stage_ms
        for fire in res.fires:
            self.fires += 1
            try:
                self.fire_sink(self.camera_id, fire, frame.image)
            except Exception:
                logger.exception("[live:%s] alert sink failed", self.camera_id)

    def _detector_dead(self) -> bool:
        det = getattr(self.pipeline, "detector", None)
        return det is not None and not getattr(det, "healthy", True)

    def _new_epoch(self, epoch: int) -> None:
        if self.pipeline is None:
            self.pipeline = self.pipeline_factory(self._fps(), self._frame_size())
        else:
            self.pipeline.reset()
            self._reset_identity()
            if hasattr(self.pipeline, "set_fps"):
                self.pipeline.set_fps(self._fps())
            elif hasattr(self.pipeline.detector, "set_fps"):
                self.pipeline.detector.set_fps(self._fps())
        self.epoch = epoch

    def _reset_identity(self) -> None:
        """Tracker ids restart: tell the optional ``on_reset`` hook (none by default)."""
        if self.on_reset is None:
            return
        try:
            self.on_reset()
        except Exception:
            logger.exception("[live:%s] reset hook failed", self.camera_id)

    def _maybe_recover(self) -> None:
        now = time.monotonic()
        if now < self._next_recover:
            return
        self._next_recover = now + self.recover_every_s
        try:
            pipe = self.pipeline_factory(self._fps(), self._frame_size())
        except Exception:
            logger.exception("[live:%s] detector rebuild failed", self.camera_id)
            return
        if not getattr(pipe.detector, "healthy", False):
            logger.warning(
                "[live:%s] detector still unavailable; next attempt in %.0f s", self.camera_id, self.recover_every_s
            )
            return
        pipe.monitoring = NOT_MONITORED  # a new FAULT is not raised again for the same outage
        self.pipeline = pipe
        logger.warning("[live:%s] detector rebuilt; resuming analysis", self.camera_id)

    def _frame_error(self, exc: BaseException) -> None:
        self.frame_errors += 1
        self._errors_in_row += 1
        n = self._errors_in_row
        if n == 1 or n % 25 == 0:
            logger.warning("[live:%s] frame failed (%d in a row)", self.camera_id, n, exc_info=exc)
        if n == MAX_FRAME_ERRORS:
            self.monitoring = NOT_MONITORED
            self.fault_sink(
                Fault(PIPELINE_ERROR, self.camera_id, f"{n} consecutive analysis errors: {type(exc).__name__}: {exc}")
            )

    def status(self) -> dict[str, Any]:
        age = None if self.last_analysed_at is None else round(time.monotonic() - self.last_analysed_at, 3)
        return {
            "monitoring": self.monitoring,
            "alive": self.alive,
            "analysed": self.analysed,
            "not_analysed": self.not_analysed,
            "frame_errors": self.frame_errors,
            "fires": self.fires,
            "proc_ms": round(self.proc_ms_ewma, 2),
            "prefetched": self.prefetched,
            "last_analysed_age_s": age,
            # [Phase 2] capture-to-screen latency over the last up-to-200 analysed frames (None until any).
            "latency_ms_p50": _percentile(self._latency_ms_window, 0.50),
            "latency_ms_p95": _percentile(self._latency_ms_window, 0.95),
            "stage_ms": {k: round(v, 2) for k, v in self._stage_ms_ewma.items()},
        }


def _percentile(values: deque[float], p: float) -> float | None:
    """Nearest-rank percentile (0 <= p <= 1) over `values`; None when empty. No numpy: this runs per status poll."""
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(p * (len(ordered) - 1)))
    return round(ordered[idx], 2)
