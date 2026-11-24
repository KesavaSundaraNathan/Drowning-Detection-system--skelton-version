"""Open the cameras while the server is still starting (plan Phase 11 WP7, gates in eval/site/levers_phase11.json
"gates_wp7").

OpenCV opens RTSP streams one at a time (~1.8 s each, measured). Without this, the cameras open only after the
provider guard and the capacity phase; with it the opens overlap them:

    lifespan:  database_ready -> PreOpener.start()  ........ opens + drains each stream (gate CLOSED)
                              -> threshold audit, provider guard, capacity phase   (in parallel)
               admission    -> LiveRuntime.start_camera: profile.admit (unchanged), then ADOPT the pre-opened
                               capture (attach queue + stream clock), build the CameraWorker + detector, open the gate

Nothing is queued, tracked or analysed before admission: a gated CaptureWorker reads and discards frames (this
keeps the TCP session drained). Every capture that is not adopted - refused for capacity, invalid source, camera
disabled, startup refused - is stopped by `close()`, which the lifespan calls on every path.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from lifeguard.core.config import Settings
from lifeguard.core.startup import TIMELINE, StartupTimeline
from lifeguard.live.capture import CaptureWorker, Opener
from lifeguard.live.frames import FrameQueue

logger = logging.getLogger(__name__)

CLOSE_BUDGET_S = 10.0  # one shared join budget for all captures


@dataclass(frozen=True)
class PreOpenSpec:
    camera_id: str
    source: str  # decrypted - never logged
    source_type: str  # only "rtsp" is pre-opened


class PreOpener:
    def __init__(
        self,
        settings: Settings,
        *,
        opener: Opener | None = None,
        timeline: StartupTimeline = TIMELINE,
    ) -> None:
        self.settings = settings
        self.opener = opener
        self.timeline = timeline
        self._lock = threading.Lock()
        self._workers: dict[str, tuple[PreOpenSpec, CaptureWorker]] = {}
        self._closed = False
        self._hold: threading.Timer | None = None
        self.adopted: list[str] = []
        self.dropped: dict[str, str] = {}  # camera id -> why its pre-opened capture was not used

    # ── start ─────────────────────────────────────────────────────────────────
    def start(self, specs: list[PreOpenSpec]) -> int:
        """Open every RTSP spec behind a closed gate. Returns how many were started."""
        s = self.settings
        started = 0
        with self._lock:
            if self._closed:
                return 0
            for spec in specs:
                if spec.source_type != "rtsp" or spec.camera_id in self._workers:
                    continue
                queue = FrameQueue(2)  # placeholder: nothing is ever queued while the gate is closed
                capture = CaptureWorker(
                    spec.camera_id,
                    spec.source,
                    spec.source_type,
                    queue,
                    s.stall_s,
                    opener=self.opener,
                    first_frame_timeout_s=s.first_frame_timeout_s,
                    fps_fallback=s.camera_fps_fallback,
                    gated=True,
                    gate_hold_s=s.preopen_max_hold_s,
                    max_fps=s.target_analysis_fps,
                )
                self._workers[spec.camera_id] = (spec, capture)
                capture.start()
                self.timeline.camera(spec.camera_id, "preopen_started")
                started += 1
            if started:
                self._hold = threading.Timer(s.preopen_max_hold_s, self.close)
                self._hold.daemon = True
                self._hold.start()
        if started:
            self.timeline.mark("preopen_started", cameras=started)
            logger.info("pre-opening %d camera(s) in parallel with the provider guard and the capacity phase", started)
        return started

    # ── adoption (called by LiveRuntime.start_camera AFTER admission) ───────────
    def adopt(self, camera_id: str, source: str, source_type: str) -> CaptureWorker | None:
        """The pre-opened capture of this camera if it is still alive and was opened for exactly this source;
        else None (a stale / mismatching one is stopped)."""
        with self._lock:
            entry = self._workers.pop(camera_id, None)
        if entry is None:
            return None
        spec, capture = entry
        if (spec.source, spec.source_type) != (source, source_type):
            self.dropped[camera_id] = "source_changed"
            capture.stop(timeout=CLOSE_BUDGET_S)
            return None
        if not capture.alive:
            self.dropped[camera_id] = "capture_ended"
            capture.stop(timeout=1.0)
            return None
        self.adopted.append(camera_id)
        capture.preopen_adopted = True
        if capture.preopened_at is not None:  # when the first frame arrived behind the gate
            self.timeline.camera(camera_id, "preopened", mono=capture.preopened_at)
        return capture

    def adopter(self) -> Callable[[str, str, str], CaptureWorker | None]:
        return self.adopt

    # ── release ───────────────────────────────────────────────────────────────
    def close(self) -> int:
        """Stop every capture that was not adopted. Idempotent; returns how many were stopped."""
        with self._lock:
            self._closed = True
            rest = list(self._workers.items())
            self._workers.clear()
            hold, self._hold = self._hold, None
        if hold is not None:
            hold.cancel()
        for _, (_, capture) in rest:
            capture.signal_stop()  # signal all first, then join under ONE shared budget
        deadline = time.monotonic() + CLOSE_BUDGET_S
        for cid, (_, capture) in rest:
            self.dropped.setdefault(cid, "not_admitted")
            capture.stop(timeout=max(0.0, deadline - time.monotonic()))
        if rest:
            logger.info(
                "pre-open released %d capture(s) that were not admitted: %s", len(rest), sorted(c for c, _ in rest)
            )
        return len(rest)

    def status(self) -> dict[str, object]:
        with self._lock:
            pending = sorted(self._workers)
        return {"pending": pending, "adopted": list(self.adopted), "dropped": dict(self.dropped)}
