"""Lockstep replay of synchronised recordings through the real LiveRuntime (Phase 11 WP6, decision 3; evaluation only -
the API never creates a "replay" camera: its source allow-list is rtsp / file).

Each recording carries absolute wall-clock frame times (site/recorder.py). A replayed camera stamps every frame with
the SESSION clock: t = start_wall - origin - offset + position, where `origin` is the earliest recording start and
`offset` the camera's measured latency offset (SYNC flash; 0 when within 2 frames). Network cameras share their pool's
tracker by that one clock, exactly as live (identity binds any source that is not a plain file).

The ReplayCoordinator keeps the cameras in step without wall-clock pacing: a camera may hand out a frame stamped t only
when no OTHER camera still has a frame older than t - step that is not yet processed (emitted-but-unprocessed, or not
yet emitted), and only while its own queue has room (no frame is ever dropped). step = one frame period. So every
camera's tracker and the shared pool tracker see the session in time order, as fast as the GPU allows, and the run is
reproducible.
"""

from __future__ import annotations

import contextlib
import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Cam:
    next_t: float = -math.inf  # the next frame this camera will hand out (inf: finished)
    pending: deque[float] = field(default_factory=deque)  # handed out, not processed yet
    capacity: int = 8


class ReplayCoordinator:
    def __init__(self, step_s: float) -> None:
        self.step_s = step_s
        self._cv = threading.Condition()
        self._cams: dict[str, _Cam] = {}

    def register(self, camera_id: str, capacity: int) -> None:
        with self._cv:
            self._cams[camera_id] = _Cam(capacity=max(1, capacity - 1))

    def _oldest_open(self, cam: _Cam) -> float:
        return min(cam.next_t, cam.pending[0] if cam.pending else math.inf)

    def _may_emit(self, camera_id: str, t: float) -> bool:
        me = self._cams[camera_id]
        if len(me.pending) >= me.capacity:
            return False
        return all(self._oldest_open(c) >= t - self.step_s - 1e-9 for cid, c in self._cams.items() if cid != camera_id)

    def wait_turn(self, camera_id: str, t: float, stop: threading.Event, poll_s: float = 0.2) -> bool:
        """Block until `camera_id` may hand out its frame stamped t; then count it as pending. False if stopped."""
        with self._cv:
            self._cams[camera_id].next_t = t
            self._cv.notify_all()
            while not self._may_emit(camera_id, t):
                if stop.is_set():
                    return False
                self._cv.wait(poll_s)
            cam = self._cams[camera_id]
            cam.pending.append(t)
            # its next frame is never older than this one (frames are in time order per camera): a lower bound,
            # NOT infinity - with infinity the others ran ahead while it decoded its next frame (test: 0.25 s)
            cam.next_t = t
            return True

    def done(self, camera_id: str, t: float) -> None:
        """The worker finished the frame stamped t (analysed, skipped or failed)."""
        with self._cv:
            cam = self._cams.get(camera_id)
            if cam is not None and cam.pending:
                if cam.pending[0] == t:
                    cam.pending.popleft()
                else:
                    with contextlib.suppress(ValueError):
                        cam.pending.remove(t)
            self._cv.notify_all()

    def finish(self, camera_id: str) -> None:
        with self._cv:
            if camera_id in self._cams:
                self._cams[camera_id].next_t = math.inf
            self._cv.notify_all()

    def idle(self) -> bool:
        """Every camera finished and every frame processed."""
        with self._cv:
            return all(c.next_t == math.inf and not c.pending for c in self._cams.values())

    def lead_s(self) -> float:
        """How far the most advanced pending frame is ahead of the oldest open frame (tests: <= step)."""
        with self._cv:
            opens = [self._oldest_open(c) for c in self._cams.values()]
            pend = [t for c in self._cams.values() for t in c.pending]
        finite = [o for o in opens if o != math.inf]
        if not pend or not finite:
            return 0.0
        return max(pend) - min(finite)


@dataclass
class ReplayFeed:
    """What a replayed camera needs: its recording's absolute start, the session origin, its measured offset, the
    shared coordinator, and optional callbacks (trace, recording-time clock for zone schedules)."""

    start_wall: float
    origin_wall: float
    coordinator: ReplayCoordinator
    offset_s: float = 0.0
    on_frame: Callable[[str, Any, Any], None] | None = None  # (camera_id, frame, worker) after each frame
    on_start: Callable[[str, Any], None] | None = None  # (camera_id, frame) before each frame

    def stamp(self, pos_s: float) -> float:
        return self.start_wall - self.origin_wall - self.offset_s + pos_s
