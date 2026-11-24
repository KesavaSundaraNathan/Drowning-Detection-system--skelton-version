"""Per-camera frame queue between capture and analysis (plan Phase 4).

Every decoded frame is queued for analysis (the tracker and behaviour stages assume every frame of
the source is seen). The queue is bounded to ``live_queue_s`` seconds of video; when analysis falls
behind, the OLDEST frame is dropped (the newest picture of the pool matters most) and counted, so
``analysed_fraction`` and ``lag_s`` are honest instead of the camera silently lagging.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    index: int  # 0-based decode index since the capture (re)opened
    t: float  # stream time in seconds (tracker clock)
    captured_at: float  # time.monotonic() when decoded
    epoch: int  # capture session; increments on every reconnect


class FrameQueue:
    def __init__(self, maxlen: int) -> None:
        if maxlen < 1:
            raise ValueError("maxlen must be >= 1")
        self.maxlen = maxlen
        self._q: deque[Frame] = deque()
        self._cv = threading.Condition()
        self.pushed = 0
        self.dropped = 0
        self.taken = 0

    def put(self, frame: Frame) -> bool:
        """Queue `frame`; returns False when an older frame had to be dropped to make room."""
        with self._cv:
            dropped = False
            if len(self._q) >= self.maxlen:
                self._q.popleft()
                self.dropped += 1
                dropped = True
            self._q.append(frame)
            self.pushed += 1
            self._cv.notify()
            return not dropped

    def get(self, timeout: float) -> Frame | None:
        with self._cv:
            if not self._q:
                self._cv.wait(timeout)
            if not self._q:
                return None
            self.taken += 1
            return self._q.popleft()

    def peek(self) -> Frame | None:
        """[v4] The frame get() would return next, left in the queue (the worker prefetches its detection)."""
        with self._cv:
            return self._q[0] if self._q else None

    def clear(self) -> int:
        with self._cv:
            n = len(self._q)
            self._q.clear()
            self.dropped += n
            return n

    def __len__(self) -> int:
        with self._cv:
            return len(self._q)

    def lag_s(self, now: float | None = None) -> float:
        """Age of the oldest frame still waiting (0 when empty)."""
        with self._cv:
            if not self._q:
                return 0.0
            return max(0.0, (time.monotonic() if now is None else now) - self._q[0].captured_at)
