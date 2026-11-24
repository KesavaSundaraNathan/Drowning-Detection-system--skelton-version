"""[2.18.0] Time base and frame rate of a finite clip (eval harness, dataset builder).

Uploads used to be stamped ``idx / CAP_PROP_FPS`` and analysed on every frame at the file's own rate (25-60 fps),
while live cameras are analysed at ``target_analysis_fps`` (18) on their real timestamps. Two consequences:

* every per-frame rule (tracker IoU between frames, Kalman steps, frame-counted windows) meant something different
  on an upload than on the live path it is supposed to reproduce;
* a variable-frame-rate phone video (iPhone ``.mov``) drifts against ``idx / fps``, so the analysis clock and the
  browser's ``currentTime`` disagree and the Model View overlay lags or jumps.

``ClipClock`` gives each decoded frame its container timestamp (``CAP_PROP_POS_MSEC``; ``idx / fps`` when the
container gives none or it runs backwards) and keeps frames on a fixed time schedule at ``CLIP_ANALYSIS_FPS``
(0 = every frame). A dropped frame is only grabbed, never decoded into an image.
"""

from __future__ import annotations

import math
from typing import Any

import cv2

from lifeguard.vision.constants import CLIP_ANALYSIS_FPS, DET_NESTED_IOA_CLIP, TRACK_BIRTH_GUARD_IOU_CLIP


def apply_clip_profile(detector: Any) -> None:
    """[2.18.0] What a finite clip (upload, eval harness, training windows) does differently from a live camera:
    nested part-boxes are dropped before tracking (DET_NESTED_IOA_CLIP). Live cameras are unchanged."""
    if hasattr(detector, "nested_ioa"):
        detector.nested_ioa = float(DET_NESTED_IOA_CLIP)
    if hasattr(detector, "set_birth_guard"):
        detector.set_birth_guard(float(TRACK_BIRTH_GUARD_IOU_CLIP) or None)


class ClipClock:
    def __init__(self, src_fps: float, target_fps: float | None = None) -> None:
        self.src_fps = src_fps if math.isfinite(src_fps) and src_fps > 0 else 25.0
        target = CLIP_ANALYSIS_FPS if target_fps is None else target_fps
        self.period = 1.0 / target if target and 0 < target < self.src_fps else 0.0
        self._tol = 0.5 / self.src_fps
        self._due = -math.inf
        self._prev_t = -math.inf

    @property
    def analysis_fps(self) -> float:
        """The rate the pipeline actually sees (what tracker and frame rules must be told)."""
        return 1.0 / self.period if self.period else self.src_fps

    def stamp(self, cap: Any, idx: int) -> float:
        """Seconds of the frame just grabbed / read: its container timestamp, else ``idx / fps``."""
        t = idx / self.src_fps
        try:
            ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        except (cv2.error, TypeError, ValueError):
            ms = math.nan
        if math.isfinite(ms) and (ms > 0.0 or idx == 0):
            pts = ms / 1000.0
            if pts >= self._prev_t - 1e-6 and abs(pts - t) < 5.0 + 0.05 * t:  # monotonic and not wildly off
                t = pts
        t = max(t, self._prev_t + 1e-6) if math.isfinite(self._prev_t) else t
        self._prev_t = t
        return t

    def keep(self, t: float) -> bool:
        """Analyse this frame? Every frame when not decimating; else one per ``period`` on a fixed schedule."""
        if not self.period:
            return True
        if not math.isfinite(self._due):
            self._due = t
        if t >= self._due - self._tol:
            # schedule from the due time, not the frame time, so the average rate is exact
            self._due = max(self._due + self.period, t - self._tol)
            return True
        return False
