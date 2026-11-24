# mypy: disable-error-code="index, misc"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""
[v3] Ported verbatim from the v2 baseline detection/motion.py (imports only).

detection/motion.py — background subtraction + per-box pixel evidence (tracking v3).

The COCO-only detector scores a swimmer seen through murky water, foam or a
splash at 0.10–0.35, the same range as its false positives: the slide on the
deck, a ball, a ripple, a drain on the pool floor. Score alone cannot separate
them, so faint boxes are checked against what the PIXELS say before they may
start a track (tracking.FaintBirths):

  motion   share of the box that is foreground in a low-resolution
           background model (OpenCV MOG2, history MOTION_HISTORY_SECONDS).
           Anything that has not moved for that long — a submerged drain, a
           toy on the pool floor, the lane-line T, the slide — is background
           and can never start a track from faint boxes.
  body     share of the box that is neither pool water nor white water:
           hair, skin, swimwear, or a body seen through the water (clearly
           DARKER than the water around it). A pure ripple / glint / foam
           patch is water or white water only.
  splash   share of the box that is white water (bright, unsaturated) —
           a surface disturbance with nothing solid inside it.

All three are computed on a ~192 px wide copy of the frame (< 1 ms). The
estimator never decides anything on its own; `None` means "not warmed up yet"
and callers treat it as "no evidence".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from lifeguard.vision.constants import (
    BODY_DARK_DELTA,
    MOTION_EVERY_N,
    MOTION_HISTORY_SECONDS,
    MOTION_VAR_THRESHOLD,
    MOTION_WARMUP_SECONDS,
    MOTION_WIDTH,
    SPLASH_SAT_MAX,
    SPLASH_VAL_MIN,
    WATER_HUE_RANGE,
    WATER_SAT_MIN,
    WATER_VAL_MIN,
)


@dataclass(frozen=True)
class BoxEvidence:
    motion: float  # foreground share of the box
    body: float  # neither water nor white water (incl. bodies darker than the water)
    splash: float  # white-water share


class SceneMotion:
    """One per camera. `update(frame, fps)` every analysed frame, then `evidence(box)`."""

    def __init__(self) -> None:
        self._mog: cv2.BackgroundSubtractor | None = None
        self._shape: tuple | None = None
        self.scale = 1.0  # frame px per low-res px
        self.frames = 0
        self.fps = 25.0
        self.fg: np.ndarray | None = None  # bool, low-res
        self._small: np.ndarray | None = None
        self._hsv: np.ndarray | None = None  # uint8, low-res — computed on first evidence() call per frame
        self._tick = 0

    def reset(self) -> None:
        self.__init__()

    @property
    def warm(self) -> bool:
        return self.fg is not None and self.frames >= max(2, int(MOTION_WARMUP_SECONDS * self.fps))

    def update(self, frame: np.ndarray, fps: float = 25.0) -> None:
        h, w = frame.shape[:2]
        sw = min(MOTION_WIDTH, w)
        sh = max(1, int(round(h * sw / float(w))))
        fps = float(fps) if fps and fps > 0 else 25.0
        if self._mog is not None and self._shape == (sh, sw) and abs(fps - self.fps) > 0.5 and self.frames > 0:
            # [dynamic fps] the rate stepped on a running stream: keep the learned background, rescale its history
            self.fps = fps
            self._mog.setHistory(max(10, int(MOTION_HISTORY_SECONDS * self.fps / MOTION_EVERY_N)))  # type: ignore[attr-defined]
        if self._mog is None or self._shape != (sh, sw):
            self.fps = fps
            self._mog = cv2.createBackgroundSubtractorMOG2(
                history=max(10, int(MOTION_HISTORY_SECONDS * self.fps / MOTION_EVERY_N)),
                varThreshold=float(MOTION_VAR_THRESHOLD),
                detectShadows=False,
            )
            self._shape, self.frames, self._tick = (sh, sw), 0, 0
        self.scale = w / float(sw)
        self._tick += 1
        self.frames += 1
        if self.fg is not None and self._tick % MOTION_EVERY_N:
            return  # the foreground of the previous update stands
        # a strided pick before the area resize: same result at a fraction of the cost
        step = max(1, int(w // (sw * 2)))
        src = frame[::step, ::step] if step > 1 else frame
        self._small = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
        fg = self._mog.apply(self._small)
        self.fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
        self._hsv = None

    @property
    def hsv(self) -> np.ndarray | None:
        if self._hsv is None and self._small is not None:
            self._hsv = cv2.cvtColor(self._small, cv2.COLOR_BGR2HSV)
        return self._hsv

    def _cells(self, box: Sequence[float], grow: float = 0.0):
        x1, y1, x2, y2 = (float(v) / self.scale for v in box[:4])
        gw, gh = grow * (x2 - x1), grow * (y2 - y1)
        H, W = self._shape
        a = (
            int(max(0, np.floor(x1 - gw))),
            int(max(0, np.floor(y1 - gh))),
            int(min(W, np.ceil(x2 + gw))),
            int(min(H, np.ceil(y2 + gh))),
        )
        return a if a[2] > a[0] and a[3] > a[1] else None

    def evidence(self, box: Sequence[float]) -> BoxEvidence | None:
        """Pixel evidence inside `box` (frame px), or None before the model is warm."""
        hsv_all = self.hsv
        if not self.warm or hsv_all is None:
            return None
        c = self._cells(box)
        if c is None:
            return None
        x1, y1, x2, y2 = c
        hsv = hsv_all[y1:y2, x1:x2].reshape(-1, 3).astype(np.int16)
        hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        water = (
            (hue >= WATER_HUE_RANGE[0]) & (hue <= WATER_HUE_RANGE[1]) & (sat >= WATER_SAT_MIN) & (val >= WATER_VAL_MIN)
        )
        splash = (sat <= SPLASH_SAT_MAX) & (val >= SPLASH_VAL_MIN)
        # Reference brightness of the water AROUND the box: a body under the
        # surface keeps the water's hue but is clearly darker than it.
        ring = self._cells(box, grow=0.6)
        ref = None
        if ring is not None:
            r = hsv_all[ring[1] : ring[3], ring[0] : ring[2]].reshape(-1, 3).astype(np.int16)
            rw = (
                (r[:, 0] >= WATER_HUE_RANGE[0])
                & (r[:, 0] <= WATER_HUE_RANGE[1])
                & (r[:, 1] >= WATER_SAT_MIN)
                & (r[:, 2] >= WATER_VAL_MIN)
            )
            if rw.sum() >= 8:
                ref = float(np.median(r[rw, 2]))
        dark = (val <= ref - BODY_DARK_DELTA) if ref is not None else np.zeros_like(water)
        body = (~water | dark) & ~splash
        motion = float(self.fg[y1:y2, x1:x2].mean())
        return BoxEvidence(motion=motion, body=float(body.mean()), splash=float(splash.mean()))
