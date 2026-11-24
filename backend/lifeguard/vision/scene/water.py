# mypy: disable-error-code="arg-type, assignment, call-overload, misc"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""
[v3] Ported verbatim from the v2 baseline detection/water_region.py (imports only; plan: the colour
estimate is the FALLBACK used only when a camera has neither zones nor a pool calibration).

detection/water_region.py — automatic water-region estimate when no zones are drawn (Phase 2.4).

With no zones the old pipeline treated the WHOLE frame as water, so everyone
on the deck was scored as a swimmer, and anyone who walked behind a pillar
"vanished inside the pool" and escalated as a Sudden Blackout.

This estimator learns where the pool is from colour, slowly:

  * every WATER_EST_INTERVAL_S a low-resolution copy of the frame is
    classified per pixel as "pool-coloured" (cyan/blue hue, some saturation,
    not black) — the colour of a chlorinated pool under daylight or pool
    lighting;
  * an exponential moving average turns that into a probability map, which is
    cleaned (morphology) and reduced to the large connected components;
  * the estimate is only CONFIDENT once it has been stable for several
    updates and covers a plausible share of the frame.

It never decides on its own that someone is out of the water. It supplies
two things the scene tracker combines with pose evidence:
  * `contains(point)` → True / False / None (None = not confident: unknown)
  * `edge_distance(point)` → signed px distance to the water edge (+ inside)

Pools that are not pool-coloured (dark ponds, black-bottom pools, heavy
night colour casts) simply never become confident, and the pipeline falls
back to pose evidence alone — which keeps everyone monitored unless their
legs are clearly on land.
"""

from __future__ import annotations

import cv2
import numpy as np

from lifeguard.vision.constants import (
    WATER_EST_ALPHA,
    WATER_EST_INTERVAL_S,
    WATER_EST_MAX_FRAC,
    WATER_EST_MIN_FRAC,
    WATER_EST_STABLE_UPDATES,
    WATER_EST_WARMUP_INTERVAL_S,
    WATER_EST_WARMUP_UPDATES,
    WATER_EST_WIDTH,
    WATER_HUE_BAND,
    WATER_HUE_RANGE,
    WATER_SAT_MIN,
    WATER_SAT_REL,
    WATER_VAL_MIN,
)


class WaterRegionEstimator:
    def __init__(self) -> None:
        self.prob: np.ndarray | None = None
        self.mask: np.ndarray | None = None  # uint8 {0,1} at low resolution
        self.dist: np.ndarray | None = None  # signed distance (low-res px), + inside
        self.scale = 1.0  # frame px per low-res px
        self.frame_wh: tuple[int, int] = (0, 0)
        self.last_t: float | None = None
        self.stable = 0
        self.updates = 0

    @property
    def confident(self) -> bool:
        return self.mask is not None and self.stable >= WATER_EST_STABLE_UPDATES

    def reset(self) -> None:
        self.__init__()

    def update(self, frame: np.ndarray, t: float) -> bool:
        """Feed a frame (cheap no-op between updates). Returns True when the map changed."""
        interval = WATER_EST_WARMUP_INTERVAL_S if self.updates < WATER_EST_WARMUP_UPDATES else WATER_EST_INTERVAL_S
        if self.last_t is not None and t - self.last_t < interval:
            return False
        self.last_t = t
        h, w = frame.shape[:2]
        if (w, h) != self.frame_wh:
            self.reset()
            self.last_t = t
            self.frame_wh = (w, h)
        sw = min(WATER_EST_WIDTH, w)
        sh = max(1, int(round(h * sw / w)))
        self.scale = w / float(sw)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        m = self._water_pixels(hue, sat, val).astype(np.float32)
        self.prob = m if self.prob is None else (1 - WATER_EST_ALPHA) * self.prob + WATER_EST_ALPHA * m
        self.updates += 1
        mask = (self.prob > 0.5).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)  # swimmers / lane ropes inside the pool
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        keep = np.zeros_like(mask)
        area = mask.size
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= 0.03 * area:
                keep[lab == i] = 1
        # fill holes (people, floats, lane lines fully surrounded by water are water)
        cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(keep)
        cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
        frac = filled.mean()
        if not (WATER_EST_MIN_FRAC <= frac <= WATER_EST_MAX_FRAC):
            self.mask, self.dist, self.stable = None, None, 0
            return True
        if self.mask is not None:
            inter = np.logical_and(self.mask, filled).sum()
            union = np.logical_or(self.mask, filled).sum()
            self.stable = self.stable + 1 if union and inter / union >= 0.85 else 0
        self.mask = filled
        inside = cv2.distanceTransform(filled, cv2.DIST_L2, 3)
        outside = cv2.distanceTransform(1 - filled, cv2.DIST_L2, 3)
        self.dist = (inside - outside).astype(np.float32)
        return True

    @staticmethod
    def _water_pixels(hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> np.ndarray:
        """
        Pool-coloured pixels. The fixed cyan→blue band alone accepted the whole
        frame of colour-cast footage (a purple-tinted phone clip: deck hue ≈131,
        wall and plants blue-ish too), so "everyone is in the water" and every
        lost person was "strictly inside the pool". The band now only proposes
        candidates; the pool is the DOMINANT hue among them (saturation-
        weighted mode), ± WATER_HUE_BAND, at least WATER_SAT_REL of that
        cluster's median saturation — pool water is the most saturated blue in
        the scene, a tinted deck or wall is not.
        """
        cand = (
            (hue >= WATER_HUE_RANGE[0]) & (hue <= WATER_HUE_RANGE[1]) & (sat >= WATER_SAT_MIN) & (val >= WATER_VAL_MIN)
        )
        if cand.sum() < 0.02 * cand.size:
            return cand
        h = hue[cand].astype(np.int32)
        hist = np.bincount(h, weights=sat[cand].astype(np.float64), minlength=181)
        hist = np.convolve(hist, np.ones(5) / 5.0, mode="same")
        h0 = int(np.argmax(hist))
        near = cand & (np.abs(hue.astype(np.int32) - h0) <= WATER_HUE_BAND)
        if not near.any():
            return cand
        s_ref = float(np.median(sat[near]))
        return near & (sat >= max(WATER_SAT_MIN, WATER_SAT_REL * s_ref))

    def _lookup(self, point: tuple[float, float]) -> float | None:
        if not self.confident or self.dist is None:
            return None
        x = int(np.clip(point[0] / self.scale, 0, self.dist.shape[1] - 1))
        y = int(np.clip(point[1] / self.scale, 0, self.dist.shape[0] - 1))
        return float(self.dist[y, x]) * self.scale

    def contains(self, point: tuple[float, float], margin_px: float = 0.0) -> bool | None:
        """True inside the water (by more than margin), False outside, None = not confident."""
        d = self._lookup(point)
        if d is None:
            return None
        if d > margin_px:
            return True
        if d < -margin_px:
            return False
        return None

    def edge_distance(self, point: tuple[float, float]) -> float | None:
        return self._lookup(point)

    def roi(self, margin: float = 0.06) -> tuple[int, int, int, int] | None:
        if not self.confident:
            return None
        ys, xs = np.nonzero(self.mask)
        w, h = self.frame_wh
        mx, my = margin * w, margin * h
        return (
            int(max(0, xs.min() * self.scale - mx)),
            int(max(0, ys.min() * self.scale - my)),
            int(min(w, (xs.max() + 1) * self.scale + mx)),
            int(min(h, (ys.max() + 1) * self.scale + my)),
        )

    def overlay_polygons(self) -> list:
        """Water outline(s) in frame px for the Model View (empty when not confident)."""
        if not self.confident:
            return []
        cnts, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [(c.reshape(-1, 2) * self.scale).astype(int) for c in cnts]
