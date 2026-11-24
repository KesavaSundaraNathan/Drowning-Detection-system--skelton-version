"""Glare mask + glare hold (Phase 11 WP3; switch GLARE_HOLD, decided by eval/site/levers_phase11.json).

Glare - sun or floodlight reflections on the water - hides a swimmer for a moment. The tracker already keeps an id
through a short straight pass (Kalman re-find, ID stitching); it loses it when the person turns inside the glare or
comes out far from where they went in. The hold handles exactly that:

* ``GlareMask``: per camera, where the glare is. AUTO: pixels that stay saturated (V >= GLARE_AUTO_V_MIN) and
  colourless (S <= GLARE_AUTO_S_MAX), an exponential average over frames thresholded at GLARE_AUTO_PERSIST (a
  moving splash is not glare). OPERATOR: GLARE zones drawn on the camera. Both are merged on a 1/8-resolution grid,
  dilated by one cell and split into connected REGIONS.
* ``GlareHold``: a track lost inside a glare region is HELD for at most GLARE_HOLD_MAX_S. A new track born from that
  same region during the hold takes the held track's id - only when exactly ONE track is held there and nobody else
  was seen in the region since the loss. Otherwise no link: a fresh id, as without the hold.

Safety: the hold only renames a newly born track. The submersion watch of the lost track opens and counts from the
real loss time exactly as without the hold; a link resolves it just as the person resurfacing would. The case the
hold could make worse - someone ELSE coming out of the glare where a person went under - is refused by the
"nobody else in the region" rule (and is the registered G5 safety scenario).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from lifeguard.vision.constants import (
    GLARE_AUTO_EVERY_S,
    GLARE_AUTO_PERSIST,
    GLARE_AUTO_S_MAX,
    GLARE_AUTO_V_MIN,
    GLARE_HOLD,
    GLARE_HOLD_MAX_S,
)

SCALE = 8  # mask cell = 8 x 8 px
MIN_REGION_CELLS = 4
EMA_ALPHA = 0.2


class GlareMask:
    def __init__(self) -> None:
        self.frame_wh: tuple[int, int] | None = None
        self._ema: np.ndarray | None = None
        self._operator: list[np.ndarray] = []  # contours, frame px
        self._labels: np.ndarray | None = None
        self._next_t = -math.inf

    def set_operator_regions(self, contours: Sequence[np.ndarray]) -> None:
        self._operator = [np.asarray(c, np.int32).reshape(-1, 1, 2) for c in contours]
        self._rebuild()

    def due(self, t: float) -> bool:
        return t >= self._next_t

    def update(self, frame: np.ndarray, t: float) -> None:
        """Every GLARE_AUTO_EVERY_S of stream time: fold this frame into the auto mask."""
        if not self.due(t):
            return
        self._next_t = t + GLARE_AUTO_EVERY_S
        h, w = frame.shape[:2]
        if self.frame_wh != (w, h):
            self.frame_wh, self._ema = (w, h), None
        small = cv2.resize(frame, (max(1, w // SCALE), max(1, h // SCALE)), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hit = ((hsv[..., 2] >= GLARE_AUTO_V_MIN) & (hsv[..., 1] <= GLARE_AUTO_S_MAX)).astype(np.float32)
        prev = np.zeros_like(hit) if self._ema is None else self._ema  # from zero: one bright frame is not glare
        self._ema = EMA_ALPHA * hit + (1.0 - EMA_ALPHA) * prev
        self._rebuild()

    def _rebuild(self) -> None:
        if self.frame_wh is None:
            return
        w, h = self.frame_wh
        grid: np.ndarray = np.zeros((max(1, h // SCALE), max(1, w // SCALE)), np.uint8)
        if self._ema is not None and self._ema.shape == grid.shape:
            grid[self._ema >= GLARE_AUTO_PERSIST] = 1
        for c in self._operator:
            cv2.fillPoly(grid, [(c // SCALE).astype(np.int32)], 1)
        grid = cv2.dilate(grid, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
        for lab in range(1, n):
            if stats[lab, cv2.CC_STAT_AREA] < MIN_REGION_CELLS:
                labels[labels == lab] = 0
        self._labels = labels

    def region_at(self, point: Sequence[float]) -> int:
        if self._labels is None:
            return 0
        gx, gy = int(point[0]) // SCALE, int(point[1]) // SCALE
        if 0 <= gy < self._labels.shape[0] and 0 <= gx < self._labels.shape[1]:
            return int(self._labels[gy, gx])
        return 0

    def region_of_box(self, box: Sequence[float]) -> int:
        """The glare region a person box touches, widened by half the box width on every side (0 = none).

        A person walking INTO glare is last seen just before its edge, and first seen again just after it: their
        box centre is outside the glare both times, so "touches" - not "centre inside" - is what holds them."""
        if self._labels is None:
            return 0
        x1, y1, x2, y2 = (float(v) for v in box)
        pad = 0.5 * (x2 - x1)
        gh, gw = self._labels.shape
        gx1, gx2 = max(0, int((x1 - pad) // SCALE)), min(gw, int((x2 + pad) // SCALE) + 1)
        gy1, gy2 = max(0, int((y1 - pad) // SCALE)), min(gh, int((y2 + pad) // SCALE) + 1)
        if gx1 >= gx2 or gy1 >= gy2:
            return 0
        window = self._labels[gy1:gy2, gx1:gx2]
        hits = window[window > 0]
        return int(np.bincount(hits).argmax()) if hits.size else 0

    @property
    def regions(self) -> int:
        return 0 if self._labels is None else int(self._labels.max())


class GlareHold:
    def __init__(self, mask: GlareMask) -> None:
        self.mask = mask
        self.links = 0
        self.refused = 0
        self._alias: dict[int, int] = {}  # new public id -> held id
        self._seen: set[int] = set()  # raw public ids ever reported
        self._last: dict[int, tuple[float, int]] = {}  # id -> (last seen t, glare region there)
        self._held: dict[int, dict[str, Any]] = {}  # id -> {"t": loss t, "region": r, "crowded": bool}

    def reset(self) -> None:
        self._alias.clear()
        self._seen.clear()
        self._last.clear()
        self._held.clear()

    def apply(self, pub: np.ndarray, boxes: np.ndarray, t: float) -> np.ndarray:
        out = pub.copy()
        raw_now = {int(r) for r in pub.tolist() if r > 0}
        for new, old in list(self._alias.items()):  # the tracker found the original track again itself:
            if old in raw_now:  # never two boxes under one id - the newcomer keeps its own id
                del self._alias[new]
        here: dict[int, int] = {}  # id now (after aliasing) -> glare region of its box
        born: list[tuple[int, int, int]] = []  # (row, raw id, region)
        for k, raw in enumerate(pub.tolist()):
            if raw <= 0:
                continue
            region = self.mask.region_of_box(boxes[k])
            if raw not in self._seen and raw not in self._alias:
                born.append((k, raw, region))
            pid = self._alias.get(raw, raw)
            out[k] = pid
            here[pid] = region
        # someone (else) seen inside a region with a held track: that hold can no longer be told apart
        # newly lost inside a region -> held (from when they were last seen); expired holds released - BEFORE the
        # newborn check, so a person who re-emerges in the very frame after a gap meets their hold
        for pid, (t_last, region) in list(self._last.items()):
            if pid not in here:
                if region and pid not in self._held:
                    self._held[pid] = {"t": t_last, "region": region, "crowded": False}
                del self._last[pid]
        for pid in list(self._held):
            if t - self._held[pid]["t"] > GLARE_HOLD_MAX_S:
                del self._held[pid]
        newborn = {raw for _k, raw, _r in born}
        for pid, region in here.items():
            if pid in newborn:
                continue  # a track born this frame is a candidate, not a witness
            for hid, h in self._held.items():
                if hid != pid and region and region == h["region"]:
                    h["crowded"] = True
        # births from a glare region with exactly one clean hold take its id
        for k, raw, region in born:
            self._seen.add(raw)
            if not region:
                continue
            holds = [hid for hid, h in self._held.items() if h["region"] == region]
            others_here = [pid for pid, r in here.items() if r == region and pid != raw]
            if len(holds) == 1 and not self._held[holds[0]]["crowded"] and not others_here:
                hid = holds[0]
                self._alias[raw] = hid
                out[k] = hid
                here[hid] = here.pop(raw, region)
                del self._held[hid]
                self.links += 1
            elif holds:
                self.refused += 1
        for pid in list(self._held):  # the tracker found a held track again itself: no longer held
            if pid in here:
                del self._held[pid]
        for pid, region in here.items():
            self._last[pid] = (t, region)
        if len(self._seen) > 8192:
            self._seen = set(here) | set(self._alias)
        return out


def prepare(pipe: Any, frame: np.ndarray, t: float) -> None:
    """Per analysed frame, before detection (vision/pipeline.py prepare): keep the camera's glare mask current and
    give the detector's (lazily created) PersonTracker its hold. GLARE_HOLD off: nothing, no hold attached."""
    tracker = getattr(getattr(pipe, "detector", None), "_tracker", None)
    if not GLARE_HOLD:
        if tracker is not None and getattr(tracker, "glare_hold", None) is not None:
            tracker.glare_hold = None
        return
    mask: GlareMask | None = getattr(pipe, "glare_mask", None)
    if mask is None:
        mask = pipe.glare_mask = GlareMask()
    geometry = getattr(getattr(pipe, "scene", None), "geometry", None)
    operator = getattr(geometry, "glare_polygons", None) or []
    if getattr(pipe, "_glare_operator_src", None) is not operator:
        pipe._glare_operator_src = operator
        if mask.frame_wh is None:
            mask.frame_wh = (int(frame.shape[1]), int(frame.shape[0]))
        mask.set_operator_regions(operator)
    mask.update(frame, t)
    if tracker is not None and getattr(tracker, "glare_hold", None) is None:
        tracker.glare_hold = GlareHold(mask)


def reset(pipe: Any) -> None:
    """Stream reconnect (tracker ids restart): no held id may survive it."""
    tracker = getattr(getattr(pipe, "detector", None), "_tracker", None)
    hold = getattr(tracker, "glare_hold", None)
    if hold is not None:
        hold.reset()


def apply_hold(tracker: Any, pub: np.ndarray, boxes: np.ndarray, t: float) -> np.ndarray:
    """PersonTracker.update's last step: public ids through the glare hold (off / no hold: unchanged)."""
    hold = getattr(tracker, "glare_hold", None)
    if not GLARE_HOLD or hold is None:
        return pub
    return hold.apply(pub, boxes, t)  # also on empty frames: that is when a person is hidden
