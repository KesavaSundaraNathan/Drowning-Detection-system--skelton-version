"""[v7 F3] Liveness: tell a real person from a person-shaped object (wall photo, poster, slide, switched-off monitor).

The COCO detector scores a lifelike printed face or a red slide as a "person" (0.36 on the Reception wall photo,
a PASSIVE alert on a deck slide). A real person's box always shows some pixel change over seconds (breathing, hands,
head, posture); a printed photo or an object shows only sensor / compression noise. Per tracked box, twice a second:

* a small grey crop of the box, mean-removed (global light changes and auto-exposure do not count as activity),
  is compared with the crop from ~1 s earlier -> mean absolute difference = "activity" (grey levels);
* the box centre's movement since the track was first seen, in box heights.

A track is STATIC when, over LIVENESS_SECONDS, activity never exceeded LIVENESS_MIN_ACTIVITY and the box never
moved more than LIVENESS_MAX_MOVE box heights. It is checked only where it cannot hide a swimmer: anywhere on a
standard CCTV camera, and on a pool camera only while the person's feet are confidently OUT of the water (zones or
the water estimate). In the water nothing is ever static - a motionless floating victim stays a person.

A static track is not reported (and not posed); the tracker keeps it, so the first activity or movement brings it
back at once. LIVENESS_CHECK off: nothing is ever static.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from lifeguard.vision.constants import (
    LIVENESS_CHECK,
    LIVENESS_LOG_STATS,
    LIVENESS_MAX_MOVE,
    LIVENESS_MIN_ACTIVITY,
    LIVENESS_SECONDS,
)

logger = logging.getLogger(__name__)

_SAMPLE_S = 0.5  # crops kept twice a second
_LAG_S = 1.0  # activity = change against the crop ~1 s earlier
_CROP = 24  # px, square grey crop
_STATS_EVERY_S = 30.0

WaterFn = Callable[[tuple[float, float]], bool | None]


@dataclass
class _Track:
    first_center: tuple[float, float]
    height: float
    crops: deque[tuple[float, np.ndarray]] = field(default_factory=lambda: deque(maxlen=8))
    activity: deque[tuple[float, float]] = field(default_factory=deque)  # (t, activity)
    max_move: float = 0.0
    since: float = 0.0  # start of the current dry, unmoving stretch
    last_sample: float = -1e9
    static: bool = False


def _crop(gray: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = (round(float(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    c = cv2.resize(gray[y1:y2, x1:x2], (_CROP, _CROP), interpolation=cv2.INTER_AREA).astype(np.float32)
    return c - float(c.mean())


class Liveness:
    """One per camera (lives in the detector next to the tracker)."""

    def __init__(self, camera_id: str = "") -> None:
        self.camera_id = camera_id
        self._tracks: dict[int, _Track] = {}
        self._last_stats = -1e9
        self.suppressed = 0  # frames x tracks hidden as static (status / tests)

    def reset(self) -> None:
        self._tracks.clear()

    def static_mask(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        ids: np.ndarray,
        t: float,
        surveillance: bool,
        water_test: WaterFn | None,
    ) -> np.ndarray:
        """Bool per box: True = a static object, not to be reported this frame."""
        n = len(boxes)
        out = np.zeros(n, bool)
        if not LIVENESS_CHECK or n == 0:
            return out
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        seen: set[int] = set()
        for i in range(n):
            tid = int(ids[i])
            if tid < 0:
                continue
            seen.add(tid)
            b = boxes[i]
            cx, cy, bh = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, max(1.0, float(b[3] - b[1]))
            tr = self._tracks.get(tid)
            if tr is None:
                tr = self._tracks[tid] = _Track((cx, cy), bh, since=t)
            dry = surveillance or (water_test is not None and water_test((cx, float(b[3]))) is False)
            move = float(np.hypot(cx - tr.first_center[0], cy - tr.first_center[1])) / max(tr.height, bh)
            tr.max_move = max(tr.max_move, move)
            if not dry or tr.max_move > LIVENESS_MAX_MOVE:
                # in (or maybe in) the water, or it moved: a person; restart the stretch from here
                tr.first_center, tr.height, tr.max_move, tr.since = (cx, cy), bh, 0.0, t
                tr.activity.clear()
                tr.static = False
                continue
            if t - tr.last_sample >= _SAMPLE_S:
                c = _crop(gray, b)
                if c is not None:
                    ref = next((cc for (tc, cc) in tr.crops if t - tc >= _LAG_S), None)
                    if ref is not None:
                        tr.activity.append((t, float(np.abs(c - ref).mean())))
                    tr.crops.appendleft((t, c))
                    tr.last_sample = t
            while tr.activity and t - tr.activity[0][0] > LIVENESS_SECONDS:
                tr.activity.popleft()
            peak = max((a for _, a in tr.activity), default=float("inf"))
            last_active = max((ta for ta, a in tr.activity if a > LIVENESS_MIN_ACTIVITY), default=None)
            if last_active is not None:
                tr.since = max(tr.since, last_active)  # the static window starts after the last active sample
            was = tr.static
            tr.static = t - tr.since >= LIVENESS_SECONDS and bool(tr.activity) and peak <= LIVENESS_MIN_ACTIVITY
            if tr.static != was:
                logger.info(
                    "[liveness:%s] track %d %s (peak activity %.2f over %.0f s, moved %.2f box heights)",
                    self.camera_id,
                    tid,
                    "classed a static object" if tr.static else "is active again",
                    peak if peak != float("inf") else -1.0,
                    LIVENESS_SECONDS,
                    tr.max_move,
                )
            out[i] = tr.static
        for tid in [k for k in self._tracks if k not in seen]:
            del self._tracks[tid]
        self.suppressed += int(out.sum())
        if LIVENESS_LOG_STATS and t - self._last_stats >= _STATS_EVERY_S:
            self._last_stats = t
            for tid, tr in sorted(self._tracks.items()):
                acts = [a for _, a in tr.activity]
                if acts:
                    logger.info(
                        "[liveness:%s] stats track %d: activity p50 %.2f max %.2f, moved %.2f, static %s, box %s",
                        self.camera_id,
                        tid,
                        float(np.median(acts)),
                        max(acts),
                        tr.max_move,
                        tr.static,
                        [int(v) for v in boxes[list(ids).index(tid)]] if tid in list(ids) else None,
                    )
        return out
