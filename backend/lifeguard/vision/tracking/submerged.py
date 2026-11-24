"""[2.40.0, on since 2.41.0] SUBMERGED_FOLLOW: follow a swimmer the detector lost under water, so they keep their id.

The COCO detector returns no box at all on a body under clear water (docs/evidence/submerged/README.md: IMG_5506,
0 boxes >= 0.02 on 20 of 20 submerged frames) although the body is plainly visible: a dark shape on the light pool
floor. This follower is image-only (CPU, no detector call), at BG_WIDTH px:

- background: a slow average of the max-channel image (SUBMERGED_FOLLOW_BG_ALPHA per frame), never updated inside a
  shown or followed box, and RE-REGISTERED to every frame (ORB + RANSAC similarity, frame to frame) so a handheld or
  swaying camera does not turn the whole scene into "change". Kept only while a swimmer is in the water or followed;
- body cue: a pixel is body when it is darker than the registered background (it was not there before) AND darker
  than the window's 20th-percentile water (it is not just a patch of deep water next to window glare - reflections
  move with the view and never register), both by SUBMERGED_FOLLOW_DARK_DELTA grey levels;
- start: a confirmed public id seen for at least SUBMERGED_FOLLOW_MIN_TRACK_SECONDS, whose last box was in the water,
  drops out of the tracker (the stitcher holds it as lost), no box shown this frame lies on it and no other confirmed
  person is within SUBMERGED_FOLLOW_ISOLATION body sizes (a crowd occlusion is not a submersion);
- confirm: a follow moves the stitcher's record and is shown only once it has held the body for
  SUBMERGED_FOLLOW_CONFIRM_SECONDS, so a short loss (an occlusion) stitches exactly as without the follower;
- step: the box is predicted (camera motion + the swimmer's decaying velocity), the body blobs that overlap it (thin
  blobs - ropes, line edges - and stripes crossing the whole window never count) give the new box, smoothed;
- stop: no blob for SUBMERGED_FOLLOW_MISS_SECONDS, SUBMERGED_FOLLOW_MAX_SECONDS reached, the id is reported again,
  or a confirmed box lies on the follower's box (the person is visible again - under their own id when the stitch
  took it, which is the point).

While a follow runs, the stitcher's lost record of that id moves with it (`IdStitcher.follow`): a swimmer who comes
up away from where they went under is stitched back to their id. The followed box is DISPLAY and IDENTITY only: it
is listed with the coasting boxes (dashed, never scored, never alert evidence); the scene still treats the id as
lost, so the submersion monitor raises its alert exactly as before (SUBMERSION is never weakened).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[float, float, float, float]

BG_WIDTH = 640  # the background and the body search run at this width
_VEL_DECAY_S = 0.7  # a swimmer who goes under keeps drifting the way they swam, slowing with this time constant


@dataclass
class FollowConfig:
    dark_delta: float = 15.0
    max_seconds: float = 60.0
    min_track_seconds: float = 1.0
    miss_seconds: float = 0.7
    search: float = 1.0  # window half-extent beyond the box, in box sizes (max side)
    bg_alpha: float = 0.03
    confirm_seconds: float = 1.0  # held this long before it moves the stitcher's record or is shown
    isolation: float = 2.0  # no other confirmed person within this many body sizes when it starts (0 = off)


@dataclass
class _Seen:
    first_t: float
    last_t: float
    box: Box
    velocity: tuple[float, float] = (0.0, 0.0)  # px/s, smoothed


@dataclass
class _Follow:
    pid: int
    started: float
    box: Box
    last_hit: float
    area0: float
    velocity: tuple[float, float] = (0.0, 0.0)
    last_t: float = 0.0


def _area(b: Box) -> float:
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = _area(a) + _area(b) - inter
    return inter / u if u > 0 else 0.0


def _ioa_small(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    m = min(_area(a), _area(b))
    return ix * iy / m if m > 0 else 0.0


class RegisteredBackground:
    """A slow average of the scene at BG_WIDTH px, warped to every new frame (camera motion compensated)."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.bg: np.ndarray | None = None
        self.v: np.ndarray | None = None  # this frame's max-channel image (BG_WIDTH)
        self.k = 1.0  # BG_WIDTH / frame width
        self._prev: tuple[np.ndarray, list[cv2.KeyPoint], np.ndarray | None] | None = None
        self._orb = cv2.ORB_create(300)  # type: ignore[attr-defined]
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.frames = 0

    def reset(self) -> None:
        self.bg = self.v = None
        self._prev = None
        self.frames = 0

    def observe(self, frame: np.ndarray) -> np.ndarray | None:
        """Take a new frame: returns the 2x3 frame-to-frame motion (frame px) or None; warps the background."""
        h, w = frame.shape[:2]
        self.k = min(1.0, BG_WIDTH / float(w))
        size = (max(8, round(w * self.k)), max(8, round(h * self.k)))
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if self.k < 1.0 else frame
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        g = cv2.resize(g, (size[0] // 2, size[1] // 2), interpolation=cv2.INTER_AREA)  # registration at half size
        v = small.max(axis=2) if small.ndim == 3 else small
        self.v = cv2.GaussianBlur(v.astype(np.float32), (5, 5), 0)
        kp, des = self._orb.detectAndCompute(g, None)
        motion = None
        if self._prev is not None and self.bg is not None and self.bg.shape == self.v.shape:
            _, pkp, pdes = self._prev
            m = None
            if pdes is not None and des is not None and len(pdes) >= 10 and len(des) >= 10:
                matches = self._bf.match(pdes, des)
                if len(matches) >= 10:
                    p1 = np.asarray([pkp[x.queryIdx].pt for x in matches], np.float32)
                    p2 = np.asarray([kp[x.trainIdx].pt for x in matches], np.float32)
                    m, _inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=1.0)
                    if m is not None:
                        m = np.asarray(m, np.float64)
                        m[:, 2] = m[:, 2] * 2.0  # half-size translation -> background px
            if m is not None:
                self.bg = cv2.warpAffine(self.bg, m, (size[0], size[1]), borderMode=cv2.BORDER_REPLICATE)
                motion = np.asarray(m, np.float64).copy()
                motion[:, 2] = motion[:, 2] / self.k  # translation in frame px (rotation / scale: size independent)
        elif self.bg is None or self.bg.shape != self.v.shape:
            self.bg = self.v.copy()
        self._prev = (g, kp, des)
        self.frames += 1
        return motion

    def update(self, exclude: Iterable[Box]) -> None:
        """Blend this frame into the background except inside `exclude` (frame px, padded)."""
        if self.bg is None or self.v is None:
            return
        keep = np.ones(self.v.shape, bool)
        hh, ww = keep.shape
        for b in exclude:
            pad = 0.15 * max(b[2] - b[0], b[3] - b[1]) * self.k + 3
            x1, y1 = int(max(0, b[0] * self.k - pad)), int(max(0, b[1] * self.k - pad))
            x2, y2 = int(min(ww, b[2] * self.k + pad)), int(min(hh, b[3] * self.k + pad))
            keep[y1:y2, x1:x2] = False
        a = self.alpha
        self.bg[keep] = (1.0 - a) * self.bg[keep] + a * self.v[keep]


def find_body(bg: np.ndarray, v: np.ndarray, k: float, box: Box, cfg: FollowConfig) -> Box | None:
    """The body blob(s) on `box` (frame px) given the registered background `bg` and the frame `v` (both BG_WIDTH px,
    max channel, k = BG_WIDTH / frame width), or None. Pure function (unit-tested)."""
    h, w = v.shape[:2]
    b = (box[0] * k, box[1] * k, box[2] * k, box[3] * k)
    side = max(b[2] - b[0], b[3] - b[1])
    if side <= 1:
        return None
    m = cfg.search * side
    x0, y0 = int(max(0, b[0] - m)), int(max(0, b[1] - m))
    x1, y1 = int(min(w, b[2] + m)), int(min(h, b[3] + m))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    vw, bw = v[y0:y1, x0:x1], bg[y0:y1, x0:x1]
    water = float(np.percentile(vw, 20))
    dark = np.minimum(bw - vw, water - vw)  # darker than before AND darker than this window's deep water
    mask = (dark > cfg.dark_delta).astype(np.uint8)
    mask = np.asarray(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)), np.uint8)
    mask = np.asarray(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), np.uint8)
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ch, cw = mask.shape
    pred = (b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0)
    grow = 0.5 * side
    zone = (pred[0] - grow, pred[1] - grow, pred[2] + grow, pred[3] + grow)
    min_area = max(4.0, 0.05 * _area(pred))
    picked: list[tuple[Box, float]] = []
    for i in range(1, n):
        cx, cy, cw_i, ch_i, area = (int(s) for s in stats[i])
        if area < min_area:
            continue
        if (cx <= 0 and cx + cw_i >= cw) or (cy <= 0 and cy + ch_i >= ch):
            continue  # a stripe through the whole window: lane line / pool edge, not a body
        if area / max(cw_i, ch_i, 1) < max(1.5, 0.08 * side):
            continue  # thin: a rope, a line's edge, a ripple crest - a body is thick
        cb = (float(cx), float(cy), float(cx + cw_i), float(cy + ch_i))
        if _ioa_small(cb, zone) > 0.3:
            picked.append((cb, float(area) * (1.0 + _iou(cb, pred))))
    if not picked:
        return None
    # the main blob (big and on the prediction) plus only the blobs right next to it: a body under ripples breaks
    # into pieces, but the rope / line fragments further along must not stretch the box
    main = max(picked, key=lambda x: x[1])[0]
    gap = 0.15 * side
    near = (main[0] - gap, main[1] - gap, main[2] + gap, main[3] + gap)
    parts = [cb for cb, _ in picked if cb is main or _ioa_small(cb, near) > 0.5]
    u = (min(p[0] for p in parts), min(p[1] for p in parts), max(p[2] for p in parts), max(p[3] for p in parts))
    # the box is fitted to the strongly dark pixels: the body is much darker than the faint band of deep water or
    # rope shadow it can touch (2nd-98th percentile extents, so a few stray pixels do not stretch it)
    ux0, uy0, ux1, uy1 = (round(v) for v in u)
    sub = np.where(mask[uy0:uy1, ux0:ux1] > 0, dark[uy0:uy1, ux0:ux1], 0.0)
    vals = sub[sub > 0]
    if vals.size >= 4:
        strong = sub >= max(cfg.dark_delta, 0.5 * float(np.percentile(vals, 95)))
        ys, xs = np.nonzero(strong)
        if xs.size >= 4:
            u = (
                ux0 + float(np.percentile(xs, 2)),
                uy0 + float(np.percentile(ys, 2)),
                ux0 + float(np.percentile(xs, 98)) + 1.0,
                uy0 + float(np.percentile(ys, 98)) + 1.0,
            )
    return ((u[0] + x0) / k, (u[1] + y0) / k, (u[2] + x0) / k, (u[3] + y0) / k)


def _warp_box(b: Box, m: np.ndarray) -> Box:
    cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    nx = m[0, 0] * cx + m[0, 1] * cy + m[0, 2]
    ny = m[1, 0] * cx + m[1, 1] * cy + m[1, 2]
    return (b[0] + nx - cx, b[1] + ny - cy, b[2] + nx - cx, b[3] + ny - cy)


class SubmergedFollower:
    """Per camera: which lost ids are followed and where. Driven once per analysed frame by the detector."""

    def __init__(self, cfg: FollowConfig | None = None) -> None:
        self.cfg = cfg or FollowConfig()
        self.seen: dict[int, _Seen] = {}
        self.follows: dict[int, _Follow] = {}
        self.background = RegisteredBackground(self.cfg.bg_alpha)
        self.started = 0
        self.prev_t: float | None = None

    @staticmethod
    def _crowded(box: Box, shown: dict[int, Box], factor: float) -> bool:
        side = max(box[2] - box[0], box[3] - box[1], 1.0)
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        for b in shown.values():
            reach = factor * max(side, b[2] - b[0], b[3] - b[1])
            if abs((b[0] + b[2]) / 2.0 - cx) <= reach and abs((b[1] + b[3]) / 2.0 - cy) <= reach:
                return True
        return False

    def reset(self) -> None:
        self.seen.clear()
        self.follows.clear()
        self.background.reset()
        self.prev_t = None

    def step(
        self,
        frame: np.ndarray,
        t: float,
        shown: dict[int, Box],
        lost_ids: set[int],
        in_water: Callable[[tuple[float, float]], bool | None] | None,
        hidden: set[int] | None = None,
    ) -> list[tuple[int, Box, float]]:
        """`shown` = confirmed public id -> box reported this frame; `lost_ids` = ids the stitcher holds as lost.
        Returns (id, followed box, seconds since the follow began) for every follow running for at least
        `confirm_seconds` (younger follows are tracked but not reported: a short loss stitches as before)."""
        cfg = self.cfg
        hidden = hidden or set()

        def wet(b: Box) -> bool:
            return in_water is not None and in_water(((b[0] + b[2]) / 2.0, b[3])) is True

        # ids seen on the previous frame and gone now: candidates to follow
        if self.prev_t is not None:
            for pid, s in list(self.seen.items()):
                if pid in shown or pid in self.follows or s.last_t < self.prev_t - 1e-6:
                    continue
                if pid not in lost_ids or pid in hidden or s.last_t - s.first_t < cfg.min_track_seconds:
                    continue
                if not wet(s.box):
                    continue
                if any(_ioa_small(s.box, b) >= 0.6 for b in shown.values()):
                    continue  # the body is visible under another id (a duplicate box): nothing went under
                if cfg.isolation > 0 and self._crowded(s.box, shown, cfg.isolation):
                    continue  # someone else right there: a crowd occlusion, and the dark-blob cue is ambiguous
                self.follows[pid] = _Follow(pid, t, s.box, t, max(_area(s.box), 1.0), s.velocity, s.last_t)
                self.started += 1
        for pid, b in shown.items():
            prev = self.seen.get(pid)
            if prev is None:
                self.seen[pid] = _Seen(t, t, b)
            else:
                dt = t - prev.last_t
                if 0 < dt <= 1.0:
                    vx = ((b[0] + b[2]) - (prev.box[0] + prev.box[2])) / 2.0 / dt
                    vy = ((b[1] + b[3]) - (prev.box[1] + prev.box[3])) / 2.0 / dt
                    prev.velocity = (0.7 * prev.velocity[0] + 0.3 * vx, 0.7 * prev.velocity[1] + 0.3 * vy)
                prev.last_t, prev.box = t, b
        for pid in [p for p, s in self.seen.items() if t - s.last_t > cfg.max_seconds]:
            del self.seen[pid]
        for pid, f in list(self.follows.items()):
            if pid in shown or pid not in lost_ids or t - f.started > cfg.max_seconds:
                del self.follows[pid]
            elif any(_ioa_small(f.box, b) >= 0.3 for b in shown.values()):
                del self.follows[pid]  # someone confirmed is on it: the person is visible again

        # the background runs only while it can matter: a swimmer in the water, or a follow
        bgm = self.background
        if not self.follows and not any(wet(b) for b in shown.values()):
            if bgm.bg is not None:
                bgm.reset()
            self.prev_t = t
            return []
        motion = bgm.observe(frame)
        out: list[tuple[int, Box, float]] = []
        warm = bgm.frames > 3
        for pid, f in list(self.follows.items()):
            if motion is not None:
                f.box = _warp_box(f.box, motion)
            dt = max(t - f.last_t, 0.0)
            f.last_t = t
            decay = float(np.exp(-dt / _VEL_DECAY_S))
            f.velocity = (f.velocity[0] * decay, f.velocity[1] * decay)
            dx, dy = f.velocity[0] * dt, f.velocity[1] * dt
            f.box = (f.box[0] + dx, f.box[1] + dy, f.box[2] + dx, f.box[3] + dy)  # predicted
            nb = None
            if warm and bgm.bg is not None and bgm.v is not None:
                nb = find_body(bgm.bg, bgm.v, bgm.k, f.box, cfg)
            if nb is not None and 0.2 * f.area0 <= _area(nb) <= 8.0 * f.area0 and _iou(nb, f.box) > 0.0:
                a = 0.5
                f.box = (
                    a * nb[0] + (1 - a) * f.box[0],
                    a * nb[1] + (1 - a) * f.box[1],
                    a * nb[2] + (1 - a) * f.box[2],
                    a * nb[3] + (1 - a) * f.box[3],
                )
                f.last_hit = t
            elif t - f.last_hit > cfg.miss_seconds:
                del self.follows[pid]
                continue
            if t - f.started >= cfg.confirm_seconds:
                out.append((pid, f.box, t - f.started))
        bgm.update([*shown.values(), *(f.box for f in self.follows.values())])
        self.prev_t = t
        return out


__all__ = ["BG_WIDTH", "FollowConfig", "RegisteredBackground", "SubmergedFollower", "find_body"]
