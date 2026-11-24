"""[2.18.0] Non-person filter: things the person detector calls a person that no person can be.

The owner's screen recordings of two phone uploads (Drowning Dataset/IMG_5495.mov, IMG_5491.mov, 2026-11-12):

* a FINGER over the lens became track #1 - "Normal Swimming", then a PASSIVE DROWNING alert at 100 % - while the real
  swimmer was missed;
* the lane-rope ANCHOR POSTS on the near pool wall (a grey post with two dark wings: a tiny person with arms out to a
  COCO-style detector) were tracked as swimmers for the whole clip.

Liveness (tracking/liveness.py) cannot help: it never runs in the water, an upload without a camera is all water, and
it measures movement from the first box position, which a hand-held phone breaks within a second.

Two rules, each built from signals measured on every track of 34 drowning / swimming clips with both detectors
(study 2026-11-12: every track either rule hides was checked by eye to be a non-person; a real face at the waterline,
a blurry child near the lens, still people on fixed cameras and all swimmers are kept):

OCCLUDER (finger, hand, strap over the lens): for NONPERSON_OCCLUDER_SECONDS the box touches the frame border on at
  least NONPERSON_OCCLUDER_BORDER_FRAC of samples, is at least NONPERSON_OCCLUDER_MIN_HEIGHT of the frame tall, and is
  out of focus - the 90th percentile of |Laplacian| inside the box is at most NONPERSON_OCCLUDER_MAX_SHARPNESS times
  the frame's median |Laplacian| (finger 1.4; nearest real person, a child touching the border, 3.0; swimmers 9-17).

FIXTURE (anchor posts, kit bags, floor logos): from its first sighting, for at least NONPERSON_FIXTURE_SECONDS, the
  track (a) never shows a body - the 90th percentile of confident keypoints is at most NONPERSON_FIXTURE_MAX_JOINTS;
  (b) never changes - median liveness activity (mean-removed 24 px grey crop vs ~1 s earlier, grey levels) at most
  NONPERSON_FIXTURE_MAX_ACTIVITY (fixtures 3.3-5.8; a face at the waterline 22, swimmers 23-25); and (c) never moves
  against the scene around it - median displacement relative to the background flow at most NONPERSON_FIXTURE_MAX_MOVE
  box heights per sample (a hand-held camera moves the fixture and its surroundings together).
  A track that EVER shows a body (NONPERSON_BODY_JOINTS confident keypoints) or sustained activity
  (NONPERSON_ACTIVE_LATCH) is a person for good: a swimmer who stops and floats is never hidden by this rule.

A hidden track is not reported (the scene forgets it: no SUBMERSION ghost for a post that "vanished"); the tracker
keeps it, so an occluder that moves off returns at once. NONPERSON_FILTER false: nothing is hidden.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from lifeguard.vision.constants import (
    ALERT_OCCLUDER_BLUR_SHARE,
    ALERT_OCCLUDER_GATE,
    ALERT_OCCLUDER_MIN_HEIGHT,
    ALERT_OCCLUDER_SLIVER_ASPECT,
    NONPERSON_ACTIVE_LATCH,
    NONPERSON_BODY_JOINTS,
    NONPERSON_DECK_STATIC_MAX_MOVE,
    NONPERSON_DECK_STATIC_SECONDS,
    NONPERSON_FILTER,
    NONPERSON_FIXTURE_MAX_ACTIVITY,
    NONPERSON_FIXTURE_MAX_JOINTS,
    NONPERSON_FIXTURE_MAX_MOVE,
    NONPERSON_FIXTURE_SECONDS,
    NONPERSON_GLARE_FRAC,
    NONPERSON_GLARE_SECONDS,
    NONPERSON_MAX_AREA_FRAC,
    NONPERSON_MAX_WIDTH_FRAC,
    NONPERSON_OCCLUDER_BORDER_FRAC,
    NONPERSON_OCCLUDER_MAX_SHARPNESS,
    NONPERSON_OCCLUDER_MIN_HEIGHT,
    NONPERSON_OCCLUDER_SECONDS,
    NONPERSON_SLIVER_ASPECT,
    NONPERSON_SLIVER_MIN_HEIGHT,
)

logger = logging.getLogger(__name__)

SAMPLE_S = 0.5  # features sampled twice a second (the study's cadence)
_ANALYSIS_SIDE = 480.0  # px, long side of the frame used for sharpness and flow
_CROP = 24  # px, liveness crop
_LAG_S = 0.95  # activity = change against the crop ~1 s earlier
_JOINT_CONF = 0.3
_BORDER = 0.01  # box edge within 1 % of the frame edge = touching it
_SHARP_FLOOR = 0.5  # floor of the frame's median |Laplacian| (a flat, letterboxed frame has ~0)
_LATCH_SAMPLES = 4
_GLARE_V, _GLARE_S = 215, 60  # HSV value floor / saturation ceiling of a specular highlight
_FORGET_S = 3.0  # state of an id not reported for this long is dropped (TRACK_STALE_SECONDS)
_BODY_MIN_FRAMES = 3  # body latch: at least this many frames with a body ...
_BODY_MIN_FRAC = 0.10  # ... and at least this share of the posed frames so far


@dataclass
class _Sample:
    t: float
    border: bool
    hfrac: float
    sharp: float
    joints: int | None
    activity: float | None
    move: float | None
    aspect: float = 1.0  # box height / width
    glare: float = 0.0  # [2.19.0] share of specular (bright, colourless) pixels in the box
    outside: bool = False  # [2.33.0] box entirely outside the camera's drawn water polygon


@dataclass
class _Track:
    born: float
    last_seen: float = 0.0
    posed: int = 0  # frames with a pose result
    body: int = 0  # ... of which showed NONPERSON_BODY_JOINTS confident keypoints
    samples: deque[_Sample] = field(default_factory=lambda: deque(maxlen=240))
    crops: deque[tuple[float, np.ndarray]] = field(default_factory=lambda: deque(maxlen=6))
    last_box: np.ndarray | None = None
    person: bool = False  # latched: showed a body or sustained activity
    body_seen: bool = False  # latched: showed a body (the reflection rule's only proof of a person)
    hidden: str | None = None  # "occluder" | "fixture" | "reflection" | "deck_static" | None


def _p90(values: list[int]) -> float:
    v = sorted(values)
    return float(v[int(0.9 * (len(v) - 1))]) if v else 0.0


class NonPersonFilter:
    """One per camera (lives in the detector next to the tracker)."""

    def __init__(self, camera_id: str = "") -> None:
        self.camera_id = camera_id
        self._tracks: dict[int, _Track] = {}
        self._last_sample = -math.inf
        self._prev_small: np.ndarray | None = None
        self._prev_t = -math.inf
        self.hidden_ids: set[int] = set()
        self.hidden_frames = 0  # frames x tracks hidden (status / tests)

    def reset(self) -> None:
        self._tracks.clear()
        self._prev_small, self._prev_t, self._last_sample = None, -math.inf, -math.inf
        self.hidden_ids = set()

    # ── per frame ─────────────────────────────────────────────────────────────
    def mask(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        ids: np.ndarray,
        joints: list[int | None],
        t: float,
        outside: list[bool] | None = None,
    ) -> np.ndarray:
        """Bool per box: True = not a person, do not report it. `joints[i]` = confident keypoints of box i on this
        frame (None: not posed this frame). `outside[i]` [2.33.0]: box i is entirely outside the drawn water polygon
        (None: no drawn zone, or the deck rule is off)."""
        n = len(boxes)
        out = np.zeros(n, bool)
        if not NONPERSON_FILTER or n == 0 or frame is None or frame.size == 0:
            self.hidden_ids = set()
            return out
        for i, tid in enumerate(ids):
            tr = self._tracks.get(int(tid))
            if tr is not None:
                tr.last_seen = t
                j = joints[i] if i < len(joints) else None
                if j is not None:
                    tr.posed += 1
                    if j >= NONPERSON_BODY_JOINTS:
                        tr.body += 1
                    # a body seen repeatedly (on any frames, not only sampled ones) makes it a person for good; one
                    # noisy frame of a post with 8 confident "joints" does not
                    if tr.body >= _BODY_MIN_FRAMES and tr.body >= _BODY_MIN_FRAC * tr.posed:
                        tr.person = tr.body_seen = True
        # a detector that misses an object on some frames must not restart its evidence: keep the state for a while
        for tid in [k for k, tr in self._tracks.items() if t - tr.last_seen > _FORGET_S]:
            del self._tracks[tid]
        if t - self._last_sample >= SAMPLE_S - 1e-6:
            self._sample(frame, boxes, ids, joints, t, outside)
            self._last_sample = t
        fh, fw = frame.shape[:2]
        hidden: set[int] = set()
        for i in range(n):
            tr = self._tracks.get(int(ids[i]))
            if (tr is not None and tr.hidden is not None) or _implausible(boxes[i], fw, fh):
                out[i] = True
                if int(ids[i]) >= 0:
                    hidden.add(int(ids[i]))
        self.hidden_ids = hidden
        self.hidden_frames += int(out.sum())
        return out

    def _sample(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        ids: np.ndarray,
        joints: list[int | None],
        t: float,
        outside: list[bool] | None = None,
    ) -> None:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        s = _ANALYSIS_SIDE / max(h, w)
        small = cv2.resize(gray, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        lap = np.abs(cv2.Laplacian(small, cv2.CV_32F))
        frame_sharp = max(float(np.median(lap)), _SHARP_FLOOR)
        flow = None
        if self._prev_small is not None and self._prev_small.shape == small.shape and t - self._prev_t <= 2.0:
            flow = cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload]  # the cv2 stubs reject flow=None
                self._prev_small, small, None, 0.5, 4, 21, 3, 5, 1.1, 0
            )
        for i in range(len(boxes)):
            tid = int(ids[i])
            if tid < 0:
                continue
            b = boxes[i]
            tr = self._tracks.get(tid)
            if tr is None:
                tr = self._tracks[tid] = _Track(born=t, last_seen=t)
            x1, y1, x2, y2 = (float(v) for v in b)
            bh = max(1.0, y2 - y1)
            border = x1 <= w * _BORDER or y1 <= h * _BORDER or x2 >= w * (1 - _BORDER) or y2 >= h * (1 - _BORDER)
            sx1, sy1 = int(max(0, x1 * s)), int(max(0, y1 * s))
            sx2, sy2 = max(int(x2 * s), sx1 + 2), max(int(y2 * s), sy1 + 2)
            patch = lap[sy1:sy2, sx1:sx2]
            sharp = float(np.percentile(patch, 90)) / frame_sharp if patch.size else math.inf
            activity = None
            crop = _crop(gray, b)
            if crop is not None:
                ref = next((c for (tc, c) in tr.crops if t - tc >= _LAG_S), None)
                if ref is not None:
                    activity = float(np.abs(crop - ref).mean())
                tr.crops.appendleft((t, crop))
            move = None
            if flow is not None and tr.last_box is not None:
                move = _relative_move(flow, tr.last_box, b, s)
            tr.last_box = np.asarray(b, np.float32)
            aspect = bh / max(1.0, x2 - x1)
            glare = _glare(frame, b)
            out = bool(outside[i]) if outside is not None and i < len(outside) else False
            tr.samples.append(_Sample(t, border, bh / h, sharp, joints[i], activity, move, aspect, glare, out))
            self._judge(tid, tr, t)
        self._prev_small, self._prev_t = small, t

    def _judge(self, tid: int, tr: _Track, t: float) -> None:
        recent = [x.activity for x in list(tr.samples)[-_LATCH_SAMPLES:] if x.activity is not None]
        if len(recent) >= 3 and statistics.median(recent) >= NONPERSON_ACTIVE_LATCH:
            tr.person = True  # sustained change: a person (the body latch is counted per frame in `mask`)
        was = tr.hidden
        tr.hidden = (
            "occluder"
            if self._occluder(tr, t)
            else "fixture"
            if self._fixture(tr, t)
            else "reflection"
            if self._reflection(tr, t)
            else "deck_static"
            if self._deck_static(tr, t)
            else None
        )
        if tr.hidden != was:
            logger.info(
                "[non-person:%s] track %d %s",
                self.camera_id,
                tid,
                f"hidden ({tr.hidden})" if tr.hidden else f"reported again (was {was})",
            )

    @staticmethod
    def _occluder(tr: _Track, t: float) -> bool:
        # a tall sliver pressed against the frame edge (a finger sliding in: 58-150 px wide, 1000 px tall at 4K) -
        # no person in a pool scene is this narrow; a person walking in from the side widens within a second
        recent = [x for x in tr.samples if t - x.t <= 1.0 + 1e-6]
        if len(recent) >= 2 and all(
            x.border and x.hfrac >= NONPERSON_SLIVER_MIN_HEIGHT and x.aspect >= NONPERSON_SLIVER_ASPECT for x in recent
        ):
            return True
        win = [x for x in tr.samples if t - x.t <= NONPERSON_OCCLUDER_SECONDS + 1e-6]
        if len(win) < 3 or t - win[0].t < NONPERSON_OCCLUDER_SECONDS - SAMPLE_S - 1e-6:
            return False
        return (
            sum(x.border for x in win) / len(win) >= NONPERSON_OCCLUDER_BORDER_FRAC
            and statistics.median(x.hfrac for x in win) >= NONPERSON_OCCLUDER_MIN_HEIGHT
            and statistics.median(x.sharp for x in win) <= NONPERSON_OCCLUDER_MAX_SHARPNESS
        )

    @staticmethod
    def _reflection(tr: _Track, t: float) -> bool:
        """[2.19.0] Light sparkling on the water (the owner's RTSP test: "water markings inside water markings", tall
        boxes on the columns of ceiling-light reflections): for NONPERSON_GLARE_SECONDS the median share of specular
        pixels in the box is at least NONPERSON_GLARE_FRAC (one such box measured 0.23; real swimmer boxes 0.001-0.03),
        and the track has never shown a body. [2.20.0] NONPERSON_GLARE_FRAC 0 = off."""
        if NONPERSON_GLARE_FRAC <= 0 or tr.body_seen:  # sparkle changes all the time: only a body proves a person here
            return False
        win = [x for x in tr.samples if t - x.t <= NONPERSON_GLARE_SECONDS + 1e-6]
        if len(win) < 3 or t - win[0].t < NONPERSON_GLARE_SECONDS - SAMPLE_S - 1e-6:
            return False
        return statistics.median(x.glare for x in win) >= NONPERSON_GLARE_FRAC

    @staticmethod
    def _deck_static(tr: _Track, t: float) -> bool:
        """[2.33.0] A thing standing on the deck (IMG_5511's sand-filter tank, Red Rescue clip 1's bag on the kickboard
        rack): for NONPERSON_DECK_STATIC_SECONDS every sampled box was entirely outside the camera's DRAWN water polygon
        and the 90th percentile of its background-relative movement was at most NONPERSON_DECK_STATIC_MAX_MOVE box
        heights per sample. No body / activity latch: RTMPose gives the tank a skeleton (8-12 confident joints) and a
        hand-held camera gives it activity, which is why the fixture rule never fired on it.

        It can never hide a person in the water: only boxes that never touch the drawn water count, and without a drawn
        zone `outside` is always False (the colour estimate is not trusted here). A person standing still on the deck,
        entirely outside the drawn water, is hidden the same way until they move (they are not scored for drowning
        there anyway); movement is not what separates them (measured: a still lifeguard 0.007, the bag 0.028)."""
        if NONPERSON_DECK_STATIC_SECONDS <= 0:
            return False
        win = [x for x in tr.samples if t - x.t <= NONPERSON_DECK_STATIC_SECONDS + 1e-6]
        if not win or t - win[0].t < NONPERSON_DECK_STATIC_SECONDS - SAMPLE_S - 1e-6 or not all(x.outside for x in win):
            return False
        move = sorted(x.move for x in win if x.move is not None)
        if len(move) < max(3, int(0.6 * NONPERSON_DECK_STATIC_SECONDS / SAMPLE_S)):
            return False
        return move[int(0.9 * (len(move) - 1))] <= NONPERSON_DECK_STATIC_MAX_MOVE

    @staticmethod
    def _fixture(tr: _Track, t: float) -> bool:
        if tr.person or t - tr.born < NONPERSON_FIXTURE_SECONDS - 1e-6:
            return False
        joints = [x.joints for x in tr.samples if x.joints is not None]
        act = [x.activity for x in tr.samples if x.activity is not None]
        move = [x.move for x in tr.samples if x.move is not None]
        if not joints or not act or not move:
            return False
        return (
            _p90(joints) <= NONPERSON_FIXTURE_MAX_JOINTS
            and statistics.median(act) <= NONPERSON_FIXTURE_MAX_ACTIVITY
            and statistics.median(move) <= NONPERSON_FIXTURE_MAX_MOVE
        )


def _glare(frame: np.ndarray, box: np.ndarray) -> float:
    """Share of specular (bright, colourless) pixels in the box, at full resolution (thumbnails blur sparkle)."""
    if frame.ndim != 3:
        return 0.0
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (round(float(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    crop = frame[y1:y2, x1:x2]
    step = max(1, int(max(crop.shape[:2]) // 160))  # at most ~160 px a side: cheap, sparkle still resolved
    hsv = cv2.cvtColor(np.ascontiguousarray(crop[::step, ::step]), cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 2] >= _GLARE_V) & (hsv[..., 1] <= _GLARE_S)).mean())


def alert_occluders(frame: np.ndarray | None, tracks: list[tuple[int, Any]]) -> set[int]:
    """[2.31.0] Alert-time lens-occluder gate (ALERT_OCCLUDER_GATE, off by default): the track ids whose box on THIS
    frame touches the frame border, is at least ALERT_OCCLUDER_MIN_HEIGHT of the frame tall and is either a sliver
    (height >= ALERT_OCCLUDER_SLIVER_ASPECT x width) or mostly out of focus (share of box pixels whose |Laplacian| is
    below the frame median >= ALERT_OCCLUDER_BLUR_SHARE). Such a track does not raise or re-raise an alert this round.

    Measured on the 4-camera live run (logs/p005_ab4*, docs/proposals/005): a finger over the lens merges with the water
    into boxes the track-level occluder rule never sees (finger alerts: blur share 0.47-0.69 or slivers; a real swimmer
    entering at the border 0.30, the busiest swimmer box 0.42). Submersion ghosts are never gated (scene_tracker)."""
    if not ALERT_OCCLUDER_GATE or frame is None or frame.size == 0 or not tracks:
        return set()
    h, w = frame.shape[:2]
    cands = []
    for tid, b in tracks:
        x1, y1, x2, y2 = (float(v) for v in b)
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        border = x1 <= w * _BORDER or y1 <= h * _BORDER or x2 >= w * (1 - _BORDER) or y2 >= h * (1 - _BORDER)
        if border and bh / h >= ALERT_OCCLUDER_MIN_HEIGHT:
            cands.append((int(tid), (x1, y1, x2, y2), bh / bw >= ALERT_OCCLUDER_SLIVER_ASPECT))
    if not cands:
        return set()
    out = {tid for tid, _, sliver in cands if sliver}
    rest = [(tid, b) for tid, b, sliver in cands if not sliver]
    if rest:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        s = _ANALYSIS_SIDE / max(h, w)
        small = cv2.resize(gray, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        lap = np.abs(cv2.Laplacian(small, cv2.CV_32F))
        fm = max(float(np.median(lap)), _SHARP_FLOOR)
        for tid, (x1, y1, x2, y2) in rest:
            sx1, sy1 = int(max(0, x1 * s)), int(max(0, y1 * s))
            patch = lap[sy1 : max(int(y2 * s), sy1 + 2), sx1 : max(int(x2 * s), sx1 + 2)]
            if patch.size and float((patch < fm).mean()) >= ALERT_OCCLUDER_BLUR_SHARE:
                out.add(tid)
    return out


def _implausible(box: np.ndarray, w: int, h: int) -> bool:
    """A box too big to be one person in a pool scene (RTMDet's 3757-px-wide "person" along the near wall of
    IMG_5495): hidden on the frame it appears, no history needed."""
    bw, bh = float(box[2] - box[0]), float(box[3] - box[1])
    return bw >= NONPERSON_MAX_WIDTH_FRAC * w or bw * bh >= NONPERSON_MAX_AREA_FRAC * w * h


def _crop(gray: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = (round(float(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    c = cv2.resize(gray[y1:y2, x1:x2], (_CROP, _CROP), interpolation=cv2.INTER_AREA).astype(np.float32)
    return c - float(c.mean())


def _relative_move(flow: np.ndarray, prev: np.ndarray, box: np.ndarray, s: float) -> float | None:
    """Box-centre displacement minus the median background flow in a ring around the box, in box heights."""
    fh, fw = flow.shape[:2]
    px1, py1, px2, py2 = (float(v) * s for v in prev)
    x1, y1, x2, y2 = (float(v) * s for v in box)
    dcx, dcy = ((x1 + x2) - (px1 + px2)) / 2.0, ((y1 + y2) - (py1 + py2)) / 2.0
    bx1, by1 = int(max(0, x1)), int(max(0, y1))
    bx2, by2 = int(min(fw, max(x2, x1 + 2))), int(min(fh, max(y2, y1 + 2)))
    pad = max(4, int(0.6 * max(bx2 - bx1, by2 - by1)))
    rx1, ry1, rx2, ry2 = max(0, bx1 - pad), max(0, by1 - pad), min(fw, bx2 + pad), min(fh, by2 + pad)
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    ring = np.ones((ry2 - ry1, rx2 - rx1), bool)
    ring[max(0, by1 - ry1) : max(0, by2 - ry1), max(0, bx1 - rx1) : max(0, bx2 - rx1)] = False
    if not ring.any():
        return None
    f = flow[ry1:ry2, rx1:rx2]
    bgx, bgy = float(np.median(f[..., 0][ring])), float(np.median(f[..., 1][ring]))
    return math.hypot(dcx - bgx, dcy - bgy) / max(1.0, y2 - y1)
