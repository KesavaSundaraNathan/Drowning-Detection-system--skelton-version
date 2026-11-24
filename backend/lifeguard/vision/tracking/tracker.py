# mypy: disable-error-code="arg-type, assignment, attr-defined, type-var"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""
[v3] Ported from the v2 baseline detection/tracking.py (verbatim except: IdStitcher now behind the
ID_STITCHING switch of thresholds.json, default on = v2 behaviour; FaintBirths stays behind
TRACK_FAINT_BIRTHS. Plan: each must win on the held-out split to stay on (Phase 7)).

detection/tracking.py — person tracking for pool CCTV (Phase 1.3 / 1.4, tracking v3).

ByteTrack (in-repo port, detection/bytetrack — MIT; replaces the deprecated
supervision.ByteTrack with identical semantics) configured for what the
pipeline really does:

  * **Frame rate = the analysed frame rate.** Every frame is analysed now, so
    ByteTrack runs at the source fps. Its lost-track buffer is set in
    SECONDS (TRACK_LOST_SECONDS), not frames, so a swimmer who ducks under
    for a few seconds keeps their Kalman track at any fps.
  * **Low-score boxes are kept.** The detector returns everything down to
    DET_LOW_SCORE_THRESHOLD. Boxes >= TRACK_HIGH_SCORE_THRESHOLD go to the
    first association; the rest go to ByteTrack's second association, which
    can only EXTEND an existing track, never start one. A new track needs
    TRACK_HIGH_SCORE_THRESHOLD + 0.1 twice in a row (ByteTrack's own rule);
    after that, faint in-water frames, splashes and partial occlusion keep it.
    A third association (tracking v3) lets a faint box RE-FIND a recently
    lost track that it overlaps (TRACK_REFIND_LOW_IOU) — a swimmer under
    murky water keeps their track instead of coming back under a new id.
  * **ID stitching.** When a person resurfaces too far from their Kalman
    prediction, ByteTrack starts a new track. `IdStitcher` hands the new
    track the id of a recently lost person when it is born close to where
    they were lost (radius grows with time, like the submersion ghost logic,
    and is measured in the SMALLER box's size) and has a similar size.
  * **Shadows (tracking v3).** A track born on top of somebody who is already
    tracked is the detector's second box on the same body (a tight torso box
    and a loose full-body box), or the same person's box changing shape when
    they stand up. It is not reported; if the person's own track drops while
    the shadow is there, the shadow inherits the person's id. Before this,
    the original id "vanished in the water" (→ a false CRITICAL BLACKOUT
    ghost) while the same person carried on under a new id.
  * **Faint births (tracking v3).** Faint boxes never start a track on score
    alone — but a submerged or splashing child is exactly what the COCO-only
    detector scores at 0.10–0.35. `FaintBirths` lets a CHAIN of faint boxes
    start a track when it is persistent, in the water, and the pixels agree
    (detection/motion.py: it moves, it has something solid in it, it is not
    just foam). Static submerged objects (drain, toy on the floor, the slide)
    never move and never qualify.

Public ids are what the rest of the system sees; -1 = not reported (shadow).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from lifeguard.vision.constants import (
    DET_ZOOM_LOST_SECONDS,
    DET_ZOOM_MAX_REGIONS,
    FAINT_BIRTH_MAX_SPLASH,
    FAINT_BIRTH_MIN_BODY,
    FAINT_BIRTH_MIN_HIT_FRAC,
    FAINT_BIRTH_MIN_MEAN,
    FAINT_BIRTH_MIN_MOTION,
    FAINT_BIRTH_MIN_SCORE,
    FAINT_BIRTH_WINDOW_SECONDS,
    ID_LINK_COLOUR_MAX,
    ID_LINK_MOTION_MAX,
    ID_LINK_NESTED,
    ID_LINK_NESTED_FRAC,
    ID_LINK_NESTED_IOA,
    ID_LINK_NESTED_MIN_SAMPLES,
    ID_LINK_NESTED_SECONDS,
    ID_LINK_REQUIRE_SKELETON,
    ID_LINK_SKELETON_SEP,
    ID_LINK_STITCH_UNIQUE,
    ID_STITCHING,
    KEYPOINT_CONF_THRESHOLD,
    SHADOW_IOA,
    SHADOW_MAX_SECONDS,
    SHADOW_MERGE,
    SHADOW_MERGE_SECONDS,
    SHADOW_NESTED_IOA,
    STITCH_BLOCK_IOA,
    STITCH_MAX_GAP_SECONDS,
    STITCH_RADIUS_BASE,
    STITCH_RADIUS_MAX,
    STITCH_RADIUS_PER_SECOND,
    STITCH_RECENT_SECONDS,
    STITCH_SIZE_RATIO_MAX,
    STITCH_SIZE_RATIO_RECENT,
    TRACK_ASSOC_COST,
    TRACK_ASSOC_DIST_RADIUS,
    TRACK_BIRTH_SCORE_MARGIN,
    TRACK_HIGH_SCORE_THRESHOLD,
    TRACK_KF_STD_POSITION,
    TRACK_KF_STD_VELOCITY,
    TRACK_LOST_SECONDS,
    TRACK_LOST_SECONDS_WATER,
    TRACK_LOW_SCORE_THRESHOLD,
    TRACK_MATCH_IOU,
    TRACK_REFIND_LOW_IOU,
    TRACK_REFIND_MAX_SECONDS,
    TRACK_SECOND_MATCH_IOU,
    TRACK_SECOND_MATCH_LAST_SEEN,
    TRACK_UNCONFIRMED_MATCH_IOU,
)
from lifeguard.vision.tracking import glare  # [v3 Phase 11] GLARE_HOLD

logger = logging.getLogger(__name__)

Box = Sequence[float]
EvidenceFn = Callable[[Box], object | None]  # → motion.BoxEvidence | None
WaterFn = Callable[[tuple[float, float]], bool | None]


def _size(b: Box) -> float:
    return math.sqrt(max((b[2] - b[0]) * (b[3] - b[1]), 1.0))


def _center(b: Box) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def ioa_small(a: Box, b: Box) -> float:
    """Intersection over the SMALLER box's area (1.0 = one box lies inside the other)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(small, 1e-9)


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _same_frame_shadows(ks: Sequence[int], boxes: np.ndarray) -> dict[int, int]:
    """Tracks born on the SAME frame that sit on each other (IoA-small >= SHADOW_IOA): {shadow k: host k}.

    The shadow rule above only sees people tracked on earlier frames, so a body box and its head / torso
    box confirmed together (a seated person at start-up or after a reconnect) both became people for good.
    The larger box is the person, the box inside it the shadow - unless the larger box holds two boxes that do
    not overlap each other and together fill at least half of it (two people in one loose box, not a head and
    an arm): then it is the shadow of its largest child and the children stay people. A shadow can still hand
    over or be released (IdStitcher.update)."""
    order = sorted(ks, key=lambda k: -float((boxes[k][2] - boxes[k][0]) * (boxes[k][3] - boxes[k][1])))
    inside = {k: [j for j in order if j != k and ioa_small(boxes[j], boxes[k]) >= SHADOW_IOA] for k in order}
    group = set()
    for k in order:
        kids = [j for j in inside[k] if _area(boxes[j]) < _area(boxes[k])]
        if any(
            ioa_small(boxes[a], boxes[b]) < SHADOW_IOA and _area(boxes[a]) + _area(boxes[b]) >= 0.5 * _area(boxes[k])
            for i, a in enumerate(kids)
            for b in kids[i + 1 :]
        ):
            group.add(k)
    out: dict[int, int] = {}
    hosts: list[int] = []
    for k in order:  # largest first: a host is never itself a shadow
        if k in group:
            kids = [j for j in inside[k] if j not in group]
            if kids:
                out[k] = max(kids, key=lambda j: _area(boxes[j]))
                continue
        host = next((h for h in hosts if ioa_small(boxes[k], boxes[h]) >= SHADOW_IOA), None)
        if host is None:
            hosts.append(k)
        else:
            out[k] = host
    for k, h in list(out.items()):  # a group box's child must be a host, not itself a shadow
        while h in out:
            h = out[h]
        out[k] = h
    return out


def _tup(b: Box) -> tuple[float, float, float, float]:
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


def _bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance of two histograms (0 = identical, 1 = no overlap), as OpenCV's HISTCMP_BHATTACHARYYA."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return 1.0
    bc = float(np.sqrt(a * b).sum()) / math.sqrt(sa * sb)
    return math.sqrt(max(0.0, 1.0 - bc))


def _area(b: Box) -> float:
    return float(max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0))


@dataclass
class _Lost:
    public_id: int
    t: float
    center: tuple[float, float]
    size: float  # sqrt(area) px
    velocity: tuple[float, float]  # px/s at loss
    box: tuple[float, float, float, float] | None = None  # [2.35.0] last observed box (TRACK_COAST_SECONDS display)


@dataclass
class _Live:
    public_id: int
    born: float
    last_t: float
    center: tuple[float, float]
    size: float
    velocity: tuple[float, float] = (0.0, 0.0)
    box: tuple[float, float, float, float] | None = None  # [2.35.0] last reported box


@dataclass
class _Shadow:
    host: int  # public id of the person this box sits on
    born: float
    merged: bool = False  # [v7 F3] made a shadow by SHADOW_MERGE: released only when the boxes really separate
    apart_since: float | None = None  # [v7 F3] merged shadow: when its box stopped overlapping the host
    linked: bool = False  # [2.35.0] ID_LINK_NESTED alias: hidden while it overlaps the host by SHADOW_IOA
    prev_pid: int = 0  # [2.35.0] the public id this alias had before the link: given back when it is released


class IdStitcher:
    """Maps ByteTrack's ids to stable public ids across short disappearances and duplicate boxes."""

    def __init__(self) -> None:
        self._live: dict[int, _Live] = {}  # internal id → state
        self._lost: dict[int, _Lost] = {}  # public id → where / when it was lost
        self._pub_of: dict[int, int] = {}  # internal id → public id (survives ByteTrack's own re-find)
        self._shadow: dict[int, _Shadow] = {}  # internal id → host public id (not reported)
        self._next_public = 1
        self.stitches = 0
        self.handovers = 0
        # [v6] standard CCTV only (no scoring, no alerts): a shadow stays hidden for as long as it overlaps its
        # host by SHADOW_IOA, instead of being released after SHADOW_MAX_SECONDS. Pool cameras keep the release:
        # there a second real person must never stay hidden behind a duplicate rule.
        self.hold_overlapping = False
        # [v7 F3] SHADOW_MERGE: where hiding can't cost a swimmer (merge_ok: standard CCTV, or out of the water), two
        # reported tracks nested by SHADOW_IOA for SHADOW_MERGE_SECONDS are one person: the newer becomes a shadow
        self.merge_ok: Callable[[Box], bool] | None = None
        self._nested_hist: dict[tuple[int, int], deque[tuple[float, bool]]] = {}
        self.merges = 0
        # [2.35.0] ID_LINK_NESTED (in-water nested duplicates): inputs and evidence
        # (t, nested, clearly apart, centre offset x / y in smaller-box sizes)
        self._link_hist: dict[tuple[int, int], deque[tuple[float, bool, bool, float, float]]] = {}
        self._pose: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}  # public id -> (t, keypoints, conf)
        self.colour_fn: Callable[[Box], np.ndarray | None] | None = None  # cap / torso histogram of a box (this frame)
        self.protected: set[int] = set()  # public ids whose alert / risk is up: never retired by a link
        self.retired: set[int] = set()  # public ids retired by a link since the last take_retired()
        self.links: list[dict] = []  # evidence of every link and release (bounded)
        self.link_vetoes: dict[str, int] = {}  # why a nested pair with a full window was not linked (evidence)
        self._group_of: dict[int, int] = {}  # ByteTrack internal id -> the public id it was linked under (alias)

    def reset(self) -> None:
        self._live.clear()
        self._lost.clear()
        self._pub_of.clear()
        self._shadow.clear()
        self._next_public = 1
        self._link_hist.clear()
        self._pose.clear()
        self.retired.clear()
        self._group_of.clear()

    def note_pose(self, pid: int, kpts: np.ndarray, conf: np.ndarray, t: float) -> None:
        """[2.35.0] The skeleton RTMPose found for a reported public id this frame (the two-skeleton veto)."""
        if pid > 0:
            self._pose[pid] = (t, np.asarray(kpts, np.float32), np.asarray(conf, np.float32))
            if len(self._pose) > 512:
                self._pose = {k: v for k, v in self._pose.items() if t - v[0] <= 2.0}

    def take_retired(self) -> set[int]:
        """[2.35.0] Public ids folded into another id by ID_LINK_NESTED since the last call (the scene forgets them:
        the same body carries on under the host id, it did not vanish in the water)."""
        out, self.retired = self.retired, set()
        return out

    def coasting(self, t: float, max_seconds: float) -> list[tuple[int, tuple[float, float, float, float], float]]:
        """[2.35.0] (public id, last observed box, seconds unseen) for ids lost at most `max_seconds` ago."""
        if max_seconds <= 0:
            return []
        return [
            (pid, l.box, t - l.t) for pid, l in self._lost.items() if l.box is not None and 0 < t - l.t <= max_seconds
        ]

    def lost_ids(self) -> set[int]:
        """[2.40.0] Public ids held as lost (stitchable)."""
        return set(self._lost)

    def follow(self, pid: int, box: tuple[float, float, float, float], t: float) -> None:
        """[2.40.0] SUBMERGED_FOLLOW: the lost id `pid` is followed under water - its lost record moves to the followed
        box (centre, last box, time; its size stays the last observed one), so a swimmer who resurfaces where the
        follower is gets their id back by the usual stitch rules."""
        lost = self._lost.get(pid)
        if lost is None:
            return
        lost.center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        lost.velocity = (0.0, 0.0)
        lost.t = t
        lost.box = box

    def alive_ids(self) -> set[int]:
        return {s.public_id for s in self._live.values()} | set(self._lost)

    def _take_live(self, iid: int, pid: int, b: Box, t: float) -> None:
        self._pub_of[iid] = pid
        self._live[iid] = _Live(pid, t, t, _center(b), _size(b), box=_tup(b))

    def update(self, internal_ids: np.ndarray, boxes: np.ndarray, t: float) -> np.ndarray:
        n = len(internal_ids)
        out = np.full(n, -1, dtype=int)
        iids = [int(i) for i in internal_ids.tolist()]
        seen: set[int] = set()
        fresh: list[int] = []
        shadows: list[int] = []
        released: set[int] = set()  # shadows that turned out to be a person of their own
        restore: dict[int, int] = {}  # [2.35.0] released link alias -> the public id it had before the link
        grouped: list[int] = []  # [2.35.0] returning internal tracks of a linked public id
        live_box: dict[int, Box] = {}  # public id → box, reported this frame
        for k, iid in enumerate(iids):
            b = boxes[k]
            st = self._live.get(iid)
            if st is None and iid in self._pub_of and self._pub_of[iid] in self._lost:
                # ByteTrack re-found its own lost track: same public id, no stitching needed
                pid = self._pub_of[iid]
                del self._lost[pid]
                st = self._live[iid] = _Live(pid, t, t, _center(b), _size(b), box=_tup(b))
            if st is None:
                if iid in self._shadow:
                    shadows.append(k)
                elif iid in self._group_of:
                    grouped.append(k)  # [2.35.0] an internal track linked under a public id comes back
                else:
                    fresh.append(k)
                continue
            c = _center(b)
            dt = max(t - st.last_t, 1e-3)
            v = ((c[0] - st.center[0]) / dt, (c[1] - st.center[1]) / dt)
            a = 0.3  # smoothed velocity
            st.velocity = (a * v[0] + (1 - a) * st.velocity[0], a * v[1] + (1 - a) * st.velocity[1])
            st.center, st.size, st.last_t, st.box = c, _size(b), t, _tup(b)
            out[k] = st.public_id
            live_box[st.public_id] = b
            seen.add(iid)
        # age out internal tracks that ByteTrack no longer reports → "lost"
        for iid in list(self._live):
            if iid in seen:
                continue
            st = self._live[iid]
            if st.last_t < t:
                self._lost[st.public_id] = _Lost(st.public_id, st.last_t, st.center, st.size, st.velocity, st.box)
                del self._live[iid]
        present = set(iids)
        for iid in [i for i in self._shadow if i not in present]:
            del self._shadow[iid]  # the shadow box itself went away
        for pid in [p for p, l in self._lost.items() if t - l.t > STITCH_MAX_GAP_SECONDS]:
            del self._lost[pid]

        # [2.35.0] ID_LINK_NESTED: a public id aliases every internal track linked under it. A returning one carries
        # the id when no other box shows it, else it is a linked alias (hidden while it overlaps, released when apart)
        for k in grouped:
            iid, b = iids[k], boxes[k]
            gp = self._group_of[iid]
            if gp in live_box:
                self._shadow[iid] = _Shadow(gp, t, merged=True, linked=True, prev_pid=0)
                shadows.append(k)
            elif gp in self._lost:
                del self._lost[gp]
                self._take_live(iid, gp, b, t)
                out[k] = gp
                live_box[gp] = b
            else:
                del self._group_of[iid]  # the id expired: a person of its own
                fresh.append(k)

        # shadows: inherit the host's id when the host dropped; stay hidden while
        # they overlap it; become a person of their own when they separate / persist
        for k in shadows:
            iid, b = iids[k], boxes[k]
            sh = self._shadow[iid]
            host_b = live_box.get(sh.host)
            if host_b is None:
                if sh.host in self._lost:  # the person's own track dropped: hand over
                    del self._lost[sh.host]
                    del self._shadow[iid]
                    for old in [i for i, p in self._pub_of.items() if p == sh.host]:
                        del self._pub_of[old]  # the old internal track must not reclaim this id
                    self._take_live(iid, sh.host, b, t)
                    out[k] = sh.host
                    live_box[sh.host] = b
                    self.handovers += 1
                    continue
                del self._shadow[iid]  # host gone for good (expired): a person of its own
                fresh.append(k)
                released.add(k)
                continue
            ov = ioa_small(b, host_b)
            if sh.linked and ov >= SHADOW_IOA:
                continue  # [2.35.0] a linked alias: hidden while it overlaps its host
            if ov >= STITCH_BLOCK_IOA:
                sh.apart_since = None  # back on the host: any separation so far was a wobble
            if ov >= SHADOW_NESTED_IOA or (
                ov >= SHADOW_IOA and (self.hold_overlapping or t - sh.born <= SHADOW_MAX_SECONDS)
            ):
                continue  # still a duplicate of the host: not reported
            if sh.merged and self.merge_ok is not None and self.merge_ok(b):
                # a merged duplicate whose box wobbles: one person until the boxes stay apart for SHADOW_MERGE_SECONDS
                if ov >= STITCH_BLOCK_IOA:
                    sh.apart_since = None
                    continue
                if sh.apart_since is None:
                    sh.apart_since = t
                if t - sh.apart_since < SHADOW_MERGE_SECONDS:
                    continue
            if sh.linked:
                restore[k] = sh.prev_pid
                self._group_of.pop(iid, None)
                self._log_link("release", t, sh.host, sh.prev_pid, {"ioa": round(float(ov), 2)})
            del self._shadow[iid]
            fresh.append(k)
            released.add(k)

        alive = {s.public_id for s in self._live.values()} | set(self._lost) | {s.host for s in self._shadow.values()}
        for iid in [i for i, p in self._pub_of.items() if p not in alive]:
            del self._pub_of[iid]
        for iid in [i for i, p in self._group_of.items() if p not in alive]:
            del self._group_of[iid]

        # new internal tracks: a shadow of someone in view, resume a lost public id, or get a new one
        taken = {s.public_id for s in self._live.values()}
        cands: list[tuple[float, int, int]] = []
        stitchable: list[int] = []
        newborn: list[int] = []
        for k in fresh:
            iid, b = iids[k], boxes[k]
            near_live = 0.0
            if live_box:
                host, near_live = max(((pid, ioa_small(b, hb)) for pid, hb in live_box.items()), key=lambda x: x[1])
                if near_live >= SHADOW_IOA and k not in released:
                    self._shadow[iid] = _Shadow(host, t)
                    continue
            newborn.append(k)
        born_shadow = _same_frame_shadows([k for k in newborn if k not in released], boxes)
        for k in newborn:
            if k in born_shadow:
                continue  # a second box on a body born this same frame: shadow of it once it has an id
            b = boxes[k]
            near_live = max((ioa_small(b, hb) for hb in live_box.values()), default=0.0)
            stitchable.append(k)
            if near_live >= STITCH_BLOCK_IOA:
                continue  # a fragment next to someone in view: never another person's id
            c, size = _center(b), _size(b)
            # Rank every lost person within reach by distance AND recency; the
            # most likely one must also pass the size check, otherwise the box
            # gets a new id — it never falls back to an older, less likely id
            # (a swimmer whose box doubled in size took a submerged child's id).
            mine: list[tuple[float, bool, int]] = []
            for pid, l in self._lost.items():
                if pid in taken:
                    continue
                gap = t - l.t
                g = min(gap, 1.0)  # extrapolate motion for at most 1 s
                px, py = l.center[0] + l.velocity[0] * g, l.center[1] + l.velocity[1] * g
                d = math.hypot(c[0] - px, c[1] - py)
                radius = min(size, l.size) * min(STITCH_RADIUS_MAX, STITCH_RADIUS_BASE + STITCH_RADIUS_PER_SECOND * gap)
                if d > radius:
                    continue
                ratio = max(size, l.size) / max(min(size, l.size), 1.0)
                max_ratio = STITCH_SIZE_RATIO_RECENT if gap <= STITCH_RECENT_SECONDS else STITCH_SIZE_RATIO_MAX
                mine.append((d / radius + gap / STITCH_MAX_GAP_SECONDS, ratio <= max_ratio, pid))
            if ID_LINK_STITCH_UNIQUE and len(mine) > 1:
                continue  # [2.35.0] two lost people within reach: ambiguous, a new id
            if mine:
                score, ok, pid = min(mine)
                if ok:
                    cands.append((score, k, pid))
        used_k, used_p = set(), set()
        for _, k, pid in sorted(cands):
            if k in used_k or pid in used_p:
                continue
            used_k.add(k)
            used_p.add(pid)
            out[k] = pid
            del self._lost[pid]
            for old in [i for i, p in self._pub_of.items() if p == pid]:
                del self._pub_of[old]  # the old internal track must not reclaim this id
            self.stitches += 1
        for k in stitchable:
            if k not in used_k:
                back = restore.get(k, 0)
                if back > 0 and back not in taken and back not in used_p:
                    out[k] = back  # [2.35.0] a released link alias gets its own id back
                    self._lost.pop(back, None)
                    taken.add(back)
                else:
                    out[k] = self._next_public
                    self._next_public += 1
            self._take_live(iids[k], int(out[k]), boxes[k], t)
        for k, host_k in born_shadow.items():
            self._shadow[iids[k]] = _Shadow(int(out[host_k]), t)
        if SHADOW_MERGE and self.merge_ok is not None:
            self._merge_nested(iids, boxes, out, t)
        if ID_LINK_NESTED:
            self._link_nested(iids, boxes, out, t)
        return out

    def _log_link(self, kind: str, t: float, host: int, alias: int, ev: dict) -> None:
        rec = {"kind": kind, "t": round(float(t), 2), "host": int(host), "alias": int(alias), **ev}
        logger.info("[id-link] %s", rec)
        self.links.append(rec)
        if len(self.links) > 1000:
            del self.links[:500]

    def _veto(self, why: str) -> None:
        self.link_vetoes[why] = self.link_vetoes.get(why, 0) + 1

    def _two_skeletons(self, pa: int, pb: int, t: float, h: float) -> bool | None:
        """True when RTMPose saw two different bodies on the two ids (recent poses, >= 4 shared confident joints whose
        median distance exceeds ID_LINK_SKELETON_SEP x the smaller box height); None = no evidence either way."""
        a, b = self._pose.get(pa), self._pose.get(pb)
        if a is None or b is None or t - a[0] > 1.0 or t - b[0] > 1.0:
            return None
        both = (a[2] >= KEYPOINT_CONF_THRESHOLD) & (b[2] >= KEYPOINT_CONF_THRESHOLD)
        if int(both.sum()) < 4:
            return None
        d = np.linalg.norm(a[1][both] - b[1][both], axis=1)
        return bool(float(np.median(d)) > ID_LINK_SKELETON_SEP * max(h, 1.0))

    def _link_nested(self, iids: list[int], boxes: np.ndarray, out: np.ndarray, t: float) -> None:
        """[2.35.0] ID_LINK_NESTED: two confirmed tracks on one body, in the water too (RTMDet's loose body box and a
        tight box inside it). On a splashing body the two ids mostly ALTERNATE (one box reported per frame), so the
        pair's history is kept while both ids are alive and sampled on the frames where both are reported.

        Linked only when every condition holds and no veto fires (ambiguous = no link): over a full
        ID_LINK_NESTED_SECONDS window at least ID_LINK_NESTED_MIN_SAMPLES joint frames, nested by ID_LINK_NESTED_IOA
        in >= ID_LINK_NESTED_FRAC of them and now, never clearly apart (< STITCH_BLOCK_IOA); the inner box not inside
        two boxes that are themselves apart; the centre offset changing by <= ID_LINK_MOTION_MAX smaller-box sizes per
        second; one skeleton on both (ID_LINK_REQUIRE_SKELETON: no pose evidence = no link); colours not clearly
        different; the id to retire carries no raised risk. The outer box is reported under the lower (older) public
        id; the inner track becomes a linked alias that is released, with its own id back, when the boxes stop
        overlapping by SHADOW_IOA. The public id aliases every linked internal track (`_group_of`)."""
        win = ID_LINK_NESTED_SECONDS
        alive = {s.public_id for s in self._live.values()} | set(self._lost)
        for key in list(self._link_hist):
            h = self._link_hist[key]
            while h and t - h[0][0] > win:
                h.popleft()
            if not h or key[0] not in alive or key[1] not in alive:
                del self._link_hist[key]
        rep = [k for k in range(len(out)) if out[k] > 0 and iids[k] in self._live]
        containers: dict[int, list[int]] = {k: [] for k in rep}
        for a_i, a in enumerate(rep):
            for b in rep[a_i + 1 :]:
                if ioa_small(boxes[a], boxes[b]) >= ID_LINK_NESTED_IOA:
                    inner, outer = (a, b) if _area(boxes[a]) <= _area(boxes[b]) else (b, a)
                    containers[inner].append(outer)
        for a_i, a in enumerate(rep):
            for b in rep[a_i + 1 :]:
                if out[a] <= 0 or out[b] <= 0:
                    continue  # linked this frame already
                key = (min(int(out[a]), int(out[b])), max(int(out[a]), int(out[b])))
                ov = ioa_small(boxes[a], boxes[b])
                nested = ov >= ID_LINK_NESTED_IOA
                if key not in self._link_hist and not nested:
                    continue
                lo, hi = (a, b) if int(out[a]) < int(out[b]) else (b, a)
                ca, cb = _center(boxes[lo]), _center(boxes[hi])
                small = math.sqrt(min(_area(boxes[a]), _area(boxes[b])) or 1.0)
                hist = self._link_hist.setdefault(key, deque())
                hist.append((t, nested, ov < STITCH_BLOCK_IOA, (cb[0] - ca[0]) / small, (cb[1] - ca[1]) / small))
                if not nested or t - hist[0][0] < 0.9 * win or len(hist) < ID_LINK_NESTED_MIN_SAMPLES:
                    continue
                frac = sum(1 for x in hist if x[1]) / len(hist)
                if frac < ID_LINK_NESTED_FRAC:
                    self._veto("frac")
                    continue
                if any(x[2] for x in hist):
                    self._veto("apart")
                    continue
                inner, outer = (a, b) if _area(boxes[a]) <= _area(boxes[b]) else (b, a)
                cs = containers[inner]
                if any(ioa_small(boxes[c1], boxes[c2]) < SHADOW_IOA for i, c1 in enumerate(cs) for c2 in cs[i + 1 :]):
                    self._veto("nested_in_two")
                    continue  # inside two boxes that are apart: two people, ambiguous
                span = max(t - hist[0][0], 1e-3)
                dxs, dys = [x[3] for x in hist], [x[4] for x in hist]
                dv = math.hypot(max(dxs) - min(dxs), max(dys) - min(dys)) / span
                ev: dict = {
                    "nested_frac": round(float(frac), 2),
                    "samples": len(hist),
                    "ioa": round(float(ov), 2),
                    "dv": round(float(dv), 2),
                }
                if dv > ID_LINK_MOTION_MAX:
                    self._veto("motion")
                    continue
                h = float(min(boxes[a][3] - boxes[a][1], boxes[b][3] - boxes[b][1]))
                two = self._two_skeletons(int(out[a]), int(out[b]), t, h)
                ev["skel"] = two
                if two:
                    self._veto("skeletons")
                    continue
                if two is None and ID_LINK_REQUIRE_SKELETON:
                    self._veto("no_pose")
                    continue  # no pose evidence that the two boxes are one body: ambiguous
                if self.colour_fn is not None:
                    ha, hb = self.colour_fn(boxes[inner]), self.colour_fn(boxes[outer])
                    if ha is not None and hb is not None:
                        dist = float(_bhattacharyya(ha, hb))
                        ev["colour"] = round(dist, 2)
                        if dist > ID_LINK_COLOUR_MAX:
                            self._veto("colour")
                            continue
                # the older person keeps the id: public ids are handed out in order (a re-found internal track's
                # `born` restarts, so it cannot tell which id is older on an alternating body)
                old, new = (a, b) if int(out[a]) < int(out[b]) else (b, a)
                keep_pid, gone_pid = int(out[old]), int(out[new])
                if gone_pid in self.protected:
                    self._veto("protected")
                    continue  # never retire an id whose alert or risk is up
                st_alias = self._live.pop(iids[inner])
                self._pub_of.pop(iids[inner], None)
                if outer == new:  # the outer box is the newer track: it takes over the older id
                    st_out = self._live[iids[outer]]
                    st_out.public_id, st_out.born = keep_pid, min(st_out.born, st_alias.born)
                    self._pub_of[iids[outer]] = keep_pid
                for i, gp in list(self._group_of.items()):
                    if gp == gone_pid:
                        self._group_of[i] = keep_pid  # the retired id's own aliases follow it
                self._group_of[iids[inner]] = keep_pid
                self._group_of[iids[outer]] = keep_pid
                self._shadow[iids[inner]] = _Shadow(keep_pid, t, merged=True, linked=True, prev_pid=gone_pid)
                out[outer], out[inner] = keep_pid, -1
                self.retired.add(gone_pid)
                self.merges += 1
                ev["inner_was"] = "older" if inner == old else "newer"
                self._log_link("link", t, keep_pid, gone_pid, ev)

    def _merge_nested(self, iids: list[int], boxes: np.ndarray, out: np.ndarray, t: float) -> None:
        """[v7 F3] Two LIVE reported tracks on one body (a wobbling full-body box released as a second person next
        to its upper-body box): nested by SHADOW_IOA in >= 70 % of the frames of the last SHADOW_MERGE_SECONDS, both
        where merge_ok allows -> the newer track becomes a (sticky) shadow of the older one. It can still hand over,
        and is never reported as lost. A box that wobbles out of the overlap now and then does not reset the count."""
        ok = self.merge_ok
        if ok is None:
            return
        rep = [k for k in range(len(out)) if out[k] > 0 and iids[k] in self._live]
        live_pairs: set[tuple[int, int]] = set()
        for a_i, a in enumerate(rep):
            for b in rep[a_i + 1 :]:
                key = (min(int(out[a]), int(out[b])), max(int(out[a]), int(out[b])))
                nested = ioa_small(boxes[a], boxes[b]) >= SHADOW_IOA
                if key not in self._nested_hist and not nested:
                    continue
                live_pairs.add(key)
                hist = self._nested_hist.setdefault(key, deque())
                hist.append((t, nested and ok(boxes[a]) and ok(boxes[b])))
                while hist and t - hist[0][0] > SHADOW_MERGE_SECONDS:
                    hist.popleft()
                span = t - hist[0][0]
                if span < 0.8 * SHADOW_MERGE_SECONDS or sum(h for _, h in hist) < 0.7 * len(hist) or not nested:
                    continue
                old, new = (a, b) if self._live[iids[a]].born <= self._live[iids[b]].born else (b, a)
                if out[new] <= 0 or out[old] <= 0:
                    continue  # already merged this frame
                host = int(out[old])
                del self._live[iids[new]]
                self._pub_of.pop(iids[new], None)
                self._shadow[iids[new]] = _Shadow(host, t, merged=True)
                out[new] = -1
                self.merges += 1
        self._nested_hist = {k: v for k, v in self._nested_hist.items() if k in live_pairs}


@dataclass
class _Chain:
    box: np.ndarray
    hits: deque[tuple[float, float, object | None]] = field(default_factory=deque)  # (t, score, evidence)
    last_t: float = 0.0


class FaintBirths:
    """
    Chains of faint boxes (FAINT_BIRTH_MIN_SCORE ≤ score < birth threshold) that
    are not on anyone already tracked. A chain that is persistent, in the water
    and backed by pixel evidence is PROMOTED: its box is handed to ByteTrack at
    birth strength and the new track is confirmed at once (the chain already
    spanned several frames; a faint swimmer rarely gives two strong frames in a row).
    """

    def __init__(self) -> None:
        self.chains: list[_Chain] = []
        self.promotions = 0

    def reset(self) -> None:
        self.chains.clear()

    def _validated(self, ch: _Chain, t: float, fps: float, in_water: WaterFn | None) -> bool:
        hits = [h for h in ch.hits if t - h[0] <= FAINT_BIRTH_WINDOW_SECONDS]
        need = max(3, int(round(FAINT_BIRTH_MIN_HIT_FRAC * FAINT_BIRTH_WINDOW_SECONDS * fps)))
        if len(hits) < need:
            return False
        if float(np.mean([h[1] for h in hits])) < FAINT_BIRTH_MIN_MEAN:
            return False
        ev = [h[2] for h in hits if h[2] is not None]
        if len(ev) < need // 2 + 1:
            return False  # background model not warm / box off-frame
        if float(np.median([e.motion for e in ev])) < FAINT_BIRTH_MIN_MOTION:
            return False  # never moved: drain, toy on the floor, slide, shadow
        if float(np.median([e.body for e in ev])) < FAINT_BIRTH_MIN_BODY:
            return False  # only water inside: a ripple / glint
        if float(np.median([e.splash for e in ev])) > FAINT_BIRTH_MAX_SPLASH:
            return False  # foam with nothing solid in it
        if in_water is None or in_water(_center(ch.box)) is not True:
            return False  # deck / unknown water: only clear detections start tracks
        return True

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        t: float,
        fps: float,
        evidence: EvidenceFn | None,
        in_water: WaterFn | None,
        occupied: Sequence[Box],
        birth_score: float,
    ) -> list[int]:
        """Indices of `boxes` to promote to birth strength this frame."""
        self.chains = [c for c in self.chains if t - c.last_t <= FAINT_BIRTH_WINDOW_SECONDS]
        if evidence is None or len(boxes) == 0:
            return []
        busy = list(occupied) + [boxes[i] for i in range(len(boxes)) if scores[i] >= birth_score]
        cand = [
            i
            for i in range(len(boxes))
            if FAINT_BIRTH_MIN_SCORE <= scores[i] < birth_score
            and not any(ioa_small(boxes[i], o) >= 0.5 or _iou(boxes[i], o) >= 0.3 for o in busy)
        ]
        pairs = []
        for i in cand:
            c, s = _center(boxes[i]), _size(boxes[i])
            for j, ch in enumerate(self.chains):
                d = math.hypot(c[0] - _center(ch.box)[0], c[1] - _center(ch.box)[1])
                if _iou(boxes[i], ch.box) >= 0.2 or d <= 0.6 * max(s, _size(ch.box)):
                    pairs.append((d, i, j))
        used_i, used_j = set(), set()
        matched: dict[int, int] = {}
        for _, i, j in sorted(pairs):
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            matched[i] = j
        promote: list[int] = []
        for i in cand:
            if i in matched:
                ch = self.chains[matched[i]]
            else:
                ch = _Chain(boxes[i].astype(np.float32).copy())
                self.chains.append(ch)
            ch.box = boxes[i].astype(np.float32).copy()
            ch.last_t = t
            ch.hits.append((t, float(scores[i]), evidence(boxes[i])))
            while ch.hits and t - ch.hits[0][0] > FAINT_BIRTH_WINDOW_SECONDS:
                ch.hits.popleft()
            if self._validated(ch, t, fps, in_water):
                promote.append(i)
        self.promotions += len(promote)
        return promote


class PersonTracker:
    """ByteTrack + IdStitcher (+ FaintBirths) for one camera. `update()` once per analysed frame."""

    def __init__(self, fps: float = 25.0) -> None:
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self._bt = None
        self.stitcher = IdStitcher()
        self.faint = FaintBirths()
        self.faint_ids: set[int] = set()  # public ids whose track started as a faint birth
        self._faint_internal: set[int] = set()
        self._reported_internal: set[int] = set()
        self._last_boxes: np.ndarray = np.zeros((0, 4), np.float32)
        self._last_t = 0.0
        self._in_water: WaterFn | None = None
        self.coast_water: WaterFn | None = None  # [2.24.0] set by the detector each frame (TRACK_LOST_SECONDS_WATER)
        self.surveillance = False  # [v6] standard CCTV camera: IdStitcher.hold_overlapping
        self.birth_guard_iou: float | None = None
        self._build()

    def _build(self) -> None:
        from lifeguard.vision.tracking.bytetrack import ByteTracker

        # max_time_lost = int(frame_rate / 30 * lost_track_buffer) frames (ByteTrack convention)
        lost_frames_at_30 = int(round(TRACK_LOST_SECONDS * 30.0))
        self._bt = ByteTracker(
            activation_threshold=TRACK_HIGH_SCORE_THRESHOLD,
            lost_track_buffer=lost_frames_at_30,
            minimum_matching_threshold=1.0 - TRACK_MATCH_IOU,
            frame_rate=self.fps,
            refind_low_iou=TRACK_REFIND_LOW_IOU,
            refind_max_frames=int(round(TRACK_REFIND_MAX_SECONDS * self.fps)),
            low_threshold=TRACK_LOW_SCORE_THRESHOLD,
            birth_margin=TRACK_BIRTH_SCORE_MARGIN,
            second_match_iou=TRACK_SECOND_MATCH_IOU,
            unconfirmed_match_iou=TRACK_UNCONFIRMED_MATCH_IOU,
            kf_std_position=TRACK_KF_STD_POSITION,
            kf_std_velocity=TRACK_KF_STD_VELOCITY,
            assoc_cost=TRACK_ASSOC_COST,
            assoc_dist_radius=TRACK_ASSOC_DIST_RADIUS,
            lost_track_buffer_water=(
                int(round(TRACK_LOST_SECONDS_WATER * 30.0)) if TRACK_LOST_SECONDS_WATER > TRACK_LOST_SECONDS else None
            ),
            second_last_seen=TRACK_SECOND_MATCH_LAST_SEEN,
        )
        self._bt.birth_guard_iou = self.birth_guard_iou

    def set_birth_guard(self, iou: float | None) -> None:
        """[2.18.0] Clips: an unmatched box overlapping a confirmed track continues it (None: standard ByteTrack)."""
        self.birth_guard_iou = iou if iou and iou > 0 else None
        if self._bt is not None:
            self._bt.birth_guard_iou = self.birth_guard_iou

    @property
    def frame_id(self) -> int:
        """ByteTrack's frame counter (advances on every update, including empty frames)."""
        return int(getattr(self._bt, "frame_id", 0))

    @property
    def max_time_lost_frames(self) -> int:
        return int(getattr(self._bt, "max_time_lost", 0))

    def set_fps(self, fps: float) -> None:
        if fps and fps > 0 and abs(fps - self.fps) > 0.5:
            self.fps = float(fps)
            self._build()
            self.stitcher.reset()
            self.faint.reset()

    def retime(self, fps: float) -> None:
        """[dynamic fps] The live analysis rate stepped (15-20 fps) on a running stream: unlike set_fps (a reopened
        stream, tracker rebuilt) every track, id and stitch survives; only the frame-count buffers are rescaled."""
        if fps and fps > 0 and abs(fps - self.fps) > 1e-6:
            self.fps = float(fps)
            self._bt.retime(self.fps, int(round(TRACK_REFIND_MAX_SECONDS * self.fps)))

    def reset(self) -> None:
        self._bt.reset()
        self.stitcher.reset()
        self.faint.reset()
        self.faint_ids.clear()
        self._faint_internal.clear()
        self._reported_internal.clear()
        self._last_boxes = np.zeros((0, 4), np.float32)

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        t: float,
        evidence: EvidenceFn | None = None,
        in_water: WaterFn | None = None,
        birth_scores: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (indices into `boxes`, public track ids) for the boxes ByteTrack
        kept this frame; id -1 = a shadow box (same person as a reported one).
        `evidence` (motion.SceneMotion.evidence) enables faint births;
        `birth_scores` are the full-frame scores they are judged on (zoom-pass
        boxes extend and re-find tracks but never seed a birth).
        """
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        scores = np.asarray(scores, np.float32).reshape(-1)
        self._last_t = t
        birth = float(self._bt.det_threshold)
        bs = scores if birth_scores is None else np.asarray(birth_scores, np.float32).reshape(-1)
        promote = self.faint.update(boxes, bs, t, self.fps, evidence, in_water, self._last_boxes, birth)
        bt_scores = scores.copy()
        # A box only the zoom pass saw (no full-frame score) may extend or
        # re-find a track but never start one: CLAHE-enlarged deck clutter and
        # a floating ball score as "people" there.
        zoom_only = (bs <= 0.0) & (scores > 0.0)
        if zoom_only.any():
            bt_scores[zoom_only] = np.minimum(bt_scores[zoom_only], TRACK_HIGH_SCORE_THRESHOLD - 1e-3)
        if promote:
            bt_scores[promote] = np.maximum(bt_scores[promote], birth + 1e-3)
        # faint boxes re-find lost tracks only in the water (never deck clutter)
        refind_ok = None
        if in_water is not None:
            refind_ok = np.asarray([in_water(_center(b)) is True for b in boxes], bool)
        self._in_water = in_water
        confirm = set(promote)
        # [2.24.0] TRACK_LOST_SECONDS_WATER: which boxes are in the water (the scene's water test, set every frame by
        # the detector as coast_water; only read when that key is on, so the default path is unchanged)
        det_wet = None
        if self._bt.max_time_lost_water is not None and self.coast_water is not None:
            det_wet = np.asarray([self.coast_water(_center(b)) is True for b in boxes], bool)
        idx, internal = self._bt.update(
            boxes, bt_scores, refind_ok=refind_ok, confirm_now=confirm or None, det_wet=det_wet
        )
        if len(idx) == 0:
            self.stitcher.update(np.zeros(0, int), np.zeros((0, 4), np.float32), t)
            self._last_boxes = np.zeros((0, 4), np.float32)
            empty = glare.apply_hold(self, np.zeros(0, int), np.zeros((0, 4), np.float32), t)  # [v3 Phase 11]
            return np.zeros(0, int), empty
        promoted = set(promote)
        for i, iid in zip(idx.tolist(), internal.tolist()):
            if iid not in self._reported_internal:  # first frame this track is confirmed
                self._reported_internal.add(iid)
                if i in promoted:
                    self._faint_internal.add(iid)
        # [v3] ID_STITCHING off: ByteTrack's own ids are public (no stitching, no shadows).
        self.stitcher.hold_overlapping = bool(self.surveillance)
        wet = in_water
        self.stitcher.merge_ok = (
            (lambda b: True)
            if self.surveillance
            else (None if wet is None else (lambda b: wet(((b[0] + b[2]) / 2.0, float(b[3]))) is False))
        )
        pub = self.stitcher.update(internal, boxes[idx], t) if ID_STITCHING else internal.copy()
        for iid, pid in zip(internal.tolist(), pub.tolist()):
            if pid > 0 and iid in self._faint_internal:
                self.faint_ids.add(pid)
        self._prune()
        self._last_boxes = boxes[idx]
        pub = glare.apply_hold(self, pub, boxes[idx], t)  # [v3 Phase 11] GLARE_HOLD (off: unchanged)
        return idx, pub

    def _prune(self) -> None:
        if len(self._reported_internal) > 4096:
            keep = {t.internal_id for t in self._bt.tracked + self._bt.lost}
            self._reported_internal &= keep
            self._faint_internal &= keep
        if len(self.faint_ids) > 256:
            self.faint_ids &= self.stitcher.alive_ids()

    def focus_regions(self) -> list[np.ndarray]:
        """
        Where to take a zoomed second look on the NEXT frame: the last-seen box
        of every track lost IN THE WATER within DET_ZOOM_LOST_SECONDS, most
        recent first. Only these — a zoom pass over live / faint tracks or
        unconfirmed chains fed fragment boxes (an arm, a torso, the ball) back
        into tracking and cost precision and ID switches (eval/reports/abl_*).
        """
        wet = self._in_water
        if wet is None:
            return []  # no water model: nothing to look closer at
        bt = self._bt
        horizon = int(round(DET_ZOOM_LOST_SECONDS * self.fps))
        lost = sorted(
            (t for t in bt.lost if t.is_activated and bt.frame_id - t.end_frame <= horizon),
            key=lambda t: bt.frame_id - t.end_frame,
        )
        out = [np.asarray(t.last_det_xyxy, np.float32) for t in lost]
        return [b for b in out if wet(_center(b)) is True][:DET_ZOOM_MAX_REGIONS]
