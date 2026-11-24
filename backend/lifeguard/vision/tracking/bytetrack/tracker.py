# mypy: disable-error-code="arg-type, misc, union-attr"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""
[v3] Ported verbatim from the v2 baseline detection/bytetrack/tracker.py (imports only).

ByteTrack association (two-stage: high-score boxes, then low-score boxes can
only extend existing tracks). Semantics mirror supervision.ByteTrack 0.2x:

  * boxes with score >= `activation_threshold` enter the first association;
    boxes in (low_threshold, activation_threshold) enter the second;
  * a new track needs score >= activation_threshold + 0.1 and is confirmed
    (reported) only after it is matched again on the next frame (except on
    the very first frame);
  * a lost track is kept for int(frame_rate / 30 * lost_track_buffer) frames;
  * external ids are consumed only by tracks that get confirmed, so short
    false-positive blips never burn ids.

Extension (off unless `refind_low_iou` is set; supervision has no such stage):
  * third association — low-score boxes left over after the second stage may
    RE-FIND a LOST track whose predicted box they overlap by at least
    `refind_low_iou` IoU. Plain ByteTrack lets only high-score boxes revive a
    lost track, so a swimmer seen faintly under murky water after a dive came
    back as a brand-new track (and a new id).

`update()` returns, for every confirmed track matched THIS frame, the index of
the input detection it matched — callers keep their own per-detection data
(detector class votes, embeddings) aligned without IoU re-matching.
"""

from __future__ import annotations

import numpy as np

from lifeguard.vision.tracking.bytetrack.kalman import KalmanFilterXYAH


class TrackState:
    NEW, TRACKED, LOST, REMOVED = 0, 1, 2, 3


def _xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2.0, b[1] + h / 2.0, w / max(h, 1e-6), h], np.float64)


class STrack:
    def __init__(self, xyxy: np.ndarray, score: float, det_idx: int) -> None:
        self._xyxy = np.asarray(xyxy, np.float64)
        self.last_det_xyxy = self._xyxy.copy()  # last OBSERVED box (the Kalman mean drifts while lost)
        self.score = float(score)
        self.det_idx = int(det_idx)
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self.state = TrackState.NEW
        self.is_activated = False
        self.internal_id = 0
        self.external_id = -1
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self.matched_this_frame = False
        self.wet = False  # [2.24.0] last matched box was in the water (TRACK_LOST_SECONDS_WATER)

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @property
    def xyxy(self) -> np.ndarray:
        if self.mean is None:
            return self._xyxy.copy()
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])

    def activate(self, kf: KalmanFilterXYAH, frame_id: int, internal_id: int, new_external) -> None:
        self.internal_id = internal_id
        self.mean, self.cov = kf.initiate(_xyxy_to_xyah(self._xyxy))
        self.tracklet_len = 1
        self.state = TrackState.TRACKED
        if frame_id == 1:  # first frame: confirmed immediately
            self.is_activated = True
            self.external_id = new_external()
        self.frame_id = self.start_frame = frame_id
        self.matched_this_frame = True

    def update(self, kf: KalmanFilterXYAH, det: STrack, frame_id: int, new_external) -> None:
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.last_det_xyxy = det._xyxy.copy()
        self.mean, self.cov = kf.update(self.mean, self.cov, _xyxy_to_xyah(det._xyxy))
        self.state = TrackState.TRACKED
        self.is_activated = True
        if self.external_id < 0:  # ids are consumed only on confirmation
            self.external_id = new_external()
        self.score = det.score
        self.det_idx = det.det_idx
        self.wet = det.wet
        self.matched_this_frame = True

    def re_activate(self, kf: KalmanFilterXYAH, det: STrack, frame_id: int) -> None:
        self.last_det_xyxy = det._xyxy.copy()
        self.mean, self.cov = kf.update(self.mean, self.cov, _xyxy_to_xyah(det._xyxy))
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.frame_id = frame_id
        self.score = det.score
        self.det_idx = det.det_idx
        self.wet = det.wet
        self.matched_this_frame = True


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _iou_cost(tracks: list[STrack], dets: list[STrack], fuse_score: bool) -> np.ndarray:
    if not tracks or not dets:
        return np.zeros((len(tracks), len(dets)))
    iou = _iou_matrix(np.stack([t.xyxy for t in tracks]), np.stack([d._xyxy for d in dets]))
    if fuse_score:
        iou = iou * np.array([d.score for d in dets])[None, :]
    return 1.0 - iou


def _iou_dist_cost(tracks: list[STrack], dets: list[STrack], radius: float) -> np.ndarray:
    """[2.23.0] TRACK_ASSOC_COST="iou_dist": plain IoU (no score fusion: a weak, dimming box is not penalised for
    its score), with a centre-distance fallback for boxes whose IoU collapsed (small or jittering swimmers): similarity
    max(IoU, 0.5 * (1 - d / (radius * h))) where d is the centre distance and h the mean box height, only for boxes
    within a 2x height ratio. Capped at 0.5, a distance-only pair never beats a real overlap in the assignment."""
    if not tracks or not dets:
        return np.zeros((len(tracks), len(dets)))
    a = np.stack([t.xyxy for t in tracks])
    b = np.stack([d._xyxy for d in dets])
    iou = _iou_matrix(a, b)
    ca = np.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2], 1)
    cb = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)
    ha = np.maximum(a[:, 3] - a[:, 1], 1.0)[:, None]
    hb = np.maximum(b[:, 3] - b[:, 1], 1.0)[None, :]
    near = np.clip(1.0 - d / (max(float(radius), 1e-6) * (ha + hb) / 2.0), 0.0, 1.0)
    near = np.where(np.maximum(ha / hb, hb / ha) <= 2.0, near, 0.0)
    return 1.0 - np.maximum(iou, 0.5 * near)


class _AtLastSeen:
    """A lost track's box where it was last OBSERVED (the Kalman prediction slides on while it is lost)."""

    def __init__(self, t: STrack) -> None:
        self.xyxy = t.last_det_xyxy


def _assign(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment keeping pairs with cost <= thresh."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    from scipy.optimize import linear_sum_assignment

    big = thresh + 1e-4  # same masking as ByteTrack/supervision
    c = np.where(cost > thresh, big, cost)
    rows, cols = linear_sum_assignment(c)
    matches = [(int(r), int(k)) for r, k in zip(rows, cols) if c[r, k] <= thresh]
    mr = {r for r, _ in matches}
    mc = {k for _, k in matches}
    return matches, [i for i in range(n) if i not in mr], [j for j in range(m) if j not in mc]


class ByteTracker:
    def __init__(
        self,
        activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: float = 30.0,
        low_threshold: float = 0.1,
        refind_low_iou: float | None = None,
        refind_max_frames: int | None = None,
        birth_margin: float = 0.1,
        second_match_iou: float = 0.5,
        unconfirmed_match_iou: float = 0.3,
        kf_std_position: float = 1.0 / 20,
        kf_std_velocity: float = 1.0 / 160,
        assoc_cost: str = "iou_fused",
        assoc_dist_radius: float = 1.0,
        lost_track_buffer_water: int | None = None,
        second_last_seen: bool = False,
    ) -> None:
        # [v6] birth margin, 2nd-association / unconfirmed IoU and the Kalman noise are configurable (thresholds.json);
        # the defaults are the values that were hardcoded here.
        self.activation_threshold = float(activation_threshold)
        self.refind_low_iou = refind_low_iou
        self.refind_max_frames = refind_max_frames
        self.det_threshold = self.activation_threshold + float(birth_margin)
        self.low_threshold = float(low_threshold)
        self.match_thresh = float(minimum_matching_threshold)
        self.second_match_thresh = 1.0 - float(second_match_iou)
        self.unconfirmed_match_thresh = 1.0 - float(unconfirmed_match_iou)
        self._lost_track_buffer = lost_track_buffer
        self._lost_track_buffer_water = lost_track_buffer_water
        self.max_time_lost = int(frame_rate / 30.0 * lost_track_buffer)
        # [2.24.0] TRACK_LOST_SECONDS_WATER: a track last seen in the water is kept longer (a swimmer under the surface
        # or behind a splash), its Kalman motion damped while lost; None = every track uses max_time_lost
        self.max_time_lost_water = (
            int(frame_rate / 30.0 * lost_track_buffer_water)
            if lost_track_buffer_water is not None and lost_track_buffer_water > lost_track_buffer
            else None
        )
        self.coast_damping = 0.9 ** (30.0 / max(float(frame_rate), 1e-6))  # velocity x0.9 per 1/30 s while lost
        self.kf = KalmanFilterXYAH(kf_std_position, kf_std_velocity)
        if assoc_cost not in ("iou_fused", "iou_dist"):
            raise ValueError(f"unknown assoc_cost {assoc_cost!r}")
        self.assoc_cost = assoc_cost  # [2.23.0] 1st association cost (TRACK_ASSOC_COST)
        self.assoc_dist_radius = float(assoc_dist_radius)
        self.birth_guard_iou: float | None = None  # [2.18.0] finite clips only (TRACK_BIRTH_GUARD_IOU_CLIP)
        self.second_last_seen = bool(second_last_seen)  # [2.35.0] TRACK_SECOND_MATCH_LAST_SEEN
        self.reset()

    def reset(self) -> None:
        self.frame_id = 0
        self.tracked: list[STrack] = []
        self.lost: list[STrack] = []
        self._next_internal = 1
        self._next_external = 1

    def retime(self, frame_rate: float, refind_max_frames: int | None) -> None:
        """[dynamic fps] The analysed frame rate changed mid-stream: rescale the frame-count buffers so they keep
        their duration in seconds. Tracks, ids and Kalman states are kept (a rebuild would drop every track)."""
        self.refind_max_frames = refind_max_frames
        self.max_time_lost = int(frame_rate / 30.0 * self._lost_track_buffer)
        water = self._lost_track_buffer_water
        self.max_time_lost_water = (
            int(frame_rate / 30.0 * water) if water is not None and water > self._lost_track_buffer else None
        )
        self.coast_damping = 0.9 ** (30.0 / max(float(frame_rate), 1e-6))

    def _new_external(self) -> int:
        self._next_external += 1
        return self._next_external - 1

    def _predict(self, tracks: list[STrack]) -> None:
        live = [t for t in tracks if t.mean is not None]
        if not live:
            return
        mean = np.stack([t.mean.copy() for t in live])
        for i, t in enumerate(live):
            if t.state != TrackState.TRACKED:
                mean[i, 7] = 0.0  # a lost track's height velocity is frozen
                if t.wet and self.max_time_lost_water is not None:
                    mean[i, 4:6] *= self.coast_damping  # [2.24.0] coast to a stop near where the swimmer went under
        cov = np.stack([t.cov for t in live])
        mean, cov = self.kf.multi_predict(mean, cov)
        for i, t in enumerate(live):
            t.mean, t.cov = mean[i], cov[i]

    def _lost_limit(self, t: STrack) -> int:
        """Frames a lost track is kept: longer if it was last seen in the water and TRACK_LOST_SECONDS_WATER is on."""
        if t.wet and self.max_time_lost_water is not None:
            return self.max_time_lost_water
        return self.max_time_lost

    def _first_cost(self, pool: list[STrack], dets: list[STrack]) -> np.ndarray:
        def cost(tracks: list) -> np.ndarray:
            if self.assoc_cost == "iou_dist":
                return _iou_dist_cost(tracks, dets, self.assoc_dist_radius)
            return _iou_cost(tracks, dets, True)

        c = cost(pool)
        if self.max_time_lost_water is not None and dets:
            # [2.24.0] a track lost in the water also matches where it was last seen: a swimmer surfaces near where
            # they went under, while the (damped) prediction has slid on
            rows = [i for i, t in enumerate(pool) if t.state == TrackState.LOST and t.wet]
            if rows:
                c[rows] = np.minimum(c[rows], cost([_AtLastSeen(pool[i]) for i in rows]))
        return c

    def _second_cost(self, remaining: list[STrack], dets: list[STrack]) -> np.ndarray:
        """2nd association cost (low-score boxes vs still-tracked confirmed tracks): 1 - IoU with the Kalman prediction.

        [2.35.0] TRACK_SECOND_MATCH_LAST_SEEN: a box that misses the prediction (which drifts on a splashing body) may
        match the track's LAST OBSERVED box instead, at the same IoU - but only when no other track fits that box (by
        prediction or last observed box): a weak box that could be two people's goes to neither. Pairs the prediction
        already matches are unchanged, so this only adds claims."""
        c = _iou_cost(remaining, dets, False)
        if not self.second_last_seen or not remaining or not dets:
            return c
        thr = self.second_match_thresh
        c_last = _iou_cost([_AtLastSeen(t) for t in remaining], dets, False)
        fits = (np.minimum(c, c_last) <= thr).sum(axis=0)
        extra = (c_last <= thr) & (c > thr) & (fits == 1)[None, :]
        return np.where(extra, c_last, c)

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        refind_ok: np.ndarray | None = None,
        confirm_now: set | None = None,
        det_wet: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (input detection indices, external ids) for confirmed tracks
        matched this frame. `refind_ok` (bool per detection) limits which
        low-score boxes the 3rd association may use (e.g. only boxes in water).
        A track born from a detection index in `confirm_now` is confirmed at
        once (the caller already confirmed it over several frames).
        """
        self.frame_id += 1
        boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
        scores = np.asarray(scores, np.float64).reshape(-1)
        high = np.where(scores >= self.activation_threshold)[0]
        second = np.where((scores > self.low_threshold) & (scores < self.activation_threshold))[0]
        dets = [STrack(boxes[i], scores[i], i) for i in high]
        dets_second = [STrack(boxes[i], scores[i], i) for i in second]
        if det_wet is not None:
            for d in dets + dets_second:
                d.wet = bool(det_wet[d.det_idx])

        for t in self.tracked + self.lost:
            t.matched_this_frame = False
        unconfirmed = [t for t in self.tracked if not t.is_activated]
        confirmed = [t for t in self.tracked if t.is_activated]
        pool = confirmed + [t for t in self.lost if t not in confirmed]
        self._predict(pool)

        activated: list[STrack] = []
        refound: list[STrack] = []
        lost_now: list[STrack] = []
        removed: list[STrack] = []

        # 1st association: high-score boxes vs every confirmed + lost track
        matches, u_track, u_det = _assign(self._first_cost(pool, dets), self.match_thresh)
        for ti, di in matches:
            t = pool[ti]
            if t.state == TrackState.TRACKED:
                t.update(self.kf, dets[di], self.frame_id, self._new_external)
                activated.append(t)
            else:
                t.re_activate(self.kf, dets[di], self.frame_id)
                refound.append(t)

        # 2nd association: low-score boxes can only extend still-tracked tracks
        remaining = [pool[i] for i in u_track if pool[i].state == TrackState.TRACKED]
        matches, u_rem, u_second = _assign(self._second_cost(remaining, dets_second), self.second_match_thresh)
        for ti, di in matches:
            remaining[ti].update(self.kf, dets_second[di], self.frame_id, self._new_external)
            activated.append(remaining[ti])

        # 3rd association (extension): leftover low-score boxes may re-find a LOST track
        if self.refind_low_iou is not None and u_second:
            lost_pool = [
                pool[i]
                for i in u_track
                if pool[i].state == TrackState.LOST
                and pool[i].is_activated
                and (
                    self.refind_max_frames is None
                    or self.frame_id - pool[i].end_frame
                    <= (
                        self._lost_limit(pool[i])
                        if pool[i].wet and self.max_time_lost_water is not None
                        else self.refind_max_frames
                    )
                )
            ]
            rest = [dets_second[i] for i in u_second if refind_ok is None or bool(refind_ok[dets_second[i].det_idx])]
            if lost_pool and rest:
                # against where the person was LAST SEEN — not the Kalman
                # prediction, which keeps sliding along the last velocity for
                # the whole lost buffer and would "re-find" a stranger
                last = _iou_matrix(np.stack([t.last_det_xyxy for t in lost_pool]), np.stack([d._xyxy for d in rest]))
                matches, _, _ = _assign(1.0 - last, 1.0 - float(self.refind_low_iou))
                for ti, di in matches:
                    lost_pool[ti].re_activate(self.kf, rest[di], self.frame_id)
                    refound.append(lost_pool[ti])
        for ti in u_rem:
            t = remaining[ti]
            if t.state != TrackState.LOST:
                t.state = TrackState.LOST
                lost_now.append(t)

        # unconfirmed (born last frame) vs leftover high-score boxes
        left = [dets[i] for i in u_det]
        matches, u_unc, u_left = _assign(_iou_cost(unconfirmed, left, True), self.unconfirmed_match_thresh)
        for ti, di in matches:
            unconfirmed[ti].update(self.kf, left[di], self.frame_id, self._new_external)
            activated.append(unconfirmed[ti])
        for ti in u_unc:
            unconfirmed[ti].state = TrackState.REMOVED
            removed.append(unconfirmed[ti])

        # [2.18.0] birth guard (finite clips): an unmatched high-score box on top of a confirmed track that found no box
        # this frame continues that track instead of starting a second one. Score fusion (IoU x score) fails a
        # splashing victim scored ~0.4 whose box swings in size, and the new track then shares the victim with the
        # old one (two ids alternating frame by frame).
        guard = [
            t
            for t in pool
            if self.birth_guard_iou is not None
            and t.is_activated
            and not t.matched_this_frame
            and (t.state == TrackState.TRACKED or self.frame_id - t.end_frame <= self.max_time_lost)
        ]
        for di in list(u_left):
            d = left[di]
            if not guard or d.score < self.det_threshold:
                continue
            ious = _iou_matrix(np.stack([t.last_det_xyxy for t in guard]), d._xyxy[None])[:, 0]
            k = int(np.argmax(ious))
            if ious[k] < float(self.birth_guard_iou or 0.0):
                continue
            t = guard.pop(k)
            if t.state == TrackState.TRACKED or t in lost_now:
                t.update(self.kf, d, self.frame_id, self._new_external)
                activated.append(t)
            else:
                t.re_activate(self.kf, d, self.frame_id)
                refound.append(t)
            u_left.remove(di)

        # births
        for di in u_left:
            d = left[di]
            if d.score < self.det_threshold:
                continue
            d.activate(self.kf, self.frame_id, self._next_internal, self._new_external)
            if confirm_now is not None and d.det_idx in confirm_now and not d.is_activated:
                d.is_activated = True  # already confirmed over several frames by the caller
                d.external_id = self._new_external()
            self._next_internal += 1
            activated.append(d)

        for t in self.lost:
            if t.state == TrackState.LOST and self.frame_id - t.end_frame > self._lost_limit(t):
                t.state = TrackState.REMOVED
                removed.append(t)

        keep_ids = set()
        tracked: list[STrack] = []
        for t in [t for t in self.tracked if t.state == TrackState.TRACKED] + activated + refound:
            if id(t) not in keep_ids:
                keep_ids.add(id(t))
                tracked.append(t)
        lost = [t for t in self.lost + lost_now if t.state == TrackState.LOST and id(t) not in keep_ids]
        tracked, lost = self._remove_duplicates(tracked, list(dict.fromkeys(lost)))
        self.tracked, self.lost = tracked, lost

        out_idx: list[int] = []
        out_ids: list[int] = []
        for t in self.tracked:
            if not (t.is_activated and t.matched_this_frame):
                continue
            out_idx.append(t.det_idx)
            out_ids.append(t.external_id)
        return np.asarray(out_idx, int), np.asarray(out_ids, int)

    @staticmethod
    def _remove_duplicates(tracked: list[STrack], lost: list[STrack]) -> tuple[list[STrack], list[STrack]]:
        """A tracked and a lost track on the same spot (IoU > 0.95): keep the longer-lived one."""
        if not tracked or not lost:
            return tracked, lost
        cost = _iou_cost(tracked, lost, False)
        drop_t, drop_l = set(), set()
        for p, q in zip(*np.where(cost < 0.05)):
            tp = tracked[p].frame_id - tracked[p].start_frame
            tq = lost[q].frame_id - lost[q].start_frame
            if tp > tq:
                drop_l.add(q)
            else:
                drop_t.add(p)
        return (
            [t for i, t in enumerate(tracked) if i not in drop_t],
            [t for i, t in enumerate(lost) if i not in drop_l],
        )
