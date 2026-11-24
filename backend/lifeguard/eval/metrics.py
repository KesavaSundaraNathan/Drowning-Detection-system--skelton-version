# mypy: disable-error-code="assignment"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""
[v3] Ported verbatim from the v2 baseline evaluation/metrics.py (imports only).

evaluation/metrics.py — metrics for the real-footage eval set.

Detection
    Per annotated frame, predictions are matched 1:1 to VISIBLE ground-truth
    boxes (Hungarian, IoU >= iou_thr). A partially submerged swimmer's GT box
    is often just a head: a prediction that covers it (IoA >= 0.6) without
    being much larger (area <= 4x) also counts. Predictions left over are
    matched against invisible / ignored GT (don't-care) before being counted
    as false positives.
    Recall is reported for in-water GT overall, by box-height bucket and by
    condition tag; precision overall and by condition.

Tracking (in-water GT tracks, over annotated frames)
    ID switches (CLEAR-MOT), fragmentations, IDF1, and continuity through
    "invisible" gaps (same predicted id before and after a dive / occlusion).

Events
    A GT event is detected when an alert fires inside
    [start - EVENT_PRE_TOLERANCE_S, end + EVENT_GRACE_S] on the event's
    person (identity via the majority pred->GT mapping, else spatial overlap
    with the GT box). Time-to-alert = first such alert - event start.
    [Phase 3 Step 0] Also reported, stricter: detected_in_window (an alert
    inside [start - EVENT_PRE_TOLERANCE_S, end], no grace, and one alert
    credits at most one event), and the victim's track continuity at the GT
    keyframes (victim_continuity). False alerts carry `notified`: whether a
    live camera would have sent a notification for it, as scene_tracker does
    it (every rising edge, plus an active alert again after
    LIVE_ALERT_COOLDOWN_SECONDS).
    Every other alert is a false alert, attributed to what the person it
    fired on was really doing (normal swimming / standing in water /
    out of water / other in-water / no person).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lifeguard.eval.annotations import (
    ACTIVITY_BUCKET,
    VISIBLE,
    ClipAnnotation,
    GTBox,
    GTEvent,
    _interval_lookup,
)

logger = logging.getLogger(__name__)

EVENT_PRE_TOLERANCE_S = 1.0  # an alert this early is still credited to the event

# [Stage 4] the per-class alert columns are gone: an alert is one "Drowning Alert" with a reason


EVENT_GRACE_S = 15.0  # alerts this long after the event end are still credited
SIZE_BUCKETS = ((0, 20, "tiny<20px"), (20, 40, "small20-40px"), (40, 80, "medium40-80px"), (80, 1e9, "large>=80px"))
GAP_BUCKETS_S = (2.0, 5.0, 10.0)


# ═════════════════════════════════════════════════════════════════════════════
# Prediction records (filled by evaluation.runner)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class PredPerson:
    track_id: int  # -1 = tentative / unconfirmed
    bbox: tuple[float, float, float, float]
    conf: float = 1.0
    in_water: bool | None = None  # pipeline's Stage-1 decision (None = unknown)
    state: str = ""
    zone_state: str | None = None  # in_water | edge | out_of_water (ML core v2), None = unknown


@dataclass
class PredAlert:
    t: float
    frame: int
    track_id: int
    scenario: str
    bbox: tuple[float, float, float, float]
    ghost: bool = False
    first: bool = True
    gate: dict[str, float] | None = None  # [2.20.0] track history at the alert (scene/temporal_gate.py)


# ═════════════════════════════════════════════════════════════════════════════
# Geometry
# ═════════════════════════════════════════════════════════════════════════════
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)).astype(np.float32)


def _similarity(gt: np.ndarray, pr: np.ndarray, iou_thr: float) -> np.ndarray:
    """IoU, with an IoA fallback for tiny head-only GT boxes (see module doc)."""
    sim = iou_matrix(gt, pr)
    if len(gt) and len(pr):
        ix1 = np.maximum(gt[:, None, 0], pr[None, :, 0])
        iy1 = np.maximum(gt[:, None, 1], pr[None, :, 1])
        ix2 = np.minimum(gt[:, None, 2], pr[None, :, 2])
        iy2 = np.minimum(gt[:, None, 3], pr[None, :, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        ag = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
        ap = (pr[:, 2] - pr[:, 0]) * (pr[:, 3] - pr[:, 1])
        ioa = inter / np.maximum(ag[:, None], 1e-9)
        ratio = ap[None, :] / np.maximum(ag[:, None], 1e-9)
        covered = (ioa >= 0.6) & (ratio <= 4.0)
        sim = np.where(covered & (sim < iou_thr), iou_thr, sim)
    return sim


def _assign(sim: np.ndarray, thr: float) -> list[tuple[int, int]]:
    if sim.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(-sim)
        return [(int(i), int(j)) for i, j in zip(r, c) if sim[i, j] >= thr]
    except Exception:  # greedy fallback
        logger.warning("Hungarian assignment failed - greedy fallback", exc_info=True)
        pairs, used_r, used_c = [], set(), set()
        for flat in np.argsort(-sim, axis=None):
            i, j = np.unravel_index(flat, sim.shape)
            if sim[i, j] < thr:
                break
            if i in used_r or j in used_c:
                continue
            pairs.append((int(i), int(j)))
            used_r.add(i)
            used_c.add(j)
        return pairs


@dataclass
class FrameMatch:
    pairs: list[tuple[GTBox, PredPerson]]  # visible GT ↔ prediction
    missed: list[GTBox]  # visible GT with no prediction
    false_pos: list[PredPerson]  # predictions on nothing
    dont_care: list[PredPerson]  # predictions on invisible / ignored GT


def match_frame(gts: Sequence[GTBox], preds: Sequence[PredPerson], iou_thr: float = 0.3) -> FrameMatch:
    vis = [g for g in gts if g.visible]
    care_not = [g for g in gts if not g.visible]
    pr = np.array([p.bbox for p in preds], np.float32).reshape(-1, 4)
    gv = np.array([g.bbox for g in vis], np.float32).reshape(-1, 4)
    pairs = _assign(_similarity(gv, pr, iou_thr), iou_thr)
    used_g = {i for i, _ in pairs}
    used_p = {j for _, j in pairs}
    rest = [j for j in range(len(preds)) if j not in used_p]
    dont_care: list[PredPerson] = []
    if rest and care_not:
        gi = np.array([g.bbox for g in care_not], np.float32).reshape(-1, 4)
        sub = _similarity(gi, pr[rest], iou_thr)
        for _, jj in _assign(sub, iou_thr):
            dont_care.append(preds[rest[jj]])
        dc = {id(p) for p in dont_care}
        rest = [j for j in rest if id(preds[j]) not in dc]
    return FrameMatch(
        pairs=[(vis[i], preds[j]) for i, j in pairs],
        missed=[vis[i] for i in range(len(vis)) if i not in used_g],
        false_pos=[preds[j] for j in rest],
        dont_care=dont_care,
    )


def size_bucket(h: float) -> str:
    for lo, hi, name in SIZE_BUCKETS:
        if lo <= h < hi:
            return name
    return SIZE_BUCKETS[-1][2]


# ═════════════════════════════════════════════════════════════════════════════
# Detection + tracking accumulation for one clip
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class _Counts:
    tp: int = 0
    fn: int = 0

    @property
    def recall(self) -> float | None:
        n = self.tp + self.fn
        return self.tp / n if n else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fn": self.fn,
            "n": self.tp + self.fn,
            "recall": round(self.recall, 4) if self.recall is not None else None,
        }


@dataclass
class ClipDetectionTracking:
    in_water: dict[str, _Counts] = field(default_factory=lambda: defaultdict(_Counts))
    out_of_water: _Counts = field(default_factory=_Counts)
    pred_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    pred_fp: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    water_confusion: Counter = field(default_factory=Counter)  # (gt_in_water, pred_in_water) → n
    zone_confusion: Counter = field(default_factory=Counter)  # (gt zone_state, pred zone_state) → n
    # crowd capacity: (visible GT people, confirmed predicted people) per annotated frame
    counts: list[tuple[int, int]] = field(default_factory=list)
    gt_seq: dict[int, list[tuple[int, int | None, str]]] = field(default_factory=lambda: defaultdict(list))
    cooc: Counter = field(default_factory=Counter)  # (gt_id, pred_id) → frames
    pred_confirmed_dets: int = 0
    gt_visible_dets: int = 0
    frames: int = 0
    # pred track → GT track (majority) for event attribution
    pred_to_gt_votes: dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))

    def add_frame(
        self, frame: int, gts: Sequence[GTBox], preds: Sequence[PredPerson], conditions: set[str], iou_thr: float
    ) -> FrameMatch:
        m = match_frame(gts, preds, iou_thr)
        self.frames += 1
        for g, p in m.pairs:
            self._gt_hit(g, conditions, True)
            if g.in_water is not None and p.in_water is not None:
                self.water_confusion[(g.in_water, p.in_water)] += 1
            if g.zone_state is not None and p.zone_state is not None:
                self.zone_confusion[(g.zone_state, p.zone_state)] += 1
            if p.track_id >= 0:
                self.pred_to_gt_votes[p.track_id][g.track_id] += 1
                self.cooc[(g.track_id, p.track_id)] += 1
        for g in m.missed:
            self._gt_hit(g, conditions, False)
        keys = ["all"] + sorted(conditions)
        for k in keys:
            self.pred_total[k] += len(m.pairs) + len(m.false_pos)
            self.pred_fp[k] += len(m.false_pos)
        self.pred_confirmed_dets += sum(1 for p in preds if p.track_id >= 0)
        self.gt_visible_dets += sum(1 for g in gts if g.visible)
        self.counts.append((sum(1 for g in gts if g.visible), sum(1 for p in preds if p.track_id >= 0)))

        matched = {id(g): p for g, p in m.pairs}
        for g in gts:
            if g.ignore or g.in_water is False:
                continue
            p = matched.get(id(g))
            pid = p.track_id if (p is not None and p.track_id >= 0) else None
            self.gt_seq[g.track_id].append((frame, pid, g.vis))
        return m

    def _gt_hit(self, g: GTBox, conditions: set[str], hit: bool) -> None:
        if g.in_water is False:
            c = self.out_of_water
            if hit:
                c.tp += 1
            else:
                c.fn += 1
            return
        keys = ["all", size_bucket(g.height)] + [f"cond:{c}" for c in sorted(conditions)] + [f"vis:{g.vis}"]
        if not conditions:
            keys.append("cond:clean")
        for k in keys:
            c = self.in_water[k]
            if hit:
                c.tp += 1
            else:
                c.fn += 1

    # ── tracking summary ──────────────────────────────────────────────────────
    def tracking(self, fps: float) -> dict[str, Any]:
        idsw = frag = 0
        gaps = {b: [0, 0] for b in GAP_BUCKETS_S}  # kept, total
        for gid, seq in self.gt_seq.items():
            seq.sort()
            last_pid: int | None = None
            was_tracked = False
            for i, (f, pid, vis) in enumerate(seq):
                if pid is not None:
                    if last_pid is not None and pid != last_pid:
                        idsw += 1
                    if not was_tracked and last_pid is not None:
                        frag += 1
                    last_pid, was_tracked = pid, True
                else:
                    was_tracked = False
            # continuity through invisible gaps
            i = 0
            while i < len(seq):
                if seq[i][2] == "invisible":
                    j = i
                    while j < len(seq) and seq[j][2] == "invisible":
                        j += 1
                    before = next((s[1] for s in reversed(seq[:i]) if s[1] is not None), None)
                    after = next((s[1] for s in seq[j:] if s[1] is not None), None)
                    if before is not None and after is not None:
                        dur = (seq[j - 1][0] - seq[i][0] + 1) / max(fps, 1e-6)
                        for b in GAP_BUCKETS_S:
                            if dur <= b:
                                gaps[b][1] += 1
                                gaps[b][0] += int(before == after)
                    i = j
                else:
                    i += 1
        idf1 = self._idf1()
        return {
            "id_switches": idsw,
            "fragmentations": frag,
            "gt_tracks": len(self.gt_seq),
            "idf1": round(idf1, 4) if idf1 is not None else None,
            "gap_continuity": {
                f"<={b:g}s": {"kept": k, "gaps": n, "rate": round(k / n, 4) if n else None}
                for b, (k, n) in gaps.items()
            },
        }

    def _idf1(self) -> float | None:
        if not self.cooc:
            return None
        gids = sorted({g for g, _ in self.cooc})
        pids = sorted({p for _, p in self.cooc})
        mat = np.zeros((len(gids), len(pids)), np.float32)
        gi = {g: i for i, g in enumerate(gids)}
        pi = {p: i for i, p in enumerate(pids)}
        for (g, p), n in self.cooc.items():
            mat[gi[g], pi[p]] = n
        idtp = sum(mat[i, j] for i, j in _assign(mat, 1.0))
        denom = self.gt_visible_dets + self.pred_confirmed_dets
        return 2.0 * idtp / denom if denom else None

    def pred_to_gt(self) -> dict[int, int]:
        return {pid: votes.most_common(1)[0][0] for pid, votes in self.pred_to_gt_votes.items() if votes}

    def detection(self) -> dict[str, Any]:
        prec = {
            k: {"preds": n, "fp": self.pred_fp[k], "precision": round(1 - self.pred_fp[k] / n, 4) if n else None}
            for k, n in self.pred_total.items()
        }
        conf = {
            f"gt_{'in' if a else 'out'}_pred_{'in' if b else 'out'}": n for (a, b), n in self.water_confusion.items()
        }
        zconf = {f"gt_{a}_pred_{b}": n for (a, b), n in sorted(self.zone_confusion.items())}
        return {
            "in_water_recall": {k: v.to_dict() for k, v in sorted(self.in_water.items())},
            "out_of_water_recall": self.out_of_water.to_dict(),
            "precision": prec,
            "water_state_confusion": conf,
            "zone_state_confusion": zconf,
            "zone_state_accuracy": zone_accuracy(self.zone_confusion),
            "crowd_count": crowd_count(self.counts),
            "frames": self.frames,
        }


ZONE_STATES = ("in_water", "edge", "out_of_water")
CROWD_MIN_PEOPLE = 15  # client requirement: 15–30 people at once


def zone_accuracy(conf: Counter) -> float | None:
    n = sum(conf.values())
    return round(sum(v for (a, b), v in conf.items() if a == b) / n, 4) if n else None


def crowd_count(counts: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Person-count error per annotated frame: all frames, and crowded (>= CROWD_MIN_PEOPLE) frames."""

    def summ(rows):
        if not rows:
            return {"frames": 0, "mae": None, "mean_gt": None, "mean_pred": None, "under_count_rate": None}
        g = np.array([r[0] for r in rows], float)
        p = np.array([r[1] for r in rows], float)
        return {
            "frames": len(rows),
            "mae": round(float(np.abs(g - p).mean()), 3),
            "mean_gt": round(float(g.mean()), 2),
            "mean_pred": round(float(p.mean()), 2),
            "under_count_rate": round(float((p < g).mean()), 4),
        }

    return {
        "all": summ(list(counts)),
        f"gt>={CROWD_MIN_PEOPLE}": summ([c for c in counts if c[0] >= CROWD_MIN_PEOPLE]),
        "max_gt_people": max((c[0] for c in counts), default=0),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Events + false alerts
# ═════════════════════════════════════════════════════════════════════════════
def _box_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    """IoU >= 0.1, or the alert box centre inside the GT box grown by 50 %."""
    if iou_matrix(np.array([a], np.float32), np.array([b], np.float32))[0, 0] >= 0.1:
        return True
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    w, h = b[2] - b[0], b[3] - b[1]
    return (b[0] - 0.5 * w <= cx <= b[2] + 0.5 * w) and (b[1] - 0.5 * h <= cy <= b[3] + 0.5 * h)


def _gt_person_at(clip: ClipAnnotation, alert: PredAlert, mapped: int | None) -> int | None:
    if mapped is not None and mapped in clip.tracks:
        return mapped
    best, best_iou = None, 0.0
    for tid in clip.tracks:
        g = clip.track_box_at(tid, alert.frame)
        if g is None or alert.frame - g.frame > 3 * clip.fps:
            continue
        iou = float(iou_matrix(np.array([alert.bbox], np.float32), np.array([g.bbox], np.float32))[0, 0])
        if _box_overlap(alert.bbox, g.bbox) and iou >= best_iou:
            best, best_iou = tid, iou
    return best


def victim_continuity(
    clip: ClipAnnotation, people: dict[int, list[PredPerson]], ev: GTEvent, iou_thr: float = 0.3
) -> dict[str, int]:
    """The victim's confirmed predicted track at each GT keyframe of the event (visible victim box, predictions
    stored for that frame), identified as score_events identifies the alerted person: a confirmed box whose centre
    lies in the GT box widened by half its size (`_box_overlap`), the highest-IoU one if several. (An IoU >= 0.3
    match found the victim on only 6-25 % of eval/heldout_v1 dev keyframes: its victim boxes are drawn tighter
    than any detector's, median IoU 0.08-0.18.) `iou_thr` is unused, kept for the signature.

    keyframes / matched: keyframes scored / with a matching confirmed track; distinct_ids: different track ids
    matched; pairs / kept: consecutive matched keyframes / with the same id (a kept pair may still hide a loss
    and a swap back between them - keyframes are sparse, 1-20 s apart on eval/heldout_v1).
    """
    ids: list[int | None] = []
    for f in clip.tracks.get(ev.track_id, {}).get("_frames", ()):
        if not (ev.start_frame <= f <= ev.end_frame) or f not in people:
            continue
        g = next((b for b in clip.boxes_by_frame.get(f, ()) if b.track_id == ev.track_id and b.vis in VISIBLE), None)
        if g is None:
            continue
        preds = [p for p in people[f] if p.track_id >= 0 and _box_overlap(p.bbox, g.bbox)]
        if not preds:
            ids.append(None)
            continue
        sim = iou_matrix(np.array([g.bbox], np.float32), np.array([p.bbox for p in preds], np.float32))[0]
        ids.append(preds[int(np.argmax(sim))].track_id)
    matched = [i for i in ids if i is not None]
    pairs = [(a, b) for a, b in zip(ids, ids[1:]) if a is not None and b is not None]
    return {
        "keyframes": len(ids),
        "matched": len(matched),
        "distinct_ids": len(set(matched)),
        "pairs": len(pairs),
        "kept": sum(a == b for a, b in pairs),
    }


def score_events(
    clip: ClipAnnotation,
    alerts: Sequence[PredAlert],
    pred_to_gt: dict[int, int],
    people: dict[int, list[PredPerson]] | None = None,
    cooldown_s: float | None = None,
) -> dict[str, Any]:
    fps = clip.fps
    used: set[int] = set()
    events_out = []
    matched_idx: list[list[int]] = []
    for ev in clip.events:
        t0, t1 = ev.start_frame / fps, ev.end_frame / fps
        hit: PredAlert | None = None
        matched: list[PredAlert] = []
        matched_idx.append([])
        for k, a in enumerate(alerts):
            if not (t0 - EVENT_PRE_TOLERANCE_S <= a.t <= t1 + EVENT_GRACE_S):
                continue
            same = pred_to_gt.get(a.track_id) == ev.track_id
            if not same:
                g = clip.track_box_at(ev.track_id, a.frame)
                same = g is not None and _box_overlap(a.bbox, g.bbox)
            if same:
                used.add(k)
                matched.append(a)
                matched_idx[-1].append(k)
                if hit is None or a.t < hit.t:
                    hit = a
        events_out.append(
            {
                "id": ev.id,
                "type": ev.type,
                "track_id": ev.track_id,
                "start_s": round(t0, 2),
                "end_s": round(t1, 2),
                "detected": hit is not None,
                "time_to_alert_s": round(max(0.0, hit.t - t0), 2) if hit else None,
                "alert_reason": hit.scenario if hit else None,
                "alerts_gate": [[a.scenario, a.ghost, a.gate] for a in matched],
            }
        )
        if people is not None:
            events_out[-1]["victim"] = victim_continuity(clip, people, ev)

    # strict credit: inside the event window (no grace), each alert credits one event, events in start order
    credited: set[int] = set()
    for i in sorted(range(len(clip.events)), key=lambda n: clip.events[n].start_frame):
        ev = clip.events[i]
        t0, t1 = ev.start_frame / fps, ev.end_frame / fps
        inside = [k for k in matched_idx[i] if k not in credited and t0 - EVENT_PRE_TOLERANCE_S <= alerts[k].t <= t1]
        k_hit = min(inside, key=lambda k: alerts[k].t) if inside else None
        if k_hit is not None:
            credited.add(k_hit)
        events_out[i]["detected_in_window"] = k_hit is not None
        events_out[i]["time_to_alert_in_window_s"] = (
            round(max(0.0, alerts[k_hit].t - t0), 2) if k_hit is not None else None
        )

    if cooldown_s is None:
        from lifeguard.vision.constants import LIVE_ALERT_COOLDOWN_SECONDS

        cooldown_s = float(LIVE_ALERT_COOLDOWN_SECONDS)
    last_notify: dict[int, float] = {}
    notified: dict[int, bool] = {}
    for k in sorted(range(len(alerts)), key=lambda n: alerts[n].t):
        a = alerts[k]
        prev = last_notify.get(a.track_id)
        notified[k] = a.first or prev is None or a.t - prev >= cooldown_s - 1e-6
        if notified[k]:
            last_notify[a.track_id] = a.t

    false_alerts = []
    for k, a in enumerate(alerts):
        if k in used:
            continue
        gid = _gt_person_at(clip, a, pred_to_gt.get(a.track_id))
        if gid is None and not clip.tracks:
            bucket, act = "unattributed", None  # events-only clip: no person labels to attribute to
        elif gid is None:
            bucket, act = "no_person", None
        else:
            tr = clip.tracks[gid]
            state = _interval_lookup(tr["state"], a.frame)
            act = _interval_lookup(tr["activity"], a.frame)
            if state == "out_of_water":
                bucket = "out_of_water"
            else:
                bucket = ACTIVITY_BUCKET.get(act or "", "other_in_water")
                if bucket == "out_of_water":  # activity says deck but state says water
                    bucket = "other_in_water"
        false_alerts.append(
            {
                "t": round(a.t, 2),
                "track_id": a.track_id,
                "gt_track": gid,
                "bucket": bucket,
                "activity": act,
                "reason": a.scenario,
                "first": a.first,
                "ghost": a.ghost,
                "gate": a.gate,
                "notified": notified[k],
            }
        )
    # [Phase 5] alerts per event: the notified alerts grouped the way alerts/events.py groups a camera's alerts (per
    # track; a new event after EVENT_IDLE_S without an alert); and, per GT event, the notified alerts matched to it
    groups, last_t = 0, dict[int, float]()
    for k in sorted(range(len(alerts)), key=lambda n: alerts[n].t):
        a = alerts[k]
        if not notified[k]:
            continue
        prev = last_t.get(a.track_id)
        groups += int(prev is None or a.t - prev > EVENT_IDLE_S)
        last_t[a.track_id] = a.t
    for i, ev_out in enumerate(events_out):
        ev_out["notified_alerts"] = sum(1 for k in matched_idx[i] if notified[k])
    out: dict[str, Any] = {
        "events": events_out,
        "false_alerts": false_alerts,
        "alert_groups": {"notified": sum(1 for v in notified.values() if v), "events": groups},
    }
    # [Phase 3] continuity-only clips (eval/underwater_v1): the subject's track over the whole clip, no events
    cont = clip.raw.get("continuity_track")
    if people is not None and cont is not None and int(cont) in clip.tracks:
        whole = GTEvent("continuity", int(cont), "CONTINUITY", 0, max(0, clip.frame_count - 1))
        out["continuity"] = victim_continuity(clip, people, whole)
    return out


EVENT_IDLE_S = 30.0  # = alerts/events.EVENT_IDLE_CLOSE_S (not imported: the harness never loads the DB layer)


def activity_exposure(clip: ClipAnnotation) -> dict[str, float]:
    """Person-seconds per false-alert bucket (context for per-hour rates)."""
    out: dict[str, float] = defaultdict(float)
    for tr in clip.tracks.values():
        for s, e, act in tr["activity"]:
            state = _interval_lookup(tr["state"], int(s))
            b = "out_of_water" if state == "out_of_water" else ACTIVITY_BUCKET.get(act, "other_in_water")
            out[b] += (int(e) - int(s) + 1) / clip.fps
    return dict(out)


# ═════════════════════════════════════════════════════════════════════════════
# [Phase 11 WP3] Entry detection + in-water consistency (eval/site/levers_phase11.json "metrics_wp3")
# ═════════════════════════════════════════════════════════════════════════════
def person_sequences_metrics(gt_seq: dict[int, list[tuple[int, int | None, str]]], fps: float) -> dict[str, Any]:
    """From each GT person's monitored annotated frames [(frame, matched confirmed pred id or None, vis)]:

    * time_to_first_detection_s: first frame a confirmed track matches the person - first frame the person is
      visible (vis != invisible); a person never matched is "missed";
    * track_breaks_per_person_minute: (id switches + fragmentations) / person-minutes (first..last monitored frame).
    Counting matches ClipDetectionTracking.tracking().
    """
    fps = max(float(fps), 1e-6)
    times: list[float] = []
    missed = 0
    breaks = 0
    minutes = 0.0
    for _gid, seq in sorted(gt_seq.items()):
        seq = sorted(seq)
        if not seq:
            continue
        first_visible = next((f for f, _p, vis in seq if vis != "invisible"), None)
        if first_visible is not None:
            hit = next((f for f, pid, _v in seq if pid is not None and f >= first_visible), None)
            if hit is None:
                missed += 1
            else:
                times.append((hit - first_visible) / fps)
        last_pid: int | None = None
        was_tracked = False
        for _f, pid, _v in seq:
            if pid is not None:
                if last_pid is not None and pid != last_pid:
                    breaks += 1  # id switch
                if not was_tracked and last_pid is not None:
                    breaks += 1  # fragmentation
                last_pid, was_tracked = pid, True
            else:
                was_tracked = False
        minutes += (seq[-1][0] - seq[0][0]) / fps / 60.0
    return {
        "time_to_first_detection_s": time_summary(times, missed),
        "track_breaks": breaks,
        "person_minutes": round(minutes, 4),
        "track_breaks_per_person_minute": round(breaks / minutes, 4) if minutes > 0 else None,
    }


def time_summary(times: Sequence[float], missed: int = 0) -> dict[str, Any]:
    if not times:
        return {"n": 0, "missed": int(missed), "median": None, "p95": None}
    a = np.asarray(times, np.float64)
    return {
        "n": int(a.size),
        "missed": int(missed),
        "median": round(float(np.median(a)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "max": round(float(a.max()), 3),
    }


def pool_entry_metrics(clips: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pooled over clips: all first-detection times together, breaks / person-minutes summed."""
    times = [t for c in clips for t in c.get("_entry_times", [])]
    missed = sum(int(c["entry"]["time_to_first_detection_s"]["missed"]) for c in clips if c.get("entry"))
    breaks = sum(int(c["entry"]["track_breaks"]) for c in clips if c.get("entry"))
    minutes = sum(float(c["entry"]["person_minutes"]) for c in clips if c.get("entry"))
    return {
        "time_to_first_detection_s": time_summary(times, missed),
        "track_breaks": breaks,
        "person_minutes": round(minutes, 4),
        "track_breaks_per_person_minute": round(breaks / minutes, 4) if minutes > 0 else None,
    }


def entry_times(gt_seq: dict[int, list[tuple[int, int | None, str]]], fps: float) -> list[float]:
    """The raw first-detection times of person_sequences_metrics (for pooling percentiles across clips)."""
    fps = max(float(fps), 1e-6)
    out = []
    for seq in gt_seq.values():
        seq = sorted(seq)
        fv = next((f for f, _p, vis in seq if vis != "invisible"), None)
        hit = None if fv is None else next((f for f, pid, _v in seq if pid is not None and f >= fv), None)
        if hit is not None and fv is not None:
            out.append((hit - fv) / fps)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Latency
# ═════════════════════════════════════════════════════════════════════════════
def latency_summary(ms: Sequence[float]) -> dict[str, Any]:
    if not ms:
        return {"n": 0}
    a = np.asarray(ms, np.float64)
    p50, p95, p99 = (float(np.percentile(a, q)) for q in (50, 95, 99))
    return {
        "n": int(a.size),
        "mean_ms": round(float(a.mean()), 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "max_ms": round(float(a.max()), 2),
        "p99_over_p50": round(p99 / p50, 2) if p50 > 0 else None,
    }
