# mypy: disable-error-code="arg-type, return-value"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""Per-track scene state (TrackState), the per-sample frame pool, and small box helpers.

[v3] Split out of the v2 baseline detection/scene_tracker.py; definitions verbatim.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lifeguard.vision.behaviour.classifier import BYSTANDER, SWIMMER, BehaviorResult
from lifeguard.vision.behaviour.engine import TrackBuffer
from lifeguard.vision.constants import (
    CATEGORY_SWITCH_SAMPLES,
    GHOST_DUPLICATE_SYNC_SECONDS,
    KEYPOINT_CONF_THRESHOLD,
    MIN_TRACK_SAMPLES,
)
from lifeguard.vision.inference.detector import TrackedPerson
from lifeguard.vision.scene import zone_state as ZS
from lifeguard.vision.scene.geometry import ZoneHit
from lifeguard.vision.scene.hysteresis import AlertHysteresis

logger = logging.getLogger(__name__)


WHOLE_FRAME_HIT = ZoneHit(None, True, True, "Whole Frame", "ALL", 1.0, None)


# No zones drawn: a person with strong on-land evidence (legs on the deck).
ESTIMATED_DECK_HIT = ZoneHit(None, False, False, "Deck (estimated)", "DECK", 1.0, None)


# Time a zone decision must hold before a person switches Swimmer ↔ Bystander:
# CATEGORY_SWITCH_SAMPLES consecutive samples at the old 8 Hz = 0.25 s span.
CATEGORY_SWITCH_SECONDS = (CATEGORY_SWITCH_SAMPLES - 1) / 8.0


# L/R hip, knee, ankle
_HIPS, _KNEES, _ANKLES = (11, 12), (13, 14), (15, 16)


_SHOULDERS = (5, 6)


class _FramePool:
    """Frames of one track since its last behaviour sample."""

    __slots__ = ("boxes", "confs", "kps", "posed")

    def __init__(self) -> None:
        self.boxes: list[tuple[float, float, float, float]] = []
        self.confs: list[float] = []
        self.kps: list[list[list[float]] | None] = []
        self.posed = 0  # frames in the pool on which pose actually ran (or that had no pose stage at all)

    def add(self, p: TrackedPerson) -> None:
        self.boxes.append(tuple(float(v) for v in p.bbox))
        self.confs.append(float(p.confidence))
        if getattr(p, "pose_skipped", False):
            return  # box-only frame: its missing keypoints mean "not looked", never "body gone"
        self.kps.append(p.keypoints)
        self.posed += 1

    def __len__(self) -> int:
        return len(self.boxes)

    def pooled(self) -> tuple[tuple[float, float, float, float], float, list[list[float]] | None]:
        """(median box, max confidence, confidence-weighted mean keypoints)."""
        n = len(self.boxes)
        if n == 1:
            return self.boxes[0], self.confs[0], self.kps[0] if self.kps else None
        b = tuple(float(v) for v in np.median(np.asarray(self.boxes, np.float32), axis=0))
        kp_list = [np.asarray(k, np.float32) for k in self.kps if k is not None and len(k) >= 13]
        kp_out: list[list[float]] | None = None
        if kp_list:
            a = np.stack([k[:, :3] for k in kp_list if k.shape == kp_list[0].shape])
            conf = a[:, :, 2]
            w = np.where(conf >= KEYPOINT_CONF_THRESHOLD, conf, 0.0)
            wsum = w.sum(axis=0)
            last = a[-1]
            xy = np.where(
                wsum[:, None] > 0,
                (a[:, :, :2] * w[:, :, None]).sum(axis=0) / np.maximum(wsum[:, None], 1e-9),
                last[:, :2],
            )
            kp_out = [[float(xy[k, 0]), float(xy[k, 1]), float(conf[:, k].mean())] for k in range(a.shape[1])]
        return b, float(max(self.confs)), kp_out


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _ioa(a: Sequence[float], b: Sequence[float]) -> float:
    """Share of box `a` that lies inside box `b`."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)
    return inter / area


def _was_swimming(ts: TrackState) -> bool:
    b = ts.buffer
    return (
        b.last_activity == "SWIMMING"
        and b.last_normal_t is not None
        and b.last_t is not None
        and b.last_t - b.last_normal_t <= 3.0
    )


def _velocity(ts: TrackState) -> tuple[float, float]:
    """Box-centre velocity (px/s) over the track's last ~1 s."""
    idx = ts.buffer.recent(1.0)
    if len(idx) < 2:
        return (0.0, 0.0)
    a, b = ts.buffer.samples[idx[0]], ts.buffer.samples[idx[-1]]
    dt = max(b.t - a.t, 1e-3)
    return ((b.center[0] - a.center[0]) / dt, (b.center[1] - a.center[1]) / dt)


def _median_conf(ts: TrackState) -> float:
    confs = sorted(s.conf for s in ts.buffer.samples)
    return confs[len(confs) // 2] if confs else 0.0


def _box_at(ts: TrackState, t: float) -> tuple[float, float, float, float] | None:
    """The track's (pooled) box at time `t`, from its behaviour samples; None if it was not seen then."""
    best, best_dt = None, GHOST_DUPLICATE_SYNC_SECONDS
    for smp in reversed(ts.buffer.samples):
        dt = abs(smp.t - t)
        if dt <= best_dt:
            best, best_dt = smp.bbox, dt
        elif smp.t < t - GHOST_DUPLICATE_SYNC_SECONDS:
            break
    if best is None and abs(ts.last_seen - t) <= GHOST_DUPLICATE_SYNC_SECONDS:
        best = ts.last_bbox
    return best


# ═════════════════════════════════════════════════════════════════════════════
# Per-track state
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class TrackState:
    track_id: int
    buffer: TrackBuffer
    first_seen: float
    last_seen: float
    last_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    last_keypoints: list[list[float]] | None = None
    last_foot: tuple[float, float] = (0.0, 0.0)
    category: str = SWIMMER
    pending_category: str | None = None
    pending_count: int = 0
    hit: ZoneHit = field(default_factory=lambda: WHOLE_FRAME_HIT)
    edge_dist_px: float = math.inf
    mode: str = "Medium"
    result: BehaviorResult | None = None
    hyst: AlertHysteresis = field(default_factory=AlertHysteresis)
    last_alert_time: float = -math.inf
    last_notify_time: float = -math.inf
    # [Phase 4 Step 4] when and as what the track's last alert cleared (re-raise damping)
    last_alert_end: float = -math.inf
    last_alert_scenario: str | None = None
    peak_risk: float = 0.0
    state: str = "ANALYZING"
    deck_edge: bool = False
    # False when the zone's ScheduleMonitoring window is closed: the track is
    # still scored and logged, but its alerts do not sound the siren.
    monitoring_active: bool = True
    # ── every-frame ingestion (Phase 3.1) ────────────────────────────────
    pool: _FramePool = field(default_factory=_FramePool)
    last_sample_t: float = -math.inf
    frames_seen: int = 0
    peak_conf: float = 0.0
    pending_since: float | None = None
    # ── in / out of water without zones (Phase 2.4): (t, on_land, feet_in_water)
    land_obs: list[tuple[float, bool, bool | None]] = field(default_factory=list)
    category_changed_at: float | None = None
    # ── position on the pool plan (this camera's calibration; no cross-camera identity, Phase 5)
    pool_xy: tuple[float, float] | None = None
    # ── zone state IN_WATER / EDGE / DECK (detection/zone_state.py) ──────
    zone_state: str = ZS.IN_WATER
    pending_zone_state: str | None = None
    pending_zone_since: float | None = None
    water_vote: ZS.WaterVote = field(default_factory=ZS.WaterVote)
    # ── tracking v3 ──────────────────────────────────────────────────────
    # When this track last came (back) into view: creation, or revival from
    # dormant (gone ≥ TRACK_STALE_SECONDS). A submersion ghost may be resolved
    # by any track that APPEARED after the loss — not only a brand-new id (the
    # person often comes back under an older id: a revived dormant track) —
    # but never by someone who was in view all along. (A neighbour whose
    # detections merely flicker must NOT count: that cleared a real victim's
    # ghost on web_slide_pool_submersion.)
    visible_since: float = 0.0
    # Started by evidence from faint boxes (tracking.FaintBirths): eligible as
    # a submersion ghost on the GHOST_FAINT_* floors.
    faint_birth: bool = False

    @property
    def warm(self) -> bool:
        return len(self.buffer) >= MIN_TRACK_SAMPLES

    @property
    def monitored(self) -> bool:
        return self.category == SWIMMER

    @property
    def alert_active(self) -> bool:
        return self.hyst.active

    @property
    def alert_started_at(self) -> float | None:
        return self.hyst.started_at

    def clear_alert(self) -> None:
        self.hyst.reset()

    def keypoints_norm(self, frame_w: int, frame_h: int) -> list[list[float] | None] | None:
        """
        17 COCO keypoints normalised to the frame, for the client-side Model
        View canvas. Landmarks below the confidence floor become null so the
        overlay skips them rather than drawing a limb to (0, 0).
        """
        if not self.last_keypoints:
            return None
        fw, fh = max(frame_w, 1), max(frame_h, 1)
        out: list[list[float] | None] = []
        for kp in self.last_keypoints:
            try:
                x, y, c = float(kp[0]), float(kp[1]), float(kp[2])
            except (TypeError, ValueError, IndexError):
                out.append(None)
                continue
            out.append(None if c < KEYPOINT_CONF_THRESHOLD else [round(x / fw, 4), round(y / fh, 4), round(c, 2)])
        return out

    @property
    def activity(self) -> str:
        """Recognised activity for the UI (includes entering / leaving the water)."""
        if self.category_changed_at is not None and self.last_seen - self.category_changed_at <= 3.0:
            return "ENTERING" if self.category == SWIMMER else "EXITING"
        if self.category == BYSTANDER:
            return "OUT_OF_WATER"
        return self.result.activity if self.result is not None else "UNKNOWN"

    def to_event(self, frame_w: int, frame_h: int) -> dict[str, Any]:
        x1, y1, x2, y2 = self.last_bbox
        r = self.result
        return {
            "track_id": self.track_id,
            "state": self.state,
            "category": self.category,
            "label": r.label if r else ("Bystander / Non-Swimmer" if self.category == BYSTANDER else "Analyzing"),
            "drowning_type": r.label if r else "Analyzing",
            "scenario": r.scenario if r else "ANALYZING",
            "reason": r.reason if r else "",
            "zone": self.hit.zone_name,
            "zone_type": self.hit.zone_type,
            "monitored": self.monitored,
            "mode": self.mode,
            "risk": round(r.risk, 4) if r else 0.0,
            "risk_score": r.risk_score if r else 0,
            "raw_risk_score": r.raw_risk_score if r else 0,
            "confidence": round(r.confidence, 3) if r else 0.0,
            "priority": r.priority if (r and (self.alert_active or r.is_drowning)) else "None",
            "alert_active": self.alert_active,
            "monitoring_active": self.monitoring_active,
            "deck_edge": self.deck_edge,
            "samples": len(self.buffer),
            "bbox_norm": [
                round(x1 / max(frame_w, 1), 4),
                round(y1 / max(frame_h, 1), 4),
                round(x2 / max(frame_w, 1), 4),
                round(y2 / max(frame_h, 1), 4),
            ],
            "keypoints_norm": self.keypoints_norm(frame_w, frame_h),
            "activity": self.activity,
            "pose_reliable": bool(r.pose_reliable) if r else False,
            "global_id": None,  # [Phase 5] wire key kept (always null): no cross-camera identity
            "pool_xy": [round(self.pool_xy[0], 2), round(self.pool_xy[1], 2)] if self.pool_xy else None,
            "zone_state": self.zone_state,
            "water_vote": round(self.water_vote.value, 3) if self.water_vote.value is not None else None,
        }
