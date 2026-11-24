# mypy: disable-error-code="union-attr"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""Zone / category logic of the scene tracker (Stage 1: Swimmer vs Bystander, and
IN_WATER / EDGE / DECK, with time hysteresis).

[v3] Split out of the v2 baseline detection/scene_tracker.py: SceneBehaviorTracker methods,
verbatim, moved into a mixin. Plan: the colour water estimate is a FALLBACK - it is consulted
only when the camera has no zones (the zone loader turns a pool calibration into a WATER zone).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import cv2

from lifeguard.vision.behaviour.classifier import BYSTANDER, SWIMMER
from lifeguard.vision.constants import (
    FALL_IN_LAST_FRAMES,
    KEYPOINT_CONF_THRESHOLD,
    LAND_EVIDENCE_FRACTION,
    LAND_EVIDENCE_SECONDS,
    LAND_FROM_WATER_ESTIMATE,
    LAND_KP_CONF,
    LAND_LEG_MIN_RATIO,
    LAND_WATER_MARGIN,
    WATER_RETURN_SECONDS,
    ZONE_DEMOTE_MAX_RISK,
)
from lifeguard.vision.frame_rules import frames_at
from lifeguard.vision.inference.detector import TrackedPerson
from lifeguard.vision.scene import zone_state as ZS
from lifeguard.vision.scene.geometry import ZoneGeometry, ZoneHit
from lifeguard.vision.scene.track_state import (
    _ANKLES,
    _HIPS,
    _KNEES,
    _SHOULDERS,
    CATEGORY_SWITCH_SECONDS,
    WHOLE_FRAME_HIT,
    TrackState,
)
from lifeguard.vision.scene.water import WaterRegionEstimator

logger = logging.getLogger(__name__)


def demotion_blocked_by_risk(ts: TrackState) -> bool:
    """[2.18.0] A swimmer whose alert is active, or whose last result is a drowning call or at least
    ZONE_DEMOTE_MAX_RISK, is not moved to the deck by the vote or colour evidence (no geometry): demotion clears the
    alert and stops the scoring, the worst outcome for a victim near the edge or with a low, out-of-domain vote.
    ZONE_DEMOTE_MAX_RISK > 1 restores the old behaviour."""
    if ZONE_DEMOTE_MAX_RISK > 1.0:
        return False
    if ts.alert_active:
        return True
    r = ts.result
    return r is not None and (r.is_drowning or float(r.risk) >= ZONE_DEMOTE_MAX_RISK)


class ZoneCategoryLogic:
    """Mixed into SceneBehaviorTracker, which owns the state these methods read."""

    geometry: ZoneGeometry | None
    water: WaterRegionEstimator
    _water_mask_cache: tuple[Any, Any, int] | None = None  # (geometry, water mask, cell px) for box_outside_drawn_water
    _gate_cache: tuple[Any, Any, int] | None = None  # (geometry, summed-area table or None, cell px) for zone_gate_mask
    frame_w: int
    frame_h: int

    @property
    def zoned(self) -> bool:
        return self.geometry is not None and self.geometry.has_zones

    def update_water(self, frame: Any, t: float) -> None:
        """Feed the automatic water estimate (cheap no-op between updates; unused with zones)."""
        if not self.zoned and frame is not None:
            self.water.update(frame, t)

    def water_contains(self, point: tuple[float, float]) -> bool | None:
        """Is this point (px) in the water? True / False / None (unknown). Zones decide when drawn."""
        if self.zoned:
            try:
                return bool(self.geometry.classify_any([point]).monitored)
            except Exception:
                logger.warning("zone classification failed", exc_info=True)
                return None
        return self.water.contains(point)

    def box_outside_drawn_water(self, box: Any) -> bool:
        """[2.33.0] True only when the camera has a drawn WATER / SHALLOW polygon and the box (px, x1 y1 x2 y2) does not
        touch it anywhere. Without a drawn zone, always False: the colour estimate is never trusted to hide anything
        (on Red Rescue clip 1 it calls real swimmers dry on up to 78 % of their frames)."""
        g = self.geometry
        if g is None or not getattr(g, "drawn_water", False):
            return False
        import numpy as np

        cache = getattr(self, "_water_mask_cache", None)
        if cache is None or cache[0] is not g:
            s = 4  # px per mask cell: a box must clear the polygon by up to 4 px to count as outside
            m = np.zeros((g.frame_h // s + 1, g.frame_w // s + 1), np.uint8)
            for z in g.zones:
                if z.zone_type in ("WATER", "SHALLOW"):
                    cv2.fillPoly(m, [np.round(z.contour.reshape(-1, 2) / s).astype(np.int32)], 1)
            cache = self._water_mask_cache = (g, m, s)
        _, m, s = cache
        x1, y1, x2, y2 = (int(float(v) // s) for v in box[:4])
        x1, y1 = max(0, x1), max(0, y1)
        return not bool(m[y1 : max(y2, y1) + 1, x1 : max(x2, x1) + 1].any())

    def zone_gate_mask(self, boxes: Any) -> Any:
        """[2.43.0] ZONE_GATE: bool mask over boxes (N x 4 px), True = inside the gate; None = no gate (whole frame).

        Only with a drawn WATER / SHALLOW polygon. A box passes when the bottom ZONE_GATE_FOOT_BAND of it (feet / the
        part of a swimmer that is in the water), grown by max(ZONE_GATE_MARGIN_BOX x box height, ZONE_GATE_MARGIN_FRAME
        x frame height) on every side, touches the drawn water. One summed-area table per geometry: O(1) per box."""
        g = self.geometry
        if g is None or not getattr(g, "drawn_water", False):
            return None
        import numpy as np

        from lifeguard.vision.constants import ZONE_GATE_FOOT_BAND, ZONE_GATE_MARGIN_BOX, ZONE_GATE_MARGIN_FRAME

        cache = getattr(self, "_gate_cache", None)
        if cache is None or cache[0] is not g:
            s = 4
            m = np.zeros((g.frame_h // s + 1, g.frame_w // s + 1), np.uint8)
            for z in g.zones:
                if z.zone_type in ("WATER", "SHALLOW") and z.contour is not None and len(z.contour) >= 3:
                    cv2.fillPoly(m, [np.round(z.contour.reshape(-1, 2) / s).astype(np.int32)], 1)
            if int(m.sum()) < max(4, m.size // 1000):  # nothing (or < 0.1 % of the frame) drawn: not a usable zone
                logger.warning("%s: drawn water zone is empty or invalid; zone gate off (whole frame)", g.camera_id)
                sat = None
            else:
                sat = np.zeros((m.shape[0] + 1, m.shape[1] + 1), np.int32)
                sat[1:, 1:] = m.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
            cache = self._gate_cache = (g, sat, s)
        _, sat, s = cache
        if sat is None:
            return None
        b = np.asarray(boxes, np.float32).reshape(-1, 4)
        if len(b) == 0:
            return np.zeros(0, bool)
        h = np.maximum(b[:, 3] - b[:, 1], 1.0)
        margin = np.maximum(ZONE_GATE_MARGIN_BOX * h, ZONE_GATE_MARGIN_FRAME * float(g.frame_h))
        ny, nx = sat.shape[0] - 1, sat.shape[1] - 1
        x1 = np.clip(((b[:, 0] - margin) // s).astype(np.int64), 0, nx - 1)
        x2 = np.clip(((b[:, 2] + margin) // s).astype(np.int64), 0, nx - 1) + 1
        y1 = np.clip(((b[:, 3] - ZONE_GATE_FOOT_BAND * h - margin) // s).astype(np.int64), 0, ny - 1)
        y2 = np.clip(((b[:, 3] + margin) // s).astype(np.int64), 0, ny - 1) + 1
        return (sat[y2, x2] - sat[y1, x2] - sat[y2, x1] + sat[y1, x1]) > 0

    def water_roi(self) -> tuple[int, int, int, int] | None:
        """Pool bbox (px): from the zones when drawn, else the confident water estimate."""
        if self.zoned:
            import numpy as np

            pts = [z.contour.reshape(-1, 2) for z in self.geometry.zones if z.monitored]
            if not pts:
                return None
            from lifeguard.vision.inference.enhance import roi_of

            return roi_of(np.concatenate(pts), self.geometry.frame_w, self.geometry.frame_h)
        return self.water.roi()

    # ── Stage 1 helpers ───────────────────────────────────────────────────────
    def _zone_hit(self, person: TrackedPerson) -> ZoneHit:
        geom = self.geometry
        if geom is None or not geom.has_zones:
            return WHOLE_FRAME_HIT
        candidates: list[tuple[float, float]] = [person.foot_point]
        if person.keypoints:
            for idx in (15, 16):
                if len(person.keypoints) > idx and person.keypoints[idx][2] >= KEYPOINT_CONF_THRESHOLD:
                    candidates.append((person.keypoints[idx][0], person.keypoints[idx][1]))
        x1, y1, x2, y2 = person.bbox
        candidates.append(((x1 + x2) / 2.0, float(y2)))
        return geom.classify_any(candidates)

    def _edge_distance(self, point: tuple[float, float]) -> float:
        if self.zoned:
            return self.geometry.edge_distance(point)
        est = self.water.edge_distance(point)
        if est is not None:
            return est
        x, y = point
        if self.frame_w and self.frame_h:
            return float(min(x, y, self.frame_w - x, self.frame_h - y))
        return math.inf

    def _switch(self, ts: TrackState, observed: str, t: float) -> None:
        ts.category = observed
        ts.pending_category, ts.pending_count, ts.pending_since = None, 0, None
        ts.category_changed_at = t
        ts.clear_alert()
        ts.result = None

    def _monitored_hit(self, p: TrackedPerson, fallback: ZoneHit) -> ZoneHit:
        """The water zone nearest a person the fusion keeps monitored although their point is outside it."""
        water = [z for z in self.geometry.zones if z.monitored]
        if not water:
            return WHOLE_FRAME_HIT
        x1, y1, x2, y2 = p.bbox
        pt = ((x1 + x2) / 2.0, float(y2))
        z = max(water, key=lambda zz: zz.signed_distance(pt))
        return ZoneHit(z, True, True, z.name, z.zone_type, z.sensitivity_multiplier, z.schedule_id)

    def _update_zone_state(self, ts: TrackState, observed: str | None, t: float) -> None:
        """
        3-state zone (IN_WATER / EDGE / DECK), consistent with the
        hysteresis-protected Swimmer/Bystander category: DECK only while the
        category is BYSTANDER; IN_WATER ↔ EDGE switches after the observation
        held CATEGORY_SWITCH_SECONDS (anti-flicker at the coping).
        """
        if ts.category == BYSTANDER:
            ts.zone_state, ts.pending_zone_state, ts.pending_zone_since = ZS.DECK, None, None
            return
        if observed is None or observed == ZS.DECK:
            observed = ts.zone_state if ts.zone_state != ZS.DECK else ZS.IN_WATER
        if observed == ts.zone_state:
            ts.pending_zone_state, ts.pending_zone_since = None, None
            return
        if ts.zone_state == ZS.DECK:  # just (re)entered: take the observation at once
            ts.zone_state, ts.pending_zone_state, ts.pending_zone_since = observed, None, None
            return
        if ts.pending_zone_state != observed:
            ts.pending_zone_state, ts.pending_zone_since = observed, t
        elif t - (t if ts.pending_zone_since is None else ts.pending_zone_since) >= CATEGORY_SWITCH_SECONDS - 1e-6:
            ts.zone_state, ts.pending_zone_state, ts.pending_zone_since = observed, None, None

    def _update_category(
        self, ts: TrackState, observed: str, t: float | None = None, hold: float = 0.0, confident: bool = False
    ) -> None:
        """Zones drawn: switch after the zone decision held for CATEGORY_SWITCH_SECONDS (or `hold`, if longer).
        `confident`: this observation is strong enough on its own to skip the clearly-visible check."""
        t = ts.last_seen if t is None else t
        if observed == ts.category:
            ts.pending_category, ts.pending_count, ts.pending_since = None, 0, None
            return
        if ts.category == SWIMMER and observed == BYSTANDER and not confident and not self._clearly_visible(ts):
            # A swimmer only leaves the water fully visible (climbing out). A
            # collapsed pose drifting past the edge line is someone going UNDER.
            ts.pending_category, ts.pending_count, ts.pending_since = None, 0, None
            return
        if ts.pending_category != observed:
            ts.pending_category, ts.pending_count, ts.pending_since = observed, 0, t
        ts.pending_count += 1
        since = t if ts.pending_since is None else ts.pending_since  # a pending start at t = 0.0 is a time, not unset
        if t - since >= max(CATEGORY_SWITCH_SECONDS, hold) - 1e-6 and ts.pending_count >= 2:
            self._switch(ts, observed, t)

    def _land_evidence(self, p: TrackedPerson) -> tuple[bool, bool | None]:
        """
        (on_land, feet_in_water) for one frame, no zones drawn.

        on_land: knees AND ankles confidently visible below the hips, legs at
        least LAND_LEG_MIN_RATIO of the torso — and not standing inside a
        confidently estimated water region (clear shallow water shows legs).
        feet_in_water: the water estimate's verdict at the feet (None = unsure).

        [v7] LAND_FROM_WATER_ESTIMATE: when the legs can't be proven by keypoints (small or far person), feet
        confidently outside the estimated water by LAND_WATER_MARGIN x box height count as land on their own -
        the same verdict the no-pose path below already uses. Sustained-evidence rules are unchanged.
        """
        feet = p.foot_point
        box_h = max(1, p.bbox[3] - p.bbox[1])
        in_water = self.water.contains(feet, margin_px=0.05 * box_h)
        dry = bool(LAND_FROM_WATER_ESTIMATE) and self.water.contains(feet, margin_px=LAND_WATER_MARGIN * box_h) is False
        kp = p.keypoints
        if not kp or len(kp) < 17:
            # No pose (POSE_ENABLED=False, or this sample's box had no crop):
            # fall back to the colour/motion water estimate alone. `in_water`
            # is None until WaterRegionEstimator is confident, so this stays
            # conservative (monitored) rather than guessing.
            return in_water is False, in_water

        def ok(i: int) -> bool:
            return kp[i][2] >= LAND_KP_CONF

        hips = [kp[i] for i in _HIPS if kp[i][2] >= KEYPOINT_CONF_THRESHOLD]
        shoulders = [kp[i] for i in _SHOULDERS if kp[i][2] >= KEYPOINT_CONF_THRESHOLD]
        if not hips or not shoulders:
            return dry, in_water
        legs = []
        for h_i, k_i, a_i in zip(_HIPS, _KNEES, _ANKLES):
            if ok(k_i) and ok(a_i):
                legs.append((kp[h_i], kp[k_i], kp[a_i]))
        if not legs:
            return dry, in_water
        hip_y = sum(h[1] for h in hips) / len(hips)
        sh_y = sum(s_[1] for s_ in shoulders) / len(shoulders)
        torso = abs(hip_y - sh_y)
        good = [(k, a) for (_h, k, a) in legs if hip_y < k[1] < a[1]]
        if not good or torso <= 1.0:
            return dry, in_water
        leg_len = max(a[1] - hip_y for (_k, a) in good)
        on_land = (leg_len >= LAND_LEG_MIN_RATIO * torso and in_water is not True) or dry
        return bool(on_land), in_water

    def _lost_land_evidence(self, ts: TrackState) -> bool:
        """
        No zones: a 'deck' person whose legs stopped being seen in their last
        second before vanishing may have fallen in (a child slipping off the
        edge). Uncertainty keeps them watched.
        """
        # the very end: a fall takes a fraction of a second, so the legs
        # disappear only in the last few frames before the person does
        last = ts.land_obs[-frames_at(FALL_IN_LAST_FRAMES, getattr(self, "fps", None)) :]
        if not last:
            return True
        return not any(o[1] for o in last)

    def _update_water_category(self, ts: TrackState, t: float) -> None:
        """No zones: only STRONG, SUSTAINED land evidence makes someone a bystander."""
        horizon = max(LAND_EVIDENCE_SECONDS, WATER_RETURN_SECONDS) + 0.5
        ts.land_obs = [o for o in ts.land_obs if t - o[0] <= horizon]
        if ts.category == SWIMMER:
            win = [o for o in ts.land_obs if t - o[0] <= LAND_EVIDENCE_SECONDS]
            span = (win[-1][0] - win[0][0]) if len(win) >= 2 else 0.0
            if span >= 0.6 * LAND_EVIDENCE_SECONDS and not demotion_blocked_by_risk(ts):
                frac = sum(1 for o in win if o[1]) / len(win)
                if frac >= LAND_EVIDENCE_FRACTION:
                    self._switch(ts, BYSTANDER, t)
        else:
            win = [o for o in ts.land_obs if t - o[0] <= WATER_RETURN_SECONDS]
            span = (win[-1][0] - win[0][0]) if len(win) >= 2 else 0.0
            if span >= 0.6 * WATER_RETURN_SECONDS:
                land_frac = sum(1 for o in win if o[1]) / len(win)
                wet_frac = sum(1 for o in win if o[2] is True) / len(win)
                if land_frac < 1.0 - LAND_EVIDENCE_FRACTION or wet_frac >= 0.5:
                    self._switch(ts, SWIMMER, t)

    @staticmethod
    def _clearly_visible(ts: TrackState) -> bool:
        if not ts.buffer.samples:
            return True
        last = ts.buffer.samples[-1]
        if not ts.buffer.had_pose:
            return True  # HOG / no pose model: nothing to judge by
        return last.has_pose and last.upper_conf == last.upper_conf and last.upper_conf >= 0.35
