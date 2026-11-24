"""Stage 3: scenario evidence (0-1 per scenario) from KinematicFeatures, with cue agreement and
positive recognition of normal activity.

[v3] Split out of the v2 baseline detection/risk_engine.py: ScenarioEvidence and step_evidence
verbatim; RiskEngine.evidence / _activity verbatim as EvidenceScorer (RiskEngine inherits it).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from lifeguard.vision.behaviour.features import KinematicFeatures
from lifeguard.vision.behaviour.numeric import clamp01, finite, inv_ramp, ramp
from lifeguard.vision.constants import (
    ACTIVE_MIN_AGREEMENT,
    ACTIVITY_CALM_ARM_AMP_MAX,
    ACTIVITY_CALM_BOB_MAX,
    ACTIVITY_HEAD_UP_FACE_CONF,
    ACTIVITY_HEAD_UP_SHOULDER_REL,
    ACTIVITY_STANDING_ASPECT,
    ACTIVITY_STANDING_CONF,
    ACTIVITY_SWIM_SPEED,
    BOX_CUE_WEIGHT_DISTRESS,
    BOX_CUE_WEIGHT_LOW,
    BOX_CUE_WEIGHT_VERTICAL,
    BOX_ONLY_ACTIVE_LOW_REQUIRED,
    CHANGE_FROM_NORMAL_SECONDS,
    FLOATABLE_DETACH_BODY_WIDTHS,
    KINEMATICS,
    PASSIVE_NEVER_TRAVELLED_SU,
    PASSIVE_STILL_ONLY_FACTOR,
    POSE_RELIABLE_MIN,
    STEP_PRE_LIMIT,
    STROKE_PERIODIC_HZ_MAX,
    STROKE_PERIODICITY_MIN,
    THRASH_HZ_MIN,
    TORSO_HORIZONTAL_DEG,
    TORSO_SWIMMING_DEG,
    TORSO_VERTICAL_DEG,
    VELOCITY_STATIONARY_SU,
    VELOCITY_SWIMMING_SU,
    WADING_STILL_CAP,
)

logger = logging.getLogger(__name__)


@dataclass
class ScenarioEvidence:
    """Per-scenario evidence in 0–1 (before the zone multiplier)."""

    normal: float = 0.0
    active: float = 0.0
    passive: float = 0.0
    floatable: float = 0.0
    submersion: float = 0.0
    resting: bool = False
    # ── Phase 1.3 canonical signatures ───────────────────────────────────
    # True when a track matches one of the spec's classification rules end to
    # end. The classifier uses these to clamp risk into the band the spec
    # pins (normal < 0.20, active > 0.85, passive > 0.90).
    sig_normal: bool = False
    sig_active: bool = False
    sig_passive: bool = False
    detachment: bool = False  # FLOATABLE_DETACHMENT_RISK (Phase 2 module)
    # ── Phase 2.2 / 2.3 ──────────────────────────────────────────────────
    activity: str = "UNKNOWN"  # SWIMMING | TREADING | STANDING_WATER | RESTING_EDGE | UNKNOWN
    activity_conf: float = 0.0
    pose_ok: bool = False
    agreement: float = 0.0  # weighted count of agreeing Active-Drowning cues
    cues: dict[str, float] = field(default_factory=dict)
    detail: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {
            "NORMAL_SWIMMING": round(self.normal, 3),
            "ACTIVE_DROWNING": round(self.active, 3),
            "PASSIVE_DROWNING": round(self.passive, 3),
            "FLOATABLE_FAILURE": round(self.floatable, 3),
            "SUBMERSION": round(self.submersion, 3),
        }

    def drowning(self) -> dict[str, float]:
        return {
            "ACTIVE_DROWNING": self.active,
            "PASSIVE_DROWNING": self.passive,
            "FLOATABLE_FAILURE": self.floatable,
            "SUBMERSION": self.submersion,
        }


def step_evidence(duration: float, threshold: float) -> float:
    """
    Duration evidence that crosses every mode threshold exactly when
    `duration` reaches `threshold`: builds to STEP_PRE_LIMIT (below every
    threshold, incl. calibrated ones — see behavior_calibration) so the
    display can show Caution, then jumps to ≥ 0.70 and grows to 1.0 at 2×.
    """
    if threshold <= 0:
        return 0.0
    if duration < threshold:
        return STEP_PRE_LIMIT * ramp(duration, 0.5 * threshold, threshold)
    return 0.70 + 0.30 * ramp(duration, threshold, 2.0 * threshold)


class EvidenceScorer:
    """Stage 3 of RiskEngine."""

    # ── Stage 3: scenario evidence ────────────────────────────────────────────
    def evidence(self, f: KinematicFeatures, params: dict[str, Any]) -> ScenarioEvidence:
        """
        Scenario evidence from reliability-weighted cues (Phase 2).

        Pose-derived cues are used only when the window's pose reliability is
        ≥ POSE_RELIABLE_MIN; otherwise box / track kinematics stand in, with a
        lower weight. Active Drowning needs its three core cues AND a weighted
        agreement ≥ ACTIVE_MIN_AGREEMENT; Passive Drowning needs stillness AND
        a posture cue; a failed agreement caps the evidence at STEP_PRE_LIMIT
        (below every threshold) so the display can still show Caution.
        """
        ev = ScenarioEvidence()
        d = ev.detail
        pose_ok = f.pose_reliability >= POSE_RELIABLE_MIN
        ev.pose_ok = pose_ok

        # ── Normal swimming ─────────────────────────────────────────────────
        # Net progress, not raw speed: bobbing and keypoint jitter have no net
        # displacement, real swimming does.
        moving = f.moving
        travel = ramp(moving, 0.25, 0.75)
        angle = f.torso_angle
        if finite(f.axis_alignment) and f.axis_alignment > 0.8 and moving > 0.5:
            angle = min(angle, 30.0) if finite(angle) else 30.0  # foreshortened lap swimmer
        horiz = inv_ramp(angle, 35.0, 55.0) if finite(angle) else 0.5
        periodic = ramp(f.arm_periodicity, 0.25, 0.60) if finite(f.arm_periodicity) else 0.5
        normal = 0.55 * travel + 0.25 * horiz + 0.20 * periodic
        if moving > 1.0:
            normal = max(normal, 0.75)
        stroking = 0.0
        if finite(f.arm_periodicity) and finite(f.arm_amp) and finite(f.arm_freq):
            stroking = (
                horiz
                * ramp(f.arm_periodicity, 0.55, 0.85)
                * ramp(f.arm_amp, 0.10, 0.25)
                * inv_ramp(f.arm_freq, 1.4, 2.0)
            )
            normal = max(normal, 0.85 * stroking + 0.15 * travel)
        ev.normal = clamp01(normal)
        d.update(travel=travel, horizontal=horiz, periodic=periodic, stroking=stroking)

        # ── Cues: (value 0–1, weight) ───────────────────────────────────────
        stationary = inv_ramp(moving, 0.15, 0.40)  # track-level: always available
        if pose_ok and finite(f.uprightness):
            vertical = ramp(f.uprightness, 48.0, 68.0)
            # Progress ALONG the apparent body axis = a horizontal swimmer seen
            # end-on (swimming away from / toward a low camera), not an upright body.
            if finite(f.axis_alignment):
                end_on = ramp(f.axis_alignment, 0.80, 0.95) * ramp(moving, 0.12, 0.30)
                vertical *= 1.0 - 0.7 * end_on
            w_vert = 1.0
        elif finite(f.box_aspect):
            vertical, w_vert = ramp(f.box_aspect, 1.1, 1.6), BOX_CUE_WEIGHT_VERTICAL
        else:
            vertical, w_vert = 0.0, 0.0
        arms: float | None = None
        if pose_ok and finite(f.arm_amp):
            energy = ramp(f.arm_amp, 0.07, 0.20)
            frantic = max(
                ramp(f.arm_freq, 0.8, 1.6) if finite(f.arm_freq) else 0.0,
                1.0 - (f.arm_periodicity if finite(f.arm_periodicity) else 0.0),
            )
            arms = energy * (0.45 + 0.55 * frantic)
        bob: float | None = None
        bob_from_pose = False
        if pose_ok and (finite(f.head_bob) or f.head_flicker > 0):
            bob = max(ramp(f.head_bob, 0.04, 0.12), ramp(f.head_flicker, 0.4, 1.2))
            bob_from_pose = True
        elif finite(f.bbox_bob):
            bob = ramp(f.bbox_bob, 0.03, 0.10)
        known = [v for v in (arms, bob) if v is not None]
        if len(known) == 2:
            distress = 0.55 * max(known) + 0.45 * min(known)
        elif known:
            distress = 0.8 * known[0]
        else:
            distress = 0.0
        w_dist = 1.0 if (arms is not None or bob_from_pose) else (BOX_CUE_WEIGHT_DISTRESS if bob is not None else 0.0)
        # Head low at the waterline: shoulders under water (pose), or the box
        # shrinking against this person's own normal height (box-level).
        if pose_ok and finite(f.shoulder_rel):
            low, w_low = inv_ramp(f.shoulder_rel, 0.40, 0.85), 1.0
        elif finite(f.box_height_ratio):
            low, w_low = inv_ramp(f.box_height_ratio, 0.60, 0.90), BOX_CUE_WEIGHT_LOW
        else:
            low, w_low = 0.0, 0.0
        change = (
            1.0
            if (
                finite(f.since_normal)
                and 0.5 <= f.since_normal <= CHANGE_FROM_NORMAL_SECONDS
                and f.last_activity not in ("", "UNKNOWN")
            )
            else 0.0
        )

        # ── Active drowning: core cues + agreement ──────────────────────────
        active = vertical * stationary * distress * (0.6 + 0.4 * (low if w_low > 0 else 0.5))
        if f.pose_lost_seconds > 0.5:
            active *= 0.5
        active = clamp01(active**0.75) if active > 0 else 0.0  # soften the product
        core = stationary >= 0.5 and vertical >= 0.5 and distress >= 0.5
        if BOX_ONLY_ACTIVE_LOW_REQUIRED and not pose_ok and w_low > 0 and low < 0.5:
            # Box cues only: a person who stops and stands up in shallow water keeps a tall, full-height box. A body in
            # trouble sinks, so its box shrinks against its own height. Only applies when that height ratio is known.
            core = False
        agreement = (
            1.0 * (stationary >= 0.5)
            + w_vert * (vertical >= 0.5)
            + w_dist * (distress >= 0.5)
            + w_low * (low >= 0.5)
            + 1.0 * change
        )
        ev.agreement = agreement
        if not (core and agreement >= ACTIVE_MIN_AGREEMENT):
            active = min(active, STEP_PRE_LIMIT)
        ev.active = active
        ev.cues = {
            "stationary": round(stationary, 3),
            "vertical": round(vertical, 3),
            "distress": round(distress, 3),
            "low_in_water": round(low, 3),
            "change_from_normal": change,
            "w_vertical": w_vert,
            "w_distress": w_dist,
            "w_low": w_low,
        }
        d.update(
            vertical=vertical,
            stationary=stationary,
            arms=arms if arms is not None else -1.0,
            bob=bob if bob is not None else -1.0,
            distress=distress,
            low_in_water=low,
            agreement=agreement,
            pose_reliability=f.pose_reliability,
        )

        # ── Passive drowning: stillness AND a posture cue ───────────────────
        tp = float(params["passive_seconds"])
        tfd = float(params["face_down_seconds"])
        still_step = step_evidence(f.still_seconds, tp)
        if pose_ok and finite(f.uprightness):
            flat = f.uprightness < TORSO_HORIZONTAL_DEG
        else:
            flat = finite(f.box_aspect) and f.box_aspect < 0.9
        posture_cue = flat or f.face_down_seconds > 0 or f.inverted_seconds > 0 or (w_low > 0 and low >= 0.6)
        resting = pose_ok and f.still_seconds > 0 and f.still_upright_face_frac >= 0.7
        if resting:
            # Upright, face visible, head up → resting at the wall / treading.
            # Safety net: an unnaturally motionless upright body still escalates.
            still_ev = step_evidence(f.still_seconds, 1.5 * tp) if f.still_ultra_frac >= 0.8 else min(still_step, 0.3)
        elif posture_cue:
            still_ev = still_step
        else:
            # Motionless but upright / posture unknown: stillness alone is what
            # standing and resting people do, so it does not alert at the normal
            # passive time. Recall safety valve: a body that stays completely
            # still for PASSIVE_STILL_ONLY_FACTOR × passive_seconds escalates
            # anyway (a vertical passive victim may only be visible as a box).
            still_ev = step_evidence(f.still_seconds, PASSIVE_STILL_ONLY_FACTOR * tp)
            standing_box = (
                finite(f.box_aspect)
                and f.box_aspect >= ACTIVITY_STANDING_ASPECT
                and finite(f.box_height_ratio)
                and f.box_height_ratio >= 0.9
                and finite(f.conf_recent)
                and f.conf_recent >= ACTIVITY_STANDING_CONF
            )
            if WADING_STILL_CAP < 1.0 and not pose_ok and standing_box and f.still_ultra_frac < 0.8:
                # A tall, steady, confidently detected box that still shows natural micro-motion is someone standing.
                still_ev = min(still_ev, WADING_STILL_CAP)
        if (
            PASSIVE_NEVER_TRAVELLED_SU > 0
            and finite(f.moved_before_still)
            and f.moved_before_still < PASSIVE_NEVER_TRAVELLED_SU
        ):
            # [Phase 4 Step 3] seen before this still stretch but never travelled (standing / bobbing in place): the
            # stillness rule waits PASSIVE_STILL_ONLY_FACTOR x passive_seconds, as for an upright still body. Delayed,
            # never muted; tracks still since they appeared (no rounds before) and face-down / inverted are unchanged.
            still_ev = min(still_ev, step_evidence(f.still_seconds, PASSIVE_STILL_ONLY_FACTOR * tp))
        face_ev = step_evidence(f.face_down_seconds, tfd)
        inv_ev = step_evidence(f.inverted_seconds, tfd)
        ev.passive = clamp01(max(still_ev, face_ev, inv_ev))
        end_on_travel = finite(f.axis_alignment) and f.axis_alignment > 0.8 and moving > 0.08
        ev.resting = bool(
            not end_on_travel
            and (
                resting
                or (
                    moving < 0.15
                    and pose_ok
                    and finite(f.uprightness)
                    and f.uprightness > 50
                    and f.face_down_seconds == 0
                )
            )
        )
        d.update(still_ev=still_ev, face_ev=face_ev, inverted_ev=inv_ev, posture_cue=float(posture_cue))

        # ── Floatable failure ───────────────────────────────────────────────
        win = KINEMATICS["event_recent_seconds"]
        decouple = 1.0 if finite(f.decoupled_ago) and f.decoupled_ago <= win else 0.0
        child_w = 0.30 + 0.70 * f.child_likelihood
        spike = ramp(f.spike_ratio, 3.0, 6.0) * child_w if finite(f.spike_ago) and f.spike_ago <= win else 0.0
        flip = ramp(f.flip_deg, 35.0, 70.0) * child_w if finite(f.flip_ago) and f.flip_ago <= win else 0.0
        event = max(decouple, 0.85 * spike, flip)
        sinking = (
            ramp(1.0 - (f.upper_conf_recent if finite(f.upper_conf_recent) else 1.0), 0.45, 0.85)
            if pose_ok
            else (ramp(1.0 - f.box_height_ratio, 0.3, 0.6) if finite(f.box_height_ratio) else 0.0)
        )
        settle = inv_ramp(moving, 0.25, 0.60)  # no swimming away after the event
        aftermath = max(
            vertical * settle * max(distress, 0.5),
            settle * ramp(f.still_seconds, 1.0, 3.0),
            settle * sinking,
            ramp(f.inverted_seconds, 0.5, 2.0),
            ramp(f.face_down_seconds, 0.5, 2.5),
        )
        ev.floatable = clamp01(event * aftermath)
        d.update(decouple=decouple, spike=spike, flip=flip, aftermath=aftermath)

        # ── Visible submersion (pose collapse corroborated by the box) ──────
        td = float(params["submersion_seconds"])
        ev.submersion = step_evidence(f.pose_lost_seconds, td) if f.pose_lost_seconds > 0 else 0.0

        # ── Phase 1.3 canonical signatures ──────────────────────────────────
        theta = f.torso_inclination
        v = f.v_horizontal if finite(f.v_horizontal) else f.moving
        ev.sig_normal = bool(
            finite(theta)
            and theta < TORSO_SWIMMING_DEG
            and finite(v)
            and v > VELOCITY_SWIMMING_SU
            and (
                (
                    finite(f.arm_periodicity)
                    and f.arm_periodicity >= STROKE_PERIODICITY_MIN
                    and (not finite(f.stroke_frequency) or f.stroke_frequency <= STROKE_PERIODIC_HZ_MAX)
                )
                or v > 2.0 * VELOCITY_SWIMMING_SU
            )
            and f.face_down_seconds == 0.0
            and f.inverted_seconds == 0.0
        )
        thrashing = bool(
            pose_ok
            and finite(f.arm_amp)
            and f.arm_amp >= 0.10
            and (
                (finite(f.arm_freq) and f.arm_freq >= THRASH_HZ_MIN)
                or (finite(f.arm_periodicity) and f.arm_periodicity < STROKE_PERIODICITY_MIN)
            )
        )
        ev.sig_active = bool(
            pose_ok
            and finite(theta)
            and theta > TORSO_VERTICAL_DEG
            and finite(f.uprightness)
            and f.uprightness > 0
            and finite(v)
            and v < VELOCITY_STATIONARY_SU
            and thrashing
            and not ev.sig_normal
            and core
            and agreement >= ACTIVE_MIN_AGREEMENT
        )
        flat_or_inverted = (
            (pose_ok and finite(f.uprightness) and f.uprightness < TORSO_HORIZONTAL_DEG)
            or f.inverted_seconds > 0.0
            or f.face_down_seconds > 0.0
        )
        ev.sig_passive = bool(
            flat_or_inverted and f.zero_velocity_seconds > float(params["passive_seconds"]) and not ev.sig_normal
        )
        ev.detachment = bool(
            finite(f.detach_ago)
            and f.detach_ago <= KINEMATICS["event_recent_seconds"]
            and f.detach_widths >= FLOATABLE_DETACH_BODY_WIDTHS
            and f.detach_vertical
            and f.child_likelihood >= 0.35
        )
        if ev.detachment:
            ev.floatable = max(ev.floatable, clamp01(0.55 + 0.45 * ramp(f.detach_widths, 2.5, 5.0)))
        d.update(
            sig_normal=float(ev.sig_normal),
            sig_active=float(ev.sig_active),
            sig_passive=float(ev.sig_passive),
            detachment=float(ev.detachment),
            thrashing=float(thrashing),
            theta_torso=theta if finite(theta) else -1.0,
            v_horizontal=v if finite(v) else -1.0,
        )

        # ── Positive recognition of normal activity (Phase 2.2) ─────────────
        ev.activity, ev.activity_conf = self._activity(f, params, pose_ok, stationary, vertical, horiz, flat)
        d.update(activity_conf=ev.activity_conf)

        # ── Normal-swimming suppression (the false-positive fix) ────────────
        gate = float(params.get("normal_gate", 0.62))
        if ev.normal >= gate and not (ev.sig_active or ev.sig_passive):
            ev.active *= 0.25
            ev.passive = min(ev.passive, max(face_ev, inv_ev) * 0.5)
            ev.floatable *= 0.5
        if ev.sig_normal and not (ev.sig_active or ev.sig_passive):
            ev.normal = max(ev.normal, 0.80)
        if ev.sig_active:
            ev.active = max(ev.active, 0.90)
        if ev.sig_passive:
            ev.passive = max(ev.passive, 0.92)
        return ev

    @staticmethod
    def _activity(
        f: KinematicFeatures,
        params: dict[str, Any],
        pose_ok: bool,
        stationary: float,
        vertical: float,
        horiz: float,
        flat: bool,
    ) -> tuple[str, float]:
        """
        What is this person doing, if it is clearly a NORMAL activity?

        Conservative by design: only strong, calm evidence counts, because a
        recognised normal activity blocks Active / Passive / Floatable alerts.
        A motionless body stops counting as "resting" after 2 × passive_seconds
        of near-zero limb motion — relaxed people make small movements.
        """
        swim = ramp(f.moving, *ACTIVITY_SWIM_SPEED)
        if pose_ok and finite(f.uprightness):
            swim *= 0.6 + 0.4 * horiz
        if swim >= 0.6 and f.face_down_seconds == 0 and f.inverted_seconds == 0:
            return "SWIMMING", swim * ramp(f.duration, 1.0, 2.0)
        if stationary < 0.5 or f.face_down_seconds > 0 or f.inverted_seconds > 0 or f.pose_lost_seconds > 0:
            return "UNKNOWN", 0.0
        tp = float(params["passive_seconds"])
        frozen = f.still_ultra_frac >= 0.8 and f.still_seconds >= 2.0 * tp
        hold = ramp(f.duration, 1.0, 2.0) * (0.5 if frozen else 1.0)
        calm_arms = (
            not finite(f.arm_amp)
            or f.arm_amp < ACTIVITY_CALM_ARM_AMP_MAX
            or (
                finite(f.arm_periodicity)
                and f.arm_periodicity >= 0.6
                and (not finite(f.arm_freq) or f.arm_freq <= STROKE_PERIODIC_HZ_MAX)
            )
        )
        if pose_ok:
            calm_bob = not finite(f.head_bob) or f.head_bob < ACTIVITY_CALM_BOB_MAX
            head_up = (
                finite(f.face_conf_recent)
                and f.face_conf_recent >= ACTIVITY_HEAD_UP_FACE_CONF
                and finite(f.shoulder_rel)
                and f.shoulder_rel >= ACTIVITY_HEAD_UP_SHOULDER_REL
                and (not finite(f.box_height_ratio) or f.box_height_ratio >= 0.85)
            )
            if head_up and calm_bob and calm_arms and vertical >= 0.5:
                near_edge = finite(f.edge_su) and 0 <= f.edge_su < 1.5 * KINEMATICS["edge_margin_su"]
                if near_edge:
                    act = "RESTING_EDGE"
                elif finite(f.box_aspect) and f.box_aspect >= ACTIVITY_STANDING_ASPECT:
                    act = "STANDING_WATER"
                else:
                    act = "TREADING"
                conf = min(stationary, vertical, 0.6 + 0.4 * ramp(f.face_conf_recent, 0.45, 0.75))
                return act, conf * hold
            # A motionless, horizontal body is NOT recognised as normal even with
            # the face up: the spec's Passive Drowning rule covers it, and
            # floating on the back is not on the list of activities that may
            # block an alert.
            return "UNKNOWN", 0.0
        # Box-only: a tall, clearly detected, steady box that is not sinking —
        # but never for longer than the still-only safety valve (see evidence()).
        if f.still_seconds >= PASSIVE_STILL_ONLY_FACTOR * tp:
            return "UNKNOWN", 0.0
        calm_box = not finite(f.bbox_bob) or f.bbox_bob < ACTIVITY_CALM_BOB_MAX
        if (
            calm_box
            and finite(f.box_aspect)
            and f.box_aspect >= ACTIVITY_STANDING_ASPECT
            and finite(f.conf_recent)
            and f.conf_recent >= ACTIVITY_STANDING_CONF
            and (not finite(f.box_height_ratio) or f.box_height_ratio >= 0.9)
        ):
            return "STANDING_WATER", 0.65 * hold
        return "UNKNOWN", 0.0
