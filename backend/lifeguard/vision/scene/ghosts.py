"""Scenario D - vanished swimmers: SubmersionGhost and SubmersionMonitor.

[v3] Split out of the v2 baseline detection/classifier.py; definitions verbatim.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from lifeguard.vision.behaviour.classifier import BehaviorClassifier
from lifeguard.vision.behaviour.engine import (
    RiskEngine,
    TrackBuffer,
    ramp,
    step_evidence,
)
from lifeguard.vision.behaviour.params import (
    clamp_multiplier,
    get_mode_params,
    normalize_mode,
    timed_params,
)
from lifeguard.vision.behaviour.result import SWIMMER, BehaviorResult, _result
from lifeguard.vision.constants import (
    BLACKOUT_CRITICAL_RISK,
    BLACKOUT_CRITICAL_STATE,
    BLACKOUT_GRACE_SECONDS,
    BLACKOUT_STRICT_INSIDE_SU,
    BLACKOUT_WATCH_STATE,
    FLOATABLE_VANISH_FACTOR,
    FLOATABLE_VANISH_MIN_SECONDS,
    GHOST_EXTRAPOLATE_SECONDS,
    GHOST_RESOLVE_SIZE_RATIO,
    KINEMATICS,
    NON_DROWNING_RISK_CAP,
    NON_DROWNING_RISK_EXPONENT,
    SUBMERSION_EDGE_FACTOR,
    SUBMERSION_GHOST_TTL_SECONDS,
    SUBMERSION_MIN_TRACK_SECONDS,
    SUBMERSION_OCCLUSION_FACTOR,
    SUBMERSION_TRAVEL_GRACE_SECONDS,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Scenario D — vanished swimmers
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class SubmersionGhost:
    track_id: int
    last_pos: tuple[float, float]
    last_bbox: tuple[int, int, int, int]
    scale_px: float
    lost_at: float
    zone_name: str
    zone_type: str
    multiplier: float
    mode: str
    occluded: bool = False
    near_edge: bool = False  # vanished within reach of the pool edge (may have climbed out)
    floatable_linked: bool = False  # vanished while / right after holding a floatable
    event_reason: str = ""
    alerted: bool = False
    result: BehaviorResult | None = None
    # ── Phase 1.1 occlusion grace timer ──────────────────────────────────
    # True when the last known coordinate was STRICTLY inside a water polygon
    # (well clear of the perimeter), i.e. the swimmer could not have climbed
    # out unseen. Those tracks run the fixed 5.0 s spec countdown and escalate
    # to CRITICAL_BLACKOUT_DROWNING with risk 1.0.
    strictly_inside_water: bool = False
    deadline: float = 0.0  # absolute time the countdown expires
    escalated: bool = False
    # Phase 1.4: where the swimmer was heading and how big they were
    velocity: tuple[float, float] = (0.0, 0.0)  # px/s at loss
    box_size: float = 0.0  # sqrt(box area) px
    travelling: bool = False  # recognised as swimming when lost

    @property
    def scenario(self) -> str:
        return "FLOATABLE_FAILURE" if self.floatable_linked else "SUBMERSION"

    @property
    def state(self) -> str:
        """Display state: watching during the countdown, critical once it expires."""
        if self.escalated:
            return BLACKOUT_CRITICAL_STATE
        return BLACKOUT_WATCH_STATE if self.strictly_inside_water else "SUBMERGED"

    def remaining(self, now: float) -> float:
        """Seconds left on the countdown (0 once it has expired)."""
        return max(0.0, self.deadline - now) if self.deadline else 0.0

    def elapsed(self, now: float) -> float:
        return max(0.0, now - self.lost_at)

    def to_dict(self, now: float, frame_w: int, frame_h: int) -> dict[str, Any]:
        x1, y1, x2, y2 = self.last_bbox
        return {
            "track_id": self.track_id,
            "elapsed": round(self.elapsed(now), 2),
            "alerted": self.alerted,
            "occluded": self.occluded,
            "zone": self.zone_name,
            "state": self.state,
            "remaining": round(self.remaining(now), 2),
            "escalated": self.escalated,
            "strictly_inside_water": self.strictly_inside_water,
            "risk": round(self.result.risk, 4) if self.result else 0.0,
            "risk_score": self.result.risk_score if self.result else 0,
            "label": "Drowning Alert" if self.alerted else "Lost from view",  # [Stage 4] one alert name
            "bbox_norm": [
                round(x1 / max(frame_w, 1), 4),
                round(y1 / max(frame_h, 1), 4),
                round(x2 / max(frame_w, 1), 4),
                round(y2 / max(frame_h, 1), 4),
            ],
        }


def _was_leaving_water(buf: TrackBuffer) -> bool:
    """
    True when the track was crossing (or closing on) the pool perimeter in its
    last ~2 s — someone climbing out, not someone going under mid-pool.
    """
    if not buf.samples:
        return True
    scale = buf.scale()
    recent = [buf.samples[i] for i in buf.recent(2.0)]
    if len(recent) < 2:
        return False
    margin = KINEMATICS["edge_margin_su"] * scale
    if any(s.edge_dist_px < margin for s in recent):
        return True  # touched the perimeter band
    first, last = recent[0].edge_dist_px, recent[-1].edge_dist_px
    if not (math.isfinite(first) and math.isfinite(last)):
        return False
    # Closing on the edge fast enough to reach it within the grace period.
    closing = (first - last) / max(recent[-1].t - recent[0].t, 1e-6)
    return closing > 0 and last - closing * BLACKOUT_GRACE_SECONDS < 0


class SubmersionMonitor:
    """
    Tracks swimmers that disappeared inside the water. A ghost resolves when a
    person re-appears near the last position (a new track id or the same id);
    otherwise, after `submersion_seconds`, it becomes a Scenario D alert.
    """

    def __init__(self, params_fn: Callable[[str], dict[str, Any]] | None = None) -> None:
        self.ghosts: dict[int, SubmersionGhost] = {}
        self.params = params_fn or get_mode_params

    def reset(self) -> None:
        self.ghosts.clear()

    def track_lost(
        self,
        buf: TrackBuffer,
        *,
        last_bbox: tuple[int, int, int, int],
        edge_dist_px: float,
        frame_w: int,
        frame_h: int,
        category: str,
        zone_name: str,
        zone_type: str,
        multiplier: float,
        mode: str,
        occluded: bool,
        travelling: bool = False,
        velocity: tuple[float, float] = (0.0, 0.0),
    ) -> SubmersionGhost | None:
        if category != SWIMMER or buf.last_t is None or not buf.samples:
            return None
        if buf.duration < SUBMERSION_MIN_TRACK_SECONDS:
            return None
        scale = buf.scale()
        if edge_dist_px < -0.25 * scale:
            return None  # last seen clearly outside the water
        x1, y1, x2, y2 = last_bbox
        bx, by = KINEMATICS["frame_border_frac"] * frame_w, KINEMATICS["frame_border_frac"] * frame_h
        at_border = x1 <= bx or y1 <= by or x2 >= frame_w - bx or y2 >= frame_h - by
        if at_border:
            # Moving OUT of the view = left the camera. Lost at the border while
            # NOT moving outward (a body drifting where the pool runs off the
            # frame) is still watched, with the longer near-edge grace.
            size = math.sqrt(max((x2 - x1) * (y2 - y1), 1.0))
            vx, vy = velocity
            outward = max(
                -vx if x1 <= bx else 0.0,
                vx if x2 >= frame_w - bx else 0.0,
                -vy if y1 <= by else 0.0,
                vy if y2 >= frame_h - by else 0.0,
            )
            if outward > 0.1 * size:
                return None  # left the camera view
        near_edge = edge_dist_px < KINEMATICS["edge_margin_su"] * scale
        if at_border:
            near_edge = True
        # Phase 1.1: "strictly inside a WATER polygon, without traversing the
        # perimeter of the pool edge". Both halves are required — the last
        # coordinate must be comfortably inside AND the track must not have
        # been drifting toward the edge on its way out of view.
        strictly_inside = (
            edge_dist_px >= BLACKOUT_STRICT_INSIDE_SU * scale and not _was_leaving_water(buf) and not at_border
        )
        last = buf.samples[-1]
        pos = last.anchor if last.anchor is not None else last.center
        linked, why = self._floatable_link(buf, zone_type)
        ghost = SubmersionGhost(
            track_id=buf.track_id,
            last_pos=pos,
            last_bbox=last_bbox,
            scale_px=scale,
            lost_at=buf.last_t,
            zone_name=zone_name,
            zone_type=zone_type,
            multiplier=clamp_multiplier(multiplier),
            mode=normalize_mode(mode),
            occluded=occluded,
            near_edge=near_edge,
            floatable_linked=linked,
            event_reason=why,
            strictly_inside_water=strictly_inside,
            velocity=velocity,
            travelling=travelling,
            box_size=math.sqrt(max((x2 - x1) * (y2 - y1), 1.0)),
        )
        # Arm the non-blocking countdown. A track that vanished strictly inside
        # the water uses the fixed 5.0 s spec grace period; anything else keeps
        # the mode/zone-resolved submersion window (which the occlusion and
        # near-edge factors stretch further in `evaluate`).
        ghost.deadline = buf.last_t + self.countdown_for(ghost)
        self.ghosts[buf.track_id] = ghost
        return ghost

    def countdown_for(self, g: SubmersionGhost) -> float:
        """Grace period (seconds) before this ghost escalates."""
        if g.strictly_inside_water and not g.floatable_linked:
            return max(BLACKOUT_GRACE_SECONDS, SUBMERSION_TRAVEL_GRACE_SECONDS if g.travelling else 0.0)
        params = timed_params(self.params(g.mode), g.multiplier)
        td = float(params["submersion_seconds"])
        if g.occluded:
            td *= SUBMERSION_OCCLUSION_FACTOR
        if g.near_edge:
            td *= SUBMERSION_EDGE_FACTOR  # may have climbed out unseen — wait longer
        if g.floatable_linked:
            td = max(FLOATABLE_VANISH_MIN_SECONDS, FLOATABLE_VANISH_FACTOR * td)
        elif g.travelling:
            td = max(td, SUBMERSION_TRAVEL_GRACE_SECONDS)
        return td

    @staticmethod
    def _floatable_link(buf: TrackBuffer, zone_type: str) -> tuple[bool, str]:
        """
        Scenario C often ENDS in a submersion: a child lets go of the ring (or
        tips over in it) and slips under in the same moment. A swimmer who
        vanished while holding a floatable, within 3 s of separating from one,
        or right after an abrupt posture failure while child-like, escalates as
        Floatable Failure on a shorter clock instead of waiting for Scenario D.
        """
        t = buf.last_t or 0.0
        if buf.last_coupled_t is not None and t - buf.last_coupled_t <= 1.5:
            return True, "was holding a floatable when it disappeared"
        if buf.decoupled_at is not None and t - buf.decoupled_at <= 3.0:
            return True, f"separated from its floatable {t - buf.decoupled_at:.1f} s before disappearing"
        recent_flip = buf.flip_t is not None and t - buf.flip_t <= 2.5
        recent_spike = buf.spike_t is not None and t - buf.spike_t <= 2.5
        if recent_flip or recent_spike:
            child = RiskEngine().extract(buf, zone_type).child_likelihood
            if child >= 0.5:
                what = "abrupt posture failure" if recent_flip else "sudden displacement"
                return True, f"{what} right before disappearing (child likelihood {child:.2f})"
        return False, ""

    def reassociate(self, now: float, tracks: Iterable[tuple[Any, ...]]) -> list[SubmersionGhost]:
        """
        `tracks` = (track_id, centre_px, appeared_at) for EVERY visible track,
        swimmers and deck bystanders alike; `appeared_at` is when that track
        last came (back) into view. A ghost resolves when the same id returns,
        or a track that APPEARED after the loss (a new id, a stitched or revived
        older id — the same person often comes back under one) is within a
        radius that grows with the time elapsed — including someone who climbed
        out onto the deck. A track that was in view all along never resolves
        it: that is a neighbour, not the person who went under. An optional
        4th element is the track's box size (sqrt area, px): a NEW track more
        than GHOST_RESOLVE_SIZE_RATIO times bigger than the vanished person
        (an adult jumping in to a toddler) is a rescuer, not them resurfacing.
        Returns resolved ghosts.
        """
        resolved: list[SubmersionGhost] = []
        tracks = list(tracks)
        for tid, g in list(self.ghosts.items()):
            el = g.elapsed(now)
            radius = max(g.scale_px, 0.5 * g.box_size) * (1.5 + 1.2 * el)
            ahead = min(el, GHOST_EXTRAPOLATE_SECONDS)
            px, py = g.last_pos[0] + g.velocity[0] * ahead, g.last_pos[1] + g.velocity[1] * ahead
            for tr in tracks:
                track_id, c, appeared_at = tr[0], tr[1], tr[2]
                size = tr[3] if len(tr) > 3 else None
                d = min(math.hypot(c[0] - px, c[1] - py), math.hypot(c[0] - g.last_pos[0], c[1] - g.last_pos[1]))
                if track_id == g.track_id:
                    # The tracker says "same person" — believed only if they are
                    # plausibly where that person could be (an id can be
                    # re-attached to a stranger's box across the frame).
                    if d <= 2.0 * radius:
                        resolved.append(self.ghosts.pop(tid))
                        break
                    continue
                newborn = appeared_at >= g.lost_at - 0.5
                if size is not None and size > GHOST_RESOLVE_SIZE_RATIO * max(g.box_size, 1.0):
                    continue  # someone much bigger arrived: a rescuer
                if newborn and d <= radius:
                    resolved.append(self.ghosts.pop(tid))
                    break
            else:
                if g.elapsed(now) > SUBMERSION_GHOST_TTL_SECONDS:
                    self.ghosts.pop(tid, None)
        return resolved

    def evaluate(self, now: float) -> list[SubmersionGhost]:
        """Refresh every ghost's BehaviorResult; returns ghosts that just crossed into alert."""
        fired: list[SubmersionGhost] = []
        for g in self.ghosts.values():
            params = timed_params(self.params(g.mode), g.multiplier)
            key = g.scenario
            td = self.countdown_for(g)
            g.deadline = g.lost_at + td
            elapsed = g.elapsed(now)

            # ── Phase 1.1: strictly-inside-water blackout escalation ───────
            # The countdown is a hard deadline, not a graded ramp: if the track
            # has not resurfaced within it, the scenario is CRITICAL with
            # risk_score = 1.0 and an immediate emergency alert.
            if g.strictly_inside_water and not g.floatable_linked:
                if elapsed >= td:
                    g.escalated = True
                    g.result = _result(
                        g.track_id,
                        SWIMMER,
                        key,
                        risk=BLACKOUT_CRITICAL_RISK,
                        raw_risk=BLACKOUT_CRITICAL_RISK,
                        confidence=0.92,
                        mode=g.mode,
                        multiplier=g.multiplier,
                        evidence={key: 1.0},
                        reason=(
                            f"CRITICAL BLACKOUT — swimmer vanished strictly inside {g.zone_name} "
                            f"{elapsed:.1f} s ago, never crossed the pool edge, and has not "
                            f"resurfaced within the {td:.1f} s grace period."
                            + (" Was partly occluded when lost." if g.occluded else "")
                        ),
                    )
                    if not g.alerted:
                        g.alerted = True
                        fired.append(g)
                else:
                    g.result = _result(
                        g.track_id,
                        SWIMMER,
                        "ANALYZING",
                        risk=NON_DROWNING_RISK_CAP * min(elapsed / max(td, 1e-6), 0.999),
                        confidence=0.5,
                        mode=g.mode,
                        multiplier=g.multiplier,
                        reason=(
                            f"Submerged watch — lost inside {g.zone_name} {elapsed:.1f} s ago; "
                            f"escalating in {max(0.0, td - elapsed):.1f} s if they do not resurface."
                        ),
                    )
                continue

            e = step_evidence(elapsed, td)
            thr = params["thresholds"][key]
            if e >= thr:
                risk = BehaviorClassifier._drowning_risk(key, e, thr)
                if g.floatable_linked:
                    reason = (
                        f"Floatable Failure — swimmer {g.event_reason} inside {g.zone_name} and has not "
                        f"resurfaced for {elapsed:.1f} s."
                    )
                else:
                    reason = (
                        f"Sudden Blackout / Submersion — swimmer vanished inside {g.zone_name} "
                        f"{elapsed:.1f} s ago, away from the pool edge, and has not resurfaced"
                        + (
                            " (was partly occluded)."
                            if g.occluded
                            else " (near the edge, no one climbed out)."
                            if g.near_edge
                            else "."
                        )
                    )
                g.result = _result(
                    g.track_id,
                    SWIMMER,
                    key,
                    risk=risk,
                    raw_risk=risk,
                    confidence=0.6 + 0.3 * ramp(e, thr, 1.0),
                    mode=g.mode,
                    multiplier=g.multiplier,
                    evidence={key: round(e, 3)},
                    reason=reason,
                )
                if not g.alerted:
                    g.alerted = True
                    fired.append(g)
            else:
                ratio = min(e / max(thr, 1e-6), 0.999)
                g.result = _result(
                    g.track_id,
                    SWIMMER,
                    "ANALYZING",
                    risk=NON_DROWNING_RISK_CAP * ratio**NON_DROWNING_RISK_EXPONENT,
                    confidence=0.4,
                    mode=g.mode,
                    multiplier=g.multiplier,
                    reason=f"Lost from view inside {g.zone_name} {elapsed:.1f} s ago — watching for resurfacing.",
                )
        return fired

    def clip_end(self, now: float, min_seconds: float) -> list[SubmersionGhost]:
        """[2.18.0] A finite clip ended at `now` while these swimmers were still under: there are no more frames in
        which they could resurface, so a ghost lost at least `min_seconds` before the end is raised now instead of
        silently dropped with the clip. Ghosts that already alerted, or were lost too recently to judge, are left."""
        fired: list[SubmersionGhost] = []
        for g in self.ghosts.values():
            elapsed = g.elapsed(now)
            if g.alerted or elapsed < min_seconds:
                continue
            key = g.scenario
            g.escalated = True
            g.result = _result(
                g.track_id,
                SWIMMER,
                key,
                risk=BLACKOUT_CRITICAL_RISK,
                raw_risk=BLACKOUT_CRITICAL_RISK,
                confidence=0.75,
                mode=g.mode,
                multiplier=g.multiplier,
                evidence={key: 1.0},
                reason=(
                    f"Sudden Blackout / Submersion — swimmer vanished inside {g.zone_name} {elapsed:.1f} s before "
                    "the clip ended and had not resurfaced by its last frame"
                    + (f" ({g.event_reason})." if g.event_reason else ".")
                ),
            )
            g.alerted = True
            fired.append(g)
        return fired
