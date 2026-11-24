# mypy: disable-error-code="arg-type, assignment, union-attr"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""detection/scene_tracker.py — Per-source behaviour state (Phase 4).

One `SceneBehaviorTracker` per camera (live) or per uploaded video. It owns
every track's `TrackBuffer`, applies Stage 1 (zone categorisation with
anti-flicker hysteresis), runs the `BehaviorClassifier`, confirms alerts with
per-mode hysteresis + cooldown, and hands vanished in-water swimmers to the
`SubmersionMonitor` (Scenario D).

The live pipeline and the uploaded-video service both drive the SAME object,
so a clip analysed offline behaves exactly like the camera it came from.

Every frame (Phase 3.1 / 2.4)
─────────────────────────────
`ingest()` is called for EVERY analysed frame. Each track pools the frames it
was seen in and emits one behaviour sample every 1 / BEHAVIOR_SAMPLE_HZ s
(median box, confidence-weighted mean keypoints), so all frames feed the
behaviour buffer and single-frame keypoint jitter is averaged away. Display
fields (box, skeleton, zone) are updated on every frame.

In / out of the water: with zones drawn, the zones decide (time-based
hysteresis). Without zones, a person is a monitored swimmer unless there is
STRONG evidence they are on land — knees and ankles clearly visible below the
hips (and, when the automatic water estimate is confident, feet outside the
water) for LAND_EVIDENCE_SECONDS. Uncertainty keeps people monitored.

    tracker.set_geometry(zone_geometry)           # zones of the camera (or None)
    tracker.ingest(frame_detections, floatables)  # every inference frame
    fires = tracker.score(now)                    # every few inference frames
    tracker.expire(now)                           # every frame
    tracker.render_items() / tracker.events(...)  # Model View + WebSocket

[v3] Split (plan): per-track state -> scene/track_state.py, AlertHysteresis / AlertFire ->
scene/hysteresis.py, submersion ghosts -> scene/ghosts.py, zone / category logic ->
scene/category.py. The remaining SceneBehaviorTracker members are verbatim.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from lifeguard.vision.behaviour.classifier import (
    BYSTANDER,
    SWIMMER,
    BehaviorClassifier,
    BehaviorResult,
    normalize_mode,
)
from lifeguard.vision.behaviour.engine import FrameSample, TrackBuffer
from lifeguard.vision.behaviour.temporal import (
    AlertFire,
    AlertHysteresis,
    TemporalClassifier,
    TemporalRiskGate,
    reraise_damped,
)
from lifeguard.vision.constants import (
    ALERT_ONGOING_SECONDS,
    BEHAVIOR_SAMPLE_HZ,
    BODY_GATE_MIN_POSE_SAMPLES,
    CLIP_END_FLUSH,
    CLIP_END_GHOST_MIN_SECONDS,
    CLIP_END_GONE_SECONDS,
    DECK_EDGE_WARNING_SU,
    DORMANT_TRACK_SECONDS,
    GHOST_DUPLICATE_IOA,
    GHOST_FAINT_MIN_MEDIAN_CONF,
    GHOST_FAINT_MIN_PEAK_CONF,
    GHOST_MIN_MEDIAN_CONF,
    GHOST_MIN_PEAK_CONF,
    KINEMATICS,
    LIVE_ALERT_COOLDOWN_SECONDS,
    SMALL_PERSON_GATE,
    SMALL_PERSON_PX,
    SUBMERSION_OCCLUSION_IOU,
    TRACK_STALE_SECONDS,
    ZONE_VOTE_DECK_HOLD_SECONDS,
    ZONE_VOTE_DECK_NEEDS_LAND,
    ZONE_VOTE_DECK_TALL_ASPECT,
    ZONE_VOTE_DECK_TALL_VOTE,
    ZONE_VOTE_DRY_CONFIDENT,
    ZONE_VOTE_NEW_TRACK_SWIMMER,
)
from lifeguard.vision.inference.detector import FrameDetections, TrackedPerson
from lifeguard.vision.scene import zone_state as ZS
from lifeguard.vision.scene.alert_reason import DROWNING_ALERT
from lifeguard.vision.scene.category import ZoneCategoryLogic, demotion_blocked_by_risk
from lifeguard.vision.scene.floatables import (
    FloatableObject,
    best_iou,
    distance_to_floatable,
    nearest_centroid_distance,
)
from lifeguard.vision.scene.geometry import ZoneGeometry, ZoneHit
from lifeguard.vision.scene.ghosts import SubmersionGhost, SubmersionMonitor
from lifeguard.vision.scene.track_state import (
    ESTIMATED_DECK_HIT,
    WHOLE_FRAME_HIT,
    TrackState,
    _box_at,
    _FramePool,
    _ioa,
    _iou,
    _median_conf,
    _velocity,
    _was_swimming,
)
from lifeguard.vision.scene.water import WaterRegionEstimator
from lifeguard.vision.tracking.non_person import alert_occluders

logger = logging.getLogger(__name__)


_MOVEMENT_SCENARIOS = ("ACTIVE_DROWNING", "FLOATABLE_FAILURE", "DISTRESS")  # never PASSIVE: see below


def small_person_gate(buf: Any, res: BehaviorResult) -> BehaviorResult:
    """[v7 F5] SMALL_PERSON_GATE: an ACTIVE / FLOATABLE / DISTRESS call on a person whose pose cannot be read and whose
    median box height over the last 2 s is under SMALL_PERSON_PX keeps its scenario and risk (shown as CAUTION) but
    cannot raise an alarm: those rules judge limb and posture motion, which is noise without a readable pose on a few
    dozen pixels. PASSIVE_DROWNING is NEVER gated - a distant motionless floating victim is small and pose-less too
    (simulation: 15 px, pose unreliable), indistinguishable by size from a misread swimmer. SUBMERSION and the
    vanished-swimmer ghosts come from a separate path and are never gated either."""
    if not SMALL_PERSON_GATE or not res.is_drowning or res.scenario not in _MOVEMENT_SCENARIOS or res.pose_reliable:
        return res
    samples = getattr(buf, "samples", None) or []
    if not samples:
        return res
    t_end = samples[-1].t
    hs = [s.bbox[3] - s.bbox[1] for s in samples if t_end - s.t <= 2.0]
    if not hs:
        return res
    h = float(np.median(hs))
    if h >= SMALL_PERSON_PX:
        return res
    return dataclasses.replace(
        res,
        is_drowning=False,
        reason=f"{res.reason} Too small to confirm ({h:.0f} px tall, < {SMALL_PERSON_PX:.0f} px): caution, no alarm.",
    )


class SceneBehaviorTracker(ZoneCategoryLogic):
    def __init__(self, source_id: str, mode: str = "Medium", classifier: BehaviorClassifier | None = None) -> None:
        self.source_id = source_id
        self.mode = normalize_mode(mode)
        # [Phase 4 Step 1] scoring round -> alert: behaviour/temporal.py (classifier, small-person gate, hysteresis,
        # temporal gate, alert label); `classifier` and `gate` below are views onto it
        self.temporal = TemporalClassifier(classifier or BehaviorClassifier(), post_classify=small_person_gate)
        self.tracks: dict[int, TrackState] = {}
        self.geometry: ZoneGeometry | None = None
        self.submersion = SubmersionMonitor(self.classifier.params)
        # [2.31.0] the current frame, for the alert-time lens-occluder gate (set by vision/pipeline.py; None = no gate)
        self.alert_frame: Any = None
        self.alerts_gated = 0  # alert rounds held by that gate
        self.frame_w = 0
        self.frame_h = 0
        self.floatables: list[FloatableObject] = []
        self.floatables_t: float | None = None
        self.last_t: float = 0.0
        # Phase 2.4: automatic water region when no zones are drawn
        self.water = WaterRegionEstimator()
        # Phase 1.4: tracks that went out of view keep their history this long
        self.dormant: dict[int, TrackState] = {}
        self.sample_period = 1.0 / BEHAVIOR_SAMPLE_HZ
        self.fps: float | None = None  # [v7] analysed fps, for the frame-count rules (vision/frame_rules.py)
        # [2.20.0] per-track rolling history: sustained evidence before an alert rises (scene/temporal_gate.py)
        self.gate = TemporalRiskGate()

    @property
    def classifier(self) -> BehaviorClassifier:
        return self.temporal.classifier

    @classifier.setter
    def classifier(self, value: BehaviorClassifier) -> None:
        self.temporal.classifier = value

    @property
    def gate(self) -> TemporalRiskGate:
        return self.temporal.gate

    @gate.setter
    def gate(self, value: TemporalRiskGate) -> None:
        self.temporal.gate = value

    # ── configuration ─────────────────────────────────────────────────────────
    def set_mode(self, mode: str) -> None:
        self.mode = normalize_mode(mode)

    def set_geometry(self, geometry: ZoneGeometry | None) -> None:
        self.geometry = geometry

    def reset(self) -> None:
        """Forget every track (stream reconnect → tracker ids restart)."""
        self.tracks.clear()
        self.dormant.clear()
        self.submersion.reset()
        self.floatables = []
        self.floatables_t = None
        self.gate = TemporalRiskGate(self.gate.cfg)

    # ── ingestion ─────────────────────────────────────────────────────────────
    def ingest(self, det: FrameDetections, floatables: Sequence[FloatableObject] | None = None) -> None:
        """
        Add one inference frame. `floatables` = None when the floatable detector
        did not run on this frame (the last result stays valid for ≤ 1 s).
        """
        if det.frame_shape and det.frame_shape[0] and det.frame_shape[1]:
            self.frame_h, self.frame_w = int(det.frame_shape[0]), int(det.frame_shape[1])
        t = float(det.timestamp)
        self.last_t = t
        if floatables is not None:
            self.floatables = list(floatables)
            self.floatables_t = t
        floatables_valid = self.floatables_t is not None and t - self.floatables_t <= 1.0 and len(self.floatables) > 0
        owner = self._floatable_owners(det) if floatables_valid else {}
        self.gate.observe_frame(t, {p.track_id: p.bbox for p in det.tracks if not p.tentative})

        for p in det.tracks:
            if p.tentative:
                continue
            ts = self.tracks.get(p.track_id)
            if ts is None and p.track_id in self.dormant:
                ts = self.tracks[p.track_id] = self.dormant.pop(p.track_id)  # same person resurfaced
                ts.visible_since = t
            vote = ts.water_vote.update(p.water_prob, t) if ts is not None else p.water_prob
            # zone state: calibration / zone geometry fused with the detector's in-water vote
            zobs = ZS.observe(p.bbox, self.geometry, vote)
            if self.zoned:
                hit = self._zone_hit(p)
                observed = BYSTANDER if not hit.monitored else SWIMMER
                if zobs is not None:
                    observed = BYSTANDER if zobs.state == ZS.DECK else SWIMMER
                    if observed == SWIMMER and not hit.monitored:
                        hit = self._monitored_hit(p, hit)
            elif zobs is not None:
                observed, hit = (BYSTANDER if zobs.state == ZS.DECK else SWIMMER), None
            else:
                on_land, in_water = self._land_evidence(p)
                observed, hit = None, WHOLE_FRAME_HIT
            # [v7] no geometry: the vote alone must not put a new person on the deck from one frame
            vote_only_start = not self.zoned and zobs is not None and ZONE_VOTE_NEW_TRACK_SWIMMER
            if ts is None:
                first_cat = SWIMMER if vote_only_start else (observed or SWIMMER)  # no zones: monitored until proven
                ts = TrackState(
                    track_id=p.track_id,
                    buffer=TrackBuffer(p.track_id),
                    first_seen=t,
                    last_seen=t,
                    category=first_cat,
                    mode=self.mode,
                    visible_since=t,
                )
                ts.water_vote.update(p.water_prob, t)
                if zobs is not None:
                    ts.zone_state = ZS.IN_WATER if first_cat == SWIMMER and zobs.state == ZS.DECK else zobs.state
                self.tracks[p.track_id] = ts
            if self.zoned:
                self._update_category(ts, observed, t)
                # keep the zone consistent with the (hysteresis-protected) category
                if hit.monitored == (ts.category == SWIMMER):
                    ts.hit = hit
            elif zobs is not None:
                # [v7] ZONE_VOTE_DECK_HOLD_SECONDS: leaving the water on the vote alone takes longer than returning;
                # ZONE_VOTE_DRY_CONFIDENT: a confidently dry vote is the evidence, no clearly visible pose needed
                dry = vote is not None and vote < ZONE_VOTE_DRY_CONFIDENT
                hold = ZONE_VOTE_DECK_HOLD_SECONDS if observed == BYSTANDER else 0.0
                if observed == BYSTANDER and ts.category == SWIMMER and self._vote_demotion_blocked(ts, p, vote):
                    observed = SWIMMER  # [2.18.0] the vote alone does not take this person out of the water
                self._update_category(ts, observed, t, hold=hold, confident=dry)
                ts.hit = WHOLE_FRAME_HIT if ts.category == SWIMMER else ESTIMATED_DECK_HIT
            else:
                ts.land_obs.append((t, on_land, in_water))
                self._update_water_category(ts, t)
                ts.hit = WHOLE_FRAME_HIT if ts.category == SWIMMER else ESTIMATED_DECK_HIT
            self._update_zone_state(ts, zobs.state if zobs is not None else None, t)
            ts.last_seen = t
            ts.last_bbox = p.bbox
            if not getattr(p, "pose_skipped", False):
                ts.last_keypoints = p.keypoints  # a skipped frame keeps showing the last real skeleton
            ts.last_foot = p.foot_point
            ts.frames_seen += 1
            ts.peak_conf = max(ts.peak_conf, float(p.confidence))
            ts.faint_birth = ts.faint_birth or bool(getattr(p, "faint_birth", False))
            # most-inside of foot / box centre: a horizontal swimmer's "foot"
            # point is unreliable, and perspective squeezes the far edge.
            ts.edge_dist_px = max(self._edge_distance(p.foot_point), self._edge_distance(p.center))
            ts.pool.add(p)
            if t - ts.last_sample_t < self.sample_period - 1e-6:
                continue  # pooled into the next behaviour sample
            if ts.pool.posed == 0:
                continue  # only pose-skipped frames so far: wait (<= POSE_STRIDE - 1 frames) for a posed one
            bbox, conf, kps = ts.pool.pooled()
            ts.pool = _FramePool()
            # phase-accumulating schedule → exactly BEHAVIOR_SAMPLE_HZ on average at any fps
            nxt = ts.last_sample_t + self.sample_period
            ts.last_sample_t = nxt if t - nxt < self.sample_period else t
            cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

            # Floatable distance: only the person NEAREST to a float can be
            # "holding" it; others can only ever measure as separated from it.
            # No float visible at all → unknown (a detector miss never looks
            # like a separation).
            fdist: float | None = None
            fiou: float | None = None
            if floatables_valid:
                own = [f for i, f in enumerate(self.floatables) if owner.get(i) == p.track_id]
                if own:
                    # Phase 2 spec: BOTH measures for the person's own float —
                    # bbox IoU (are they overlapping it?) and centroid Euclidean
                    # distance (how far has it drifted?).
                    fdist = min(distance_to_floatable((cx, cy), own), nearest_centroid_distance((cx, cy), own))
                    fiou = best_iou(bbox, own)
                else:
                    floor = KINEMATICS["floatable_decoupled_su"] * ts.buffer.scale()
                    fdist = max(distance_to_floatable((cx, cy), self.floatables), floor)
                    fiou = best_iou(bbox, self.floatables)
            ts.buffer.add(
                FrameSample(
                    t=t,
                    bbox=bbox,
                    conf=conf,
                    keypoints=kps,
                    in_water=ts.category == SWIMMER,
                    edge_dist_px=ts.edge_dist_px,
                    floatable_dist_px=fdist,
                    floatable_iou=fiou,
                )
            )

        self._update_pool_xy(t)

        # a re-appearing person resolves any matching submersion ghost
        if self.submersion.ghosts:
            current = [
                (
                    ts.track_id,
                    ts.buffer.samples[-1].center
                    if ts.buffer.samples
                    else ((ts.last_bbox[0] + ts.last_bbox[2]) / 2.0, (ts.last_bbox[1] + ts.last_bbox[3]) / 2.0),
                    ts.visible_since,
                    math.sqrt(max((ts.last_bbox[2] - ts.last_bbox[0]) * (ts.last_bbox[3] - ts.last_bbox[1]), 1.0)),
                )
                for ts in self.tracks.values()
                if ts.last_seen == t
            ]
            self.submersion.reassociate(t, current)

    def _vote_demotion_blocked(self, ts: TrackState, p: TrackedPerson, vote: float | None) -> bool:
        """[2.18.0] No geometry, the detector's vote says "dry": keep the person monitored when they are at risk
        (ZONE_DEMOTE_MAX_RISK) or nothing but the vote says land (ZONE_VOTE_DECK_NEEDS_LAND). Land evidence: the
        legs-visible / feet-outside-the-water evidence of `_land_evidence`, or a clearly dry vote (below
        ZONE_VOTE_DECK_TALL_VOTE) on a tall, narrow full-body box (height/width at least ZONE_VOTE_DECK_TALL_ASPECT):
        someone standing on the deck too far away for the pose to prove the legs. A victim in the water shows a squat
        head-and-shoulders box."""
        if demotion_blocked_by_risk(ts):
            return True
        if not ZONE_VOTE_DECK_NEEDS_LAND or self._land_evidence(p)[0]:
            return False
        x1, y1, x2, y2 = p.bbox
        tall = (y2 - y1) >= ZONE_VOTE_DECK_TALL_ASPECT * max(1.0, x2 - x1)
        return not (tall and vote is not None and vote < ZONE_VOTE_DECK_TALL_VOTE)

    # ── pool plan position (this camera's own calibration) ────────────────────
    # [Phase 5] Cameras are independent (owner directive 2026-11-13): no cross-camera identity, and nothing another
    # camera sees resolves, hides or mutes this camera's tracks, ghosts or alerts.
    @property
    def calibration(self) -> Any:
        return getattr(self.geometry, "calibration", None) if self.geometry is not None else None

    def _update_pool_xy(self, t: float) -> None:
        cal = self.calibration
        if cal is None:
            return
        for ts in self.tracks.values():
            if ts.last_seen != t:
                continue
            x1, y1, x2, y2 = ts.last_bbox
            ts.pool_xy = cal.to_pool(((x1 + x2) / 2.0, float(y2)))

    def _floatable_owners(self, det: FrameDetections) -> dict[int, int]:
        """floatable index → track_id of the nearest person (distance in body-scale units)."""
        owners: dict[int, int] = {}
        people = [p for p in det.tracks if not p.tentative]
        for i, f in enumerate(self.floatables):
            best, best_id = math.inf, None
            for p in people:
                ts = self.tracks.get(p.track_id)
                scale = (
                    ts.buffer.scale()
                    if ts is not None and len(ts.buffer)
                    else max(8.0, 0.3 * math.sqrt((p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1])))
                )
                d = distance_to_floatable(p.center, [f]) / scale
                if d < best:
                    best, best_id = d, p.track_id
            if best_id is not None:
                owners[i] = best_id
        return owners

    # ── scoring ───────────────────────────────────────────────────────────────
    def mode_for(self, ts: TrackState) -> str:
        mode = self.mode if not self.zoned else self.geometry.mode_for(ts.hit, self.mode)
        # EDGE (sitting on the coping, holding the wall): monitored one step less sensitive
        return ZS.lower_mode(mode) if ts.zone_state == ZS.EDGE else mode

    def monitoring_active_for(self, ts: TrackState) -> bool:
        """Is this track's zone inside an active ScheduleMonitoring window?"""
        if not self.zoned:
            return True
        try:
            return self.geometry.monitoring_active(ts.hit)
        except Exception:
            logger.warning("schedule lookup failed - alert stays audible", exc_info=True)
            return True  # never let a bad schedule row mute a real alert

    def at_risk_track_ids(self) -> set[int]:
        """[Phase 2] Tracks already showing a rising or alarming signal (CAUTION/DROWNING, see
        _hysteresis below) - these stay on every-frame scoring instead of waiting for the next
        BEHAVIOR_SCORE_INTERVAL_S tick, so a worsening swimmer is caught sooner."""
        return {tid for tid, ts in self.tracks.items() if ts.state in ("CAUTION", "DROWNING")}

    def score(self, now: float, track_ids: set[int] | None = None) -> list[AlertFire]:
        """track_ids=None scores every track plus the submersion-ghost sweep (the normal cadence
        tick); a given subset (at-risk tracks between ticks) only re-scores those tracks."""
        fires: list[AlertFire] = []
        for ts in list(self.tracks.values()):
            if track_ids is not None and ts.track_id not in track_ids:
                continue
            if ts.last_seen < now - 1.0:
                continue  # not currently visible; handled by expiry
            ts.mode = self.mode_for(ts)
            ts.monitoring_active = self.monitoring_active_for(ts)
            if ts.category == BYSTANDER:
                ts.result = self.classifier.classify(ts.buffer, category=BYSTANDER, mode=ts.mode)
                ts.clear_alert()
                ts.deck_edge = (
                    math.isfinite(ts.edge_dist_px)
                    and ts.edge_dist_px < 0
                    and -ts.edge_dist_px <= DECK_EDGE_WARNING_SU * ts.buffer.scale()
                )
                ts.state = "DECK_EDGE" if ts.deck_edge else "DECK"
                continue
            ts.deck_edge = False
            # classifier -> small_person_gate ([v7 F5] too small to judge: caution, never an alarm) -> gate history
            res = self.temporal.classify(
                ts.buffer,
                track_id=ts.track_id,
                now=now,
                mode=ts.mode,
                sensitivity_multiplier=ts.hit.sensitivity_multiplier,
                zone_type=ts.hit.zone_type,
            )
            ts.result = res
            fire = self._hysteresis(ts, res, now)
            if fire is not None:
                fires.append(fire)

        if track_ids is None:
            for g in self.submersion.evaluate(now):
                fire = self._ghost_fire(g, now)
                if fire is not None:
                    fires.append(fire)
        return fires

    def _hysteresis(self, ts: TrackState, res: BehaviorResult, now: float) -> AlertFire | None:
        # hysteresis, then the temporal gate ([2.20.0] defers a rising edge, never cancels it)
        was_active, prev_scenario = ts.hyst.active, ts.hyst.scenario
        rising, gate_feats = self.temporal.confirm(ts.track_id, ts.hyst, ts.buffer, res, ts.mode, now)
        if was_active and not ts.hyst.active:  # the alert cleared: remembered for re-raise damping
            ts.last_alert_end, ts.last_alert_scenario = now, prev_scenario

        ts.peak_risk = max(ts.peak_risk, res.risk)
        if res.scenario == "ANALYZING":
            ts.state = "ANALYZING"
        elif ts.alert_active:
            ts.state = "DROWNING"
        elif res.is_drowning or res.risk >= 0.26:
            ts.state = "CAUTION"
        else:
            ts.state = "SWIMMER"

        # [2.19.0] while the alert is active it fires again every ALERT_ONGOING_SECONDS (the incident stays loud and
        # timed for its whole duration); a notification (outbox, alert row of an upload) goes out on the rising edge
        # and then every LIVE_ALERT_COOLDOWN_SECONDS, as before.
        every = ALERT_ONGOING_SECONDS if ALERT_ONGOING_SECONDS > 0 else LIVE_ALERT_COOLDOWN_SECONDS
        # [2.31.0] a lens occluder on this frame raises / re-raises nothing; its notification state is left
        # untouched, so the alert notifies once the box is no longer an occluder (ghosts are never gated)
        if (
            ts.alert_active
            and self.alert_frame is not None
            and ts.last_bbox is not None
            and alert_occluders(self.alert_frame, [(ts.track_id, ts.last_bbox)])
        ):
            self.alerts_gated += 1
            return None
        if ts.alert_active and (rising or now - ts.last_alert_time >= every - 1e-6):
            # [Phase 4 Step 4] a damped re-raise is a continuation of the alert that just cleared: raised, not new
            new = rising and not reraise_damped(ts.last_alert_scenario, ts.last_alert_end, res.scenario, now)
            notify = new or now - ts.last_notify_time >= LIVE_ALERT_COOLDOWN_SECONDS
            ts.last_alert_time = now
            if notify:
                ts.last_notify_time = now
            started = ts.hyst.started_at if ts.hyst.started_at is not None else now
            return AlertFire(
                ts.track_id,
                self.temporal.report(res, ts.buffer),  # [Stage 4] one Drowning Alert + reason; timing untouched
                ts.last_bbox,
                ts.hit.zone_name,
                ts.hit.zone_type,
                ts.mode,
                first=new,
                buffer=ts.buffer,
                siren=ts.monitoring_active,
                t=now,
                onset_t=ts.hyst.onset,
                ongoing_s=max(0.0, now - started),
                notify=notify,
                gate=gate_feats.as_dict() if gate_feats is not None else None,
            )
        return None

    def _ghost_fire(self, g: SubmersionGhost, now: float) -> AlertFire | None:
        """The alert of a vanished swimmer, unless the temporal gate rules its history out (final: it is frozen)."""
        if g.result is None:
            return None
        ok, feats = self.gate.allow(g.track_id, now, ghost=True)
        if not ok:
            return None
        return AlertFire(
            g.track_id,
            self.temporal.report(g.result, getattr(g, "buffer", None), ghost=True),  # [Stage 4] Submersion reason
            g.last_bbox,
            g.zone_name,
            g.zone_type,
            g.mode,
            first=True,
            ghost=True,
            siren=self._ghost_siren(g),
            t=now,
            gate=feats.as_dict() if feats is not None else None,
        )

    def _ghost_siren(self, g: SubmersionGhost) -> bool:
        """A vanished swimmer inherits the schedule state of the zone they were in."""
        if self.geometry is None:
            return True
        for z in self.geometry.zones:
            if z.name == g.zone_name:
                try:
                    return self.geometry.monitoring_active(
                        ZoneHit(
                            z, z.monitored, z.monitored, z.name, z.zone_type, z.sensitivity_multiplier, z.schedule_id
                        )
                    )
                except Exception:
                    logger.warning("schedule lookup failed - alert stays audible", exc_info=True)
                    return True
        return True

    # ── expiry → Scenario D ───────────────────────────────────────────────────
    def expire(self, now: float) -> list[TrackState]:
        for tid in [tid for tid, d in self.dormant.items() if now - d.last_seen > DORMANT_TRACK_SECONDS]:
            del self.dormant[tid]
        stale = [ts for ts in self.tracks.values() if now - ts.last_seen > TRACK_STALE_SECONDS]
        return self._lose(stale, now)

    def forget(self, track_ids: set[int]) -> None:
        """[2.18.0] These ids are not people (tracking/non_person.py): drop their state without a submersion ghost -
        an anchor post that stops being reported has not "vanished in the water"."""
        for tid in track_ids:
            self.tracks.pop(tid, None)
            self.dormant.pop(tid, None)
            self.submersion.ghosts.pop(tid, None)
            self.gate.forget(tid)

    def finish(self, now: float) -> list[AlertFire]:
        """[2.18.0] The clip ended at `now` (eval harness; a live stream never calls this). Visible tracks are
        scored once more; swimmers gone for more than CLIP_END_GONE_SECONDS are treated as lost now (the usual ghost
        filters apply), and a ghost under for at least CLIP_END_GHOST_MIN_SECONDS raises SUBMERSION - the clip has no
        more frames in which they could resurface."""
        fires = self.score(now)
        if not CLIP_END_FLUSH:
            return fires
        gone = [ts for ts in self.tracks.values() if now - ts.last_seen > CLIP_END_GONE_SECONDS]
        self._lose(gone, now)
        for g in self.submersion.clip_end(now, CLIP_END_GHOST_MIN_SECONDS):
            fire = self._ghost_fire(g, now)
            if fire is not None:
                fires.append(fire)
        return fires

    def _lose(self, stale: list[TrackState], now: float) -> list[TrackState]:
        """Move `stale` tracks to dormant and hand vanished swimmers to the SubmersionMonitor (Scenario D)."""
        if not stale:
            return []
        others = [ts for ts in self.tracks.values() if ts not in stale]
        for ts in stale:
            del self.tracks[ts.track_id]
            self.dormant[ts.track_id] = ts
            self.gate.freeze(ts.track_id, now)
            if ts.track_id in self.submersion.ghosts:
                continue
            fell_in = ts.category == BYSTANDER and not self.zoned and self._lost_land_evidence(ts)
            if ts.category != SWIMMER and not fell_in:
                continue
            min_peak, min_med = (
                (GHOST_FAINT_MIN_PEAK_CONF, GHOST_FAINT_MIN_MEDIAN_CONF)
                if ts.faint_birth
                else (GHOST_MIN_PEAK_CONF, GHOST_MIN_MEDIAN_CONF)
            )
            if ts.peak_conf < min_peak or _median_conf(ts) < min_med:
                continue  # never solidly a person (reflection, splash, glint, faint duplicate)
            if ts.buffer.good_pose_samples < BODY_GATE_MIN_POSE_SAMPLES:
                continue  # [2.19.0] never showed a body: a crowd fragment / reflection, not a swimmer who went under
            if self._duplicate_at_loss(ts, others):
                continue  # a duplicate box on a person who stayed in view
            occluded = any(_iou(ts.last_bbox, o.last_bbox) >= SUBMERSION_OCCLUSION_IOU for o in others)
            self.submersion.track_lost(
                ts.buffer,
                last_bbox=ts.last_bbox,
                edge_dist_px=ts.edge_dist_px,
                frame_w=self.frame_w or 1,
                frame_h=self.frame_h or 1,
                category=SWIMMER,
                zone_name=ts.hit.zone_name,
                zone_type=ts.hit.zone_type,
                multiplier=ts.hit.sensitivity_multiplier,
                mode=ts.mode,
                occluded=occluded,
                travelling=_was_swimming(ts),
                velocity=_velocity(ts),
            )
        return stale

    @staticmethod
    def _duplicate_at_loss(ts: TrackState, others: Sequence[TrackState]) -> bool:
        """
        Was the vanished track a second box on someone who carried on? Compared
        with the other track's box AT THE MOMENT OF LOSS — `o.last_bbox` is from
        TRACK_STALE_SECONDS later, by when that person has swum elsewhere and
        the old check never matched (a false CRITICAL BLACKOUT on a swimmer who
        was in plain view the whole time).
        """
        for o in others:
            if o.last_seen <= ts.last_seen:
                continue  # did not carry on after the loss
            ob = _box_at(o, ts.last_seen)
            if ob is None:
                continue
            if _ioa(ts.last_bbox, ob) >= GHOST_DUPLICATE_IOA and _median_conf(o) >= _median_conf(ts):
                return True
        return False

    # ── views ─────────────────────────────────────────────────────────────────
    @property
    def ghosts(self) -> list[SubmersionGhost]:
        return list(self.submersion.ghosts.values())

    def counts(self) -> dict[str, int]:
        visible = [ts for ts in self.tracks.values() if ts.last_seen >= self.last_t - 1.0]
        return {
            "person_count": len(visible),
            "swimmer_count": sum(1 for ts in visible if ts.category == SWIMMER),
            "bystander_count": sum(1 for ts in visible if ts.category == BYSTANDER),
            "in_water_count": sum(1 for ts in visible if ts.category == SWIMMER and ts.zone_state == ZS.IN_WATER),
            "edge_count": sum(1 for ts in visible if ts.category == SWIMMER and ts.zone_state == ZS.EDGE),
            "deck_count": sum(1 for ts in visible if ts.category == BYSTANDER),
            "active_alerts": sum(1 for ts in visible if ts.alert_active)
            + sum(1 for g in self.submersion.ghosts.values() if g.alerted),
            "submerged": len(self.submersion.ghosts),
        }

    def events(self) -> list[dict[str, Any]]:
        w, h = self.frame_w or 1, self.frame_h or 1
        out = [ts.to_event(w, h) for ts in self.tracks.values() if ts.last_seen >= self.last_t - 1.0]
        for g in self.submersion.ghosts.values():
            ev = g.to_dict(self.last_t, w, h)
            r = g.result
            out.append(
                {
                    "track_id": g.track_id,
                    "state": ev["state"],
                    "category": SWIMMER,
                    "label": ev["label"],
                    "drowning_type": ev["label"],
                    "scenario": "SUBMERSION" if r else "ANALYZING",
                    "reason": r.reason if r else "",
                    "zone": g.zone_name,
                    "zone_type": g.zone_type,
                    "monitored": True,
                    "mode": g.mode,
                    "risk": ev["risk"],
                    "risk_score": ev["risk_score"],
                    "raw_risk_score": ev["risk_score"],
                    "confidence": round(r.confidence, 3) if r else 0.0,
                    "priority": r.priority if (r and g.alerted) else "None",
                    "alert_active": g.alerted,
                    "deck_edge": False,
                    "samples": 0,
                    "monitoring_active": True,
                    "bbox_norm": ev["bbox_norm"],
                    "keypoints_norm": None,
                    "submerged_seconds": ev["elapsed"],
                    "blackout_remaining": ev["remaining"],
                    "escalated": ev["escalated"],
                }
            )
        return out

    def render_items(self) -> list[dict[str, Any]]:
        """Pixel-space data for the Model View renderer (current tracks + ghosts)."""
        items: list[dict[str, Any]] = []
        for ts in self.tracks.values():
            if ts.last_seen < self.last_t - 0.5:
                continue
            r = ts.result
            items.append(
                {
                    "track_id": ts.track_id,
                    "bbox": ts.last_bbox,
                    "keypoints": ts.last_keypoints,
                    "foot": ts.last_foot,
                    "state": ts.state,
                    "category": ts.category,
                    "label": r.label if r else ("Bystander / Non-Swimmer" if ts.category == BYSTANDER else "Analyzing"),
                    "risk": r.risk if r else 0.0,
                    "zone": ts.hit.zone_name,
                    "alert": ts.alert_active,
                    "deck_edge": ts.deck_edge,
                    "ghost": False,
                    "activity": ts.activity,
                    "zone_state": ts.zone_state,
                }
            )
        for g in self.submersion.ghosts.values():
            live = self.tracks.get(g.track_id)
            if live is not None and live.last_seen >= self.last_t - 0.5 and _iou(live.last_bbox, g.last_bbox) > 0.3:
                continue
            items.append(
                {
                    "track_id": g.track_id,
                    "bbox": g.last_bbox,
                    "keypoints": None,
                    "foot": g.last_pos,
                    "state": g.state,
                    "category": SWIMMER,
                    "label": DROWNING_ALERT if g.alerted else "Lost from view",
                    "risk": g.result.risk if g.result else 0.0,
                    "zone": g.zone_name,
                    "alert": g.alerted,
                    "deck_edge": False,
                    "ghost": True,
                    "elapsed": g.elapsed(self.last_t),
                }
            )
        return items

    @property
    def geometry_source(self) -> str:
        """Where in / out of the water comes from: a pool calibration, drawn zones, or - as the
        FALLBACK only - the automatic colour estimate ("none" until it is confident)."""
        if self.calibration is not None and not getattr(self.geometry, "drawn_water", False):
            return "calibration"
        if self.zoned:
            return "zones"
        return "estimated" if self.water.confident else "none"


__all__ = ["AlertFire", "AlertHysteresis", "SceneBehaviorTracker", "TrackState"]
