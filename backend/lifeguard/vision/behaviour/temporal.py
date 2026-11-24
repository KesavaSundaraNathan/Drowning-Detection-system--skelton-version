"""[Phase 4 Step 1] Per-track temporal classification: one module from scoring round to raised alert.

Plan (docs/LIFEGUARD_REBUILD_PLAN.md, Phase 4): `classifier.py`, `hysteresis.py` and `temporal_gate.py` become
`behaviour/temporal.py` - a sliding window per track, 6 classes, confidence and reason; rules now, the learned model
(`learned/`) when labelled data exists. Step 1 MOVES the code, verbatim, with no behaviour change (proven by
bit-identical dev traces, docs/evidence/phase4/RULE.md). The old modules re-export from here.

A scoring round, in order (TemporalClassifier):
  1. BehaviorClassifier.classify  - the rule scenarios over the track's buffer (behaviour/{features,evidence}.py)
  2. post_classify                - the scene's small-person gate (scene_tracker.small_person_gate)
  3. TemporalRiskGate.observe_score
  4. AlertHysteresis.update       - positive_windows rounds (ACTIVE: active_sustain_seconds) raise, negatives clear
  5. TemporalRiskGate.allow       - defers (never cancels) a rising edge without sustained history; off by default
  6. alert reason                 - what the AlertFire reports (scene/alert_reason.py: one "Drowning Alert")
SUBMERSION ghosts (scene/ghosts.py) keep their own path; nothing here mutes SUBMERSION or FAULT.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from lifeguard.vision import constants as C
from lifeguard.vision.behaviour.engine import (
    KinematicFeatures,
    RiskEngine,
    ScenarioEvidence,
    TrackBuffer,
    clamp01,
    finite,
    ramp,
)
from lifeguard.vision.behaviour.params import (
    clamp_multiplier,
    effective_thresholds,
    get_mode_params,
    normalize_mode,
    priority_for,
    thresholds_source,
    timed_params,
)
from lifeguard.vision.behaviour.result import BYSTANDER, SWIMMER, BehaviorResult, _result
from lifeguard.vision.constants import (
    ACTIVE_DROWNING_RISK_MIN,
    ALERT_RERAISE_DAMP_S,
    ALERT_SCENARIOS,
    BEHAVIOR_MODEL_DEFAULT_THRESHOLDS,
    BEHAVIOR_MODEL_MIN_SECONDS,
    BEHAVIOR_MODEL_MODE_SCALE,
    BEHAVIOR_MODEL_RULE_NORMAL_BLOCKS,
    BEHAVIOR_MODEL_THRESHOLD_BOUNDS,
    BODY_GATE_MIN_POSE_SAMPLES,
    DECK_PERSON_RISK,
    DROWNING_SCENARIOS,
    DROWNING_TYPES,
    MIN_TRACK_SAMPLES,
    NON_DROWNING_RISK_CAP,
    NON_DROWNING_RISK_EXPONENT,
    NORMAL_ACTIVITIES,
    NORMAL_ACTIVITY_CONF_MIN,
    NORMAL_SWIMMING_RISK_MAX,
    PASSIVE_DROWNING_RISK_MIN,
    RAISE_MIN_POSITIVE_S,
    SCENARIO_BASE_RISK,
    SCENARIO_RISK_MAX,
    SCENARIO_SEVERITY,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Rule scenarios per round (was behaviour/classifier.py, verbatim)
# ═════════════════════════════════════════════════════════════════════════════
class BehaviorClassifier:
    """Stateless multi-stage classifier; all per-track state lives in TrackBuffer."""

    def __init__(
        self,
        engine: RiskEngine | None = None,
        params_fn: Callable[[str], dict[str, Any]] | None = None,
        behavior_model: Any = "default",
    ) -> None:
        self.engine = engine or RiskEngine()
        # calibration injects candidate parameters without touching live cameras
        self.params = params_fn or get_mode_params
        # Learned model (detection/behavior_model): "default" = the configured
        # BEHAVIOR_MODEL_PATH (resolved lazily), None = rules only, or an instance.
        self._behavior_model = behavior_model

    @property
    def behavior_model(self) -> Any:
        if isinstance(self._behavior_model, str):
            from lifeguard.vision.behaviour.learned import get_behavior_model

            return get_behavior_model()
        return self._behavior_model

    def learned_candidates(
        self, buf: TrackBuffer, f: KinematicFeatures, mode: str, multiplier: float
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """
        (probabilities, effective thresholds, candidates ≥ threshold) from the
        learned model; three empty dicts when no model is configured or the
        track is too short. Thresholds come from the model sidecar (tuned on
        out-of-fold predictions), scaled by the monitoring mode and divided by
        the zone multiplier like graded rule scenarios.
        """
        model = self.behavior_model
        if model is None or buf.duration < BEHAVIOR_MODEL_MIN_SECONDS:
            return {}, {}, {}
        from lifeguard.vision.behaviour.learned import window_features

        try:
            seq, summ = window_features(buf, f)
            probs = model.predict_one(seq, summ)
        except Exception as exc:  # a broken model never takes the rules down
            logger.warning("[classifier] behaviour model failed on track %s: %s", buf.track_id, exc, exc_info=True)
            return {}, {}, {}
        lo, hi = BEHAVIOR_MODEL_THRESHOLD_BOUNDS
        scale = BEHAVIOR_MODEL_MODE_SCALE.get(mode, 1.0) / clamp_multiplier(multiplier)
        thr: dict[str, float] = {}
        for k in ALERT_SCENARIOS:
            if k not in probs:
                continue
            base = getattr(model, "thresholds", {}).get(k, BEHAVIOR_MODEL_DEFAULT_THRESHOLDS.get(k, 0.6))
            thr[k] = float(min(hi, max(lo, base * scale)))
        cands = {k: probs[k] for k in thr if probs[k] >= thr[k]}
        return probs, thr, cands

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    @staticmethod
    def categorize(in_water: bool) -> str:
        """Bystander / Non-Swimmer outside the water polygon, Swimmer inside."""
        return SWIMMER if in_water else BYSTANDER

    # ── Stages 2–4 ────────────────────────────────────────────────────────────
    def classify(
        self,
        buf: TrackBuffer,
        *,
        category: str = SWIMMER,
        mode: str = "Medium",
        sensitivity_multiplier: float = 1.0,
        zone_type: str | None = None,
    ) -> BehaviorResult:
        mode = normalize_mode(mode)
        m = clamp_multiplier(sensitivity_multiplier)
        params = self.params(mode)
        tid = buf.track_id

        if category == BYSTANDER:
            # Spec DECK_PERSON rule: centre-bottom outside every water polygon
            # → risk_score = 0.0, water alerts ignored entirely.
            return _result(
                tid,
                BYSTANDER,
                "BYSTANDER",
                risk=DECK_PERSON_RISK,
                raw_risk=DECK_PERSON_RISK,
                confidence=0.9,
                mode=mode,
                multiplier=m,
                reason="Deck / non-swimmer — centre-bottom outside every water polygon, tracked but not risk-scored.",
            )

        if len(buf) < MIN_TRACK_SAMPLES or buf.duration < params["min_window_seconds"]:
            return _result(
                tid,
                SWIMMER,
                "ANALYZING",
                confidence=0.4,
                mode=mode,
                multiplier=m,
                reason=f"Collecting motion history ({buf.duration:.1f}/{params['min_window_seconds']:.1f} s).",
            )

        f = self.engine.extract(buf, zone_type)
        ev_raw = self.engine.evidence(f, params)
        ev_adj = self.engine.evidence(f, timed_params(params, m)) if m != 1.0 else ev_raw
        raw = ev_raw.drowning()
        adj = ev_adj.drowning()
        base_thr = params["thresholds"]
        thr = effective_thresholds(params, m)

        # Phase 2.2: a person positively recognised as doing something normal
        # cannot raise Active / Passive / Floatable alerts. Visible submersion
        # is never blocked (it has its own box-corroborated evidence).
        activity, act_conf = ev_adj.activity, ev_adj.activity_conf
        normal_block = activity in NORMAL_ACTIVITIES and act_conf >= NORMAL_ACTIVITY_CONF_MIN
        if normal_block:
            buf.last_normal_t = buf.last_t
            for k in ("ACTIVE_DROWNING", "PASSIVE_DROWNING", "FLOATABLE_FAILURE"):
                adj[k] = min(adj[k], 0.95 * thr[k])
                raw[k] = min(raw[k], 0.95 * base_thr[k])
        if activity != "UNKNOWN":
            buf.last_activity = activity
        if buf.good_pose_samples < BODY_GATE_MIN_POSE_SAMPLES:
            # [2.19.0] Floatable Failure is a person losing their float / falling: a track that never showed a body is
            # a crowd fragment whose box jumps between people (owner RTSP test: 'sudden displacement x10-20')
            adj["FLOATABLE_FAILURE"] = min(adj["FLOATABLE_FAILURE"], 0.95 * thr["FLOATABLE_FAILURE"])
            raw["FLOATABLE_FAILURE"] = min(raw["FLOATABLE_FAILURE"], 0.95 * base_thr["FLOATABLE_FAILURE"])
        extra: dict[str, Any] = dict(
            activity=activity, activity_conf=act_conf, pose_reliable=ev_adj.pose_ok, cues=ev_adj.cues
        )
        # Phase 1.2: the zone multiplier scales the THRESHOLDS, not the evidence
        # (Effective Threshold = Base Threshold / sensitivity_multiplier), so a
        # deep-end zone clears the bar on less evidence than a splash zone.
        quality = 0.4 + 0.6 * clamp01(f.pose_quality / 0.6)
        evidence = {k: round(v, 3) for k, v in adj.items()}
        evidence["NORMAL_SWIMMING"] = round(ev_adj.normal, 3)

        # ML core v2: learned model probabilities (rules stay as the safety net)
        l_probs, l_thr, l_cands = self.learned_candidates(buf, f, mode, m)
        if l_probs:
            for k_, v_ in l_probs.items():
                evidence[f"LEARNED_{k_}"] = round(v_, 3)
            if normal_block and BEHAVIOR_MODEL_RULE_NORMAL_BLOCKS:
                l_cands = {}
        rule_cands = [k for k in DROWNING_SCENARIOS if adj[k] >= thr[k]]
        # a learned call on a scenario the rules ALSO see keeps the rule path
        # (its risk bands / signature floors); learned-only calls go below
        learned_only = {k: p for k, p in l_cands.items() if k not in rule_cands}
        if learned_only and (
            not rule_cands
            or max(SCENARIO_SEVERITY[k] * p for k, p in learned_only.items())
            > max(SCENARIO_SEVERITY[k] * adj[k] for k in rule_cands)
        ):
            key = max(learned_only, key=lambda k: (learned_only[k] * SCENARIO_SEVERITY[k], SCENARIO_SEVERITY[k]))
            p, t_ = learned_only[key], l_thr[key]
            risk = min(self._drowning_risk(key, p, t_), SCENARIO_RISK_MAX.get(key, 1.0))
            margin = ramp(p, t_, min(1.0, t_ + 0.35))
            r = _result(
                tid,
                SWIMMER,
                key,
                risk=risk,
                raw_risk=risk,
                confidence=0.5 + 0.45 * margin,
                mode=mode,
                multiplier=m,
                evidence=evidence,
                features=f.to_dict(),
                reason=self._learned_reason(key, p, t_),
            )
            for k_, v_ in extra.items():
                setattr(r, k_, v_)
            return r
        cands = rule_cands
        if cands:
            key = max(cands, key=lambda k: (adj[k] * SCENARIO_SEVERITY[k], SCENARIO_SEVERITY[k]))
            risk = self._drowning_risk(key, adj[key], thr[key])
            raw_risk = (
                self._drowning_risk(key, raw[key], base_thr[key])
                if raw[key] >= base_thr[key]
                else self._non_drowning_risk(raw, base_thr)
            )
            risk = self._apply_signature_floor(key, risk, ev_adj)
            raw_risk = self._apply_signature_floor(key, raw_risk, ev_raw)
            margin = ramp(adj[key], thr[key], min(1.0, thr[key] + 0.35))
            conf = 0.45 + 0.5 * quality * (0.5 + 0.5 * margin)
            r = _result(
                tid,
                SWIMMER,
                key,
                risk=risk,
                raw_risk=raw_risk,
                confidence=conf,
                mode=mode,
                multiplier=m,
                evidence=evidence,
                features=f.to_dict(),
                reason=self._reason(key, f, ev_adj),
            )
            for k_, v_ in extra.items():
                setattr(r, k_, v_)
            return r

        risk = self._non_drowning_risk(adj, thr)
        raw_risk = self._non_drowning_risk(raw, base_thr)
        closeness = max(adj[k] / max(thr[k], 1e-6) for k in DROWNING_SCENARIOS)
        conf = 0.45 + 0.5 * quality * max(ev_adj.normal, 1.0 - closeness)
        scenario = "RESTING" if (ev_adj.resting and f.moving < 0.2) else "NORMAL_SWIMMING"
        if normal_block:
            scenario = "NORMAL_SWIMMING" if activity == "SWIMMING" else "RESTING"
        if scenario == "NORMAL_SWIMMING" and ev_adj.sig_normal:
            # Spec: a textbook freestyle signature scores strictly below 0.20.
            risk = min(risk, NORMAL_SWIMMING_RISK_MAX - 0.01)
            raw_risk = min(raw_risk, NORMAL_SWIMMING_RISK_MAX - 0.01)
        r = _result(
            tid,
            SWIMMER,
            scenario,
            risk=risk,
            raw_risk=raw_risk,
            confidence=conf,
            mode=mode,
            multiplier=m,
            evidence=evidence,
            features=f.to_dict(),
            reason=self._reason(scenario, f, ev_adj),
        )
        for k_, v_ in extra.items():
            setattr(r, k_, v_)
        return r

    # ── risk mapping ──────────────────────────────────────────────────────────
    @staticmethod
    def _apply_signature_floor(key: str, risk: float, ev: ScenarioEvidence) -> float:
        """
        Phase 1.3 spec pins risk bands for the canonical signatures:
            ACTIVE_DROWNING  (θ > 60°, |v| ≈ 0, vertical thrashing)  → > 0.85
            PASSIVE_DROWNING (flat/inverted, zero velocity > 8 s)    → > 0.90
        The graded evidence already put these well above threshold; this only
        guarantees the floor so the stated rules hold exactly.
        """
        if key == "ACTIVE_DROWNING" and ev.sig_active:
            return max(risk, ACTIVE_DROWNING_RISK_MIN)
        if key == "PASSIVE_DROWNING" and ev.sig_passive:
            return max(risk, PASSIVE_DROWNING_RISK_MIN)
        return risk

    @staticmethod
    def _drowning_risk(key: str, e: float, thr: float) -> float:
        base = SCENARIO_BASE_RISK[key]
        return clamp01(base + (1.0 - base) * ramp(e, thr, 1.0))

    @staticmethod
    def _non_drowning_risk(ev: dict[str, float], thr: dict[str, float]) -> float:
        ratio = max(min(ev[k] / max(thr[k], 1e-6), 0.999) for k in DROWNING_SCENARIOS)
        return clamp01(NON_DROWNING_RISK_CAP * ratio**NON_DROWNING_RISK_EXPONENT)

    # ── explanation ───────────────────────────────────────────────────────────
    def _learned_reason(self, key: str, p: float, thr: float) -> str:
        from lifeguard.vision.constants import BEHAVIOR_MODEL_SEQ_SECONDS  # was via learned.features (same value)

        name = getattr(self.behavior_model, "name", "behaviour model")
        what = {
            "DISTRESS": "struggling / signalling for help",
            "ACTIVE_DROWNING": "instinctive drowning response",
            "PASSIVE_DROWNING": "motionless at or below the surface",
        }.get(key, key)
        return (
            f"{DROWNING_TYPES[key]} — learned model '{name}': {what}, probability {p:.2f} "
            f"(threshold {thr:.2f}) over the last {BEHAVIOR_MODEL_SEQ_SECONDS:.0f} s."
        )

    @staticmethod
    def _reason(key: str, f: KinematicFeatures, ev: ScenarioEvidence) -> str:
        def fmt(v: Any, spec: str = ".2f", unit: str = "") -> str:
            return f"{float(v):{spec}}{unit}" if finite(v) else "n/a"

        d = ev.detail
        basis = "" if ev.pose_ok else " [pose unreliable — box/track cues]"
        act_text = {
            "SWIMMING": "swimming",
            "TREADING": "treading water",
            "STANDING_WATER": "standing in the water",
            "RESTING_EDGE": "resting at the pool edge",
        }
        if (
            key in ("NORMAL_SWIMMING", "RESTING")
            and ev.activity in act_text
            and ev.activity_conf >= NORMAL_ACTIVITY_CONF_MIN
        ):
            head = "Normal Swimming" if key == "NORMAL_SWIMMING" else "Resting / Treading Water"
            return (
                f"{head} — {act_text[ev.activity]} recognised (confidence {ev.activity_conf:.2f}), travel "
                f"{fmt(f.moving)} torso-lengths/s, torso {fmt(f.torso_angle, '.0f', '°')}, stroke periodicity "
                f"{fmt(f.arm_periodicity)}.{basis}"
            )
        if key == "ACTIVE_DROWNING":
            parts = [
                f"vertical torso {fmt(f.uprightness, '.0f', '°')}",
                f"net forward progress {fmt(f.progress_speed)} torso-lengths/s",
            ]
            if d.get("arms", -1) >= 0:
                parts.append(
                    f"frantic vertical arm motion (amp {fmt(f.arm_amp)}, "
                    f"{fmt(f.arm_freq, '.1f', ' Hz')}, periodicity {fmt(f.arm_periodicity)})"
                )
            if d.get("bob", -1) >= 0:
                parts.append(f"head bobbing at the surface ({fmt(f.head_bob)}, {f.head_flicker:.1f} dips/s)")
            parts.append(f"{ev.agreement:.1f} cues agreeing")
            return "Active Drowning — " + ", ".join(parts) + "." + basis
        if key == "PASSIVE_DROWNING":
            if d.get("face_ev", 0) >= max(d.get("still_ev", 0), d.get("inverted_ev", 0)):
                return (
                    f"Passive Drowning — face-down and motionless for {f.face_down_seconds:.1f} s "
                    f"(progress {fmt(f.moving)} torso-lengths/s)."
                )
            if d.get("inverted_ev", 0) >= d.get("still_ev", 0):
                return f"Passive Drowning — inverted (head below hips) for {f.inverted_seconds:.1f} s."
            return (
                f"Passive Drowning — negligible movement for {f.still_seconds:.1f} s on the surface "
                f"(progress {fmt(f.moving)} torso-lengths/s, limb motion {fmt(f.pose_motion)})."
            )
        if key == "FLOATABLE_FAILURE":
            parts = []
            if d.get("decouple", 0) > 0:
                parts.append(f"separated from floatable {fmt(f.decoupled_ago, '.1f', ' s')} ago")
            if d.get("spike", 0) > 0:
                parts.append(f"sudden displacement ×{f.spike_ratio:.1f}")
            if d.get("flip", 0) > 0:
                parts.append(f"abrupt posture failure {f.flip_deg:.0f}° → {fmt(f.flip_to, '.0f', '°')}")
            if f.child_likelihood >= 0.5:
                parts.append(f"child likelihood {f.child_likelihood:.2f}")
            parts.append("followed by distress / no progress")
            return "Floatable Failure — " + ", ".join(parts) + "."
        if key == "SUBMERSION":
            return (
                f"Sudden Blackout / Submersion — pose lost inside the pool for {f.pose_lost_seconds:.1f} s, "
                f"away from the pool edge."
            )
        if key == "RESTING":
            return (
                f"Resting / Treading Water — stationary ({fmt(f.moving)} torso-lengths/s), upright "
                f"{fmt(f.uprightness, '.0f', '°')}, head above water."
            )
        return (
            f"Normal Swimming — travel {fmt(f.travel_speed)} torso-lengths/s, torso "
            f"{fmt(f.torso_angle, '.0f', '°')}, stroke periodicity {fmt(f.arm_periodicity)}."
        )


# ═════════════════════════════════════════════════════════════════════════════
# 2. Alert confirmation and the alert record (was scene/hysteresis.py, verbatim)
# ═════════════════════════════════════════════════════════════════════════════
# [Phase 4 Step 2] the classes RAISE_MIN_POSITIVE_S applies to; never SUBMERSION, never ACTIVE (own rule)
RAISE_MIN_POSITIVE_SCENARIOS = ("PASSIVE_DROWNING", "FLOATABLE_FAILURE")


@dataclass
class AlertHysteresis:
    """
    An alert rises after `positive_windows` consecutive positive scoring
    rounds (and, for Active Drowning, `active_sustain_seconds` of continuous
    evidence); it clears after `negative_windows` negative rounds.
    """

    consecutive_positive: int = 0
    consecutive_negative: int = 0
    positive_since: float | None = None
    active: bool = False
    scenario: str | None = None
    started_at: float | None = None
    onset: float | None = None  # [2.19.0] when the evidence that raised the alert began

    def reset(self) -> None:
        self.consecutive_positive = self.consecutive_negative = 0
        self.positive_since = None
        self.active = False
        self.scenario = None
        self.started_at = None
        self.onset = None

    def update(self, res: BehaviorResult, params: dict[str, Any], now: float) -> bool:
        """Feed one scoring round; returns True on the rising edge."""
        positive = res.should_alert
        if positive:
            self.consecutive_positive += 1
            self.consecutive_negative = 0
            if self.positive_since is None:
                self.positive_since = now
        else:
            self.consecutive_negative += 1
            self.consecutive_positive = 0
            self.positive_since = None

        sustained = True
        if positive and res.scenario == "ACTIVE_DROWNING" and self.positive_since is not None:
            sustained = now - self.positive_since >= float(params.get("active_sustain_seconds", 0.0))
        elif (
            positive
            and RAISE_MIN_POSITIVE_S > 0
            and res.scenario in RAISE_MIN_POSITIVE_SCENARIOS
            and self.positive_since is not None
        ):
            # [Phase 4 Step 2] 2 rounds is ~0.13 s once at-risk tracks are scored every frame; PASSIVE / FLOATABLE
            # need RAISE_MIN_POSITIVE_S of unbroken positive rounds. SUBMERSION and the ghosts never wait.
            sustained = now - self.positive_since >= RAISE_MIN_POSITIVE_S

        rising = False
        if not self.active and sustained and self.consecutive_positive >= int(params["positive_windows"]):
            self.active, self.scenario, self.started_at = True, res.scenario, now
            self.onset = self.positive_since if self.positive_since is not None else now
            rising = True
        elif self.active and self.consecutive_negative >= int(params["negative_windows"]):
            self.reset()
        if self.active and positive:
            self.scenario = res.scenario
        return rising


@dataclass
class AlertFire:
    """An alert that should be persisted / broadcast now."""

    track_id: int
    result: BehaviorResult
    bbox: tuple[int, int, int, int]
    zone_name: str
    zone_type: str
    mode: str
    first: bool  # True on the rising edge, False for cooldown re-fires
    ghost: bool = False
    buffer: TrackBuffer | None = None
    # False when the zone's schedule window is closed. The alert is still
    # persisted (background telemetry) but must not sound the siren.
    siren: bool = True
    # [2.19.0] incident timing: pipeline time of this fire, when its evidence began, and how long the alert has been
    # active. `notify` False = an ONGOING update between two notifications (UI / alarm only: no outbox message, no
    # upload alert row, not counted by the eval harness).
    t: float | None = None
    onset_t: float | None = None
    ongoing_s: float = 0.0
    notify: bool = True
    # [2.20.0] the track's history when the alert rose (scene/temporal_gate.GateFeatures.as_dict), diagnostics
    gate: dict[str, float] | None = None


# ═════════════════════════════════════════════════════════════════════════════
# 3. Temporal risk gate (was scene/temporal_gate.py, verbatim; its module docstring stays in that shim)
# ═════════════════════════════════════════════════════════════════════════════
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class TemporalGateConfig:
    """Gate settings (thresholds.json TEMPORAL_GATE_*). A limit of 0 disables that check."""

    enabled: bool = False
    window_s: float = 4.0
    min_positive_frac: float = 0.0
    min_age_s: float = 0.0
    min_visible_frac: float = 0.0
    ghost_min_age_s: float = 0.0
    ghost_min_visible_frac: float = 0.0
    keep_s: float = 30.0  # history kept after a track was last seen (re-found ids, late ghost fires)

    @classmethod
    def from_constants(cls) -> TemporalGateConfig:
        return cls(
            enabled=bool(C.TEMPORAL_GATE),
            window_s=float(C.TEMPORAL_GATE_WINDOW_S),
            min_positive_frac=float(C.TEMPORAL_GATE_MIN_POSITIVE_FRAC),
            min_age_s=float(C.TEMPORAL_GATE_MIN_AGE_S),
            min_visible_frac=float(C.TEMPORAL_GATE_MIN_VISIBLE_FRAC),
            ghost_min_age_s=float(C.TEMPORAL_GATE_GHOST_MIN_AGE_S),
            ghost_min_visible_frac=float(C.TEMPORAL_GATE_GHOST_MIN_VISIBLE_FRAC),
        )


@dataclass(frozen=True)
class GateFeatures:
    """One track's history at a decision time (all times in pipeline seconds)."""

    age_s: float  # first seen -> now (or -> loss, for a ghost)
    visible_frac_life: float  # analysed frames with the track visible / analysed frames since its birth
    visible_frac_win: float  # the same over the last window_s
    positive_frac_win: float  # positive scoring rounds / scoring rounds in the window (1.0 when none yet)
    displacement_win: float  # net centre displacement over the window, in median box heights
    frames: int

    def as_dict(self) -> dict[str, float]:
        return {
            "age_s": round(self.age_s, 2),
            "vis_life": round(self.visible_frac_life, 3),
            "vis_win": round(self.visible_frac_win, 3),
            "pos_win": round(self.positive_frac_win, 3),
            "disp_win": round(self.displacement_win, 3),
            "frames": self.frames,
        }


@dataclass
class _History:
    born: float
    born_frame: int  # global analysed-frame counter at birth
    last_seen: float
    last_frame: int = 0
    seen: int = 0
    obs: deque[tuple[float, int, float, float, float]] = field(default_factory=deque)  # (t, frame_no, cx, cy, h)
    scores: deque[tuple[float, bool]] = field(default_factory=deque)


class TemporalRiskGate:
    """Rolling per-track history and the sustained-evidence decision (see module docstring)."""

    def __init__(self, cfg: TemporalGateConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else TemporalGateConfig.from_constants()
        self._h: dict[int, _History] = {}
        self._frames: deque[tuple[float, int]] = deque()  # (t, frame_no) of recent analysed frames
        self._frame_no = 0
        self._frozen: dict[int, GateFeatures] = {}  # features at loss, for submersion ghosts
        self.deferred = 0  # rising edges held back (diagnostics)

    # ── feeding ────────────────────────────────────────────────────────────
    def observe_frame(self, t: float, boxes: dict[int, Box]) -> None:
        """One analysed frame: the ids visible on it and their boxes."""
        self._frame_no += 1
        self._frames.append((t, self._frame_no))
        horizon = t - max(self.cfg.window_s, 1.0) - 1.0
        while self._frames and self._frames[0][0] < horizon:
            self._frames.popleft()
        for tid, b in boxes.items():
            h = self._h.get(tid)
            if h is None:
                h = self._h[tid] = _History(born=t, born_frame=self._frame_no, last_seen=t)
            h.last_seen, h.last_frame = t, self._frame_no
            h.seen += 1
            h.obs.append((t, self._frame_no, (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, max(1.0, b[3] - b[1])))
            while h.obs and h.obs[0][0] < t - self.cfg.window_s:
                h.obs.popleft()
        for tid in [k for k, h in self._h.items() if t - h.last_seen > self.cfg.keep_s]:
            del self._h[tid]
            self._frozen.pop(tid, None)

    def observe_score(self, track_id: int, t: float, positive: bool) -> None:
        h = self._h.get(track_id)
        if h is None:
            return
        h.scores.append((t, bool(positive)))
        while h.scores and h.scores[0][0] < t - self.cfg.window_s:
            h.scores.popleft()

    def freeze(self, track_id: int, now: float) -> None:
        """The track was lost: keep its history as of its last sighting for a later ghost decision."""
        h = self._h.get(track_id)
        if h is not None:
            self._frozen[track_id] = self._features(h, h.last_seen, h.last_frame)

    def forget(self, track_id: int) -> None:
        self._h.pop(track_id, None)
        self._frozen.pop(track_id, None)

    # ── deciding ───────────────────────────────────────────────────────────
    def features(self, track_id: int, now: float, ghost: bool = False) -> GateFeatures | None:
        if ghost and track_id in self._frozen:
            return self._frozen[track_id]
        h = self._h.get(track_id)
        if h is None:
            return None
        return self._features(h, h.last_seen, h.last_frame) if ghost else self._features(h, now, self._frame_no)

    def allow(self, track_id: int, now: float, ghost: bool = False) -> tuple[bool, GateFeatures | None]:
        """May this track's alert rise now? Unknown tracks are always allowed (never block on missing data)."""
        f = self.features(track_id, now, ghost)
        if not self.cfg.enabled or f is None:
            return True, f
        cfg = self.cfg
        if ghost:
            ok = _at_least(f.age_s, cfg.ghost_min_age_s) and _at_least(f.visible_frac_life, cfg.ghost_min_visible_frac)
        else:
            ok = (
                _at_least(f.age_s, cfg.min_age_s)
                and _at_least(f.visible_frac_win, cfg.min_visible_frac)
                and _at_least(f.positive_frac_win, cfg.min_positive_frac)
            )
        if not ok:
            self.deferred += 1
            logger.debug("temporal gate held track %s (ghost=%s): %s", track_id, ghost, f.as_dict())
        return ok, f

    def _features(self, h: _History, now: float, end_frame: int) -> GateFeatures:
        lo = now - self.cfg.window_s
        frames_life = max(1, end_frame - h.born_frame + 1)
        win_frames = [n for (t, n) in self._frames if lo <= t <= now and h.born_frame <= n <= end_frame]
        obs = [o for o in h.obs if lo <= o[0] <= now]
        scores = [p for (t, p) in h.scores if lo <= t <= now]
        disp = 0.0
        if len(obs) >= 2:
            hs = sorted(o[4] for o in obs)
            med_h = hs[len(hs) // 2]
            disp = math.hypot(obs[-1][2] - obs[0][2], obs[-1][3] - obs[0][3]) / med_h
        return GateFeatures(
            age_s=max(0.0, now - h.born),
            visible_frac_life=min(1.0, h.seen / frames_life),
            visible_frac_win=min(1.0, len(obs) / max(1, len(win_frames))),
            positive_frac_win=(sum(scores) / len(scores)) if scores else 1.0,
            displacement_win=disp,
            frames=h.seen,
        )


def _at_least(value: float, limit: float) -> bool:
    return limit <= 0 or value >= limit - 1e-9


# ═════════════════════════════════════════════════════════════════════════════
# 4. Re-raise damping ([Stage 4] the per-class alert label vote / relabel that lived here is removed: an alert reports
#    one "Drowning Alert" with a reason, vision/scene/alert_reason.py)
# ═════════════════════════════════════════════════════════════════════════════
# [Phase 4 Step 4] re-raise damping. Never SUBMERSION (live/policy.NEVER_MUTED; FAULT never comes through here).
NEVER_DAMPED = frozenset({"SUBMERSION"})


def reraise_damped(
    prev_scenario: str | None, prev_end: float, scenario: str, now: float, *, window_s: float | None = None
) -> bool:
    """True when a rising alert repeats, within ALERT_RERAISE_DAMP_S of the track's last alert clearing, a class of
    the same or lower severity: it is raised (UI, alarm) as a continuation of that alert, not notified anew. A worse
    class always notifies; SUBMERSION is never damped. 0 (the default) = never."""
    w = ALERT_RERAISE_DAMP_S if window_s is None else window_s
    if w <= 0 or prev_scenario is None or scenario in NEVER_DAMPED or now - prev_end >= w:
        return False
    return float(SCENARIO_SEVERITY.get(scenario, 0.0)) <= float(SCENARIO_SEVERITY.get(prev_scenario, 0.0))


# ═════════════════════════════════════════════════════════════════════════════
# 5. The temporal classifier: one scoring round, start to finish
# ═════════════════════════════════════════════════════════════════════════════
TEMPORAL_CLASSES = ("NORMAL", "RESTING", "ACTIVE", "PASSIVE", "FLOATABLE", "SUBMERSION")
_SCENARIO_CLASS = {
    "NORMAL_SWIMMING": "NORMAL",
    "NON_DROWNING": "NORMAL",
    "RESTING": "RESTING",
    "ACTIVE_DROWNING": "ACTIVE",
    "DISTRESS": "ACTIVE",
    "PASSIVE_DROWNING": "PASSIVE",
    "CRITICAL_BLACKOUT_DROWNING": "PASSIVE",
    "FLOATABLE_FAILURE": "FLOATABLE",
    "FLOATABLE_DETACHMENT_RISK": "FLOATABLE",
    "SUBMERSION": "SUBMERSION",
}


def temporal_class(scenario: str) -> str | None:
    """One of TEMPORAL_CLASSES; None for a person not scored (ANALYZING, BYSTANDER / DECK_PERSON)."""
    return _SCENARIO_CLASS.get(scenario)


@dataclass(frozen=True)
class TemporalState:
    """What a track shows now: class, the classifier's confidence and reason, and whether its alert is raised."""

    cls: str | None
    scenario: str
    confidence: float
    reason: str
    alerting: bool


class TemporalClassifier:
    """One scene's scoring rounds (order in the module docstring). Per-track state stays with the caller's track:
    its TrackBuffer, AlertHysteresis and label history; the gate keeps its own per-track history by track id."""

    def __init__(
        self,
        classifier: BehaviorClassifier | None = None,
        gate: TemporalRiskGate | None = None,
        post_classify: Callable[[Any, BehaviorResult], BehaviorResult] | None = None,
    ) -> None:
        self.classifier = classifier or BehaviorClassifier()
        self.gate = gate or TemporalRiskGate()
        self.post_classify = post_classify

    def classify(
        self,
        buf: TrackBuffer,
        *,
        track_id: int,
        now: float,
        mode: str,
        sensitivity_multiplier: float = 1.0,
        zone_type: str | None = None,
    ) -> BehaviorResult:
        """Steps 1-3: the round's result (what the track shows), fed to the gate's history."""
        res = self.classifier.classify(
            buf,
            category=SWIMMER,
            mode=mode,
            sensitivity_multiplier=sensitivity_multiplier,
            zone_type=zone_type,
        )
        if self.post_classify is not None:
            res = self.post_classify(buf, res)
        self.gate.observe_score(track_id, now, res.should_alert)
        return res

    def confirm(
        self,
        track_id: int,
        hyst: AlertHysteresis,
        buf: TrackBuffer,
        res: BehaviorResult,
        mode: str,
        now: float,
    ) -> tuple[bool, GateFeatures | None]:
        """Steps 4-5: (rising edge, gate features at the rising edge)."""
        rising = hyst.update(res, self.classifier.params(mode), now)
        gate_ok, gate_feats = self.gate.allow(track_id, now) if rising else (True, None)
        if not gate_ok:
            # [2.20.0] deferred, not cancelled: the counts stay, so the next positive round asks again
            hyst.active, hyst.scenario, hyst.started_at, hyst.onset = False, None, None, None
            rising = False
        return rising, gate_feats

    @staticmethod
    def report(res: BehaviorResult, buf: TrackBuffer, *, ghost: bool = False) -> BehaviorResult:
        """Step 6: the result an AlertFire carries - a copy collapsed to one "Drowning Alert" and its reason
        (scene/alert_reason.py). `res` itself, the hysteresis and the track's state are not touched."""
        from lifeguard.vision.scene.alert_reason import collapse

        out: BehaviorResult = collapse(res, ghost=ghost, buf=buf)
        return out

    @staticmethod
    def state(res: BehaviorResult, alerting: bool) -> TemporalState:
        return TemporalState(temporal_class(res.scenario), res.scenario, float(res.confidence), res.reason, alerting)


__all__ = [
    "BYSTANDER",
    "SWIMMER",
    "TEMPORAL_CLASSES",
    "AlertFire",
    "AlertHysteresis",
    "BehaviorClassifier",
    "BehaviorResult",
    "GateFeatures",
    "TemporalClassifier",
    "TemporalGateConfig",
    "TemporalRiskGate",
    "TemporalState",
    "_result",
    "clamp_multiplier",
    "effective_thresholds",
    "get_mode_params",
    "normalize_mode",
    "priority_for",
    "reraise_damped",
    "temporal_class",
    "thresholds_source",
    "timed_params",
]
