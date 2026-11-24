"""BehaviorResult: one classification of one track.

[v3] Split out of the v2 baseline detection/classifier.py; definitions verbatim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from lifeguard.vision.behaviour.engine import clamp01
from lifeguard.vision.behaviour.params import priority_for
from lifeguard.vision.constants import ALERT_MIN_RISK, ALERT_SCENARIOS, DROWNING_TYPES

logger = logging.getLogger(__name__)


SWIMMER = "SWIMMER"

BYSTANDER = "BYSTANDER"


# ═════════════════════════════════════════════════════════════════════════════
# Result
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class BehaviorResult:
    track_id: int
    category: str  # SWIMMER | BYSTANDER
    scenario: str  # key of DROWNING_TYPES
    label: str  # human label
    is_drowning: bool
    risk: float  # 0.0–1.0 after zone sensitivity
    raw_risk: float  # 0.0–1.0 at multiplier 1.0
    confidence: float  # 0.0–1.0
    reason: str  # the specific triggering reason
    mode: str
    sensitivity_multiplier: float = 1.0
    evidence: dict[str, float] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    # Phase 2: recognised normal activity (SWIMMING, TREADING, STANDING_WATER,
    # RESTING_EDGE or UNKNOWN) and whether pose cues were trusted.
    activity: str = "UNKNOWN"
    activity_conf: float = 0.0
    pose_reliable: bool = False
    cues: dict[str, float] = field(default_factory=dict)

    @property
    def risk_score(self) -> int:
        return int(round(clamp01(self.risk) * 100))

    @property
    def raw_risk_score(self) -> int:
        return int(round(clamp01(self.raw_risk) * 100))

    @property
    def priority(self) -> str:
        return priority_for(self.risk_score)[0]

    @property
    def risk_status(self) -> str:
        return priority_for(self.risk_score)[1]

    @property
    def detection_result(self) -> str:
        return "Drowning" if self.is_drowning else "Non-Drowning"

    @property
    def should_alert(self) -> bool:
        return self.is_drowning and self.risk >= ALERT_MIN_RISK

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "category": self.category,
            "scenario": self.scenario,
            "label": self.label,
            "drowning_type": self.label,
            "detection_result": self.detection_result,
            "is_drowning": self.is_drowning,
            "risk": round(self.risk, 4),
            "raw_risk": round(self.raw_risk, 4),
            "risk_score": self.risk_score,
            "raw_risk_score": self.raw_risk_score,
            "priority": self.priority,
            "risk_status": self.risk_status,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "mode": self.mode,
            "sensitivity_multiplier": self.sensitivity_multiplier,
            "evidence": self.evidence,
            "activity": self.activity,
            "activity_conf": round(self.activity_conf, 3),
            "pose_reliable": self.pose_reliable,
            "cues": self.cues,
        }


def _result(
    track_id: int,
    category: str,
    scenario: str,
    *,
    risk: float = 0.0,
    raw_risk: float = 0.0,
    confidence: float = 0.5,
    reason: str = "",
    mode: str = "Medium",
    multiplier: float = 1.0,
    evidence: dict[str, float] | None = None,
    features: dict[str, Any] | None = None,
) -> BehaviorResult:
    return BehaviorResult(
        track_id=track_id,
        category=category,
        scenario=scenario,
        label=DROWNING_TYPES[scenario],
        is_drowning=scenario in ALERT_SCENARIOS,
        risk=clamp01(risk),
        raw_risk=clamp01(raw_risk),
        confidence=float(min(0.97, max(0.3, confidence))),
        reason=reason,
        mode=mode,
        sensitivity_multiplier=multiplier,
        evidence=evidence or {},
        features=features or {},
    )
