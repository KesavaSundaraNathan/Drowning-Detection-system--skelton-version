"""Re-export shim: the behaviour classifier moved to behaviour/temporal.py (Phase 4 Step 1, verbatim)."""

from lifeguard.vision.behaviour.temporal import (
    BYSTANDER,
    SWIMMER,
    BehaviorClassifier,
    BehaviorResult,
    _result,
    clamp_multiplier,
    effective_thresholds,
    get_mode_params,
    normalize_mode,
    priority_for,
    thresholds_source,
    timed_params,
)

__all__ = [
    "BYSTANDER",
    "SWIMMER",
    "BehaviorClassifier",
    "BehaviorResult",
    "_result",
    "clamp_multiplier",
    "effective_thresholds",
    "get_mode_params",
    "normalize_mode",
    "priority_for",
    "thresholds_source",
    "timed_params",
]
