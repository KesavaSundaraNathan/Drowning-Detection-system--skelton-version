"""Mode parameters and sensitivity arithmetic (zone multiplier, timed limits, risk bands).

[v3] Split out of the v2 baseline detection/classifier.py; definitions verbatim except
get_mode_params / thresholds_source (no calibration override file - see get_mode_params).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from lifeguard.vision.constants import (
    BEHAVIOR_MODE_PARAMS,
    RISK_RANGES,
    THRESHOLD_CEILING,
    THRESHOLD_FLOOR,
    ZONE_SENSITIVITY_MAX,
    ZONE_SENSITIVITY_MIN,
)

logger = logging.getLogger(__name__)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def normalize_mode(mode: str | None) -> str:
    return mode if mode in BEHAVIOR_MODE_PARAMS else "Medium"


def clamp_multiplier(m: float | None) -> float:
    try:
        v = float(m) if m is not None else 1.0
    except (TypeError, ValueError):
        v = 1.0
    return max(ZONE_SENSITIVITY_MIN, min(ZONE_SENSITIVITY_MAX, v))


def priority_for(risk_score: int) -> tuple[str, str]:
    """int 0–100 → (alert_priority, risk_status)."""
    for band in RISK_RANGES:
        if band["min"] <= risk_score <= band["max"]:
            return band["alert_level"], band["status"]
    return "None", "Normal"


def timed_params(params: dict[str, Any], multiplier: float) -> dict[str, Any]:
    """Divide timed-scenario limits by the zone multiplier (bounded 0.5×–2×)."""
    out = dict(params)
    factor = max(0.5, min(2.0, 1.0 / max(multiplier, 1e-6)))
    for key in ("passive_seconds", "submersion_seconds", "face_down_seconds"):
        out[key] = float(params[key]) * factor
    return out


# Scenarios whose evidence is a graded 0–1 score: the zone multiplier moves
# their THRESHOLD. The rest are timed (their evidence is a step function of a
# duration), so the same multiplier moves their TIME LIMIT in `timed_params`
# instead — scaling both would penalise a low-sensitivity zone twice and could
# push a timed scenario out of reach, which the spec forbids.
GRADED_SCENARIOS = ("ACTIVE_DROWNING", "FLOATABLE_FAILURE")


def effective_thresholds(params: dict[str, Any], multiplier: float) -> dict[str, float]:
    """
    Phase 1.2 spec:

        Effective Threshold = Base Threshold / sensitivity_multiplier

    A deep-end zone (multiplier > 1) therefore needs LESS evidence and alerts
    sooner than a shallow-end or splash zone (multiplier < 1), which needs more.
    Timed scenarios (Passive, Submersion) carry the same sensitivity through
    their time limits via `timed_params`, so every zone's multiplier moves both
    halves of the decision — but a low multiplier can only ever DELAY a timed
    alert, never disable it.
    """
    m = clamp_multiplier(multiplier)
    out: dict[str, float] = {}
    for k, v in params["thresholds"].items():
        if k in GRADED_SCENARIOS:
            out[k] = float(min(THRESHOLD_CEILING, max(THRESHOLD_FLOOR, float(v) / m)))
        else:
            out[k] = float(v)
    return out


def get_mode_params(mode: str) -> dict[str, Any]:
    """Parameters of one monitoring mode, from the versioned thresholds file.

    v2 also deep-merged a calibration override file (models/behavior_thresholds.json); v3 has no
    second source of thresholds - a calibrated change is a new thresholds.json version.
    """
    base = BEHAVIOR_MODE_PARAMS.get(mode) or BEHAVIOR_MODE_PARAMS["Medium"]
    return copy.deepcopy(base)


def thresholds_source() -> dict[str, Any]:
    """v2's shape (evaluation reports compare it): there is no override file in v3."""
    return {"calibrated": False, "meta": {}}
