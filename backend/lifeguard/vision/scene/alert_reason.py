"""[Stage 4, thresholds 2.37.0] One alert: "Drowning Alert", with one short reason.

The behaviour rules (ACTIVE / PASSIVE / FLOATABLE evidence, the submersion monitor) still DETECT exactly as before:
when an alert fires, how long it lasts and whom it is on are unchanged. Only what the alert REPORTS is collapsed:

  SUBMERSION  "Submersion / sudden blackout"  a submersion ghost (a tracked swimmer who vanished in the water and did
                                              not come back up) or the SUBMERSION rule (pose lost inside the pool)
  ACTIVE      "Active drowning"               every other detection whose confidence is at least
                                              ALERT_ACTIVE_MIN_CONFIDENCE
  DROWNING    "Drowning"                      the same detection with a lower confidence: generic, never dropped
  FLOATABLE   "Floatable failure"             only with ALERT_FLOATABLE_REASON on (off: there is no float detector) and
                                              a float seen near the person; otherwise as ACTIVE / DROWNING

The code is what an alert row stores in `scenario` (String(32)). Rows written before Stage 4 hold the old type names;
`reason_code` maps them on read (old and new are accepted, nothing is rewritten). FAULT is not a drowning alert and
passes through unchanged.
"""

from __future__ import annotations

import copy
from typing import Any

from lifeguard.vision.constants import ALERT_ACTIVE_MIN_CONFIDENCE, ALERT_FLOATABLE_REASON

DROWNING_ALERT = "Drowning Alert"
API_SCENARIO = "DROWNING"  # what the API reports as an alert's scenario (FAULT stays FAULT)

SUBMERSION = "SUBMERSION"
ACTIVE = "ACTIVE"
DROWNING = "DROWNING"
FLOATABLE = "FLOATABLE"
FAULT = "FAULT"

REASON_TEXT = {
    SUBMERSION: "Submersion / sudden blackout",
    ACTIVE: "Active drowning",
    DROWNING: "Drowning",
    FLOATABLE: "Floatable failure",
}
REASON_CODES = frozenset(REASON_TEXT)
# the first line of a snapshot's recommended action (alerts/snapshots.py)
REASON_ACTIONS = {
    SUBMERSION: "Swimmer went under and has not resurfaced - dive to the last seen position now.",
    ACTIVE: "Swimmer in distress - reach or throw a flotation aid, enter the water if needed.",
    DROWNING: "Possible drowning - check this swimmer now.",
    FLOATABLE: "Swimmer separated from their float - retrieve them before they submerge.",
}

# rows and frozen files written before Stage 4 (accepted on read, never rewritten)
LEGACY = {
    "SUBMERSION": SUBMERSION,
    "CRITICAL_BLACKOUT_DROWNING": SUBMERSION,
    "ACTIVE_DROWNING": ACTIVE,
    "DISTRESS": ACTIVE,
    "PASSIVE_DROWNING": DROWNING,
    "FLOATABLE_FAILURE": DROWNING,
    "FLOATABLE_DETACHMENT_RISK": DROWNING,
}
# the human labels old reason texts start with ("Passive Drowning: Passive Drowning — face-down ...")
_LEGACY_LABELS = (
    "Sudden Blackout / Submersion",
    "Active Drowning",
    "Passive Drowning",
    "Floatable Failure",
    "Distress",
)


def reason_code(stored: str | None) -> str:
    """The reason code of a stored scenario value, old or new. FAULT stays FAULT; anything unknown is DROWNING."""
    if stored is None:
        return DROWNING
    s = str(stored)
    if s == FAULT:
        return FAULT
    if s in REASON_CODES:
        return s
    return LEGACY.get(s, DROWNING)


def reason_text(stored: str | None) -> str | None:
    """'Active drowning' etc. for a drowning alert; None for a FAULT."""
    code = reason_code(stored)
    return None if code == FAULT else REASON_TEXT[code]


def is_drowning_alert(stored: str | None) -> bool:
    return reason_code(stored) != FAULT


def api_scenario(stored: str | None) -> str:
    """What the API, WebSocket, exports and messages report as an alert's scenario: DROWNING or FAULT."""
    return FAULT if reason_code(stored) == FAULT else API_SCENARIO


def title_of(stored: str | None) -> str | None:
    """'Drowning Alert' for a drowning alert, None for a FAULT (its fault code names it)."""
    return None if reason_code(stored) == FAULT else DROWNING_ALERT


def clean_reason(stored_reason: str | None, stored_scenario: str | None) -> str:
    """An alert's reason text for output: an old row's leading type label ("Passive Drowning: Passive Drowning — ")
    becomes the reason text; new rows already start with it. FAULT reasons are untouched."""
    text = stored_reason or ""
    code = reason_code(stored_scenario)
    if code == FAULT:
        return text
    rest = text
    changed = True
    while changed:  # strip any number of leading "<label>: " / "<label> — " heads
        changed = False
        for lab in _LEGACY_LABELS:
            for sep in (": ", " — ", " - "):
                if rest.startswith(lab + sep):
                    rest = rest[len(lab) + len(sep) :]
                    changed = True
    if rest == text:
        return text
    return f"{REASON_TEXT[code]} — {rest}" if rest else REASON_TEXT[code]


def _float_near(buf: Any) -> bool:
    return bool(
        getattr(buf, "floatable_seen", False)
        or getattr(buf, "decoupled_at", None) is not None
        or getattr(buf, "detach_t", None) is not None
    )


def code_for(res: Any, *, ghost: bool = False, buf: Any = None) -> str:
    """The reason an alert raised from the classifier result `res` reports."""
    scenario = str(getattr(res, "scenario", ""))
    if ghost or LEGACY.get(scenario) == SUBMERSION:
        return SUBMERSION
    if scenario in REASON_CODES:  # already collapsed (a reason code): kept as it is
        return scenario
    if ALERT_FLOATABLE_REASON and scenario == "FLOATABLE_FAILURE" and buf is not None and _float_near(buf):
        return FLOATABLE
    return ACTIVE if float(getattr(res, "confidence", 0.0)) >= ALERT_ACTIVE_MIN_CONFIDENCE else DROWNING


def _cues(reason: str) -> str:
    """The classifier's reason without its leading type label ('Passive Drowning — face-down' -> 'face-down')."""
    for lab in _LEGACY_LABELS:
        for sep in (" — ", ": "):
            if reason.startswith(lab + sep):
                return reason[len(lab) + len(sep) :]
    return reason


def collapse(res: Any, *, ghost: bool = False, buf: Any = None) -> Any:
    """A copy of `res` as the alert reports it: scenario = the reason code, label = 'Drowning Alert', reason =
    '<reason text> — <the classifier's cues>'. `res` itself (the track's state, the hysteresis) is not touched."""
    code = code_for(res, ghost=ghost, buf=buf)
    out = copy.copy(res)
    out.scenario = code
    out.label = DROWNING_ALERT
    cues = _cues(str(getattr(res, "reason", "") or ""))
    out.reason = f"{REASON_TEXT[code]} — {cues}" if cues else REASON_TEXT[code]
    return out
