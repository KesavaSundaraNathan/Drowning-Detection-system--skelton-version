"""The alert outbox (plan Phase 5): what to send, written in the SAME transaction as what it announces.

Every FAULT and every AUDIBLE alert gets one `alert_outbox` row per configured channel, inserted by
the same DB transaction that inserts the alert, so an alert can never be recorded without its
delivery being queued (and vice versa). The dispatcher (alerts/dispatcher.py) delivers the rows.

Not queued, by design (they are in the alert list and the WebSocket feed):
  * alerts muted by a zone schedule (audible=False; never SUBMERSION or FAULT - live/policy.py);

Event ids are stable, so a receiver can de-duplicate the at-least-once deliveries:
  alert-<id> | fault-<id> | fault-<id>-cleared | alert-<id>-esc<n> | selftest-<YYYYmmddTHHMMSS>-<hex>
  | repeat-<oldest id>-<n>-<time> (one last-tier repeat for n alarms of a camera)
The process-wide channel list is set once at startup with `configure(settings)` (API,
CLI); an unconfigured process queues nothing (development/test default).
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from lifeguard.alerts.channels.base import Notification
from lifeguard.alerts.events import alert_severity
from lifeguard.core.clock import utcnow
from lifeguard.core.config import Settings
from lifeguard.db.models import Alert, AlertOutbox
from lifeguard.vision.scene import alert_reason

# kinds
ALERT, FAULT, FAULT_CLEARED, ESCALATION, SELF_TEST = "ALERT", "FAULT", "FAULT_CLEARED", "ESCALATION", "SELF_TEST"
# delivery order inside one channel: alarms first
KIND_ORDER = {FAULT: 0, ALERT: 0, ESCALATION: 1, FAULT_CLEARED: 2, SELF_TEST: 3}

_lock = threading.Lock()
_channels: tuple[str, ...] = ()


def configure(settings: Settings | None) -> None:
    global _channels
    with _lock:
        _channels = tuple(settings.channels_configured) if settings is not None else ()


def channels() -> tuple[str, ...]:
    with _lock:
        return _channels


def enqueue(
    db: Session,
    note: Notification,
    kind: str,
    *,
    alert_id: int | None = None,
    channel_list: Sequence[str] | None = None,
    now: datetime | None = None,
) -> int:
    """Insert one pending row per channel in the CALLER'S transaction; returns the number of rows."""
    now = now or utcnow()
    chans = channels() if channel_list is None else tuple(channel_list)
    payload = note.to_dict()
    for ch in chans:
        db.add(
            AlertOutbox(
                alert_id=alert_id,
                event_id=note.event_id,
                kind=kind,
                channel=ch,
                tier=note.tier,
                payload=payload,
                status="pending",
                attempts=0,
                next_attempt_at=now,
                created_at=now,
            )
        )
    return len(chans)


def note_for_alert(alert: Alert, kind: str, *, tier: int = 0, event_id: str | None = None) -> Notification:
    """The Notification announcing `alert` (ALERT, FAULT, FAULT_CLEARED or ESCALATION)."""
    is_fault = alert.scenario == "FAULT"
    where = f"camera {alert.camera_id}" if alert.camera_id else "system"
    if kind == FAULT_CLEARED:
        title = f"CLEARED {alert.fault_code}: {where} back to normal"
        severity, message = (
            "info",
            f"FAULT {alert.fault_code} on {alert.fault_source or where} has cleared. ({alert.reason})",
        )
    elif is_fault:
        title = f"{alert.fault_code or 'FAULT'}: {where} NOT MONITORING"
        severity, message = "critical", alert.reason
    else:
        # [Phase 6] owner decision 2026-11-15: every drowning alarm goes off-site at the most urgent channel level
        # (ntfy 5, SMTP X-Priority 1)
        # [Stage 4] one alert name, one short reason ("DROWNING ALERT - Active drowning in Deep end (camera 1)")
        zone = f" in {alert.zone_name}" if alert.zone_name else ""
        title = f"DROWNING ALERT - {alert_reason.reason_text(alert.scenario)}{zone} ({where})"
        severity = "critical"
        message = (
            f"{alert_reason.clean_reason(alert.reason, alert.scenario)} - risk {alert.risk_score}, "
            f"confidence {alert.confidence:.2f}"
        )
        if alert.incident_id is not None and alert.incident_id != alert.id:
            # [Phase 11] linked, never suppressed: the receiver sees it is the same person as an earlier alarm
            title = f"{title} - same person as #{alert.incident_id}"
            message = f"Same person as alert #{alert.incident_id}. {message}"
    if kind == ESCALATION:
        title = f"ESCALATED (unacknowledged {alert.escalation_level}x) {title}"
        severity = "critical"
    detail: dict[str, Any] = {
        "alert_id": alert.id,
        "scenario": alert_reason.api_scenario(alert.scenario),
        "alert_reason": alert_reason.reason_text(alert.scenario),
        "priority": alert.priority,
        "severity": alert_severity(alert),
        "camera_id": alert.camera_id,
        "zone": alert.zone_name,
        "status": alert.status,
        "global_id": alert.global_id,
        "incident_id": alert.incident_id,
    }
    if is_fault:
        detail["fault_code"] = alert.fault_code
    return Notification(
        kind=kind,
        code=(alert.fault_code or "FAULT") if is_fault else alert_reason.API_SCENARIO,
        title=title[:180],
        message=message[:2000],
        severity=severity,
        source=alert.fault_source or where,
        at=(alert.cleared_at if kind == FAULT_CLEARED and alert.cleared_at else alert.created_at).isoformat(
            timespec="seconds"
        ),
        detail=detail,
        event_id=event_id or default_event_id(alert, kind),
        tier=tier,
        alert_id=alert.id,
    )


REPEAT_LIST_MAX = 20  # alarms spelled out in a coalesced repeat's message; the rest are counted
REPEAT_IDS_MAX = 500  # alarm ids carried in its detail


def note_for_repeat(alerts: Sequence[Alert], *, tier: int, now: datetime) -> Notification:
    """[Phase 6] ONE last-tier repeat for several unacknowledged alarms of one camera (or of the system).

    Escalation repeats every pending alarm; this note announces all of them at once so N unacknowledged alarms cost
    one outbox row per channel per repeat interval instead of N (soak R5: 4 021 unacknowledged alarms -> 1 400 rows a
    minute). A single alarm keeps its own note_for_alert(ESCALATION) - callers pass two or more.
    """
    ordered = sorted(alerts, key=lambda a: a.id)
    first = ordered[0]
    where = f"camera {first.camera_id}" if first.camera_id else "system"
    worst = note_for_alert(max(ordered, key=lambda a: (a.scenario == "FAULT", a.risk_score or 0, -a.id)), ALERT).title
    lines = [
        f"#{a.id} {(a.fault_code or 'FAULT') if a.scenario == 'FAULT' else 'Drowning Alert'}"
        f"{' in ' + a.zone_name if a.zone_name else ''} since {a.created_at.isoformat(timespec='seconds')}"
        f" (escalated {a.escalation_level}x)"
        for a in ordered[:REPEAT_LIST_MAX]
    ]
    if len(ordered) > REPEAT_LIST_MAX:
        lines.append(f"... and {len(ordered) - REPEAT_LIST_MAX} more")
    n = len(ordered)
    return Notification(
        kind=ESCALATION,
        code="UNACKNOWLEDGED",
        title=f"ESCALATED: {n} UNACKNOWLEDGED alarms ({where}) - worst: {worst}"[:180],
        message=f"{n} alarms on {where} are still not acknowledged:\n" + "\n".join(lines)[:1900],
        severity="critical",
        source=first.fault_source or where,
        at=now.isoformat(timespec="seconds"),
        detail={
            "camera_id": first.camera_id,
            "count": n,
            "alert_ids": [a.id for a in ordered[:REPEAT_IDS_MAX]],
            "oldest_alert_id": first.id,
        },
        event_id=f"repeat-{first.id}-{n}-{now.strftime('%Y%m%dT%H%M%S%f')}",
        tier=tier,
        alert_id=first.id,
    )


def default_event_id(alert: Alert, kind: str) -> str:
    base = f"{'fault' if alert.scenario == 'FAULT' else 'alert'}-{alert.id}"
    if kind == FAULT_CLEARED:
        return f"{base}-cleared"
    if kind == ESCALATION:
        return f"{base}-esc{alert.escalation_level}"
    return base


def alert_event(alert: Alert, event_type: str) -> dict[str, Any]:
    """Compact JSON event for the WebSocket feed (never contains camera sources)."""
    return {
        "type": event_type,
        "alert": {
            "id": alert.id,
            "created_at": alert.created_at.isoformat(timespec="seconds"),
            "camera_id": alert.camera_id,
            "scenario": alert_reason.api_scenario(alert.scenario),
            "title": alert_reason.title_of(alert.scenario),
            "alert_reason": alert_reason.reason_text(alert.scenario),
            "priority": alert.priority,
            "severity": alert_severity(alert),
            "risk_score": alert.risk_score,
            "zone_name": alert.zone_name,
            "reason": alert_reason.clean_reason(alert.reason, alert.scenario),
            "status": alert.status,
            "audible": alert.audible,
            "fault_code": alert.fault_code,
            "cleared_at": alert.cleared_at.isoformat(timespec="seconds") if alert.cleared_at else None,
            "escalation_level": alert.escalation_level,
            "global_id": alert.global_id,
            "incident_id": alert.incident_id,
            "event_id": alert.event_id,
            "snapshot": alert.snapshot_path is not None,
            "recording": alert.recording_path is not None,
        },
    }
