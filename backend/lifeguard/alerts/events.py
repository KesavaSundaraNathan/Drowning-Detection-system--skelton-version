"""[Phase 5] Events: one per person and incident on ONE camera (cameras are independent).

A behaviour alert joins the open event of its camera and track when that event was last updated less than
EVENT_IDLE_CLOSE_S ago; otherwise it opens a new one. An ONGOING update of a raised alert (writer.record_ongoing)
keeps its event alive. An event closes once it has been quiet that long AND none of its alerts is still pending
(`close_idle`, run on the live heartbeat): an unacknowledged alarm never leaves the AlertCenter. Alerts are never
suppressed or merged: every fire stays its own row and sounds as before; the event only groups them (one
AlertCenter card, one acknowledge, the alerts-per-event metric).

- severity: [Stage 4] critical for every Drowning Alert (one alarm tier on screen, sounded and sent off-site; the
  per-type tiers are removed, alarms are ordered by risk). Events stored earlier may hold high / caution.
- scenario: the reason code of the event's latest alert (vision/scene/alert_reason.py), reported as "Drowning Alert"
  with its reason text. [2.38.0] With ALERT_EVENT_CATEGORY_LOCK on it is the FIRST alert's reason, fixed until the
  event closes: a later alert of the event (a submersion ghost after an active struggle, a re-notification whose
  confidence crossed ALERT_ACTIVE_MIN_CONFIDENCE) stores the event's category and keeps what it observed in its reason
  text, so one incident never flips category from alert to alert.
- an alert that joins an ACKNOWLEDGED event re-opens it: an incident that continues must not go quiet.
- FAULT rows never belong to an event.
- WebSocket: event.opened / event.updated / event.closed (app/ws.FORWARDED).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from lifeguard.db.models import Alert, Event
from lifeguard.vision import constants as vision_constants
from lifeguard.vision.scene import alert_reason

EVENT_IDLE_CLOSE_S = 30.0
_RANK = {"critical": 3, "high": 2, "caution": 1}  # events stored before Stage 4 may hold high / caution


def severity_of(scenario: str, risk_score: int | None = None) -> str:
    """[Stage 4] Every Drowning Alert is critical (one alarm tier everywhere; the per-type tiers are removed - lists
    order alarms by risk score). `risk_score` is kept for the callers' signature."""
    return "critical"


def alert_severity(alert: Alert) -> str | None:
    """The tier of an alert row (None for a FAULT: a technical alarm, not a drowning tier)."""
    if alert.scenario == "FAULT":
        return None
    return severity_of(alert.scenario, int(alert.risk_score) if alert.risk_score is not None else None)


def attach(db: Session, alert: Alert, now: datetime, *, idle_s: float = EVENT_IDLE_CLOSE_S) -> tuple[Event, bool]:
    """Put `alert` (flushed, not a FAULT) into its camera + track's open event, or open one. Returns (event, opened)."""
    sev = alert_severity(alert) or "caution"
    ev = db.execute(
        select(Event)
        .where(
            Event.camera_id == alert.camera_id,
            Event.track_id == alert.track_id,
            Event.closed_at.is_(None),
            Event.status != "resolved",
            Event.updated_at >= now - timedelta(seconds=idle_s),
        )
        .order_by(Event.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    opened = ev is None
    if ev is None:
        ev = Event(
            camera_id=alert.camera_id,
            track_id=alert.track_id,
            opened_at=now,
            updated_at=now,
            severity=sev,
            scenario=alert.scenario,
            status="open",
            alert_count=0,
            first_alert_id=alert.id,
        )
        db.add(ev)
        db.flush()
    elif lock_category(ev, alert):
        pass  # [2.38.0] the event keeps its first alert's category; the alert now reports it
    else:
        # [Stage 4] the event shows the reason of its latest alert (an active struggle that turns into a submersion
        # reads "Submersion / sudden blackout"); an event stored before Stage 4 with a lower tier is lifted
        ev.scenario = alert.scenario
        if _RANK[sev] > _RANK.get(ev.severity, 0):
            ev.severity = sev
    ev.alert_count = int(ev.alert_count or 0) + 1
    ev.last_alert_id = alert.id
    ev.updated_at = now
    if ev.status == "acknowledged":
        ev.status = "open"  # the incident continues: it alarms again
    alert.event_id = ev.id
    return ev, opened


def lock_category(ev: Event, alert: Alert, *, on: bool | None = None) -> bool:
    """[2.38.0] ALERT_EVENT_CATEGORY_LOCK: `alert` joins the open event `ev`, whose category was fixed by its first
    alert. The alert takes the event's category (its `scenario`); when that differs from what it observed, its reason
    keeps the observation ("[observed: Submersion / sudden blackout]"). Returns False (nothing changed) when the lock
    is off, or for an event stored without a drowning category (a FAULT, or an empty value)."""
    if not (vision_constants.ALERT_EVENT_CATEGORY_LOCK if on is None else on):
        return False
    if not ev.scenario or not alert_reason.is_drowning_alert(ev.scenario):
        return False
    if ev.severity in _RANK and _RANK[severity_of(alert.scenario)] > _RANK[ev.severity]:
        ev.severity = severity_of(alert.scenario)  # an event stored before Stage 4 with a lower tier is lifted
    fixed = alert_reason.reason_code(ev.scenario)
    seen = alert_reason.reason_code(alert.scenario)
    if seen != fixed:
        note = f" [observed: {alert_reason.REASON_TEXT[seen]}]"
        alert.reason = ((alert.reason or "") + note)[:2000]
        alert.scenario = fixed
    return True


def touch(db: Session, event_id: int | None, now: datetime, alert: Alert | None = None) -> tuple[Event | None, bool]:
    """An ongoing alert of the event is still firing: the event stays open (updated_at = now). [Phase 6] With the
    alert, a risk rise that lifts its tier lifts the event. Returns (event, whether its severity rose)."""
    ev = db.get(Event, event_id) if event_id is not None else None
    lifted = False
    if ev is not None and ev.closed_at is None:
        ev.updated_at = now
        sev = alert_severity(alert) if alert is not None else None
        if alert is not None and sev is not None and _RANK[sev] > _RANK.get(ev.severity, 0):
            ev.severity, ev.scenario, lifted = sev, alert.scenario, True
    return ev, lifted


def close_idle(db: Session, now: datetime, *, idle_s: float = EVENT_IDLE_CLOSE_S) -> list[Event]:
    """Close every event quiet for `idle_s` with no pending alert left. Returns the events closed now."""
    pending = exists().where(Alert.event_id == Event.id, Alert.status == "pending")
    rows = list(
        db.execute(
            select(Event).where(Event.closed_at.is_(None), Event.updated_at < now - timedelta(seconds=idle_s), ~pending)
        ).scalars()
    )
    for ev in rows:
        ev.closed_at = now
    return rows


def event_payload(ev: Event, event_type: str) -> dict[str, Any]:
    """Compact JSON event for the WebSocket feed."""
    return {"type": event_type, "event": event_dict(ev)}


def event_dict(ev: Event) -> dict[str, Any]:
    def iso(d: datetime | None) -> str | None:
        return d.isoformat(timespec="seconds") if d is not None else None

    return {
        "id": ev.id,
        "camera_id": ev.camera_id,
        "track_id": ev.track_id,
        "opened_at": iso(ev.opened_at),
        "updated_at": iso(ev.updated_at),
        "closed_at": iso(ev.closed_at),
        "severity": ev.severity,
        "scenario": alert_reason.api_scenario(ev.scenario),
        "title": alert_reason.title_of(ev.scenario),
        "alert_reason": alert_reason.reason_text(ev.scenario),
        "status": ev.status,
        "alert_count": ev.alert_count,
        "first_alert_id": ev.first_alert_id,
        "last_alert_id": ev.last_alert_id,
    }
