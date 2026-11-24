"""Behaviour alerts from the live vision pipeline -> `alerts` rows.

One transaction per alert: the row (with its `audible` flag), its audit event and - for an audible
alert - its outbox rows (alerts/outbox.py). Then "alert.created" is published (WebSocket, dispatcher
wake-up). Snapshot and recording files are written afterwards (alerts/media.py) and attached with
`attach_media`, so delivery never waits for disk I/O.

[Phase 5] Cameras are independent: alerts are no longer linked across cameras (the Phase 11 pool-wide
global id / incident linking is removed; the `global_id` / `incident_id` columns stay, NULL for new rows).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from lifeguard.alerts import events as event_engine
from lifeguard.alerts import outbox
from lifeguard.core import audit, events
from lifeguard.core.clock import utcnow
from lifeguard.db.models import Alert
from lifeguard.db.writer import DBWriter
from lifeguard.vision.scene import alert_reason
from lifeguard.vision.scene.hysteresis import AlertFire

logger = logging.getLogger(__name__)


def record_fire(
    writer: DBWriter,
    fire: AlertFire,
    *,
    camera_id: str | None = None,
    audible: bool = True,
    muted_reason: str | None = None,
) -> int:
    res = fire.result
    if res.scenario != alert_reason.FAULT and res.label != alert_reason.DROWNING_ALERT:
        # [Stage 4] the live scene already reports a reason code; anything else (tools, tests) is collapsed here
        res = alert_reason.collapse(res, ghost=bool(fire.ghost), buf=getattr(fire, "buffer", None))
    now = utcnow()
    live = camera_id is not None
    audible = audible and live  # a row without a camera (tests, tools) never sounds or escalates
    # why a live alert does not sound: a closed schedule window
    muted = f" [{muted_reason or 'MUTED by schedule'}]" if live and not audible else ""
    reason = f"{res.label}: {res.reason}{_onset_note(fire, now)}{muted}"[:2000]

    def work(db: Session) -> tuple[int, list[dict[str, object]]]:
        row = Alert(
            created_at=now,
            camera_id=camera_id,
            track_id=int(fire.track_id),
            scenario=str(res.scenario)[:32],
            priority=str(res.priority)[:16],
            risk_score=int(res.risk_score),
            confidence=float(res.confidence),
            zone_name=(fire.zone_name or None),
            reason=reason,
            status="pending",
            audible=audible,
            escalation_level=0,
        )
        db.add(row)
        db.flush()
        published: list[dict[str, object]] = []
        if live and row.scenario != "FAULT":  # [Phase 5] the person's event on this camera (alerts/events.py)
            ev, opened = event_engine.attach(db, row, now)
            published.append(event_engine.event_payload(ev, "event.opened" if opened else "event.updated"))
        audit.record(
            db,
            action="alert.raised",
            actor=None,
            actor_username="system",
            ip="local",
            target_type="alert",
            target_id=row.id,
            details={
                "scenario": res.scenario,
                "camera_id": camera_id,
                "track_id": int(fire.track_id),
                "audible": audible,
                "first": bool(fire.first),
                "ghost": bool(fire.ghost),
            },
        )
        if audible and getattr(fire, "notify", True):
            outbox.enqueue(db, outbox.note_for_alert(row, outbox.ALERT), outbox.ALERT, alert_id=row.id, now=now)
        return int(row.id), [outbox.alert_event(row, "alert.created"), *published]

    alert_id, published = writer.run(work)
    log = logger.warning if audible else logger.info
    log(
        "ALERT %d %s (%s) camera=%s audible=%s",
        alert_id,
        res.scenario,
        res.priority,
        camera_id,
        audible,
    )
    for event in published:
        events.publish(event)
    return alert_id


def close_idle_events(writer: DBWriter) -> int:
    """[Phase 5] Close events quiet for EVENT_IDLE_CLOSE_S and announce them (event.closed). Returns how many."""
    now = utcnow()

    def work(db: Session) -> list[dict[str, object]]:
        return [event_engine.event_payload(ev, "event.closed") for ev in event_engine.close_idle(db, now)]

    closed = writer.run(work)
    for event in closed:
        events.publish(event)
    return len(closed)


def _onset_note(fire: AlertFire, now: datetime) -> str:
    """[2.19.0] ' Began HH:MM:SS UTC (N s before this alert).' - when the evidence that raised the alert started."""
    t, onset = getattr(fire, "t", None), getattr(fire, "onset_t", None)
    if t is None or onset is None or t < onset:
        return ""
    lead = float(t) - float(onset)
    began = now - timedelta(seconds=lead)
    return f" Began {began.strftime('%H:%M:%S')} UTC ({lead:.1f} s before this alert)."


_ONGOING_RE = re.compile(r" \[ONGOING \d+ s\]$")
ONGOING_MAX_AGE_S = 900.0


def record_ongoing(writer: DBWriter, fire: AlertFire, *, camera_id: str) -> int | None:
    """[2.19.0] The drowning is still going on: mark it on the person's unacknowledged alert of this camera
    ('[ONGOING 8 s]', highest risk so far) and tell the UI (alert.updated). Returns the alert id, or None when there
    is no such alert any more (acknowledged / resolved) - the caller then raises a NEW alert: an acknowledged
    incident that continues must not go quiet."""
    res = fire.result
    now = utcnow()

    def work(db: Session) -> tuple[int | None, tuple[dict[str, object], dict[str, Any] | None] | None]:
        row = db.execute(
            select(Alert)
            .where(
                Alert.camera_id == camera_id,
                Alert.track_id == int(fire.track_id),
                Alert.status == "pending",
                Alert.created_at >= now - timedelta(seconds=ONGOING_MAX_AGE_S),
            )
            .order_by(Alert.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        if row is None:
            return None, None
        base = _ONGOING_RE.sub("", row.reason or "")
        row.reason = f"{base} [ONGOING {round(float(getattr(fire, 'ongoing_s', 0.0)))} s]"[:2000]
        row.risk_score = max(int(row.risk_score or 0), int(res.risk_score))
        # [Phase 5] the drowning goes on: its event stays open; [Phase 6] a risk rise can lift its tier
        ev, lifted = event_engine.touch(db, row.event_id, now, row)
        lift = event_engine.event_payload(ev, "event.updated") if (ev is not None and lifted) else None
        return int(row.id), (outbox.alert_event(row, "alert.updated"), lift)

    alert_id, payloads = writer.run(work)
    if payloads is not None:
        for event in payloads:
            if event is not None:
                events.publish(event)
    return alert_id


def attach_media(writer: DBWriter, alert_id: int, *, snapshot: str | None = None, recording: str | None = None) -> None:
    """Record snapshot / recording file NAMES (relative to their directories) on an alert."""

    def work(db: Session) -> dict[str, object] | None:
        row = db.get(Alert, alert_id)
        if row is None:
            return None
        if snapshot is not None:
            row.snapshot_path = snapshot
        if recording is not None:
            row.recording_path = recording
        return outbox.alert_event(row, "alert.updated")

    event = writer.run(work)
    if event is not None:
        events.publish(event)
