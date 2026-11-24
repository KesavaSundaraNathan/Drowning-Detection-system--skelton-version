"""[Phase 5] Events API: one event per person and incident on one camera (alerts/events.py).

GET  /events                     newest first; `status=open` = not closed and not resolved (the AlertCenter)
GET  /events/{id}                one event with its alerts' ids
POST /events/{id}/acknowledge    acknowledges every pending alert of the event (each audited), event -> acknowledged
POST /events/{id}/resolve        supervisor: resolves every open alert of the event with the note, event -> resolved
A resolved event is closed; an acknowledged one re-opens when a new alert joins it (alerts/events.attach).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from lifeguard.alerts import events as event_engine
from lifeguard.alerts import outbox
from lifeguard.app.api.v1.alerts import ack_event_alerts
from lifeguard.app.api.v1.schemas import AlertCloseIn, EventOut
from lifeguard.app.deps import AnyUser, Ctx, Supervisor, client_ip
from lifeguard.core import audit, events
from lifeguard.core.clock import utcnow
from lifeguard.db.models import Alert, Event

router = APIRouter(prefix="/events", tags=["events"])


@router.get("", response_model=list[EventOut])
def list_events(
    ctx: Ctx,
    _: AnyUser,
    status: Annotated[Literal["open", "all"], Query()] = "open",
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[EventOut]:
    with ctx.db.read_sessions() as rs:
        q = select(Event).order_by(Event.id.desc()).limit(limit)
        if status == "open":
            q = q.where(Event.closed_at.is_(None), Event.status != "resolved")
        return [EventOut.model_validate(e) for e in rs.execute(q).scalars()]


@router.get("/{event_id}")
def get_event(event_id: int, ctx: Ctx, _: AnyUser) -> dict[str, Any]:
    with ctx.db.read_sessions() as rs:
        ev = rs.get(Event, event_id)
        if ev is None:
            raise HTTPException(404, "not_found")
        ids = [
            int(i) for i in rs.execute(select(Alert.id).where(Alert.event_id == event_id).order_by(Alert.id)).scalars()
        ]
        return {**EventOut.model_validate(ev).model_dump(mode="json"), "alert_ids": ids}


@router.post("/{event_id}/acknowledge", response_model=EventOut)
def acknowledge(event_id: int, request: Request, ctx: Ctx, me: AnyUser) -> EventOut:
    now, ip = utcnow(), client_ip(request)

    def work(db: Session) -> tuple[EventOut, list[dict[str, Any]]]:
        ev = db.get(Event, event_id)
        if ev is None:
            raise HTTPException(404, "not_found")
        if ev.status == "resolved":
            raise HTTPException(409, "invalid_transition: resolved -> acknowledged")
        evs = ack_event_alerts(db, event_id, me, ip, now)
        audit.record(db, action="event.acknowledge", actor=me, ip=ip, target_type="event", target_id=ev.id, details={})
        return EventOut.model_validate(ev), evs

    out, evs = ctx.writer.run(work)
    for e in evs:
        events.publish(e)
    return out


@router.post("/{event_id}/resolve", response_model=EventOut)
def resolve(event_id: int, body: AlertCloseIn, request: Request, ctx: Ctx, me: Supervisor) -> EventOut:
    now, ip = utcnow(), client_ip(request)

    def work(db: Session) -> tuple[EventOut, list[dict[str, Any]]]:
        ev = db.get(Event, event_id)
        if ev is None:
            raise HTTPException(404, "not_found")
        out: list[dict[str, Any]] = []
        rows = db.execute(
            select(Alert).where(Alert.event_id == event_id, Alert.status.in_(("pending", "acknowledged")))
        ).scalars()
        for a in rows:
            before = a.status
            a.status, a.closed_by, a.closed_at, a.close_note = "resolved", me.user_id, now, body.note
            audit.record(
                db,
                action="alert.resolve",
                actor=me,
                ip=ip,
                target_type="alert",
                target_id=a.id,
                details={"from": before, "to": "resolved", "note": body.note, "via_event": event_id},
            )
            out.append(outbox.alert_event(a, "alert.updated"))
        ev.status, ev.resolved_by, ev.resolved_at = "resolved", me.user_id, now
        if ev.closed_at is None:
            ev.closed_at = now
        audit.record(
            db,
            action="event.resolve",
            actor=me,
            ip=ip,
            target_type="event",
            target_id=ev.id,
            details={"note": body.note},
        )
        out.append(event_engine.event_payload(ev, "event.closed"))
        return EventOut.model_validate(ev), out

    result, evs = ctx.writer.run(work)
    for e in evs:
        events.publish(e)
    return result
