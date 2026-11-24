"""/api/v1/alerts: review workflow. Alerts are created by the pipelines (Phase 4/5).

  pending --acknowledge (any role)--> acknowledged
  pending | acknowledged --resolve / false-alarm (supervisor+, note required)--> resolved | false_alarm
Any other transition is 409.
Clear all (supervisor+, note required): resolves every open (pending / acknowledged) alert in one audited step -
nothing is deleted; each alert keeps its own audit row. Every review action is audited; CSV export
(supervisor+) is audited and formula-injection-safe. Acknowledging stops escalation (Phase 5); every transition
is published on the WebSocket feed. Snapshot (JPEG) and recording (MP4) downloads need a session (any role) and
only ever serve a file strictly inside its directory.

[Phase 5] One event per person and camera (alerts/events.py): acknowledging an alert also acknowledges the OTHER
alerts of its event that are pending at that moment - never another track's, never an alert raised after the click
- and marks the event acknowledged. Each is audited separately with `via_event`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from lifeguard.alerts import events as event_engine
from lifeguard.alerts import outbox
from lifeguard.alerts.media import MEDIA_TYPES, contained
from lifeguard.app.api.v1.schemas import AlertCloseIn, AlertOut
from lifeguard.app.deps import AnyUser, Ctx, Supervisor, client_ip
from lifeguard.core import audit, events
from lifeguard.core.clock import utcnow
from lifeguard.core.csvsafe import to_csv
from lifeguard.core.security.rbac import Principal
from lifeguard.db.models import Alert, Event
from lifeguard.vision.scene import alert_reason

router = APIRouter(prefix="/alerts", tags=["alerts"])
_NO_STORE = {"Cache-Control": "no-store, private"}  # footage of people: never cached
Status = Literal["pending", "acknowledged", "resolved", "false_alarm"]
EXPORT_COLUMNS = (
    "id",
    "created_at",
    "camera_id",
    "track_id",
    "scenario",
    "alert_reason",
    "priority",
    "risk_score",
    "confidence",
    "zone_name",
    "reason",
    "status",
    "acknowledged_by",
    "acknowledged_at",
    "closed_by",
    "closed_at",
    "close_note",
    "global_id",
    "incident_id",
    "event_id",
)


@router.get("", response_model=list[AlertOut])
def list_alerts(
    ctx: Ctx,
    _: AnyUser,
    status: Annotated[Status | None, Query()] = None,
    audible: Annotated[bool | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    before_id: Annotated[int | None, Query(ge=1)] = None,
) -> list[AlertOut]:
    q = select(Alert).order_by(Alert.id.desc()).limit(limit)
    if status:
        q = q.where(Alert.status == status)
    if audible is not None:
        q = q.where(Alert.audible.is_(audible))
    if before_id:
        q = q.where(Alert.id < before_id)
    with ctx.db.read_sessions() as rs:
        return [AlertOut.model_validate(a) for a in rs.execute(q).scalars()]


def _export_value(a: Alert, column: str) -> Any:
    """[Stage 4] Exports say what the API says: scenario DROWNING / FAULT, the reason text, the cleaned reason."""
    if column == "scenario":
        return alert_reason.api_scenario(a.scenario)
    if column == "alert_reason":
        return alert_reason.reason_text(a.scenario)
    if column == "reason":
        return alert_reason.clean_reason(a.reason, a.scenario)
    return getattr(a, column)


@router.get("/export.csv")
def export_csv(
    request: Request,
    ctx: Ctx,
    me: Supervisor,
    since: Annotated[datetime | None, Query()] = None,
    until: Annotated[datetime | None, Query()] = None,
) -> Response:
    q = select(Alert).order_by(Alert.id)
    if since:
        q = q.where(Alert.created_at >= since)
    if until:
        q = q.where(Alert.created_at < until)
    with ctx.db.read_sessions() as rs:
        rows = [[_export_value(a, c) for c in EXPORT_COLUMNS] for a in rs.execute(q).scalars()]
    ip = client_ip(request)
    ctx.writer.run(
        lambda db: audit.record(
            db,
            action="alert.export",
            actor=me,
            ip=ip,
            target_type="alert",
            details={"rows": len(rows), "since": str(since), "until": str(until)},
        )
    )
    return Response(
        to_csv(EXPORT_COLUMNS, rows),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="alerts.csv"'},
    )


@router.get("/{alert_id}", response_model=AlertOut)
def get_alert(alert_id: int, ctx: Ctx, _: AnyUser) -> AlertOut:
    with ctx.db.read_sessions() as rs:
        alert = rs.get(Alert, alert_id)
        if alert is None:
            raise HTTPException(404, "not_found")
        return AlertOut.model_validate(alert)


def _transition(
    ctx: Ctx,
    request: Request,
    me: Principal,
    alert_id: int,
    action: str,
    allowed_from: tuple[str, ...],
    new_status: str,
    note: str | None,
) -> AlertOut:
    now, ip = utcnow(), client_ip(request)

    def work(db: Session) -> tuple[AlertOut, list[dict[str, Any]]]:
        alert = db.get(Alert, alert_id)
        if alert is None:
            raise HTTPException(404, "not_found")
        if alert.status not in allowed_from:
            raise HTTPException(409, f"invalid_transition: {alert.status} -> {new_status}")
        before = alert.status
        alert.status = new_status
        if new_status == "acknowledged":
            alert.acknowledged_by, alert.acknowledged_at = me.user_id, now
        else:
            alert.closed_by, alert.closed_at, alert.close_note = me.user_id, now, note
        audit.record(
            db,
            action=action,
            actor=me,
            ip=ip,
            target_type="alert",
            target_id=alert.id,
            details={"from": before, "to": new_status, "note": note},
        )
        evs = [outbox.alert_event(alert, "alert.updated")]
        if new_status == "acknowledged" and alert.event_id is not None:
            evs += ack_event_alerts(db, alert.event_id, me, ip, now, clicked=alert.id)
        return AlertOut.model_validate(alert), evs

    out, evs = ctx.writer.run(work)
    for event in evs:
        events.publish(event)
    return out


def ack_event_alerts(
    db: Session, event_id: int, me: Principal, ip: str, now: datetime, *, clicked: int | None = None
) -> list[dict[str, Any]]:
    """[Phase 5] Acknowledge the event's other PENDING alerts and mark the event acknowledged (alerts/events.py)."""
    rows = db.execute(
        select(Alert).where(
            Alert.event_id == event_id,
            Alert.id != (clicked or -1),
            Alert.status == "pending",
            Alert.scenario != "FAULT",
        )
    ).scalars()
    out: list[dict[str, Any]] = []
    for a in rows:
        a.status = "acknowledged"
        a.acknowledged_by, a.acknowledged_at = me.user_id, now
        audit.record(
            db,
            action="alert.acknowledge",
            actor=me,
            ip=ip,
            target_type="alert",
            target_id=a.id,
            details={"from": "pending", "to": "acknowledged", "via_event": event_id, "clicked": clicked},
        )
        out.append(outbox.alert_event(a, "alert.updated"))
    ev = db.get(Event, event_id)
    if ev is not None and ev.status == "open":
        ev.status, ev.acknowledged_by, ev.acknowledged_at = "acknowledged", me.user_id, now
        out.append(event_engine.event_payload(ev, "event.updated"))
    return out


@router.post("/clear-all")
def clear_all(body: AlertCloseIn, request: Request, ctx: Ctx, me: Supervisor) -> dict[str, int]:
    """Resolve every open alert (pending or acknowledged). Alerts are kept; each gets its own audit row."""
    now, ip = utcnow(), client_ip(request)

    def work(db: Session) -> list[dict[str, Any]]:
        rows = db.execute(select(Alert).where(Alert.status.in_(("pending", "acknowledged")))).scalars().all()
        evs: list[dict[str, Any]] = []
        for a in rows:
            before = a.status
            a.status = "resolved"
            a.closed_by, a.closed_at, a.close_note = me.user_id, now, body.note
            audit.record(
                db,
                action="alert.resolve",
                actor=me,
                ip=ip,
                target_type="alert",
                target_id=a.id,
                details={"from": before, "to": "resolved", "note": body.note, "via": "clear_all"},
            )
            evs.append(outbox.alert_event(a, "alert.updated"))
        return evs

    evs = ctx.writer.run(work)
    for event in evs:
        events.publish(event)
    return {"cleared": len(evs)}


@router.post("/{alert_id}/acknowledge", response_model=AlertOut)
def acknowledge(alert_id: int, request: Request, ctx: Ctx, me: AnyUser) -> AlertOut:
    return _transition(ctx, request, me, alert_id, "alert.acknowledge", ("pending",), "acknowledged", None)


@router.post("/{alert_id}/resolve", response_model=AlertOut)
def resolve(alert_id: int, body: AlertCloseIn, request: Request, ctx: Ctx, me: Supervisor) -> AlertOut:
    return _transition(ctx, request, me, alert_id, "alert.resolve", ("pending", "acknowledged"), "resolved", body.note)


@router.post("/{alert_id}/false-alarm", response_model=AlertOut)
def false_alarm(alert_id: int, body: AlertCloseIn, request: Request, ctx: Ctx, me: Supervisor) -> AlertOut:
    return _transition(
        ctx, request, me, alert_id, "alert.false_alarm", ("pending", "acknowledged"), "false_alarm", body.note
    )


def _media(ctx: Ctx, alert_id: int, which: str) -> Path:
    with ctx.db.read_sessions() as rs:
        alert = rs.get(Alert, alert_id)
        if alert is None:
            raise HTTPException(404, "not_found")
        name = alert.snapshot_path if which == "snapshot" else alert.recording_path
    base = ctx.settings.snapshots_dir if which == "snapshot" else ctx.settings.recordings_dir
    path = contained(base, name)
    if path is None:
        raise HTTPException(404, f"no_{which}")
    return path


@router.get("/{alert_id}/snapshot", response_class=FileResponse)
def snapshot(alert_id: int, ctx: Ctx, _: AnyUser) -> FileResponse:
    return FileResponse(_media(ctx, alert_id, "snapshot"), media_type="image/jpeg", headers=_NO_STORE)


@router.get("/{alert_id}/recording", response_class=FileResponse)
def recording(alert_id: int, ctx: Ctx, _: AnyUser) -> FileResponse:
    path = _media(ctx, alert_id, "recording")
    headers = dict(_NO_STORE)
    if path.suffix.lower() != ".webm":  # legacy MPEG-4 Part 2: not playable in a browser -> download
        headers["Content-Disposition"] = f'attachment; filename="{path.name}"'
    return FileResponse(
        path, media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"), headers=headers
    )
