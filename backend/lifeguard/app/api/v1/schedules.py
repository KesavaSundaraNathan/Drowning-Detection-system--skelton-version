"""Monitoring schedules. Read: any role. Write: supervisor+ (schedules can mute sirens in Phase 4)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from lifeguard.app.api.v1.schemas import ScheduleIn, ScheduleOut
from lifeguard.app.api.v1.zones import zones_changed
from lifeguard.app.deps import AnyUser, Ctx, Supervisor, client_ip
from lifeguard.core import audit
from lifeguard.core.clock import utcnow
from lifeguard.db.models import Schedule

router = APIRouter(prefix="/schedules", tags=["schedules"])


def _summary(s: Schedule) -> dict[str, object]:
    return {
        "name": s.name,
        "days": s.days,
        "start_time": s.start_time,
        "end_time": s.end_time,
        "monitoring_mode": s.monitoring_mode,
        "status": s.status,
    }


@router.get("", response_model=list[ScheduleOut])
def list_schedules(ctx: Ctx, _: AnyUser) -> list[ScheduleOut]:
    with ctx.db.read_sessions() as rs:
        return [ScheduleOut.model_validate(s) for s in rs.execute(select(Schedule).order_by(Schedule.name)).scalars()]


@router.get("/{schedule_id}", response_model=ScheduleOut)
def get_schedule(schedule_id: int, ctx: Ctx, _: AnyUser) -> ScheduleOut:
    with ctx.db.read_sessions() as rs:
        sched = rs.get(Schedule, schedule_id)
        if sched is None:
            raise HTTPException(404, "not_found")
        return ScheduleOut.model_validate(sched)


@router.post("", response_model=ScheduleOut, status_code=201)
def create_schedule(body: ScheduleIn, request: Request, ctx: Ctx, me: Supervisor) -> ScheduleOut:
    now, ip = utcnow(), client_ip(request)

    def work(db: Session) -> ScheduleOut:
        sched = Schedule(
            name=body.name,
            days=list(body.days),
            start_time=body.start_time,
            end_time=body.end_time,
            monitoring_mode=body.monitoring_mode,
            status=body.status,
            created_at=now,
            updated_at=now,
        )
        db.add(sched)
        db.flush()
        audit.record(
            db,
            action="schedule.create",
            actor=me,
            ip=ip,
            target_type="schedule",
            target_id=sched.id,
            details=_summary(sched),
        )
        return ScheduleOut.model_validate(sched)

    try:
        out: ScheduleOut = ctx.writer.run(work)
    except IntegrityError:
        raise HTTPException(409, "schedule_name_taken") from None
    zones_changed(ctx, None)  # any camera's zones may use this schedule
    return out


@router.put("/{schedule_id}", response_model=ScheduleOut)
def replace_schedule(schedule_id: int, body: ScheduleIn, request: Request, ctx: Ctx, me: Supervisor) -> ScheduleOut:
    now, ip = utcnow(), client_ip(request)

    def work(db: Session) -> ScheduleOut:
        sched = db.get(Schedule, schedule_id)
        if sched is None:
            raise HTTPException(404, "not_found")
        before = _summary(sched)
        sched.name, sched.days, sched.start_time, sched.end_time = (
            body.name,
            list(body.days),
            body.start_time,
            body.end_time,
        )
        sched.monitoring_mode, sched.status, sched.updated_at = body.monitoring_mode, body.status, now
        audit.record(
            db,
            action="schedule.update",
            actor=me,
            ip=ip,
            target_type="schedule",
            target_id=sched.id,
            details={"before": before, "after": _summary(sched)},
        )
        return ScheduleOut.model_validate(sched)

    try:
        out: ScheduleOut = ctx.writer.run(work)
    except IntegrityError:
        raise HTTPException(409, "schedule_name_taken") from None
    zones_changed(ctx, None)  # any camera's zones may use this schedule
    return out


@router.delete("/{schedule_id}", status_code=204)
def delete_schedule(schedule_id: int, request: Request, ctx: Ctx, me: Supervisor) -> Response:
    ip = client_ip(request)

    def work(db: Session) -> None:
        sched = db.get(Schedule, schedule_id)
        if sched is None:
            raise HTTPException(404, "not_found")
        audit.record(
            db,
            action="schedule.delete",
            actor=me,
            ip=ip,
            target_type="schedule",
            target_id=sched.id,
            details=_summary(sched),
        )
        db.delete(sched)

    ctx.writer.run(work)
    zones_changed(ctx, None)
    return Response(status_code=204)
