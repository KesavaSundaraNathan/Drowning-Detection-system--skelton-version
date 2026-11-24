"""/api/v1/audit: read the audit trail (admin only). There is no write or delete endpoint."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query
from sqlalchemy import select

from lifeguard.app.api.v1.schemas import AuditOut
from lifeguard.app.deps import Admin, Ctx
from lifeguard.db.models import AuditEvent

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("", response_model=list[AuditOut])
def list_audit(
    ctx: Ctx,
    _: Admin,
    action: Annotated[str | None, Query(max_length=64)] = None,
    actor: Annotated[str | None, Query(max_length=64)] = None,
    since: Annotated[datetime | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    before_id: Annotated[int | None, Query(ge=1)] = None,
) -> list[AuditOut]:
    q = select(AuditEvent).order_by(AuditEvent.id.desc()).limit(limit)
    if action:
        q = q.where(AuditEvent.action == action)
    if actor:
        q = q.where(AuditEvent.actor_username == actor.lower())
    if since:
        q = q.where(AuditEvent.at >= since)
    if before_id:
        q = q.where(AuditEvent.id < before_id)
    with ctx.db.read_sessions() as rs:
        return [AuditOut.model_validate(e) for e in rs.execute(q).scalars()]
