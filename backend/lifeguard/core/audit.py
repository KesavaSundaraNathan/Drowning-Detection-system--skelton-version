"""Audit trail: every mutation (users, cameras, zones, schedules, thresholds, alert reviews,
backups, retention) and every login attempt.

`record()` is called INSIDE the writer transaction that performs the change, so the change and
its audit row commit or roll back together - there is never a mutation without its audit
event, or an audit event for a change that did not happen. Details are scrubbed of secrets.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from lifeguard.core.clock import utcnow
from lifeguard.core.logging import redact
from lifeguard.core.security.rbac import Principal
from lifeguard.db.models import AuditEvent

_SECRET_KEYS = {
    "password",
    "new_password",
    "current_password",
    "source",
    "source_url",
    "token",
    "csrf",
    "secret",
    "credential",
    "key",
}


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("***" if k.lower() in _SECRET_KEYS else _scrub(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, str):
        return redact(value)
    return value


def record(
    session: Session,
    *,
    action: str,
    actor: Principal | None,
    ip: str,
    target_type: str = "",
    target_id: object = "",
    outcome: str = "success",
    details: dict[str, Any] | None = None,
    actor_username: str | None = None,
) -> None:
    session.add(
        AuditEvent(
            at=utcnow(),
            actor_user_id=actor.user_id if actor else None,
            actor_username=(actor.username if actor else actor_username) or "-",
            ip=ip[:45],
            action=action,
            target_type=target_type,
            target_id=str(target_id)[:64],
            outcome=outcome,
            details=_scrub(details or {}),
        )
    )
