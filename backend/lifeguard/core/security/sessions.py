"""Server-side sessions.

* Token: 256-bit random (core.security.tokens); stored only as HMAC-SHA256 under the session key.
* Expiry: absolute (`session_absolute_hours`, default 12 h from login) AND idle
  (`session_idle_minutes`, default 30 min since the last authenticated request).
* CSRF: a second random token per session (synchronizer + double-submit): the client must send
  it in ``X-CSRF-Token`` on every state-changing request, matching both the CSRF cookie and the
  hash stored with the session.
* `last_seen_at` is refreshed at most once per `TOUCH_INTERVAL` to avoid a DB write per request;
  idle expiry is therefore precise to within that interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from lifeguard.core.security.rbac import Principal, Role
from lifeguard.core.security.tokens import new_token, token_hash
from lifeguard.db.models import User, UserSession

TOUCH_INTERVAL = timedelta(seconds=60)


@dataclass(frozen=True)
class IssuedSession:
    session_id: int
    token: str  # raw - goes to the browser once, never stored
    csrf_token: str  # raw - cookie + response body, never stored
    expires_at: datetime


@dataclass(frozen=True)
class SessionCheck:
    principal: Principal | None
    reason: str  # "ok" | "not_authenticated" | "session_expired" | "account_disabled"
    csrf_hash: str = ""
    needs_touch: bool = False


def issue(
    session: Session, *, session_key: bytes, user_id: int, ip: str, user_agent: str, now: datetime, absolute: timedelta
) -> IssuedSession:
    token, csrf = new_token(), new_token()
    row = UserSession(
        user_id=user_id,
        token_hash=token_hash(session_key, token),
        csrf_hash=token_hash(session_key, csrf),
        created_at=now,
        last_seen_at=now,
        expires_at=now + absolute,
        revoked_at=None,
        ip=ip[:45],
        user_agent=user_agent[:256],
    )
    session.add(row)
    session.flush()
    return IssuedSession(row.id, token, csrf, row.expires_at)


def check(session: Session, *, session_key: bytes, token: str, now: datetime, idle: timedelta) -> SessionCheck:
    row = session.execute(
        select(UserSession, User)
        .join(User, User.id == UserSession.user_id)
        .where(UserSession.token_hash == token_hash(session_key, token))
    ).first()
    if row is None:
        return SessionCheck(None, "not_authenticated")
    sess, user = row
    if sess.revoked_at is not None:
        return SessionCheck(None, "not_authenticated")
    if now >= sess.expires_at or now - sess.last_seen_at >= idle:
        return SessionCheck(None, "session_expired")
    if not user.is_active:
        return SessionCheck(None, "account_disabled")
    principal = Principal(user_id=user.id, username=user.username, role=Role(user.role), session_id=sess.id)
    return SessionCheck(principal, "ok", sess.csrf_hash, now - sess.last_seen_at >= TOUCH_INTERVAL)


def touch(session: Session, session_id: int, now: datetime) -> None:
    session.execute(update(UserSession).where(UserSession.id == session_id).values(last_seen_at=now))


def revoke(session: Session, session_id: int, now: datetime) -> None:
    session.execute(
        update(UserSession).where(UserSession.id == session_id, UserSession.revoked_at.is_(None)).values(revoked_at=now)
    )


def revoke_all_for_user(session: Session, user_id: int, now: datetime, except_session: int | None = None) -> int:
    q = update(UserSession).where(UserSession.user_id == user_id, UserSession.revoked_at.is_(None))
    if except_session is not None:
        q = q.where(UserSession.id != except_session)
    result = session.execute(q.values(revoked_at=now))
    return int(getattr(result, "rowcount", 0) or 0)
