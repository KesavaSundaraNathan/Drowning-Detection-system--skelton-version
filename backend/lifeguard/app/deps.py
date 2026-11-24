"""Request dependencies: context, client IP, session authentication, CSRF, role checks.

Authentication is cookie-only (``__Host-lifeguard_session``: HttpOnly, Secure, SameSite=Strict).
There is no ``?token=`` and no Authorization header path - the browser never exposes the token to
JavaScript and it never lands in URLs, logs or browser history (audit #5).

State-changing methods additionally need ``X-CSRF-Token`` equal to the CSRF cookie AND to the
hash stored with the session. Errors are machine-readable codes in ``detail``:
401 not_authenticated | session_expired | account_disabled, 403 csrf_failed | forbidden.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response

from lifeguard.app.context import AppContext
from lifeguard.core.clock import utcnow
from lifeguard.core.config import Settings
from lifeguard.core.security import sessions
from lifeguard.core.security.rbac import Principal, Role
from lifeguard.core.security.tokens import constant_time_equal, token_hash

SESSION_COOKIE = "__Host-lifeguard_session"
CSRF_COOKIE = "__Host-lifeguard_csrf"
CSRF_HEADER = "x-csrf-token"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def get_ctx(request: Request) -> AppContext:
    ctx: AppContext = request.app.state.ctx
    return ctx


def client_ip(request: Request) -> str:
    # Behind Caddy, uvicorn runs with --proxy-headers --forwarded-allow-ips 127.0.0.1 (Phase 8),
    # so request.client is the real client; X-Forwarded-For is never trusted here directly.
    return request.client.host if request.client else "unknown"


def require_session(request: Request) -> Principal:
    ctx = get_ctx(request)
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="not_authenticated")
    now = utcnow()
    with ctx.db.read_sessions() as s:
        result = sessions.check(
            s,
            session_key=ctx.keys.session_key,
            token=token,
            now=now,
            idle=timedelta(minutes=ctx.settings.session_idle_minutes),
        )
    if result.principal is None:
        raise HTTPException(status_code=401, detail=result.reason)
    if request.method not in SAFE_METHODS:
        header = request.headers.get(CSRF_HEADER, "")
        cookie = request.cookies.get(CSRF_COOKIE, "")
        if not (
            header
            and cookie
            and constant_time_equal(header, cookie)
            and constant_time_equal(token_hash(ctx.keys.session_key, header), result.csrf_hash)
        ):
            raise HTTPException(status_code=403, detail="csrf_failed")
    if result.needs_touch:
        sid = result.principal.session_id
        ctx.writer.run(lambda s: sessions.touch(s, sid, now))
    request.state.principal = result.principal
    return result.principal


class RequireRole:
    """Route-level role requirement. tests/api/test_auth_matrix.py discovers these on every route."""

    def __init__(self, role: Role) -> None:
        self.role = role

    def __call__(self, principal: Annotated[Principal, Depends(require_session)]) -> Principal:
        if not principal.role.at_least(self.role):
            raise HTTPException(status_code=403, detail="forbidden")
        return principal

    def __repr__(self) -> str:
        return f"RequireRole({self.role.value})"


def set_session_cookies(response: Response, settings: Settings, token: str, csrf: str, max_age: int) -> None:
    common = {"max_age": max_age, "path": "/", "secure": settings.cookie_secure, "samesite": "strict"}
    response.set_cookie(SESSION_COOKIE, token, httponly=True, **common)  # type: ignore[arg-type]
    response.set_cookie(CSRF_COOKIE, csrf, httponly=False, **common)  # type: ignore[arg-type]


def clear_session_cookies(response: Response, settings: Settings) -> None:
    for name in (SESSION_COOKIE, CSRF_COOKIE):
        response.delete_cookie(
            name, path="/", secure=settings.cookie_secure, httponly=name == SESSION_COOKIE, samesite="strict"
        )


Ctx = Annotated[AppContext, Depends(get_ctx)]
AnyUser = Annotated[Principal, Depends(RequireRole(Role.LIFEGUARD))]
Supervisor = Annotated[Principal, Depends(RequireRole(Role.SUPERVISOR))]
Admin = Annotated[Principal, Depends(RequireRole(Role.ADMIN))]
