"""/api/v1/auth: login (public), logout, me, password change."""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from lifeguard.app.api.v1.schemas import LoginIn, LoginOut, PasswordChangeIn, UserOut
from lifeguard.app.deps import AnyUser, Ctx, clear_session_cookies, client_ip, set_session_cookies
from lifeguard.core import audit
from lifeguard.core.clock import utcnow
from lifeguard.core.security import ratelimit, sessions
from lifeguard.core.security.passwords import PasswordPolicyError, check_policy, hash_password, verify_password
from lifeguard.db.models import User

public = APIRouter(prefix="/auth", tags=["auth"])
router = APIRouter(prefix="/auth", tags=["auth"])


@public.post("/login", response_model=LoginOut)
def login(body: LoginIn, request: Request, response: Response, ctx: Ctx) -> LoginOut:
    s, ip, now = ctx.settings, client_ip(request), utcnow()
    window = timedelta(minutes=s.login_lockout_minutes)
    username = body.username

    def reserve(db: Session) -> ratelimit.Reservation:
        res = ratelimit.reserve(db, username, ip, now, s.login_max_failures, window)
        if res.attempt_id is None:
            audit.record(
                db,
                action="auth.login.locked",
                actor=None,
                actor_username=username,
                ip=ip,
                target_type="user",
                target_id=username,
                outcome="denied",
                details={"by": res.locked_by},
            )
        return res

    reservation = ctx.writer.run(reserve)
    if reservation.attempt_id is None:
        raise HTTPException(429, "too_many_attempts", headers={"Retry-After": str(reservation.retry_after_s)})
    attempt_id = reservation.attempt_id

    # bcrypt runs here, outside the writer thread, so a slow hash never stalls other writes.
    with ctx.db.read_sessions() as rs:
        user = rs.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if user is None:
        ok, reason = ctx.dummy_hasher.burn(body.password), "unknown_user"
    elif not user.is_active:
        verify_password(body.password, user.password_hash)
        ok, reason = False, "account_disabled"
    else:
        ok = verify_password(body.password, user.password_hash)
        reason = "ok" if ok else "bad_password"

    def finish(db: Session) -> sessions.IssuedSession | None:
        ratelimit.resolve(db, attempt_id, ok)
        if not ok or user is None:
            audit.record(
                db,
                action="auth.login.failure",
                actor=None,
                actor_username=username,
                ip=ip,
                target_type="user",
                target_id=username,
                outcome="denied",
                details={"reason": reason},
            )
            return None
        issued = sessions.issue(
            db,
            session_key=ctx.keys.session_key,
            user_id=user.id,
            ip=ip,
            user_agent=request.headers.get("user-agent", ""),
            now=now,
            absolute=timedelta(hours=s.session_absolute_hours),
        )
        audit.record(
            db,
            action="auth.login.success",
            actor=None,
            actor_username=username,
            ip=ip,
            target_type="user",
            target_id=user.id,
            details={"session_id": issued.session_id},
        )
        return issued

    issued = ctx.writer.run(finish)
    if issued is None or user is None:
        raise HTTPException(401, "invalid_credentials")
    set_session_cookies(response, s, issued.token, issued.csrf_token, int(s.session_absolute_hours * 3600))
    return LoginOut(user=UserOut.model_validate(user), csrf_token=issued.csrf_token, expires_at=issued.expires_at)


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, ctx: Ctx, me: AnyUser) -> Response:
    now, ip = utcnow(), client_ip(request)

    def work(db: Session) -> None:
        sessions.revoke(db, me.session_id, now)
        audit.record(db, action="auth.logout", actor=me, ip=ip, target_type="session", target_id=me.session_id)

    ctx.writer.run(work)
    out = Response(status_code=204)
    clear_session_cookies(out, ctx.settings)
    return out


@router.get("/me", response_model=UserOut)
def me(ctx: Ctx, principal: AnyUser) -> UserOut:
    with ctx.db.read_sessions() as rs:
        user = rs.get(User, principal.user_id)
        if user is None:
            raise HTTPException(401, "not_authenticated")
        return UserOut.model_validate(user)


@router.post("/password", status_code=204)
def change_password(body: PasswordChangeIn, request: Request, ctx: Ctx, principal: AnyUser) -> Response:
    s, ip, now = ctx.settings, client_ip(request), utcnow()
    with ctx.db.read_sessions() as rs:
        user = rs.get(User, principal.user_id)
    if user is None or not verify_password(body.current_password, user.password_hash):
        ctx.writer.run(
            lambda db: audit.record(
                db,
                action="auth.password.change",
                actor=principal,
                ip=ip,
                target_type="user",
                target_id=principal.user_id,
                outcome="denied",
                details={"reason": "bad_current_password"},
            )
        )
        raise HTTPException(403, "invalid_current_password")
    try:
        check_policy(body.new_password, principal.username, s.password_min_length)
    except PasswordPolicyError as exc:
        raise HTTPException(422, f"password_policy: {exc}") from None
    new_hash = hash_password(body.new_password, s.bcrypt_rounds)

    def work(db: Session) -> None:
        row = db.get(User, principal.user_id)
        if row is None:
            raise HTTPException(401, "not_authenticated")
        row.password_hash, row.password_changed_at, row.updated_at = new_hash, now, now
        revoked = sessions.revoke_all_for_user(db, principal.user_id, now, except_session=principal.session_id)
        audit.record(
            db,
            action="auth.password.change",
            actor=principal,
            ip=ip,
            target_type="user",
            target_id=principal.user_id,
            details={"other_sessions_revoked": revoked},
        )

    ctx.writer.run(work)
    return Response(status_code=204)
