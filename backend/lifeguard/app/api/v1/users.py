"""/api/v1/users: account administration (admin only).

* passwords must pass the policy; usernames are lower-case ``[a-z0-9._-]``, 3-64 chars;
* a role change, a deactivation or an admin password reset revokes that user's sessions, so the
  new privileges take effect immediately (no stale session keeps old rights);
* the last active admin can never be demoted or deactivated (no lock-out of the whole system).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from lifeguard.app.api.v1.schemas import PasswordResetIn, UserCreateIn, UserOut, UserPatchIn
from lifeguard.app.deps import Admin, Ctx, client_ip
from lifeguard.core import audit
from lifeguard.core.clock import utcnow
from lifeguard.core.security import sessions
from lifeguard.core.security.passwords import PasswordPolicyError, check_policy, hash_password
from lifeguard.core.security.rbac import Role
from lifeguard.db.models import User

router = APIRouter(prefix="/users", tags=["users"])


def _policy(password: str, username: str, min_length: int) -> None:
    try:
        check_policy(password, username, min_length)
    except PasswordPolicyError as exc:
        raise HTTPException(422, f"password_policy: {exc}") from None


def _other_active_admins(db: Session, excluding: int) -> int:
    return int(
        db.execute(
            select(func.count(User.id)).where(
                User.role == Role.ADMIN.value, User.is_active.is_(True), User.id != excluding
            )
        ).scalar_one()
    )


@router.get("", response_model=list[UserOut])
def list_users(ctx: Ctx, _: Admin) -> list[UserOut]:
    with ctx.db.read_sessions() as rs:
        return [UserOut.model_validate(u) for u in rs.execute(select(User).order_by(User.id)).scalars()]


@router.post("", response_model=UserOut, status_code=201)
def create_user(body: UserCreateIn, request: Request, ctx: Ctx, me: Admin) -> UserOut:
    _policy(body.password, body.username, ctx.settings.password_min_length)
    pw_hash, now, ip = hash_password(body.password, ctx.settings.bcrypt_rounds), utcnow(), client_ip(request)

    def work(db: Session) -> UserOut:
        user = User(
            username=body.username,
            password_hash=pw_hash,
            role=body.role.value,
            is_active=True,
            created_at=now,
            updated_at=now,
            password_changed_at=now,
        )
        db.add(user)
        db.flush()
        audit.record(
            db,
            action="user.create",
            actor=me,
            ip=ip,
            target_type="user",
            target_id=user.id,
            details={"username": user.username, "role": user.role},
        )
        return UserOut.model_validate(user)

    try:
        return ctx.writer.run(work)
    except IntegrityError:
        raise HTTPException(409, "username_taken") from None


@router.patch("/{user_id}", response_model=UserOut)
def update_user(user_id: int, body: UserPatchIn, request: Request, ctx: Ctx, me: Admin) -> UserOut:
    now, ip = utcnow(), client_ip(request)

    def work(db: Session) -> UserOut:
        user = db.get(User, user_id)
        if user is None:
            raise HTTPException(404, "not_found")
        before = {"role": user.role, "is_active": user.is_active}
        losing_admin = (
            user.role == Role.ADMIN.value
            and user.is_active
            and ((body.role is not None and body.role != Role.ADMIN) or body.is_active is False)
        )
        if losing_admin and _other_active_admins(db, user.id) == 0:
            raise HTTPException(409, "last_admin")
        if body.role is not None:
            user.role = body.role.value
        if body.is_active is not None:
            user.is_active = body.is_active
        user.updated_at = now
        revoked = 0
        if (user.role, user.is_active) != (before["role"], before["is_active"]):
            revoked = sessions.revoke_all_for_user(db, user.id, now)
        audit.record(
            db,
            action="user.update",
            actor=me,
            ip=ip,
            target_type="user",
            target_id=user.id,
            details={
                "before": before,
                "after": {"role": user.role, "is_active": user.is_active},
                "sessions_revoked": revoked,
            },
        )
        return UserOut.model_validate(user)

    return ctx.writer.run(work)


@router.post("/{user_id}/password", status_code=204)
def reset_password(user_id: int, body: PasswordResetIn, request: Request, ctx: Ctx, me: Admin) -> Response:
    now, ip = utcnow(), client_ip(request)
    with ctx.db.read_sessions() as rs:
        target = rs.get(User, user_id)
    if target is None:
        raise HTTPException(404, "not_found")
    _policy(body.new_password, target.username, ctx.settings.password_min_length)
    pw_hash = hash_password(body.new_password, ctx.settings.bcrypt_rounds)

    def work(db: Session) -> None:
        user = db.get(User, user_id)
        if user is None:
            raise HTTPException(404, "not_found")
        user.password_hash, user.password_changed_at, user.updated_at = pw_hash, now, now
        revoked = sessions.revoke_all_for_user(db, user.id, now)
        audit.record(
            db,
            action="user.password.reset",
            actor=me,
            ip=ip,
            target_type="user",
            target_id=user.id,
            details={"sessions_revoked": revoked},
        )

    ctx.writer.run(work)
    return Response(status_code=204)
