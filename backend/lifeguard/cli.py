"""Administrative CLI:  python -m lifeguard <command>

keys init [--machine]          create the protected session + credential keys (once; DPAPI on Windows,
                               owner-only 0600 files on Linux / WSL)
db upgrade                     migrate the schema to head (back up first)
db backup                      online backup + integrity check + prune (scheduled task)
admin create --username NAME   create the FIRST admin; refuses if any admin exists.
                               The password is read from a hidden prompt or, with
                               --password-stdin, from standard input - never from argv
                               (other users can read process command lines).
admin exists                   exit 0 if an active admin exists, 1 if none (install.ps1)
admin reset-password --username NAME [--reactivate]
                               offline lock-out recovery for ANY account: new password (same
                               input rules as create), all sessions revoked, lock-out cleared,
                               audited. Server console only (needs the env file + DPAPI keys).
retention run                 purge footage older than RETENTION_DAYS (scheduled task)
config check                   load settings + keys + schema state; print status, no secrets
"""

from __future__ import annotations

import argparse
import getpass
import sys
from collections.abc import Sequence
from datetime import timedelta

from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session

from lifeguard.core import audit
from lifeguard.core.clock import utcnow
from lifeguard.core.config import Settings, load_settings
from lifeguard.core.errors import StartupError
from lifeguard.core.logging import configure_logging
from lifeguard.core.security import sessions
from lifeguard.core.security.keys import WINDOWS, init_key_files, load_keys
from lifeguard.core.security.passwords import PasswordPolicyError, check_policy, hash_password
from lifeguard.core.security.rbac import Role
from lifeguard.db import migrate
from lifeguard.db.backup import backup_sqlite
from lifeguard.db.models import LoginAttempt, User
from lifeguard.db.session import Database, make_database
from lifeguard.db.writer import DBWriter
from lifeguard.jobs.retention import run_retention


def _out(msg: str) -> None:
    sys.stdout.write(msg + "\n")


def _writer(settings: Settings) -> tuple[DBWriter, Database]:
    db = make_database(settings.effective_database_url)
    migrate.assert_at_head(db.write_engine)
    writer = DBWriter(db.write_sessions, name="cli-writer")
    writer.start()
    return writer, db


def cmd_keys_init(settings: Settings, args: argparse.Namespace) -> int:
    paths = init_key_files(settings.keys_dir, machine_scope=args.machine)
    how = ("DPAPI, " + ("machine" if args.machine else "user") + " scope") if WINDOWS else "owner-only files, mode 0600"
    _out("created " + ", ".join(str(p) for p in paths) + f" ({how})")
    return 0


def cmd_db_upgrade(settings: Settings, _: argparse.Namespace) -> int:
    db = make_database(settings.effective_database_url)
    try:
        before = migrate.current(db.write_engine)
        migrate.upgrade(db.write_engine)
        _out(f"schema: {before or 'empty'} -> {migrate.current(db.write_engine)}")
    finally:
        db.dispose()
    return 0


def cmd_db_backup(settings: Settings, _: argparse.Namespace) -> int:
    if not settings.is_sqlite:
        raise StartupError("db backup handles SQLite; use pg_dump for PostgreSQL")
    result = backup_sqlite(settings.sqlite_path, settings.backups_dir, settings.backup_keep)
    writer, db = _writer(settings)
    try:
        writer.run(
            lambda s: audit.record(
                s,
                action="db.backup",
                actor=None,
                actor_username="system",
                ip="local",
                target_type="database",
                target_id=result.path.name,
                details={"size_bytes": result.size_bytes, "pruned": result.pruned},
            )
        )
    finally:
        writer.stop()
        db.dispose()
    _out(f"backup {result.path} ({result.size_bytes} bytes, integrity {result.integrity}, pruned {result.pruned})")
    return 0


def _read_password(args: argparse.Namespace) -> str:
    if args.password_stdin:
        return sys.stdin.readline().rstrip("\r\n")
    first = getpass.getpass(f"New password for '{args.username.strip().lower()}': ")
    if first != getpass.getpass("Repeat password: "):
        raise StartupError("passwords do not match")
    return first


def cmd_admin_create(settings: Settings, args: argparse.Namespace) -> int:
    username = args.username.strip().lower()
    password = _read_password(args)
    try:
        check_policy(password, username, settings.password_min_length)
    except PasswordPolicyError as exc:
        raise StartupError(f"password policy: {exc}") from None
    pw_hash, now = hash_password(password, settings.bcrypt_rounds), utcnow()
    writer, db = _writer(settings)

    def work(s: Session) -> int:
        if s.execute(select(User.id).where(User.role == Role.ADMIN.value)).first() is not None:
            raise StartupError("an admin already exists - create further accounts through the API")
        user = User(
            username=username,
            password_hash=pw_hash,
            role=Role.ADMIN.value,
            is_active=True,
            created_at=now,
            updated_at=now,
            password_changed_at=now,
        )
        s.add(user)
        s.flush()
        audit.record(
            s,
            action="admin.bootstrap",
            actor=None,
            actor_username="system",
            ip="local",
            target_type="user",
            target_id=user.id,
            details={"username": username},
        )
        return user.id

    try:
        user_id = writer.run(work)
    finally:
        writer.stop()
        db.dispose()
    _out(f"created admin '{username}' (id {user_id})")
    return 0


def cmd_admin_exists(settings: Settings, _: argparse.Namespace) -> int:
    """Exit 0 if an active admin account exists, 1 if none (install.ps1 asks before prompting for a password)."""
    db = make_database(settings.effective_database_url)
    try:
        with db.read_sessions() as s:
            found = (
                s.execute(select(User.id).where(User.role == Role.ADMIN.value, User.is_active.is_(True))).first()
                is not None
            )
    finally:
        db.dispose()
    _out("an admin account exists" if found else "no admin account")
    return 0 if found else 1


def cmd_admin_reset_password(settings: Settings, args: argparse.Namespace) -> int:
    """Offline lock-out recovery (Phase 9): set a new password for an existing account from the server console.

    Only a server administrator can run it (it needs the protected env file and the DPAPI keys, i.e. the
    ACLs of install.ps1 step 10), so it is not a network back door. It revokes every session of the account,
    optionally re-activates it, and lifts the lock-outs that keep its owner out: the account's own, and the
    per-IP lock-out of every address that failed to log in AS THIS ACCOUNT inside the lock-out window (the
    owner mistyping 5 times locks their workstation's IP too, and a login success does not reset an IP).
    Only failed/unresolved attempt rows of the current window are removed (they are throttling state; the
    audit log keeps every login). Audited as `user.password.reset` by `local-console` with the counts.
    """
    username = args.username.strip().lower()
    password = _read_password(args)
    try:
        check_policy(password, username, settings.password_min_length)
    except PasswordPolicyError as exc:
        raise StartupError(f"password policy: {exc}") from None
    pw_hash, now = hash_password(password, settings.bcrypt_rounds), utcnow()
    writer, db = _writer(settings)

    window_start = now - timedelta(minutes=settings.login_lockout_minutes)
    failed = or_(LoginAttempt.success.is_(False), LoginAttempt.success.is_(None))

    def work(s: Session) -> tuple[int, bool, int, int]:
        user = s.execute(select(User).where(User.username == username)).scalar_one_or_none()
        if user is None:
            raise StartupError(f"no account '{username}' (the accounts are listed in the UI under Users)")
        user.password_hash, user.password_changed_at, user.updated_at = pw_hash, now, now
        reactivated = bool(args.reactivate and not user.is_active)
        if reactivated:
            user.is_active = True
        revoked = sessions.revoke_all_for_user(s, user.id, now)
        recent = (LoginAttempt.at >= window_start) & failed
        ips = sorted(set(s.execute(select(LoginAttempt.ip).where(recent, LoginAttempt.username == username)).scalars()))
        blocking = recent & or_(LoginAttempt.username == username, LoginAttempt.ip.in_(ips))
        cleared = int(s.execute(select(func.count(LoginAttempt.id)).where(blocking)).scalar_one())
        s.execute(delete(LoginAttempt).where(blocking))
        audit.record(
            s,
            action="user.password.reset",
            actor=None,
            actor_username="local-console",
            ip="local",
            target_type="user",
            target_id=user.id,
            details={
                "via": "cli",
                "sessions_revoked": revoked,
                "reactivated": reactivated,
                "lockout_rows_cleared": cleared,
                "lockout_ips_cleared": ips,
            },
        )
        return revoked, user.is_active, user.id, cleared

    try:
        revoked, active, user_id, cleared = writer.run(work)
    finally:
        writer.stop()
        db.dispose()
    _out(
        f"password reset for '{username}' (id {user_id}); {revoked} session(s) signed out; "
        f"{cleared} failed login(s) cleared (lock-out lifted)"
    )
    if not active:
        _out("NOTE: the account is deactivated - add --reactivate to sign in with it")
    return 0


def cmd_retention(settings: Settings, _: argparse.Namespace) -> int:
    writer, db = _writer(settings)
    try:
        r = run_retention(settings, writer, utcnow())
    finally:
        writer.stop()
        db.dispose()
    _out(f"retention cutoff {r.cutoff.isoformat()}: files {r.files_deleted}, login attempts {r.login_attempts_deleted}")
    return 0


def cmd_config_check(settings: Settings, _: argparse.Namespace) -> int:
    load_keys(settings)
    db = make_database(settings.effective_database_url)
    try:
        migrate.assert_at_head(db.write_engine)
    finally:
        db.dispose()
    _out(f"OK: env={settings.env} data_dir={settings.data_dir} keys=loaded schema=head")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m lifeguard", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="group", required=True)
    keys = sub.add_parser("keys").add_subparsers(dest="cmd", required=True)
    k = keys.add_parser("init")
    k.add_argument("--machine", action="store_true", help="DPAPI machine scope instead of the current account")
    k.set_defaults(fn=cmd_keys_init)
    db = sub.add_parser("db").add_subparsers(dest="cmd", required=True)
    db.add_parser("upgrade").set_defaults(fn=cmd_db_upgrade)
    db.add_parser("backup").set_defaults(fn=cmd_db_backup)
    admin = sub.add_parser("admin").add_subparsers(dest="cmd", required=True)
    a = admin.add_parser("create")
    a.add_argument("--username", required=True)
    a.add_argument("--password-stdin", action="store_true")
    a.set_defaults(fn=cmd_admin_create)
    admin.add_parser("exists").set_defaults(fn=cmd_admin_exists)
    r = admin.add_parser("reset-password", help="offline lock-out recovery for any account (server console only)")
    r.add_argument("--username", required=True)
    r.add_argument("--password-stdin", action="store_true")
    r.add_argument("--reactivate", action="store_true", help="also re-activate a deactivated account")
    r.set_defaults(fn=cmd_admin_reset_password)
    ret = sub.add_parser("retention").add_subparsers(dest="cmd", required=True)
    ret.add_parser("run").set_defaults(fn=cmd_retention)
    cfg = sub.add_parser("config").add_subparsers(dest="cmd", required=True)
    cfg.add_parser("check").set_defaults(fn=cmd_config_check)
    return p


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = settings or load_settings()
    cfg.ensure_dirs()
    configure_logging(cfg.log_dir, cfg.log_level, cfg.log_max_bytes, cfg.log_backups, to_stderr=False)
    try:
        code: int = args.fn(cfg, args)
    except StartupError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    return code
