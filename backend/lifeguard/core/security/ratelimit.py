"""Login throttling: per user AND per client IP, N failures in the window -> lockout.

Both limits are evaluated atomically in the DB-writer thread. Each attempt RESERVES a row
(success=None) before the password is checked; an unresolved reservation counts as a failure.
So parallel guessing cannot slip past the limit (at most `max_failures` guesses are ever in
flight), and a crash between reservation and result fails safe (it counts against the caller).

* user limit: failures for that username since its last successful login, inside the window;
  a successful login of that user resets it;
* IP limit: failures from that IP inside the window, NOT reset by a success - otherwise an
  attacker holding one valid account could reset the counter between guesses at other accounts.
The lockout lasts until the oldest counted failure leaves the window (default 15 minutes after
the 5th failure); attempts made while locked are audited but not counted, so they do not
extend it. (A per-user lockout lets an attacker lock a legitimate user out for 15 minutes:
accepted trade-off, documented in docs/SECURITY.md.)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.orm import InstrumentedAttribute, Session

from lifeguard.db.models import LoginAttempt


@dataclass(frozen=True)
class Reservation:
    attempt_id: int | None  # None -> locked, nothing reserved
    locked_by: str | None = None  # "user" | "ip"
    retry_after_s: int = 0


def _failures(
    session: Session, column: InstrumentedAttribute[str], value: str, since: datetime, after_id: int
) -> tuple[int, datetime | None]:
    q = select(func.count(LoginAttempt.id), func.min(LoginAttempt.at)).where(
        column == value,
        LoginAttempt.at >= since,
        LoginAttempt.id > after_id,
        or_(LoginAttempt.success.is_(False), LoginAttempt.success.is_(None)),
    )
    count, oldest = session.execute(q).one()
    return int(count), oldest


def reserve(
    session: Session, username: str, ip: str, now: datetime, max_failures: int, window: timedelta
) -> Reservation:
    window_start = now - window
    # "Since the last success" is decided by attempt id, not timestamp: ids are strictly ordered by
    # the single writer, timestamps can tie (coarse clocks) and would let pre-success failures count.
    last_ok_id = session.execute(
        select(func.max(LoginAttempt.id)).where(LoginAttempt.username == username, LoginAttempt.success.is_(True))
    ).scalar_one_or_none()
    for kind, column, value, after_id in (
        ("user", LoginAttempt.username, username, last_ok_id or 0),
        ("ip", LoginAttempt.ip, ip, 0),
    ):
        count, oldest = _failures(session, column, value, window_start, after_id)
        if count >= max_failures and oldest is not None:
            retry = max(1, int((oldest + window - now).total_seconds()))
            return Reservation(None, kind, retry)
    row = LoginAttempt(at=now, username=username, ip=ip, success=None)
    session.add(row)
    session.flush()
    return Reservation(row.id)


def resolve(session: Session, attempt_id: int, success: bool) -> None:
    row = session.get(LoginAttempt, attempt_id)
    if row is not None:
        row.success = success


def purge_older_than(session: Session, cutoff: datetime) -> int:
    rows = session.execute(select(LoginAttempt).where(LoginAttempt.at < cutoff)).scalars().all()
    for r in rows:
        session.delete(r)
    return len(rows)
