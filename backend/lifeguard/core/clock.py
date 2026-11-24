"""Single source of time for the server (timezone-aware UTC only).

Every expiry, lockout window and retention cutoff reads `utcnow()`, so tests can
move time deterministically with `set_clock()` instead of sleeping.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

_now: Callable[[], datetime] = lambda: datetime.now(UTC)  # noqa: E731 - replaceable in tests


def utcnow() -> datetime:
    value = _now()
    if value.tzinfo is None:
        raise RuntimeError("clock returned a naive datetime")
    return value.astimezone(UTC)


def set_clock(fn: Callable[[], datetime] | None) -> None:
    """Override the clock (tests). `None` restores the real clock."""
    global _now
    _now = fn if fn is not None else (lambda: datetime.now(UTC))
