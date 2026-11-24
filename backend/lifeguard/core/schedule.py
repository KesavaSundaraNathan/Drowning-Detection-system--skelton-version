"""Monitoring schedule windows (wall-clock local time of the site).

A schedule lists weekdays and a [start, end) time window. When ``end < start`` the window runs
overnight and BELONGS TO THE DAY IT STARTS: "Mon 22:00 -> 06:00" covers Monday 22:00-24:00 and
Tuesday 00:00-06:00.

Fixes a v2 bug (services/zone_service.schedule_is_now): v2 checked the weekday of the *current*
time for both halves of an overnight window, so Tuesday 01:00 was reported OUTSIDE the Monday-night
window, while Monday 01:00 (Sunday night's tail) was reported INSIDE it. In Phase 4 this decides
whether a siren is muted, so the difference is safety-relevant.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def parse_hhmm(value: str) -> time:
    try:
        hh, mm = value.split(":")
        t = time(int(hh), int(mm))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"time must be HH:MM, got {value!r}") from exc
    if len(hh) != 2 or len(mm) != 2:
        raise ValueError(f"time must be HH:MM, got {value!r}")
    return t


def window_contains(days: list[str], start: str, end: str, at: datetime) -> bool:
    """True if `at` (local wall-clock time) is inside the schedule window."""
    t0, t1 = parse_hhmm(start), parse_hhmm(end)
    if t0 == t1:
        raise ValueError("start and end must differ")
    today = WEEKDAYS[at.weekday()]
    yesterday = WEEKDAYS[(at - timedelta(days=1)).weekday()]
    now_t = at.time().replace(second=0, microsecond=0)
    if t0 < t1:
        return today in days and t0 <= now_t < t1
    # overnight: [start, 24:00) on a listed day, plus [00:00, end) on the following day
    return (today in days and now_t >= t0) or (yesterday in days and now_t < t1)
