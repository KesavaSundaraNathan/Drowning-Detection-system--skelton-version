"""Outbox dispatcher (plan Phase 5): at-least-once delivery with retry and backoff, persisted across restarts.

One thread PER CHANNEL, so a slow SMTP server can never delay the webhook or ntfy. Each thread:
  * reads its due rows (status "pending", next_attempt_at <= now), alarms first, oldest first;
  * sends one row; success -> "sent"; failure -> attempts+1, next_attempt_at = now + backoff
    (base * 2^(attempts-1), capped, +-20 % jitter), and the channel itself cools down for the same
    backoff (a dead server is not hammered row by row);
  * rows older than outbox_max_age_h are marked "dead" and raise an OUTBOX_DEAD FAULT.
A channel failing `channel_down_after` times in a row raises CHANNEL_DOWN:<channel> (delivered on the
OTHER channels); its next success clears it (FAULT_CLEARED notice).

At-least-once: a row is marked "sent" only after the channel accepted it. A crash between the two
re-sends it after restart; receivers de-duplicate on the event id (webhook body / header, e-mail
X-LifeGuard-Event, ntfy message text).
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from lifeguard.alerts import outbox
from lifeguard.alerts.channels import Notification, send_one
from lifeguard.core import events
from lifeguard.core.clock import utcnow
from lifeguard.core.config import Settings
from lifeguard.db.models import AlertOutbox
from lifeguard.db.session import Database
from lifeguard.db.writer import DBWriter
from lifeguard.vision.faults import Fault

logger = logging.getLogger(__name__)

CHANNEL_DOWN = "CHANNEL_DOWN"
OUTBOX_DEAD = "OUTBOX_DEAD"
BATCH = 20

Sender = Callable[[Settings, str, Notification], None]
FaultSink = Callable[[Fault], Any]
ClearSink = Callable[[str, str], Any]


def backoff_s(attempts: int, base: float, cap: float, rng: random.Random | None = None) -> float:
    """Delay after the `attempts`-th failure: base * 2^(attempts-1), capped, +-20 % jitter."""
    raw = min(cap, base * (2 ** max(0, min(attempts - 1, 30))))
    r = rng or random
    return float(raw * (0.8 + 0.4 * r.random()))


@dataclass
class ChannelHealth:
    consecutive_failures: int = 0
    down: bool = False
    sent: int = 0
    failed_attempts: int = 0
    last_success_at: datetime | None = None
    last_error: str | None = None
    cooldown_until: float = 0.0


class Dispatcher:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        writer: DBWriter,
        *,
        fault_sink: FaultSink,
        clear_sink: ClearSink,
        sender: Sender = send_one,
        channels: tuple[str, ...] | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.writer = writer
        self.fault_sink = fault_sink
        self.clear_sink = clear_sink
        self.sender = sender
        self.channels = tuple(settings.channels_configured) if channels is None else channels
        self.health = {ch: ChannelHealth() for ch in self.channels}
        self._stop = threading.Event()
        self._wake = {ch: threading.Event() for ch in self.channels}
        self._threads: list[threading.Thread] = []
        self._unsubscribe: Callable[[], None] | None = None
        self.rng = random.Random()  # noqa: S311  # nosec B311 - jitter only

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        self._stop.clear()
        self._unsubscribe = events.subscribe(self._on_event)
        for ch in self.channels:
            t = threading.Thread(target=self._loop, args=(ch,), name=f"outbox-{ch}", daemon=True)
            t.start()
            self._threads.append(t)
        logger.info("outbox dispatcher started for %s", list(self.channels) or "NO channel")

    def stop(self) -> None:
        self._stop.set()
        if self._unsubscribe is not None:
            self._unsubscribe()
        for ev in self._wake.values():
            ev.set()
        for t in self._threads:
            t.join(timeout=15)

    @property
    def alive(self) -> bool:
        return all(t.is_alive() for t in self._threads)

    def wake(self) -> None:
        for ev in self._wake.values():
            ev.set()

    def _on_event(self, event: dict[str, Any]) -> None:
        if event.get("type") in ("alert.created", "fault.cleared", "alert.escalated", "outbox.enqueued"):
            self.wake()

    # ── per-channel loop ──────────────────────────────────────────────────────
    def _loop(self, channel: str) -> None:
        while not self._stop.is_set():
            try:
                self.run_once(channel)
            except Exception:
                logger.exception("outbox %s round failed", channel)
            self._wake[channel].wait(self.settings.outbox_poll_s)
            self._wake[channel].clear()

    def due(self, channel: str, now: datetime) -> list[AlertOutbox]:
        order = case(outbox.KIND_ORDER, value=AlertOutbox.kind, else_=9)
        with self.db.read_sessions() as rs:
            rows = list(
                rs.execute(
                    select(AlertOutbox)
                    .where(
                        AlertOutbox.status == "pending",
                        AlertOutbox.channel == channel,
                        AlertOutbox.next_attempt_at <= now,
                    )
                    .order_by(order, AlertOutbox.id)
                    .limit(BATCH)
                ).scalars()
            )
            for r in rows:
                rs.expunge(r)
            return rows

    def run_once(self, channel: str, now: datetime | None = None) -> int:
        """One delivery round for `channel`; returns rows sent (public for tests)."""
        h = self.health[channel]
        if time.monotonic() < h.cooldown_until:
            return 0
        now = now or utcnow()
        sent = 0
        max_age = timedelta(hours=self.settings.outbox_max_age_h)
        for row in self.due(channel, now):
            if self._stop.is_set():
                break
            if now - row.created_at > max_age:
                self._mark_dead(row)
                continue
            note = Notification.from_dict(row.payload)
            try:
                self.sender(self.settings, channel, note)
            except Exception as exc:  # noqa: BLE001 - any send failure is retried; error kept on the row
                self._failed(row, channel, exc)
                break  # the channel is unhealthy: leave the rest for after its cooldown
            self._sent(row, channel)
            sent += 1
        return sent

    # ── bookkeeping ───────────────────────────────────────────────────────────
    def _sent(self, row: AlertOutbox, channel: str) -> None:
        now = utcnow()

        def work(db: Session) -> None:
            r = db.get(AlertOutbox, row.id)
            if r is not None:
                r.status, r.sent_at, r.attempts, r.last_error = "sent", now, r.attempts + 1, None

        self.writer.run(work)
        h = self.health[channel]
        was_down = h.down
        h.consecutive_failures, h.down, h.sent, h.last_success_at = 0, False, h.sent + 1, now
        h.cooldown_until = 0.0
        events.publish({"type": "outbox.sent", "channel": channel, "event_id": row.event_id, "kind": row.kind})
        if was_down:
            logger.warning("channel %s is delivering again", channel)
            self.clear_sink(CHANNEL_DOWN, f"channel:{channel}")

    def _failed(self, row: AlertOutbox, channel: str, exc: BaseException) -> None:
        h = self.health[channel]
        h.consecutive_failures += 1
        h.failed_attempts += 1
        h.last_error = f"{type(exc).__name__}: {exc}"[:500]
        delay = backoff_s(
            row.attempts + 1, self.settings.outbox_retry_base_s, self.settings.outbox_retry_max_s, self.rng
        )
        h.cooldown_until = time.monotonic() + delay
        retry_at = utcnow() + timedelta(seconds=delay)

        def work(db: Session) -> None:
            r = db.get(AlertOutbox, row.id)
            if r is not None:
                r.attempts, r.next_attempt_at, r.last_error = r.attempts + 1, retry_at, h.last_error

        self.writer.run(work)
        logger.warning(
            "outbox %s: %s %s attempt %d failed (%s); retry in %.1f s",
            channel,
            row.kind,
            row.event_id,
            row.attempts + 1,
            h.last_error,
            delay,
        )
        if h.consecutive_failures >= self.settings.channel_down_after and not h.down:
            h.down = True
            self.fault_sink(
                Fault(
                    CHANNEL_DOWN,
                    f"channel:{channel}",
                    f"alert channel {channel} failed {h.consecutive_failures} times in a row: {h.last_error}",
                    {"channel": channel},
                )
            )

    def _mark_dead(self, row: AlertOutbox) -> None:
        def work(db: Session) -> None:
            r = db.get(AlertOutbox, row.id)
            if r is not None:
                r.status = "dead"

        self.writer.run(work)
        logger.error(
            "outbox %s: %s %s undeliverable for %g h - DEAD",
            row.channel,
            row.kind,
            row.event_id,
            self.settings.outbox_max_age_h,
        )
        if row.kind != outbox.SELF_TEST:
            self.fault_sink(
                Fault(
                    OUTBOX_DEAD,
                    f"channel:{row.channel}",
                    f"{row.kind} {row.event_id} could not be delivered on {row.channel} within "
                    f"{self.settings.outbox_max_age_h:g} h",
                    {"event_id": row.event_id},
                )
            )

    # ── status ────────────────────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        now = utcnow()
        with self.db.read_sessions() as rs:
            pend = dict(
                rs.execute(
                    select(AlertOutbox.channel, func.min(AlertOutbox.created_at))
                    .where(AlertOutbox.status == "pending")
                    .group_by(AlertOutbox.channel)
                ).all()
            )
            counts = dict(
                rs.execute(
                    select(AlertOutbox.channel, func.count())
                    .where(AlertOutbox.status == "pending")
                    .group_by(AlertOutbox.channel)
                ).all()
            )
        out: dict[str, Any] = {"alive": self.alive, "channels": {}}
        for ch in self.channels:
            h = self.health[ch]
            oldest = pend.get(ch)
            out["channels"][ch] = {
                "pending": int(counts.get(ch, 0)),
                "oldest_pending_s": None if oldest is None else round((now - oldest).total_seconds(), 1),
                "consecutive_failures": h.consecutive_failures,
                "down": h.down,
                "sent": h.sent,
                "last_success_at": h.last_success_at.isoformat(timespec="seconds") if h.last_success_at else None,
                "last_error": h.last_error,
            }
        return out
