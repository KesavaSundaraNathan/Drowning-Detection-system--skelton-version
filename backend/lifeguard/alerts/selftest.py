"""Daily channel self-test (plan Phase 5).

Every day at `self_test_time` (site local time) - and on demand (POST /api/v1/system/self-test) - a
SELF_TEST message is queued on every configured channel through the normal outbox, so the test
exercises the real delivery path. After `self_test_timeout_s` each channel needs a DELIVERY RECEIPT:

  webhook  the receiver answered 2xx to the signed POST   (row "sent")
  smtp     the server accepted every recipient           (row "sent")
  ntfy     row "sent" AND the message is readable back from the topic (ntfy JSON poll API)

A channel without a receipt raises CHANNEL_SELF_TEST_FAILED:<channel> (delivered on the others);
a later passing self-test clears it. "Already ran today" is read from the outbox itself (a SELF_TEST
row created today), so a restart does not double-send.
"""

from __future__ import annotations

import logging
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from lifeguard.alerts import outbox
from lifeguard.alerts.channels import Notification
from lifeguard.alerts.channels.ntfy import verify_ntfy
from lifeguard.core import events
from lifeguard.core.clock import utcnow
from lifeguard.core.config import Settings
from lifeguard.core.schedule import parse_hhmm
from lifeguard.db.models import AlertOutbox
from lifeguard.db.session import Database
from lifeguard.db.writer import DBWriter
from lifeguard.vision.faults import Fault

logger = logging.getLogger(__name__)

SELF_TEST_FAILED = "CHANNEL_SELF_TEST_FAILED"
PERIOD_S = 15.0

Verifier = Callable[[Settings, str, int, int], bool]


@dataclass
class _Run:
    event_id: str
    started: datetime
    results: dict[str, str | None] = field(default_factory=dict)  # channel -> None (ok) | reason
    done: bool = False


class SelfTest:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        writer: DBWriter,
        *,
        fault_sink: Callable[[Fault], Any],
        clear_sink: Callable[[str, str], Any],
        verifier: Verifier = verify_ntfy,
        local_now: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.writer = writer
        self.fault_sink = fault_sink
        self.clear_sink = clear_sink
        self.verifier = verifier
        self.local_now = local_now or (lambda: utcnow().astimezone())
        self.runs: list[_Run] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="self-test", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _loop(self) -> None:
        while not self._stop.wait(PERIOD_S):
            try:
                if self.settings.self_test_is_enabled and self.due():
                    self.run()
                self.check()
            except Exception:
                logger.exception("self-test step failed")

    # ── schedule ──────────────────────────────────────────────────────────────
    def due(self, local_now: datetime | None = None) -> bool:
        local = local_now or self.local_now()
        at = parse_hhmm(self.settings.self_test_time)
        if local.time().replace(second=0, microsecond=0) < at:
            return False
        day_start = local.replace(hour=0, minute=0, second=0, microsecond=0)
        with self.db.read_sessions() as rs:
            ran = rs.execute(
                select(AlertOutbox.id)
                .where(AlertOutbox.kind == outbox.SELF_TEST, AlertOutbox.created_at >= day_start)
                .limit(1)
            ).first()
        return ran is None

    # ── run / check ───────────────────────────────────────────────────────────
    def run(self, now: datetime | None = None) -> str:
        """Queue a SELF_TEST on every channel; returns its event id."""
        now = now or utcnow()
        event_id = f"selftest-{now.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(3)}"  # unique even within 1 s
        note = Notification(
            kind=outbox.SELF_TEST,
            code="SELF_TEST",
            title="daily channel self-test (no action needed)",
            message=f"Life Guard channel self-test {event_id}. No action needed.",
            severity="info",
            source="api",
            at=now.isoformat(timespec="seconds"),
            event_id=event_id,
        )

        def work(db: Session) -> int:
            return outbox.enqueue(db, note, outbox.SELF_TEST, now=now)

        n = self.writer.run(work)
        self.runs.append(_Run(event_id, now))
        self.runs = self.runs[-30:]
        logger.info("channel self-test %s queued on %d channel(s)", event_id, n)
        events.publish({"type": "outbox.enqueued", "event_id": event_id})
        return event_id

    def check(self, now: datetime | None = None) -> list[_Run]:
        """Evaluate every finished-or-timed-out run once; returns the runs evaluated now."""
        now = now or utcnow()
        evaluated = []
        for run in self.runs:
            if run.done:
                continue
            rows = self._rows(run.event_id)
            timed_out = now - run.started >= timedelta(seconds=self.settings.self_test_timeout_s)
            if not timed_out and any(r.status != "sent" for r in rows):
                continue  # still delivering
            for r in rows:
                run.results[r.channel] = self._receipt(r, run)
            run.done = True
            evaluated.append(run)
            for ch, problem in run.results.items():
                if problem is None:
                    self.clear_sink(SELF_TEST_FAILED, f"channel:{ch}")
                else:
                    self.fault_sink(
                        Fault(
                            SELF_TEST_FAILED,
                            f"channel:{ch}",
                            f"daily self-test {run.event_id}: channel {ch} gave no delivery receipt ({problem})",
                            {"channel": ch, "event_id": run.event_id},
                        )
                    )
            logger.log(
                logging.INFO if all(v is None for v in run.results.values()) else logging.ERROR,
                "self-test %s results: %s",
                run.event_id,
                run.results,
            )
        return evaluated

    def _rows(self, event_id: str) -> list[AlertOutbox]:
        with self.db.read_sessions() as rs:
            rows = list(rs.execute(select(AlertOutbox).where(AlertOutbox.event_id == event_id)).scalars())
            for r in rows:
                rs.expunge(r)
            return rows

    def _receipt(self, row: AlertOutbox, run: _Run) -> str | None:
        if row.status != "sent":
            return f"not delivered within {self.settings.self_test_timeout_s:g} s ({row.last_error or row.status})"
        if row.channel == "ntfy":
            try:
                since = int(run.started.timestamp()) - 5
                if not self.verifier(self.settings, run.event_id, since, 0):
                    return "accepted but not readable back from the topic"
            except Exception as exc:  # noqa: BLE001 - a failed receipt check IS the failure
                return f"receipt check failed: {type(exc).__name__}: {exc}"
        return None

    def last(self) -> dict[str, Any] | None:
        done = [r for r in self.runs if r.done]
        if not done:
            return None
        r = done[-1]
        return {"event_id": r.event_id, "started": r.started.isoformat(timespec="seconds"), "results": r.results}
