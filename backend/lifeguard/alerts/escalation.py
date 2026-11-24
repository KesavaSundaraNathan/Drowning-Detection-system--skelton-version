"""Acknowledgement-driven escalation (plan Phase 5).

An AUDIBLE alert (and an uncleared FAULT, unless escalate_faults=false) still "pending" (not
acknowledged / resolved / false alarm) after ESCALATE_AFTER_S goes to the next contact tier on EVERY
channel; each further ESCALATE_AFTER_S moves one tier up. At the last tier the alarm is repeated
every escalation_repeat_s until somebody acknowledges it - an alarm never goes quiet on its own.

`escalation_level` counts the escalations already sent; the tier of escalation n is min(n, tiers-1).
Escalation n, its outbox rows and the "alert.escalated" event are one transaction + one publish.
Muted (schedule) alerts are never escalated.

[Phase 6] Last-tier repeats are coalesced per camera (camera-less faults: one "system" group): when any repeating
alarm of a camera is due, EVERY repeating alarm of that camera is repeated now in one transaction - pulled forward,
never delayed - with one outbox note listing them (outbox.note_for_repeat). Each alarm still gets its level, its
escalated_at and its audit row; a group of one keeps its own note. Tier climbs stay per alarm. Soak R5: 4 021
unacknowledged alarms had cost one note each per repeat (1 400 outbox rows a minute -> OUTBOX_BACKLOG).
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from lifeguard.alerts import outbox
from lifeguard.core import audit, events
from lifeguard.core.clock import utcnow
from lifeguard.core.config import Settings
from lifeguard.db.models import Alert
from lifeguard.db.session import Database
from lifeguard.db.writer import DBWriter

logger = logging.getLogger(__name__)
PERIOD_S = 1.0


class Escalator:
    def __init__(self, settings: Settings, db: Database, writer: DBWriter) -> None:
        self.settings = settings
        self.db = db
        self.writer = writer
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.escalations = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="escalator", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop.wait(PERIOD_S):
            try:
                self.step()
            except Exception:
                logger.exception("escalation step failed")

    def _repeating(self, level: int) -> bool:
        """True when the escalation after `level` is a last-tier repeat (no tier left to climb)."""
        return level >= max(1, self.settings.escalation_tiers - 1)

    def _wait_for(self, level: int) -> timedelta:
        """Silence allowed after escalation `level` (0 = after the original alert)."""
        repeating = self._repeating(level)
        return timedelta(seconds=self.settings.escalation_repeat_s if repeating else self.settings.escalate_after_s)

    def _eligible(self) -> list[Any]:
        s = self.settings
        cond: list[Any] = [Alert.status == "pending", Alert.audible.is_(True)]
        fault_ok = Alert.cleared_at.is_(None) if s.escalate_faults else Alert.scenario != "FAULT"
        cond.append(or_(Alert.scenario != "FAULT", fault_ok))
        return cond

    def _due_rows(self, now: datetime) -> list[tuple[int, str | None, int]]:
        """(id, camera_id, level) of every alarm due for its next escalation."""
        s = self.settings
        oldest = now - timedelta(seconds=min(s.escalate_after_s, s.escalation_repeat_s))
        with self.db.read_sessions() as rs:
            rows = rs.execute(
                select(Alert.id, Alert.camera_id, Alert.created_at, Alert.escalated_at, Alert.escalation_level)
                .where(*self._eligible(), Alert.created_at <= oldest)
                .order_by(Alert.id)
            ).all()
        due = []
        for aid, cam, created, escalated, level in rows:
            last = escalated or created
            if now - last >= self._wait_for(int(level)):
                due.append((int(aid), cam, int(level)))
        return due

    def candidates(self, now: datetime) -> list[int]:
        return [aid for aid, _, _ in self._due_rows(now)]

    def step(self, now: datetime | None = None) -> list[int]:
        """Escalate every due alarm once; returns the ids escalated (public for tests).

        Tier climbs go one alarm at a time; a due last-tier repeat repeats every repeating alarm of its camera.
        """
        now = now or utcnow()
        done: list[int] = []
        repeat_cams: dict[str | None, None] = {}
        for aid, cam, level in self._due_rows(now):
            if self._repeating(level):
                repeat_cams.setdefault(cam, None)
            elif self.escalate(aid, now):
                done.append(aid)
        for cam in repeat_cams:
            done.extend(self.repeat_camera(cam, now))
        return sorted(done)

    def repeat_camera(self, camera_id: str | None, now: datetime) -> list[int]:
        """One last-tier repeat for every repeating pending alarm of `camera_id` (None = camera-less), one note."""
        tier = self.settings.escalation_tiers - 1
        threshold = max(1, tier)

        def work(db: Session) -> tuple[list[int], dict[str, Any]] | None:
            cam_cond = Alert.camera_id.is_(None) if camera_id is None else Alert.camera_id == camera_id
            alarms = list(
                db.scalars(
                    select(Alert)
                    .where(*self._eligible(), cam_cond, Alert.escalation_level >= threshold)
                    .order_by(Alert.id)
                )
            )
            if not alarms:
                return None  # all acknowledged / cleared since the read
            for a in alarms:
                a.escalation_level += 1
                a.escalated_at = now
                audit.record(
                    db,
                    action="alert.escalated",
                    actor=None,
                    actor_username="system",
                    ip="local",
                    target_type="alert",
                    target_id=a.id,
                    details={"level": a.escalation_level, "tier": tier, "repeat_group": len(alarms)},
                )
            if len(alarms) == 1:
                note = outbox.note_for_alert(alarms[0], outbox.ESCALATION, tier=tier)
            else:
                note = outbox.note_for_repeat(alarms, tier=tier, now=now)
            outbox.enqueue(db, note, outbox.ESCALATION, alert_id=alarms[0].id, now=now)
            ev = outbox.alert_event(alarms[0], "alert.escalated")
            ev["tier"] = tier
            ev["alert_ids"] = [a.id for a in alarms]
            return [a.id for a in alarms], ev

        result = self.writer.run(work)
        if result is None:
            return []
        ids, event = result
        self.escalations += len(ids)
        logger.warning("%d alarm(s) on %s UNACKNOWLEDGED - repeated at tier %s", len(ids), camera_id or "system", tier)
        events.publish(event)
        return ids

    def escalate(self, alert_id: int, now: datetime) -> bool:
        tiers = self.settings.escalation_tiers

        def work(db: Session) -> dict[str, Any] | None:
            a = db.get(Alert, alert_id)
            # re-check inside the transaction: an acknowledgement may have just landed
            if a is None or a.status != "pending" or not a.audible:
                return None
            if a.scenario == "FAULT" and (a.cleared_at is not None or not self.settings.escalate_faults):
                return None
            if now - (a.escalated_at or a.created_at) < self._wait_for(a.escalation_level):
                return None
            a.escalation_level += 1
            a.escalated_at = now
            tier = min(a.escalation_level, tiers - 1)
            audit.record(
                db,
                action="alert.escalated",
                actor=None,
                actor_username="system",
                ip="local",
                target_type="alert",
                target_id=a.id,
                details={"level": a.escalation_level, "tier": tier},
            )
            outbox.enqueue(
                db, outbox.note_for_alert(a, outbox.ESCALATION, tier=tier), outbox.ESCALATION, alert_id=a.id, now=now
            )
            ev = outbox.alert_event(a, "alert.escalated")
            ev["tier"] = tier
            return ev

        event = self.writer.run(work)
        if event is None:
            return False
        self.escalations += 1
        logger.warning("alert %d UNACKNOWLEDGED - escalated to tier %s", alert_id, event["tier"])
        events.publish(event)
        return True
