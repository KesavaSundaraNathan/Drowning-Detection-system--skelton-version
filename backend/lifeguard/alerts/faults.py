"""FAULT alerts: "this camera / this system is NOT MONITORING" (Phase 3; out of band from Phase 5).

record_fault: a vision Fault becomes an `alerts` row (scenario FAULT, Critical, audible), its audit
event AND its outbox rows - one transaction - then an "alert.created" event is published.

clear_fault: when the condition ends (stream back, detector rebuilt, supervisor back, channel up ...),
every OPEN fault with that code and source gets `cleared_at` and a FAULT_CLEARED notice is queued on
every channel - again in one transaction. Clearing a code that is not open is a no-op, so callers
may clear on every healthy transition (also after a restart, for faults a dead process left open).
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from lifeguard.alerts import outbox
from lifeguard.core import audit, events
from lifeguard.core.clock import utcnow
from lifeguard.db.models import Alert
from lifeguard.db.writer import DBWriter
from lifeguard.vision.faults import Fault

logger = logging.getLogger(__name__)

FAULT_SCENARIO = "FAULT"
FAULT_PRIORITY = "Critical"


def record_fault(writer: DBWriter, fault: Fault, camera_id: str | None = None) -> int:
    """Persist `fault` as a FAULT alert + audit event + outbox rows; returns the alert id."""
    now = utcnow()
    reason = f"{fault.code}: {fault.message}"[:2000]

    def work(db: Session) -> tuple[int, dict[str, object]]:
        row = Alert(
            created_at=now,
            camera_id=camera_id,
            track_id=None,
            scenario=FAULT_SCENARIO,
            priority=FAULT_PRIORITY,
            risk_score=100,
            confidence=1.0,
            zone_name=None,
            reason=reason,
            status="pending",
            audible=True,
            fault_code=fault.code[:48],
            fault_source=fault.source_id[:64],
            escalation_level=0,
        )
        db.add(row)
        db.flush()
        audit.record(
            db,
            action="fault.raised",
            actor=None,
            actor_username="system",
            ip="local",
            target_type="alert",
            target_id=row.id,
            details={"code": fault.code, "source": fault.source_id, **fault.detail},
        )
        outbox.enqueue(db, outbox.note_for_alert(row, outbox.FAULT), outbox.FAULT, alert_id=row.id, now=now)
        return int(row.id), outbox.alert_event(row, "alert.created")

    alert_id, event = writer.run(work)
    logger.error("FAULT alert %d raised: %s (%s)", alert_id, reason, fault.detail or "")
    events.publish(event)
    return alert_id


def clear_fault(writer: DBWriter, code: str, source_id: str) -> list[int]:
    """Mark every open FAULT (`code`, `source_id`) cleared and queue FAULT_CLEARED notices."""
    now = utcnow()

    def work(db: Session) -> list[tuple[int, dict[str, object]]]:
        rows = list(
            db.execute(
                select(Alert).where(
                    Alert.scenario == FAULT_SCENARIO,
                    Alert.fault_code == code,
                    Alert.fault_source == source_id,
                    Alert.cleared_at.is_(None),
                )
            ).scalars()
        )
        out = []
        for row in rows:
            row.cleared_at = now
            audit.record(
                db,
                action="fault.cleared",
                actor=None,
                actor_username="system",
                ip="local",
                target_type="alert",
                target_id=row.id,
                details={"code": code, "source": source_id},
            )
            outbox.enqueue(
                db, outbox.note_for_alert(row, outbox.FAULT_CLEARED), outbox.FAULT_CLEARED, alert_id=row.id, now=now
            )
            out.append((int(row.id), outbox.alert_event(row, "fault.cleared")))
        return out

    cleared: list[tuple[int, dict[str, object]]] = writer.run(work)
    for _id, ev in cleared:
        events.publish(ev)
    if cleared:
        logger.warning("FAULT %s on %s cleared (%d alert(s))", code, source_id, len(cleared))
    return [i for i, _ in cleared]
