"""Out-of-band delivery in the API process (plan Phase 5): outbox dispatcher, escalation, daily self-test.

    delivery = DeliveryService(settings, db, writer); delivery.start(); ...; delivery.stop()

FAULTs raised by the delivery machinery itself (CHANNEL_DOWN, OUTBOX_DEAD, CHANNEL_SELF_TEST_FAILED)
go through `record_fault`, i.e. into the outbox like every other FAULT - the healthy channels
deliver them.
"""

from __future__ import annotations

import logging
from typing import Any

from lifeguard.alerts import outbox
from lifeguard.alerts.dispatcher import Dispatcher
from lifeguard.alerts.escalation import Escalator
from lifeguard.alerts.faults import clear_fault, record_fault
from lifeguard.alerts.selftest import SelfTest
from lifeguard.core.config import Settings
from lifeguard.db.session import Database
from lifeguard.db.writer import DBWriter
from lifeguard.vision.faults import Fault

logger = logging.getLogger(__name__)


class DeliveryService:
    def __init__(self, settings: Settings, db: Database, writer: DBWriter) -> None:
        self.settings = settings
        self.writer = writer
        self.dispatcher = Dispatcher(settings, db, writer, fault_sink=self.fault, clear_sink=self.clear)
        self.escalator = Escalator(settings, db, writer)
        self.selftest = SelfTest(settings, db, writer, fault_sink=self.fault, clear_sink=self.clear)

    def fault(self, fault: Fault) -> int:
        return record_fault(self.writer, fault)

    def clear(self, code: str, source_id: str) -> list[int]:
        return clear_fault(self.writer, code, source_id)

    def start(self) -> None:
        outbox.configure(self.settings)
        if not self.settings.channels_configured:
            logger.warning("NO out-of-band channel configured: alerts and FAULTs are NOT delivered off-site")
        self.dispatcher.start()
        self.escalator.start()
        self.selftest.start()

    def stop(self) -> None:
        self.selftest.stop()
        self.escalator.stop()
        self.dispatcher.stop()

    @property
    def alive(self) -> bool:
        return self.dispatcher.alive and self.escalator.alive

    def status(self) -> dict[str, Any]:
        st = self.dispatcher.stats()
        oldest = [c["oldest_pending_s"] for c in st["channels"].values() if c["oldest_pending_s"] is not None]
        return {
            "channels": st["channels"],
            "dispatcher_alive": st["alive"],
            "escalator_alive": self.escalator.alive,
            "escalations": self.escalator.escalations,
            "oldest_pending_s": max(oldest) if oldest else None,
            "channels_down": sorted(ch for ch, c in st["channels"].items() if c["down"]),
            "escalation_tiers": self.settings.escalation_tiers,
            "self_test": {
                "enabled": self.settings.self_test_is_enabled,
                "time": self.settings.self_test_time,
                "last": self.selftest.last(),
            },
        }
