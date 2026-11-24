"""Everything a request handler needs, built once at startup and held on ``app.state.ctx``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lifeguard.core.config import Settings
from lifeguard.core.security.keys import ServerKeys
from lifeguard.core.security.passwords import DummyHasher
from lifeguard.db.session import Database
from lifeguard.db.writer import DBWriter

if TYPE_CHECKING:
    from lifeguard.app.delivery import DeliveryService
    from lifeguard.app.live_startup import LiveService


@dataclass(frozen=True)
class AppContext:
    settings: Settings
    keys: ServerKeys
    db: Database
    writer: DBWriter
    dummy_hasher: DummyHasher
    live: LiveService | None = None  # live runtime + heartbeat + supervisor watchdog (Phase 4)
    delivery: DeliveryService | None = None  # outbox dispatcher, escalation, self-test (Phase 5)
