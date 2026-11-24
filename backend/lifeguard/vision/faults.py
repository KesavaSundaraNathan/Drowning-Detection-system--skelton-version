"""FAULT records raised by the vision pipeline (plan Phase 3: no silent degradation).

A FAULT means "this camera / this system is NOT MONITORING" - never a degraded best effort. The
vision package only DESCRIBES the fault; where it goes is the caller's FaultSink:

  * the API (lifeguard.alerts.faults.record_fault) persists it as a FAULT alert row and logs it;
  * Phase 5 delivers it out of band (outbox -> webhook / ntfy / SMTP);
  * the default sink only logs at ERROR level (a library must never swallow a fault).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Fault codes
DETECTOR_UNAVAILABLE = "DETECTOR_UNAVAILABLE"  # model missing / corrupt / repeated frame failures
PROVIDER_NOT_ACTIVE = "PROVIDER_NOT_ACTIVE"  # asked for CUDA, onnxruntime runs elsewhere

MONITORED = "MONITORED"
NOT_MONITORED = "NOT_MONITORED"


@dataclass(frozen=True)
class Fault:
    code: str
    source_id: str  # camera id or "system"
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


FaultSink = Callable[[Fault], None]


def log_fault(fault: Fault) -> None:
    logger.error("FAULT %s on %s: %s %s", fault.code, fault.source_id, fault.message, fault.detail or "")
