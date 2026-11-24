"""Schedule muting policy (plan Phase 4).

A zone's schedule may MUTE the siren for ordinary behaviour alerts outside its monitoring window.
It never mutes:
  * SUBMERSION (a swimmer who vanished under water), including a submersion ghost;
  * FAULT (the system cannot see) - FAULTs do not pass through here at all; they are always loud.

The ported scene tracker (Phase 3, verbatim for parity) lets a schedule mute a submersion ghost;
this layer overrides that, so the rule holds without changing the vision code.

A muted zone is reported with a persistent ``MUTED`` state (live status API -> UI banner), never as
silently quiet. Only the supervisor role can create or change schedules / zone schedule links (the
schedules and zones routers require ``Role.SUPERVISOR``; enforced by tests/api/test_auth_matrix.py).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from lifeguard.vision.scene.hysteresis import AlertFire

NEVER_MUTED = frozenset({"SUBMERSION", "FAULT"})

MONITORED = "MONITORED"
MUTED = "MUTED"


def audible(fire: AlertFire) -> bool:
    """Should this alert sound / escalate? (It is persisted either way.)"""
    if fire.ghost or fire.result.scenario in NEVER_MUTED:
        return True
    return bool(fire.siren)


def zone_states(geometry: Any, now: datetime | None = None) -> list[dict[str, Any]]:
    """[{zone, zone_type, state: MONITORED|MUTED, schedule_id}] for a camera's ZoneGeometry."""
    from lifeguard.vision.scene.geometry import ZoneHit

    if geometry is None:
        return []
    out = []
    for z in geometry.zones:
        hit = ZoneHit(z, z.monitored, z.monitored, z.name, z.zone_type, z.sensitivity_multiplier, z.schedule_id)
        try:
            active = geometry.monitoring_active(hit, now)
        except Exception:  # noqa: BLE001 - a broken schedule row never mutes
            active = True
        out.append(
            {
                "zone": z.name,
                "zone_type": z.zone_type,
                "state": MONITORED if active else MUTED,
                "schedule_id": z.schedule_id,
                "never_muted": sorted(NEVER_MUTED),
            }
        )
    return out
