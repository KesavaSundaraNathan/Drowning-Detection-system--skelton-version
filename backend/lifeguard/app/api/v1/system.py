"""/api/v1/system: out-of-band delivery status and the manual channel self-test (Phase 5).

GET  /system/delivery    supervisor+  per channel: pending rows, oldest pending age, consecutive
                                      failures, DOWN flag, last success; escalation tiers; last self-test
POST /system/self-test   admin        queue a SELF_TEST on every channel now (audited); the result
                                      (delivery receipts) appears under /system/delivery and, on
                                      failure, as a CHANNEL_SELF_TEST_FAILED FAULT
GET  /system/release     any role     release-gate status for the permanent ASSIST / PILOT banner. Until a
                                      signed acceptance report from the site's own footage passes (plan
                                      Phase 7, eval/release_gate.py) the answer is ASSIST_PILOT.
Channel URLs, topics, addresses and secrets are never returned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.orm import Session

from lifeguard.app.deps import Admin, AnyUser, Ctx, Supervisor, client_ip
from lifeguard.core import audit

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/delivery")
def delivery_status(ctx: Ctx, _: Supervisor) -> dict[str, Any]:
    if ctx.delivery is None:
        raise HTTPException(503, "delivery_not_running")
    status: dict[str, Any] = ctx.delivery.status()
    return status


@router.post("/self-test", status_code=202)
def self_test(request: Request, ctx: Ctx, me: Admin) -> dict[str, Any]:
    if ctx.delivery is None:
        raise HTTPException(503, "delivery_not_running")
    if not ctx.settings.channels_configured:
        raise HTTPException(409, "no_channel_configured")
    event_id = ctx.delivery.selftest.run()
    ip = client_ip(request)

    def work(db: Session) -> None:
        audit.record(db, action="system.self_test", actor=me, ip=ip, target_type="outbox", target_id=event_id)

    ctx.writer.run(work)
    return {"event_id": event_id, "channels": ctx.settings.channels_configured}


@router.get("/release")
def release_status(ctx: Ctx, _: AnyUser) -> dict[str, Any]:
    """Phase 8: the signed release gate (lifeguard.eval.release_gate). ASSIST / PILOT unless a report signed
    with this site's key, bound to the detector + thresholds this server RUNS, says the gate passed."""
    from lifeguard.eval import release_gate
    from lifeguard.vision import config as vision_config

    s = ctx.settings
    det = Path(s.detector_model) if s.detector_model else vision_config.MODELS_DIR / vision_config.DEFAULT_DET_MODEL
    return release_gate.status(s.data_dir, s.keys_dir, det)
