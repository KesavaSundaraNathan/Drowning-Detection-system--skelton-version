"""E2E-only: raise behaviour alerts in the E2E stack's database through the real alert writer.

    echo '{"fires": [{"camera": "Cam A", "track": 1, "scenario": "ACTIVE"}]}' | python tools/e2e_inject.py

Run by tools/e2e_stack.py (POST /alerts/inject on its loopback control port) with the stack's environment, so
it writes to the scratch data dir of that run only. Each fire goes through `alerts.writer.record_fire` exactly
like a live camera's (its event on that camera and track - Phase 5, cameras are independent - outbox, audit).
Cameras are looked up by NAME. Prints {"alert_ids": [...]}.
Never part of the product: the installed services have no way to inject alerts.
"""

from __future__ import annotations

import json
import sys

from sqlalchemy import select

from lifeguard.alerts.writer import record_fire
from lifeguard.cli import _writer
from lifeguard.core.config import load_settings
from lifeguard.db.models import Camera
from lifeguard.vision.behaviour.classifier import BehaviorResult
from lifeguard.vision.scene.hysteresis import AlertFire


def main() -> int:
    req = json.loads(sys.stdin.read())
    settings = load_settings()
    writer, db = _writer(settings)
    ids = []
    try:
        with db.read_sessions() as rs:
            cams = {c.name: c.id for c in rs.execute(select(Camera)).scalars()}
        for f in req["fires"]:
            track = int(f.get("track", 1))
            res = BehaviorResult(
                track_id=track,
                category="SWIMMER",
                scenario=f["scenario"],
                label=f["scenario"].replace("_", " ").title(),
                is_drowning=True,
                risk=0.9,
                raw_risk=0.9,
                confidence=0.85,
                reason="E2E injected alert",
                mode="Medium",
            )
            fire = AlertFire(track, res, (0, 0, 10, 10), "Pool (calibrated)", "WATER", "Medium", first=True)
            ids.append(record_fire(writer, fire, camera_id=cams[f["camera"]], audible=True))
    finally:
        writer.stop()
    sys.stdout.write(json.dumps({"alert_ids": ids}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
