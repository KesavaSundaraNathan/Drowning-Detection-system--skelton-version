"""Re-export shim: AlertHysteresis / AlertFire moved to behaviour/temporal.py (Phase 4 Step 1, verbatim)."""

from lifeguard.vision.behaviour.temporal import AlertFire, AlertHysteresis

__all__ = ["AlertFire", "AlertHysteresis"]
