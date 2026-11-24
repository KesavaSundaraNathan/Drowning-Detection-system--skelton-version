"""Vision startup checks, run by the app lifespan (plan Phase 3).

1. The active thresholds file (version + sha256 of its exact bytes) is written to the audit log at
   EVERY startup, so every alert can be traced to the thresholds that produced it.
2. Provider guard: the detector is loaded on `inference_provider`. If that fails - CUDA requested but
   onnxruntime runs on the CPU (missing CUDA/cuDNN DLLs), model missing or corrupt - a FAULT alert is
   recorded, the missing DLL list is logged, and startup is REFUSED (StartupError). There is no
   silent CPU fallback and no degraded mode.
   `vision_startup_check=false` skips the guard; that choice is itself audited at every startup.
3. (Phase 4) The live runtime starts: capacity is measured on the real GPU and every enabled camera
   is admitted or gets a CAPACITY_EXCEEDED FAULT. If the runtime itself cannot start, a
   LIVE_RUNTIME_FAILED FAULT is recorded and startup is REFUSED.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from lifeguard.alerts.faults import record_fault
from lifeguard.core import audit
from lifeguard.core.config import Settings
from lifeguard.core.errors import StartupError
from lifeguard.core.security.keys import ServerKeys
from lifeguard.db.session import Database
from lifeguard.db.writer import DBWriter

if TYPE_CHECKING:
    from lifeguard.app.live_startup import LiveService
    from lifeguard.live.preopen import PreOpener

logger = logging.getLogger(__name__)


def _audit(writer: DBWriter, action: str, target_id: str, details: dict[str, Any]) -> None:
    def work(db: Session) -> None:
        audit.record(
            db,
            action=action,
            actor=None,
            actor_username="system",
            ip="local",
            target_type="vision",
            target_id=target_id,
            details=details,
        )

    writer.run(work)


def audit_thresholds(writer: DBWriter) -> dict[str, str]:
    from lifeguard.vision.thresholds import active

    info = active().describe()
    _audit(writer, "vision.thresholds.active", info["version"], info)
    logger.info("vision thresholds %s (sha256 %s)", info["version"], info["sha256"])
    return info


def check_vision(cfg: Settings, writer: DBWriter) -> dict[str, Any]:
    """Configure the vision runtime and prove the detector runs on the requested provider."""
    from lifeguard.vision import config as vision_config
    from lifeguard.vision import pipeline

    vcfg = vision_config.from_settings(cfg)
    vision_config.configure(vcfg)
    det = vcfg.det_model_path
    if not cfg.vision_check_enabled:
        _audit(
            writer,
            "vision.startup_check.disabled",
            "provider-guard",
            {"provider": cfg.inference_provider, "detector_model": det, "pose_model": vcfg.pose_model_path},
        )
        logger.warning("vision startup check DISABLED by configuration (vision_startup_check=false)")
        return {"checked": False}
    status = pipeline.warmup_models(force=True)
    if status.get("healthy"):
        _audit(
            writer,
            "vision.provider.ok",
            cfg.inference_provider,
            {
                "provider": cfg.inference_provider,
                "active": status.get("active_provider"),
                "detector_model": det,
                "pose": status.get("pose"),
                "behaviour_model": status.get("behaviour_model"),
            },
        )
        return status
    fault = pipeline.fault_for("system", status.get("load_exception"), str(status.get("error")))
    record_fault(writer, fault)
    logger.error(
        "vision startup check failed: %s; missing DLLs: %s", fault.message, fault.detail.get("missing_dlls", [])
    )
    raise StartupError(f"detector not available on {cfg.inference_provider}: {fault.message}")


LIVE_RUNTIME_FAILED = "LIVE_RUNTIME_FAILED"


def start_preopen(cfg: Settings, keys: ServerKeys, db: Database) -> PreOpener | None:
    """[Phase 11 WP7] Open the enabled RTSP cameras now, in parallel with the provider guard and the capacity phase
    (frames are discarded until admission, live/preopen.py). None when the live runtime or the switch is off."""
    if not cfg.live_runtime_enabled or not cfg.camera_preopen:
        return None
    from lifeguard.app.live_startup import preopen_specs
    from lifeguard.live.capture import force_rtsp_tcp
    from lifeguard.live.preopen import PreOpener

    force_rtsp_tcp()  # before the first open, as LiveRuntime.start does later
    pre = PreOpener(cfg)
    try:
        pre.start(preopen_specs(cfg, keys, db))
    except Exception:
        logger.exception("camera pre-open failed - cameras will open at admission")
        pre.close()
        return None
    return pre


def start_live(
    cfg: Settings, keys: ServerKeys, db: Database, writer: DBWriter, preopener: PreOpener | None = None
) -> LiveService:
    """Start the live service (runtime when enabled, heartbeat, supervisor watchdog) or refuse startup."""
    from lifeguard.app.live_startup import LiveService
    from lifeguard.vision import config as vision_config
    from lifeguard.vision.faults import Fault

    live = LiveService(
        cfg,
        keys,
        db,
        writer,
        vision_config.current().det_model_path,
        ready_fn=lambda: writer.alive,
        preopener=preopener,
    )
    try:
        live.start()
    except Exception as exc:
        live.stop()
        record_fault(writer, Fault(LIVE_RUNTIME_FAILED, "system", f"{type(exc).__name__}: {exc}"))
        raise StartupError(f"live runtime failed to start: {exc}") from exc
    return live


STARTUP_FAULTS = (LIVE_RUNTIME_FAILED, "DETECTOR_UNAVAILABLE", "PROVIDER_NOT_ACTIVE")


def clear_startup_faults(writer: DBWriter) -> None:
    """This process started: system FAULTs that refused an earlier start are over (FAULT_CLEARED)."""
    from lifeguard.alerts.faults import clear_fault

    for code in STARTUP_FAULTS:
        clear_fault(writer, code, "system")
