"""
The per-source analysis pipeline, every frame.

    sp = SourcePipeline("cam-1", mode="Medium", geometry=zones, fps=25.0)
    res = sp.process(frame, frame_index, t)      # EVERY decoded frame
    res.detections   FrameDetections (people, ids)
    res.fires        alerts to persist / broadcast
    res.scored       behaviour was re-scored on this frame

Every frame is decoded, detected and tracked; every frame feeds the behaviour tracker (pooled into
BEHAVIOR_SAMPLE_HZ samples). Behaviour is re-scored on a fixed cadence of BEHAVIOR_SCORE_INTERVAL_S
of SOURCE time and uses the data of every frame since the last round. The same object serves
uploaded videos, the evaluation harness and (split into `prepare` / detect / `after_detect`) the
multi-camera live runtime (Phase 4).

[v3] From the v2 baseline detection/pipeline.py, interface unchanged so the evaluation harness's
PipelineAdapter drives this tree exactly as it drives the v2 baseline (`--backend-dir`). Changes:
  * the detector comes from vision.config (model path + provider guard), not a settings module;
  * the floatable DETECTOR is gone (plan: it had no model) - `after_detect` still accepts floatables;
  * NOT_MONITORED: when the detector is unavailable, `monitoring` becomes NOT_MONITORED, ONE FAULT
    is sent to the fault sink, and DetectorUnavailableError propagates to the caller (no empty
    frames that would read as "nobody in the pool"). It returns to MONITORED on the next good frame.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lifeguard.vision import config as vision_config
from lifeguard.vision.behaviour.learned.infer import behavior_model_status
from lifeguard.vision.constants import BEHAVIOR_SCORE_INTERVAL_S, ID_LINK_NESTED, POSE_SKIP_MAX_RISK
from lifeguard.vision.faults import (
    DETECTOR_UNAVAILABLE,
    MONITORED,
    NOT_MONITORED,
    PROVIDER_NOT_ACTIVE,
    Fault,
    FaultSink,
    log_fault,
)
from lifeguard.vision.inference import models
from lifeguard.vision.inference.detector import DetectorUnavailableError, FrameDetections, MultiPersonDetector
from lifeguard.vision.inference.session import ProviderError
from lifeguard.vision.scene import roi_config
from lifeguard.vision.scene.floatables import FloatableObject
from lifeguard.vision.scene.scene_tracker import AlertFire, SceneBehaviorTracker
from lifeguard.vision.scene.scene_type import POOL, SCENE_TYPES, STANDARD_CCTV, SceneTypeEstimator
from lifeguard.vision.tracking import glare

logger = logging.getLogger(__name__)

_warm_lock = threading.Lock()
_warm_status: dict[str, Any] | None = None


def warmup_models(force: bool = False) -> dict[str, Any]:
    """
    Load + warm the shared detector session once (idempotent, thread-safe). Returns the detector
    status plus session info. A failed load is returned (healthy=False, error=...), not raised:
    the caller decides between FAULT + refuse-to-start (server) and reporting (harness).
    """
    global _warm_status
    with _warm_lock:
        if _warm_status is not None and not force:
            return dict(_warm_status)
        cfg = vision_config.current()
        t0 = time.perf_counter()
        det = MultiPersonDetector("warmup", cfg.det_model_path, cfg.provider, pose_model_path=cfg.pose_model_path)
        backend = det.load()
        st = det.status()
        st["sessions"] = {k[0]: models.session_info(m) for k, m in models.cached_models().items()}
        st["warmup_total_s"] = round(time.perf_counter() - t0, 2)
        st["behaviour_model"] = behavior_model_status()
        st["backend"] = backend
        st["load_exception"] = det.load_exception
        _warm_status = st
        logger.info(
            "[pipeline] models warm in %.2fs: %s",
            st["warmup_total_s"],
            {k: (v.get("active"), v.get("warmup_s")) for k, v in st["sessions"].items()},
        )
        return dict(st)


def warm_status() -> dict[str, Any] | None:
    return dict(_warm_status) if _warm_status else None


def fault_for(source_id: str, exc: BaseException | None, message: str) -> Fault:
    """The FAULT describing a dead detector (a ProviderError carries the missing DLL list)."""
    cause = exc.__cause__ if isinstance(exc, DetectorUnavailableError) else exc
    if isinstance(cause, ProviderError):
        return Fault(
            PROVIDER_NOT_ACTIVE,
            source_id,
            message,
            {"requested": cause.requested, "active": cause.active, "missing_dlls": cause.missing_dlls},
        )
    return Fault(DETECTOR_UNAVAILABLE, source_id, message)


@dataclass
class FrameResult:
    detections: FrameDetections
    fires: list[AlertFire]
    scored: bool = False
    stages_ms: dict[str, float] = field(default_factory=dict)


class SourcePipeline:
    """Detection -> tracking -> behaviour for ONE video source, every frame."""

    def __init__(
        self,
        source_id: str,
        mode: str = "Medium",
        geometry: Any = None,
        fps: float = 25.0,
        frame_size: tuple[int, int] | None = None,
        detector: Any | None = None,
        fault_sink: FaultSink | None = None,
        scene_type: str = POOL,
        scene_classifier: Any = None,
    ) -> None:
        self.source_id = source_id
        if detector is None:
            cfg = vision_config.current()
            detector = MultiPersonDetector(
                source_id, cfg.det_model_path, cfg.provider, pose_model_path=cfg.pose_model_path
            )
        self.detector = detector
        self.fps = float(fps) if fps and fps > 0 else 25.0
        if hasattr(self.detector, "set_fps"):
            self.detector.set_fps(self.fps)
        self.scene = SceneBehaviorTracker(source_id, mode)
        self.scene.fps = self.fps
        # [2.20.0] zones from LIFEGUARD_ROI_FILE (vision/scene/roi_config.py) for sources without database zones,
        # and MASK polygons blacked out before detection. Unset = no change.
        roi = roi_config.from_env()
        self._roi_masker = roi.masker_for(source_id) if roi is not None else None
        self._roi = roi
        self._roi_geometry: Any = None
        self._caller_geometry: Any = geometry
        self.scene.set_geometry(geometry)
        self.frame_size = frame_size
        self.last_score_t = -math.inf
        self.frames = 0
        self.fault_sink: FaultSink = fault_sink or log_fault
        self.monitoring = MONITORED
        # [v4] scene_classifier: the live runtime's keyframe scene model (None: colour-only suggestion)
        self.scene_type_estimate = SceneTypeEstimator(scene_classifier)
        self.scene_type = POOL
        self.set_scene_type(scene_type)

    # ── configuration ─────────────────────────────────────────────────────────
    def set_geometry(self, geometry: Any) -> None:
        self._caller_geometry = geometry
        self.scene.set_geometry(self._effective_geometry())

    def _effective_geometry(self) -> Any:
        """Database / caller zones win; the ROI file fills in only when the caller has none."""
        g = self._caller_geometry
        if g is not None and getattr(g, "has_zones", True):
            return g
        return self._roi_geometry if self._roi_geometry is not None else g

    @property
    def surveillance(self) -> bool:
        """[v4] Standard CCTV: detection + tracking only. No pose, no behaviour scoring - no drowning alert."""
        return self.scene_type == STANDARD_CCTV

    def set_scene_type(self, scene_type: str) -> None:
        if scene_type not in SCENE_TYPES:
            raise ValueError(f"unknown scene type {scene_type!r}")
        self.scene_type = scene_type
        if hasattr(self.detector, "surveillance"):
            self.detector.surveillance = self.surveillance

    def set_mode(self, mode: str) -> None:
        self.scene.set_mode(mode)

    def set_fps(self, fps: float) -> None:
        """[v7] The analysed fps changed (stream reopened): detector, tracker and scene frame rules follow it."""
        if fps and fps > 0:
            self.fps = float(fps)
            if hasattr(self.detector, "set_fps"):
                self.detector.set_fps(self.fps)
            self.scene.fps = self.fps

    def retime(self, fps: float) -> None:
        """[dynamic fps] The live analysis rate stepped (live/rate.py) on the SAME stream: tracker ids, behaviour state
        and the motion model are kept; frame-count rules follow the new rate (set_fps is for a reopened stream)."""
        if fps and fps > 0 and abs(fps - self.fps) > 1e-6:
            self.fps = float(fps)
            if hasattr(self.detector, "retime"):
                self.detector.retime(self.fps)
            elif hasattr(self.detector, "set_fps"):
                self.detector.set_fps(self.fps)
            self.scene.fps = self.fps

    def reset(self) -> None:
        """Stream reconnect: tracker ids restart, so behaviour state must too."""
        self.detector.reset()
        self.scene.reset()
        self.scene_type_estimate.reset()
        glare.reset(self)  # [v3 Phase 11] no glare-held id survives the tracker reset
        self.last_score_t = -math.inf

    # ── monitoring state ──────────────────────────────────────────────────────
    def _not_monitored(self, exc: DetectorUnavailableError) -> None:
        if self.monitoring != NOT_MONITORED:
            self.monitoring = NOT_MONITORED
            cause = getattr(self.detector, "load_exception", None) or exc
            self.fault_sink(fault_for(self.source_id, cause, str(exc)))

    # ── per frame ─────────────────────────────────────────────────────────────
    def prepare(self, frame: np.ndarray, t: float) -> None:
        """Before detection: update the water estimate and hand the pool ROI to the detector."""
        if self._roi_masker is not None:
            self._roi_masker.apply(frame)  # MASK zones only: overlays that can never hold a swimmer
        if self._roi is not None and self._roi_geometry is None:
            h, w = frame.shape[:2]
            self._roi_geometry = self._roi.geometry_for(self.source_id, w, h)
            if self._roi_geometry is not None:
                logger.info("%s: zones from %s", self.source_id, roi_config.ENV_VAR)
                self.scene.set_geometry(self._effective_geometry())
        self.scene.update_water(frame, t)
        # [v4] the pool / CCTV suggestion runs in BOTH modes, so a camera set to CCTV can still be told "pool"
        self.scene_type_estimate.update(frame, t, self.scene.water, self.scene.geometry)
        if hasattr(self.detector, "set_roi"):
            self.detector.set_roi(self.scene.water_roi())
        if hasattr(self.detector, "set_water_test"):  # tracking v3: faint births only in water
            self.detector.set_water_test(self.scene.water_contains)
        if hasattr(self.detector, "set_outside_water_test"):  # [2.33.0] deck rules: drawn zones only
            self.detector.set_outside_water_test(self.scene.box_outside_drawn_water)
        if hasattr(self.detector, "set_zone_gate"):  # [2.43.0] ZONE_GATE: drawn water zones only
            self.detector.set_zone_gate(self.scene.zone_gate_mask)
        glare.prepare(self, frame, t)  # [v3 Phase 11 WP3] glare mask + hold (GLARE_HOLD off: nothing)

    def process(self, frame: np.ndarray, frame_index: int, t: float) -> FrameResult:
        """Single-source path (eval harness). Raises DetectorUnavailableError (camera NOT MONITORED)."""
        t0 = time.perf_counter()
        self.prepare(frame, t)
        try:
            det = self.detector.detect(frame, frame_index=frame_index, timestamp=t)
        except DetectorUnavailableError as exc:
            self._not_monitored(exc)
            raise
        self.monitoring = MONITORED
        t1 = time.perf_counter()
        self.scene.alert_frame = frame  # [2.31.0] the alert-time lens-occluder gate judges alert boxes on this frame
        res = self.after_detect(det, None, t)
        res.stages_ms["detect_track_pose"] = (t1 - t0) * 1000.0
        return res

    def finish(self, t_end: float) -> list[AlertFire]:
        """[2.18.0] End of a finite clip (eval harness, dataset builder): the alerts the last frames still owe (scene
        `finish`). Standard CCTV is never scored, so nothing."""
        if self.surveillance:
            return []
        return self.scene.finish(t_end)

    def pose_stable_tracks(self) -> set[int]:
        """Track ids that may skip pose on alternate frames: last result not drowning, risk below
        POSE_SKIP_MAX_RISK and pose-reliable. Everything else (and anything unscored) is posed every frame."""
        stable: set[int] = set()
        for tid, ts in self.scene.tracks.items():
            r = getattr(ts, "result", None)
            if r is None or r.is_drowning or not r.pose_reliable or r.risk >= POSE_SKIP_MAX_RISK:
                continue
            stable.add(int(tid))
        return stable

    def after_detect(self, det: FrameDetections, floatables: list[FloatableObject] | None, t: float) -> FrameResult:
        """Behaviour stage (CPU): ingest every frame, score on the fixed cadence, expire."""
        t0 = time.perf_counter()
        self.frames += 1
        hidden = getattr(getattr(self.detector, "non_person", None), "hidden_ids", None)
        if hidden:
            self.scene.forget(hidden)  # [2.18.0] occluders / fixtures: no state, no submersion ghost
        take_linked = getattr(self.detector, "take_linked_ids", None)
        linked = take_linked() if take_linked is not None else None
        if linked:
            self.scene.forget(linked)  # [2.35.0] ID_LINK_NESTED: the body carries on under the host id
        self.scene.ingest(det, floatables)
        fires: list[AlertFire] = []
        scored = False
        if self.surveillance:  # [v4] standard CCTV: tracks kept (presence, Model View), never scored
            self.scene.expire(t)
            return FrameResult(det, fires, scored, {"behaviour": (time.perf_counter() - t0) * 1000.0})
        if t - self.last_score_t >= BEHAVIOR_SCORE_INTERVAL_S - 1e-6:
            fires = self.scene.score(t)
            self.last_score_t = t
            scored = True
            if hasattr(self.detector, "set_pose_stable"):
                self.detector.set_pose_stable(self.pose_stable_tracks())
        else:
            # [Phase 2] at-risk tracks (CAUTION/DROWNING) don't wait for the next cadence tick -
            # every other track keeps the fixed-interval cost this gate was added for.
            at_risk = self.scene.at_risk_track_ids()
            if at_risk:
                fires = self.scene.score(t, track_ids=at_risk)
                scored = True
        self.scene.expire(t)
        if ID_LINK_NESTED and hasattr(self.detector, "set_link_protected"):
            self.detector.set_link_protected(self.scene.at_risk_track_ids())  # [2.35.0] never retire a risky id
        return FrameResult(det, fires, scored, {"behaviour": (time.perf_counter() - t0) * 1000.0})
