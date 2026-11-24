"""Process-wide vision runtime configuration: which detector model, on which execution provider.

    configure(VisionConfig(det_model_path=..., provider="auto"))   # the app (Phase 4) / the harness
    current()                                                      # what SourcePipeline uses

Unconfigured processes read LIFEGUARD_DET_MODEL (default: the COCO-only RTMDet-m in backend/models/),
LIFEGUARD_POSE_MODEL (default: the top-down RTMPose-m COCO in backend/models/; "none" = no pose model, which the
detector status and the startup audit show) and LIFEGUARD_INFERENCE_PROVIDER (default "auto": CUDA, then
OpenVINO, then CPU - see inference/session.py).
Thresholds and behaviour switches are NOT here: they live in the versioned thresholds file.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lifeguard.vision.inference import session
from lifeguard.vision.inference.session import PROVIDERS

DEFAULT_DET_MODEL = "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c_dynbatch.onnx"
DEFAULT_POSE_MODEL = "rtmpose-m_8xb256-420e_coco-256x192_dynbatch.onnx"
POSE_OFF_VALUES = ("none", "off", "false", "0")
# [v4] pool-vs-other scene suggestion (vision/scene/scene_classifier.py; tools/export_scene_clip.py builds it)
DEFAULT_SCENE_MODEL = "clip_vit_b32_scene.onnx"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"  # backend/models


@dataclass(frozen=True)
class VisionConfig:
    det_model_path: str
    provider: str = "auto"
    pose_model_path: str | None = None  # None = no pose model
    scene_model_path: str | None = None  # [v4] None = colour-only scene suggestion

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"inference provider must be one of {PROVIDERS}, got {self.provider!r}")


_lock = threading.Lock()
_current: VisionConfig | None = None


def pose_model_from(value: str | None) -> str | None:
    """A configured pose model value -> path (empty = the default model; none/off = no pose model)."""
    v = (value or "").strip()
    if v.lower() in POSE_OFF_VALUES:
        return None
    return v or str(MODELS_DIR / DEFAULT_POSE_MODEL)


def scene_model_from(value: str | None) -> str | None:
    """[v4] A configured scene model value -> path. Empty = the default file IF it was exported (it is optional: the
    scene suggestion then uses colour only); none/off = no scene model."""
    v = (value or "").strip()
    if v.lower() in POSE_OFF_VALUES:
        return None
    if v:
        return v
    default = MODELS_DIR / DEFAULT_SCENE_MODEL
    return str(default) if default.exists() else None


def from_env() -> VisionConfig:
    path = os.environ.get("LIFEGUARD_DET_MODEL", "").strip() or str(MODELS_DIR / DEFAULT_DET_MODEL)
    return VisionConfig(
        path,
        os.environ.get("LIFEGUARD_INFERENCE_PROVIDER", "auto").strip().lower(),
        pose_model_from(os.environ.get("LIFEGUARD_POSE_MODEL")),
        scene_model_from(os.environ.get("LIFEGUARD_SCENE_MODEL")),
    )


def from_settings(settings: Any) -> VisionConfig:
    """The app configuration (core.config.Settings); also points TensorRT's engine cache at
    Settings.trt_engine_dir (trt_cache_dir, default <data_dir>/trt_cache) so engines built at install are reused."""
    session.set_trt_cache_dir(Path(settings.trt_engine_dir))
    det = str(settings.detector_model) if settings.detector_model else str(MODELS_DIR / DEFAULT_DET_MODEL)
    return VisionConfig(
        det,
        settings.inference_provider,
        pose_model_from(settings.pose_model),
        scene_model_from(getattr(settings, "scene_model", None)),
    )


def configure(cfg: VisionConfig | None) -> None:
    """Set (or with None, forget) the process configuration."""
    global _current
    with _lock:
        _current = cfg


def current() -> VisionConfig:
    global _current
    with _lock:
        if _current is None:
            _current = from_env()
        return _current
