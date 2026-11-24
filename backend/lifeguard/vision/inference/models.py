# mypy: disable-error-code="union-attr"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""ONNX models: DetModel (letterbox / normalise / run), PoseModel (top-down RTMPose), metadata, the shared
session cache.

[v3] From the v2 baseline detection/rtm_backend.py: DetModel, PoseModel, _SessionModel, metadata helpers and
the cache; sessions come from inference/session.py (provider guard). Removed: rtmlib backends, URL downloads
(models are local files verified against MODELS.json), TensorRT.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lifeguard.vision.inference.session import active_provider, make_session

logger = logging.getLogger(__name__)


# ONNX metadata keys written by detection/trainer.py's exporter.
META_FORMAT = "lifeguard.det_format"  # "xyxy_scores" for layout (b)
META_CLASSES = "lifeguard.classes"  # JSON list of class names
META_TASK = "lifeguard.task"  # "detector" | "pose"
FORMAT_XYXY_SCORES = "xyxy_scores"
# DETR-family exports (training/dfine): raw `logits (1,Q,C)` + `boxes (1,Q,4)`
# cxcywh normalised to [0,1]; input RGB / 255, plain resize (no letterbox).
FORMAT_DETR = "detr_logits"
# RTMDet (mmdet) normalises BGR input; RTMPose (mmpose) was trained on RGB.
DET_MEAN_BGR = (103.53, 116.28, 123.675)
DET_STD_BGR = (57.375, 57.12, 58.395)
POSE_MEAN_RGB = (123.675, 116.28, 103.53)
POSE_STD_RGB = (58.395, 57.12, 57.375)
POSE_INPUT_WH = (192, 256)
# A dynamic-batch export only ever sees these batch sizes (v2: 1, 4, 8; 16 and 32 for four cameras' crops
# gathered into one call by live/pose_batcher.py). They are also the TensorRT optimisation profile points.
POSE_BATCH_BUCKETS = (1, 4, 8, 16, 32)


def _thread_buffers() -> threading.local:
    return threading.local()


class _SessionModel:
    """Common base: session + metadata + per-thread preallocated buffers."""

    kind = "model"

    def __init__(self, path: str, session: Any, info: dict[str, Any]) -> None:
        self.onnx_model = path
        self.session = session
        self.info = info
        inp = session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = list(inp.shape)
        self.output_names = [o.name for o in session.get_outputs()]
        b = self.input_shape[0]
        self.static_batch: int | None = int(b) if isinstance(b, int) and b > 0 else None
        self._tls = _thread_buffers()
        self.warm = False
        self.warmup_s = 0.0
        self.calls = 0

    def inference(self, img: np.ndarray) -> list[np.ndarray]:
        """rtmlib-compatible: HWC float image (or NCHW batch) → outputs."""
        x = img if img.ndim == 4 else np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32)
        self.calls += 1
        return self.session.run(self.output_names, {self.input_name: x})


class DetModel(_SessionModel):
    """RTMDet / YOLOX-style person (or object) detector with a fixed input size."""

    kind = "det"

    def __init__(self, path: str, session: Any, info: dict[str, Any], imgsz: int) -> None:
        super().__init__(path, session, info)
        h, w = self.input_shape[2], self.input_shape[3]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            self.model_input_size = (int(h), int(w))
        else:
            self.model_input_size = (int(imgsz), int(imgsz))
        self.fmt = model_metadata(self).get(META_FORMAT, "")
        # RTMDet (mmdet) normalises with ImageNet mean/std; YOLOX exports take
        # raw 0–255 BGR (what rtmlib's YOLOX wrapper feeds them); DETR exports
        # take RGB / 255 at a plain (non-letterboxed) resize.
        self.detr = self.fmt == FORMAT_DETR
        raw_input = "yolox" in Path(path).name.lower()
        if self.detr:
            self.mean, self.std = np.zeros(3, np.float32), np.full(3, 255.0, np.float32)
        else:
            self.mean = np.zeros(3, np.float32) if raw_input else np.array(DET_MEAN_BGR, np.float32)
            self.std = np.ones(3, np.float32) if raw_input else np.array(DET_STD_BGR, np.float32)
        self._inv_std = (1.0 / self.std).astype(np.float32)

    def _bufs(self) -> tuple[np.ndarray, np.ndarray]:
        tls = self._tls
        if getattr(tls, "canvas", None) is None:
            h, w = self.model_input_size
            tls.canvas = np.full((h, w, 3), 114, np.uint8)
            tls.x = np.empty((1, 3, h, w), np.float32)
        return tls.canvas, tls.x

    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Letterbox (top-left aligned, pad 114) + normalise. Returns (NCHW float32 view, ratio).

        DETR exports: plain resize to the input size, RGB, / 255 (ratio 1.0 —
        their boxes are normalised and scaled by the caller's frame size)."""
        canvas, x = self._bufs()
        h, w = self.model_input_size
        ih, iw = img.shape[:2]
        if self.detr:
            rs = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            for c in range(3):  # BGR frame -> RGB planes
                np.multiply(rs[:, :, 2 - c], self._inv_std[c], out=x[0, c], dtype=np.float32)
            return x, 1.0
        ratio = min(h / ih, w / iw)
        nh, nw = int(ih * ratio), int(iw * ratio)
        canvas.fill(114)
        if (nh, nw) == (ih, iw):
            canvas[:nh, :nw] = img
        else:
            canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        for c in range(3):
            np.subtract(canvas[:, :, c], self.mean[c], out=x[0, c], dtype=np.float32)
            x[0, c] *= self._inv_std[c]
        return x, ratio

    def run(self, frame: np.ndarray) -> tuple[list[np.ndarray], float]:
        x, ratio = self.preprocess(frame)
        self.calls += 1
        return self.session.run(self.output_names, {self.input_name: x}), ratio


class PoseModel(_SessionModel):
    """Top-down RTMPose (SimCC: crops in, simcc_x (B,K,Wx) + simcc_y (B,K,Hy) out) with bucketed batches
    (v2 rtm_backend.PoseModel). One-stage pose exports (dets + keypoints outputs, e.g. RTMO) are refused: the
    pipeline is top-down by design - every RTMDet box is cropped and enlarged before pose."""

    kind = "pose"

    def __init__(self, path: str, session: Any, info: dict[str, Any], rgb: bool = False) -> None:
        super().__init__(path, session, info)
        ranks = [len(o.shape) for o in session.get_outputs()]
        if len(ranks) < 2 or any(r != 3 for r in ranks[:2]):
            raise ValueError(
                f"{Path(path).name}: not a top-down RTMPose SimCC export (outputs simcc_x, simcc_y of rank 3); "
                "one-stage pose models are not supported"
            )
        h, w = self.input_shape[2], self.input_shape[3]
        static_hw = isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0
        self.model_input_size = (int(w), int(h)) if static_hw else POSE_INPUT_WH  # (W, H) like rtmlib
        self.buckets = [self.static_batch] if self.static_batch is not None else list(POSE_BATCH_BUCKETS)
        self.max_batch = self.buckets[-1]
        self.rgb = bool(rgb)
        self.mean = np.array(POSE_MEAN_RGB, np.float32)
        self.std = np.array(POSE_STD_RGB, np.float32)
        self._inv_std = (1.0 / self.std).astype(np.float32)

    def bufs(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-thread (batch, crop) scratch buffers."""
        tls = self._tls
        if getattr(tls, "x", None) is None:
            w, h = self.model_input_size
            tls.x = np.zeros((self.max_batch, 3, h, w), np.float32)
            tls.crop = np.zeros((h, w, 3), np.uint8)
        return tls.x, tls.crop

    def bucket_for(self, n: int) -> int:
        for b in self.buckets:
            if b >= n:
                return b
        return self.buckets[-1]


# ═════════════════════════════════════════════════════════════════════════════
# Model cache (one session per model file/device, shared across cameras)
# ═════════════════════════════════════════════════════════════════════════════
_cache_lock = threading.Lock()


_model_cache: dict[tuple[str, str, str, str], Any] = {}


def clear_model_cache() -> None:
    """Drop every cached session (used by a detector-status refresh and tests)."""
    with _cache_lock:
        _model_cache.clear()


def cached_models() -> dict[tuple[str, str, str, str], Any]:
    with _cache_lock:
        return dict(_model_cache)


def model_metadata(model: Any) -> dict[str, str]:
    """Custom ONNX metadata (onnxruntime backend only; {} otherwise).

    Frozen model files exported before the Life Guard rename carry their keys under the old
    ``aquaguard.*`` namespace (``aquaguard.det_format``, ``aquaguard.classes``, ``aquaguard.task``); they  # legacy-name
    are not re-exported for a name change alone. A key still under the old namespace is copied to its
    current ``lifeguard.*`` name here (once, in memory) so every caller only ever matches on the current name.
    """
    session = getattr(model, "session", None)
    try:
        meta = dict(session.get_modelmeta().custom_metadata_map or {})
    except Exception:
        logger.warning("cannot read ONNX metadata", exc_info=True)
        return {}
    for key in list(meta):
        if key.startswith("aquaguard."):  # legacy-name
            new_key = "lifeguard." + key[len("aquaguard.") :]
            meta.setdefault(new_key, meta[key])
    return meta


# ═════════════════════════════════════════════════════════════════════════════
# Detector output decoding
# ═════════════════════════════════════════════════════════════════════════════
def class_names_for(model: Any, path: str) -> list[str] | None:
    """Class names from ONNX metadata, then a `<model>.classes.json` sidecar."""
    meta = model_metadata(model)
    if META_CLASSES in meta:
        try:
            names = json.loads(meta[META_CLASSES])
            if isinstance(names, list) and names:
                return [str(n) for n in names]
        except Exception:
            logger.warning("unreadable class names in ONNX metadata", exc_info=True)
    local = Path(getattr(model, "onnx_model", "") or path)
    for sidecar in (local.with_suffix(".classes.json"), Path(str(local) + ".classes.json")):
        try:
            if sidecar.exists():
                names = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(names, list) and names:
                    return [str(n) for n in names]
        except Exception:
            logger.warning("unreadable class-name sidecar %s", sidecar, exc_info=True)
            continue
    return None


DET_TRT_BATCH = (1, 4, 8)  # TensorRT profile: one frame .. 4 cameras (opt) .. batch_max


def load_detector_model(
    path: str, provider: str, imgsz: int, trt_batch: tuple[int, int, int] = DET_TRT_BATCH
) -> DetModel:
    """DetModel for a LOCAL .onnx `path` on exactly `provider` (session.ProviderError otherwise),
    cached per (path, provider) and shared by every camera.

    [v3] v2's URL / .zip download (rtmlib cache), fp16-sibling switch and openvino / opencv
    backends are gone: models are local files verified against MODELS.json (tools/verify_models.py).
    """
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"detector model not found: {path!r}")
    key = ("det", str(path), provider, "onnxruntime")
    with _cache_lock:
        if key in _model_cache:
            return _model_cache[key]
        sess, info = make_session(str(path), provider, trt_batch=trt_batch)
        model = DetModel(str(path), sess, info, imgsz)
        _model_cache[key] = model
        return model


def load_pose_model(path: str, provider: str, rgb: bool = False) -> PoseModel:
    """PoseModel for a LOCAL RTMPose .onnx on exactly `provider`, cached and shared like the detector."""
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"pose model not found: {path!r} (export it with tools/export_rtmpose.py)")
    key = ("pose", str(path), provider, f"onnxruntime:rgb={int(bool(rgb))}")
    with _cache_lock:
        if key in _model_cache:
            return _model_cache[key]
        b = POSE_BATCH_BUCKETS
        sess, info = make_session(str(path), provider, trt_batch=(b[0], b[-2], b[-1]))
        model = PoseModel(str(path), sess, info, rgb)
        _model_cache[key] = model
        return model


def warmup(model: Any, repeats: int = 2) -> float:
    """Run the detector's input shape so no algorithm search happens on the first live frame (v2, det only)."""
    if model is None or getattr(model, "warm", False) or not hasattr(model, "session"):
        return 0.0
    t0 = time.perf_counter()
    if isinstance(model, DetModel):
        h, w = model.model_input_size
        dummy = np.full((h, w, 3), 114, np.uint8)
        for _ in range(repeats):
            model.run(dummy)
    elif isinstance(model, PoseModel):
        w, h = model.model_input_size
        for b in model.buckets:
            for _ in range(repeats):
                model.session.run(model.output_names, {model.input_name: np.zeros((b, 3, h, w), np.float32)})
    model.warm = True
    model.warmup_s = round(time.perf_counter() - t0, 2)
    return model.warmup_s


def session_info(model: Any) -> dict[str, Any]:
    info = dict(getattr(model, "info", {}) or {})
    info.update(
        {
            "path": str(getattr(model, "onnx_model", "")),
            "active": active_provider(model),
            "warm": bool(getattr(model, "warm", False)),
            "warmup_s": getattr(model, "warmup_s", None),
        }
    )
    if isinstance(model, DetModel):
        info["input_hw"] = list(model.model_input_size)
    elif isinstance(model, PoseModel):
        info["input_wh"] = list(model.model_input_size)
        info["buckets"] = list(model.buckets)
    return info


__all__ = [
    "DetModel",
    "PoseModel",
    "active_provider",
    "cached_models",
    "class_names_for",
    "clear_model_cache",
    "load_detector_model",
    "load_pose_model",
    "model_metadata",
    "session_info",
    "warmup",
]
