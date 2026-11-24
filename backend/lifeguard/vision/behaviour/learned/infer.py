"""
[v3] Ported from the v2 baseline detection/behavior_model/infer.py (verbatim except: the model path).
The learned model is ON by default (owner decision 2026-10-14): LIFEGUARD_BEHAVIOR_MODEL unset = the default
backend/models/behavior/behavior_v1.onnx (what training/lifeguard_training/behavior/cli.py writes); "none" =
rules only. A missing or mismatched model is logged once and reported by behavior_model_status(); the
rules always run (the learned model only adds evidence on top of them).

Live inference for the learned behaviour classifier (onnxruntime only — no
torch in the live path, CLAUDE.md).

A model is two files written by training/behavior/train.py:

    <name>.onnx   inputs  seq (N, C, T) float32, summary (N, 2S) float32
                  output  probs (N, K) float32 (softmax)
    <name>.json   {"format": "aquaguard-behavior/1", "classes": [...],
                   "seq_channels": [...], "seq_len": T, "summary_fields": [...],
                   "norm": {"seq_mean": [C], "seq_std": [C], "sum_mean": [2S], "sum_std": [2S]},
                   "thresholds": {"DISTRESS": 0.6, ...},      # tuned on out-of-fold predictions
                   "cv": {...}, "trained_on": {...}}

A sidecar whose feature layout differs from features.py is REFUSED (logged),
never silently fed mismatched inputs.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import numpy as np

from lifeguard.vision.behaviour.learned.features import CLASSES, SEQ_CHANNELS, SEQ_LEN, SUMMARY_FIELDS

logger = logging.getLogger(__name__)

FORMAT = "aquaguard-behavior/1"


class BehaviorModelError(RuntimeError):
    pass


class BehaviorModel:
    def __init__(self, path: str, device: str = "cpu") -> None:
        self.path = Path(path)
        meta_path = self.path.with_suffix(".json")
        if not self.path.exists() or not meta_path.exists():
            raise BehaviorModelError(f"behaviour model needs {self.path.name} + {meta_path.name}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("format") != FORMAT:
            raise BehaviorModelError(f"{meta_path.name}: format {meta.get('format')!r} != {FORMAT!r}")
        if (
            tuple(meta.get("seq_channels", ())) != SEQ_CHANNELS
            or int(meta.get("seq_len", -1)) != SEQ_LEN
            or tuple(meta.get("summary_fields", ())) != SUMMARY_FIELDS
        ):
            raise BehaviorModelError(
                f"{meta_path.name}: feature layout differs from this code version — "
                "retrain the model (training/behavior/train.py)"
            )
        self.meta = meta
        self.classes: list[str] = list(meta["classes"])
        unknown = set(self.classes) - set(CLASSES)
        if unknown:
            raise BehaviorModelError(f"{meta_path.name}: unknown classes {sorted(unknown)}")
        n = meta["norm"]
        self._seq_mean = np.asarray(n["seq_mean"], np.float32).reshape(1, -1, 1)
        self._seq_std = np.maximum(np.asarray(n["seq_std"], np.float32), 1e-6).reshape(1, -1, 1)
        self._sum_mean = np.asarray(n["sum_mean"], np.float32).reshape(1, -1)
        self._sum_std = np.maximum(np.asarray(n["sum_std"], np.float32), 1e-6).reshape(1, -1)
        self.thresholds: dict[str, float] = {k: float(v) for k, v in (meta.get("thresholds") or {}).items()}
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"]
        )
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(self.path), so, providers=providers)
        self.name = meta.get("name") or self.path.stem
        self._lock = threading.Lock()

    def predict(self, seq: np.ndarray, summary: np.ndarray) -> np.ndarray:
        """seq (N,C,T) or (C,T); summary (N,2S) or (2S,) → probs (N,K)."""
        seq = np.asarray(seq, np.float32)
        summary = np.asarray(summary, np.float32)
        if seq.ndim == 2:
            seq, summary = seq[None], summary[None]
        x = (seq - self._seq_mean) / self._seq_std
        s = (summary - self._sum_mean) / self._sum_std
        with self._lock:  # one session shared by every camera thread
            (probs,) = self.session.run(["probs"], {"seq": x.astype(np.float32), "summary": s.astype(np.float32)})
        return probs

    def predict_one(self, seq: np.ndarray, summary: np.ndarray) -> dict[str, float]:
        p = self.predict(seq, summary)[0]
        return {c: float(p[i]) for i, c in enumerate(self.classes)}


_cache_lock = threading.Lock()
_cache: dict[str, tuple[BehaviorModel | None, str | None]] = {}


DEFAULT_BEHAVIOR_MODEL = Path(__file__).resolve().parents[4] / "models" / "behavior" / "behavior_v1.onnx"
OFF_VALUES = ("none", "off", "false", "0")


def _configured_path() -> str:
    v = os.environ.get("LIFEGUARD_BEHAVIOR_MODEL")
    if v is None or not v.strip():
        return str(DEFAULT_BEHAVIOR_MODEL)
    return "" if v.strip().lower() in OFF_VALUES else v.strip()


def get_behavior_model(path: str | None = None) -> BehaviorModel | None:
    """Cached model for `path` (default settings.BEHAVIOR_MODEL_PATH); None when not configured or broken."""
    path = _configured_path() if path is None else path
    if not path:
        return None
    with _cache_lock:
        if path not in _cache:
            try:
                _cache[path] = (BehaviorModel(path), None)
                logger.info("[behavior-model] loaded %s", path)
            except Exception as exc:  # missing / mismatched model: rules only
                _cache[path] = (None, str(exc))
                if isinstance(exc, BehaviorModelError):  # not trained / not installed yet: expected, one line
                    logger.warning("[behavior-model] not used (rules only) — %s", exc)
                else:
                    logger.error("[behavior-model] not used — %s", exc, exc_info=True)
        return _cache[path][0]


def behavior_model_status(path: str | None = None) -> dict[str, object]:
    path = _configured_path() if path is None else path
    m = get_behavior_model(path)
    if m is None:
        err = _cache.get(path, (None, None))[1] if path else "LIFEGUARD_BEHAVIOR_MODEL=none (rules only)"
        return {"loaded": False, "path": path, "error": err}
    return {
        "loaded": True,
        "path": path,
        "name": m.name,
        "classes": m.classes,
        "thresholds": m.thresholds,
        "cv": m.meta.get("cv", {}),
        "trained_on": m.meta.get("trained_on", {}),
    }


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
