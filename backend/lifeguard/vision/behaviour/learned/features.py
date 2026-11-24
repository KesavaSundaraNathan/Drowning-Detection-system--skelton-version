"""
[v3] Ported verbatim from the v2 baseline detection/behavior_model/features.py (imports only).

Window features for the learned behaviour classifier.

One sample = the last BEHAVIOR_MODEL_SEQ_SECONDS of one track, built from the
same TrackBuffer the rule engine reads (10 Hz pooled samples), so training
and live inference see identical inputs:

  seq      (len(SEQ_CHANNELS), SEQ_LEN) — resampled on a uniform grid ending
           at the latest sample. Positions/sizes are in body-scale units
           (resolution- and distance-independent, like the rule engine).
  summary  (2 * len(SUMMARY_FIELDS),) — the rule engine's KinematicFeatures
           for the track (value, then a 0/1 "was measured" mask per field;
           pose-derived fields are unmeasured while POSE_ENABLED is off).

Only box/track signals are required — the project runs without a pose model
(licensing, see config/settings.py POSE_ENABLED).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from lifeguard.vision.constants import BEHAVIOR_MODEL_SEQ_SECONDS, BEHAVIOR_SAMPLE_HZ

CLASSES = ("NORMAL", "DISTRESS", "ACTIVE_DROWNING", "PASSIVE_DROWNING")

SEQ_CHANNELS = ("vx_su", "vy_su", "log_h", "log_w", "log_aspect", "conf", "edge_su", "valid")
SEQ_LEN = int(round(BEHAVIOR_MODEL_SEQ_SECONDS * BEHAVIOR_SAMPLE_HZ))

SUMMARY_FIELDS = (
    # box / track level — always available
    "travel_speed",
    "progress_speed",
    "travel_max",
    "bbox_bob",
    "aspect_ratio",
    "still_seconds",
    "spike_ratio",
    "box_aspect",
    "box_height_ratio",
    "conf_recent",
    "edge_su",
    "head_flicker",
    "zero_velocity_seconds",
    "pose_lost_seconds",
    "duration",
    # pose level — masked out when no pose model runs
    "torso_angle",
    "arm_amp",
    "arm_periodicity",
    "head_bob",
    "face_down_seconds",
    "inverted_seconds",
    "pose_quality",
)
_CLIP = 8.0


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def summary_vector(features: Any) -> np.ndarray:
    """KinematicFeatures (or a dict of its fields) → value/mask vector."""
    get = features.get if isinstance(features, dict) else (lambda k, d=None: getattr(features, k, d))
    out = np.zeros(2 * len(SUMMARY_FIELDS), np.float32)
    for i, name in enumerate(SUMMARY_FIELDS):
        v = _finite(get(name, None))
        if v is not None:
            out[2 * i] = float(np.clip(v, -1e3, 1e3))
            out[2 * i + 1] = 1.0
    return out


def sequence_matrix(samples: Sequence[Any], scale: float, t_end: float | None = None) -> np.ndarray:
    """FrameSample list → (C, T) on a uniform grid of SEQ_LEN steps ending at t_end."""
    C, T = len(SEQ_CHANNELS), SEQ_LEN
    seq = np.zeros((C, T), np.float32)
    if not samples:
        return seq
    ts = np.array([s.t for s in samples], np.float64)
    t_end = float(ts[-1]) if t_end is None else float(t_end)
    dt = 1.0 / BEHAVIOR_SAMPLE_HZ
    grid = t_end - dt * np.arange(T - 1, -1, -1)
    idx = np.searchsorted(ts, grid + 1e-6, side="right") - 1  # latest sample at or before each step
    valid = idx >= 0
    idx = np.clip(idx, 0, len(samples) - 1)
    # samples older than 1.5 steps from the grid point are stale (track gap)
    valid &= np.abs(ts[idx] - grid) <= 1.5 * dt + 1e-6
    scale = max(float(scale), 1.0)
    cx = np.array([samples[i].center[0] for i in idx], np.float64) / scale
    cy = np.array([samples[i].center[1] for i in idx], np.float64) / scale
    h = np.array([max(samples[i].bbox[3] - samples[i].bbox[1], 1.0) for i in idx], np.float64)
    w = np.array([max(samples[i].bbox[2] - samples[i].bbox[0], 1.0) for i in idx], np.float64)
    conf = np.array([samples[i].conf for i in idx], np.float64)
    edge = np.array([samples[i].edge_dist_px for i in idx], np.float64)
    edge = np.where(np.isfinite(edge), edge / scale, _CLIP)
    vx = np.gradient(cx, dt) if T > 1 else np.zeros(T)
    vy = np.gradient(cy, dt) if T > 1 else np.zeros(T)
    seq[0] = np.clip(vx, -_CLIP, _CLIP)
    seq[1] = np.clip(vy, -_CLIP, _CLIP)
    seq[2] = np.log(h / scale)
    seq[3] = np.log(w / scale)
    seq[4] = np.clip(np.log(h / w), -2.0, 2.0)
    seq[5] = conf
    seq[6] = np.clip(edge, -_CLIP, _CLIP)
    seq[7] = valid.astype(np.float32)
    seq[:7, ~valid] = 0.0
    return seq


def window_features(buf: Any, features: Any, t_end: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(seq (C,T), summary (2S,)) for one track: `buf` a TrackBuffer, `features` its KinematicFeatures."""
    samples = list(buf.samples)
    if t_end is not None:
        samples = [s for s in samples if s.t <= t_end + 1e-6]
    return sequence_matrix(samples, buf.scale(), t_end), summary_vector(features)
