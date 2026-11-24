# mypy: disable-error-code="arg-type, operator"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""Numeric helpers and COCO-17 keypoint indices of the behaviour engine.

[v3] Split out of the v2 baseline detection/risk_engine.py; definitions verbatim.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── COCO-17 keypoint indices ──────────────────────────────────────────────────
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10

L_HIP, R_HIP = 11, 12

FACE = (NOSE, L_EYE, R_EYE)

HEAD = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)

UPPER = tuple(range(11))

# Rigid-ish points used for frame-to-frame translation (no wrists / elbows /
# knees / ankles, which move independently of the body).
CORE = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH, L_HIP, R_HIP)

NAN = float("nan")

Point = tuple[float, float]


# ═════════════════════════════════════════════════════════════════════════════
# Small numeric helpers
# ═════════════════════════════════════════════════════════════════════════════
def finite(x: Any) -> bool:
    try:
        return x is not None and not isinstance(x, bool) and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def ramp(x: float | None, lo: float, hi: float) -> float:
    """0 at `lo`, 1 at `hi`, clamped. Non-finite input → 0."""
    if not finite(x):
        return 0.0
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return float(min(1.0, max(0.0, (float(x) - lo) / (hi - lo))))


def inv_ramp(x: float | None, lo: float, hi: float) -> float:
    """1 at `lo`, 0 at `hi`, clamped. Non-finite input → 0."""
    if not finite(x):
        return 0.0
    return 1.0 - ramp(x, lo, hi)


def clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _median(vals: Sequence[float]) -> float:
    """Median of a small list of floats (much faster than numpy for n < 100)."""
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return NAN
    h = n // 2
    return float(v[h]) if n % 2 else 0.5 * (v[h - 1] + v[h])


def _percentile(vals: Sequence[float], q: float) -> float:
    v = sorted(vals)
    if not v:
        return NAN
    pos = (len(v) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return float(v[lo] + (v[hi] - v[lo]) * (pos - lo))


def _nanmedian(values: Sequence[float]) -> float:
    return _median([float(v) for v in values if v is not None and v == v and math.isfinite(v)])


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Centred moving average with shrinking windows at the edges (no phase lag)."""
    n = len(x)
    if n == 0 or win <= 1:
        return x.astype(float)
    half = win // 2
    c = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx + half + 1, 0, n)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def _resample(t: np.ndarray, y: np.ndarray, grid: np.ndarray, min_valid: float = 0.6) -> np.ndarray | None:
    """Linear resample of a NaN-holed series onto `grid`; None if too sparse."""
    ok = np.isfinite(y)
    if ok.sum() < max(4, int(min_valid * len(y))):
        return None
    return np.interp(grid, t[ok], y[ok])


def _periodicity(x: np.ndarray, rate: float, f_lo: float = 0.35, f_hi: float = 3.0) -> tuple[float, float]:
    """
    Normalised autocorrelation peak in the [f_lo, f_hi] Hz band.
    Returns (periodicity 0–1, dominant frequency Hz or NaN).
    """
    n = len(x)
    x = x - x.mean()
    energy = float((x * x).sum())
    if n < 8 or energy <= 1e-9 or rate <= 0:
        return 0.0, NAN
    k_min = max(1, int(math.floor(rate / f_hi)))
    k_max = min(n // 2, int(math.ceil(rate / f_lo)))
    acs: list[tuple[int, float]] = []
    for k in range(k_min, k_max + 1):
        acs.append((k, float((x[:-k] * x[k:]).sum()) / energy * (n / max(n - k, 1))))
    best = max((a for _, a in acs), default=0.0)
    if best <= 0:
        return 0.0, NAN
    # The unbiased correction inflates long lags, so a periodic signal's 2nd/3rd
    # harmonic lag can edge out the fundamental: take the FIRST local peak that
    # is within 85 % of the best one.
    for i, (k, a) in enumerate(acs):
        left = acs[i - 1][1] if i > 0 else -math.inf
        right = acs[i + 1][1] if i + 1 < len(acs) else -math.inf
        if a >= 0.85 * best and a >= left and a >= right:
            return clamp01(a), rate / k
    k, a = max(acs, key=lambda ka: ka[1])
    return clamp01(a), rate / k
