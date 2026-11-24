"""Pool-level zones (Phase 11 WP2): one zone drawn on the pool plan in metres, projected into every calibrated camera.

    contour = project_polygon(calibration, [[x_m, y_m], ...], frame_w, frame_h)   # int32 contour, or None

The projection goes through `PoolCalibration.to_image` (pool metres -> image px). A homography maps straight
pool edges to straight image edges, so a point lies inside the projected contour exactly when its pool position
lies inside the zone: every camera that sees a spot gives it the same zone (up to pixel rounding at the edge).

Two clips keep it exact:
  * pool side: the polygon is clipped to the half-plane of the pool plan that lies IN FRONT of the camera (the
    homogeneous w of `H_inv`, signed by the side the calibration points are on - a homography's sign is
    arbitrary), before dividing by w: points behind the camera would otherwise project to mirrored garbage;
  * image side: the projected outline is clipped to the frame (Sutherland-Hodgman, pool_geometry._clip_to_frame).
Unlike `PoolCalibration.water_contour` there is no minimum area: a small zone at the far end stays a zone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from lifeguard.vision.scene.pool_geometry import _clip_to_frame

Point = tuple[float, float]


DEFAULT_POOL = "pool"


def pool_key(calibration: Any) -> str:
    """The pool a calibrated camera belongs to (its calibration's pool_id; "pool" when unset). Groups cameras for the
    shared pool-plan zones only - never for identity or alerts (cameras are independent, Phase 5)."""
    raw = getattr(calibration, "raw", None) or {}
    return str(raw.get("pool_id") or DEFAULT_POOL)[:64]


def front_halfplane(cal: Any) -> tuple[float, float, float]:
    """(a, b, c) with a*x + b*y + c > 0 for pool points in front of the camera."""
    row = np.asarray(cal.H_inv, np.float64)[2]
    ref = np.hstack([cal.pool_points, np.ones((len(cal.pool_points), 1))]) @ row
    sign = 1.0 if np.median(ref) > 0 else -1.0
    a, b, c = (sign * row).tolist()
    return float(a), float(b), float(c)


def _clip_halfplane(pts: list[np.ndarray], a: float, b: float, c: float, eps: float) -> list[np.ndarray]:
    def w(p: np.ndarray) -> float:
        return float(a * p[0] + b * p[1] + c)

    out: list[np.ndarray] = []
    for i in range(len(pts)):
        cur, prev = pts[i], pts[i - 1]
        wc, wp = w(cur) - eps, w(prev) - eps
        if wc > 0:
            if wp <= 0:
                out.append(prev + (cur - prev) * (wp / (wp - wc)))
            out.append(cur)
        elif wp > 0:
            out.append(prev + (cur - prev) * (wp / (wp - wc)))
    return out


def project_polygon(
    cal: Any,
    poly_m: Sequence[Sequence[float]],
    frame_w: int,
    frame_h: int,
    samples_per_edge: int = 24,
    *,
    rounded: bool = True,
) -> np.ndarray | None:
    """The pool-plan polygon as an int32 contour (N, 1, 2) in a `frame_w` x `frame_h` frame, None if not in view.
    `rounded=False`: the same contour before rounding to whole pixels (float64)."""
    pts = [np.asarray(p, np.float64) for p in poly_m]
    if len(pts) < 3:
        return None
    a, b, c = front_halfplane(cal)
    scale = max(abs(a), abs(b), abs(c), 1e-12)
    pts = _clip_halfplane(pts, a, b, c, 1e-4 * scale)
    if len(pts) < 3:
        return None
    dense: list[np.ndarray] = []
    for i in range(len(pts)):
        p, q = pts[i], pts[(i + 1) % len(pts)]
        dense += [p + (q - p) * k / samples_per_edge for k in range(samples_per_edge)]
    sx, sy = frame_w / cal.frame_wh[0], frame_h / cal.frame_wh[1]
    img = np.array([cal.to_image((float(p[0]), float(p[1]))) for p in dense], np.float64) * [sx, sy]
    poly = _clip_to_frame(img, frame_w, frame_h)
    if poly is None or len(poly) < 3 or cv2.contourArea(poly.astype(np.float32)) < 1.0:
        return None
    if not rounded:
        return np.asarray(poly, np.float64).reshape(-1, 1, 2)
    return np.round(poly).astype(np.int32).reshape(-1, 1, 2)


def contains_m(poly_m: Sequence[Sequence[float]], point_m: Point) -> bool:
    """Is a pool-plan point inside the zone (boundary counts as inside)?"""
    c = np.asarray(poly_m, np.float32).reshape(-1, 1, 2)
    return bool(cv2.pointPolygonTest(c, (float(point_m[0]), float(point_m[1])), False) >= 0)


def edge_distance_m(poly_m: Sequence[Sequence[float]], point_m: Point) -> float:
    """Metres from the zone boundary (unsigned)."""
    c = np.asarray(poly_m, np.float32).reshape(-1, 1, 2)
    return abs(float(cv2.pointPolygonTest(c, (float(point_m[0]), float(point_m[1])), True)))
