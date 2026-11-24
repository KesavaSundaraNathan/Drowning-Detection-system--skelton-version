# mypy: disable-error-code="assignment"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""
[v3] Ported verbatim from the v2 baseline detection/pool_geometry.py (imports only).

detection/pool_geometry.py — one pool, several cameras: image ↔ pool floor plan.

Each camera is calibrated once by clicking ≥ 4 points of the pool outline (the
corners) in its view and giving their positions on the pool in metres
(corner A = (0, 0), B = (length, 0), C = (length, width), D = (0, width)).
A homography then maps any image point on the water plane to pool metres.

What it gives the pipeline
  * `to_pool(point_px)`          — where a person is on the shared pool plan
  * `pool_signed_distance(m)`    — metres inside (+) / outside (−) the pool edge
  * `water_contour(w, h)`        — the pool outline projected into the camera
                                   image, used as the camera's WATER zone when
                                   no zones are drawn: people on the deck are
                                   ignored until they enter the pool
[Phase 5] There is no cross-camera tracker any more (cameras are independent);
pool metres serve the pool-plan zones and the coverage check.

A person's pool position is taken at the bottom-centre of their box (feet on
the deck, or where the body meets the water for a swimmer): that point lies on
the water / deck plane the homography describes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

Point = tuple[float, float]


class CalibrationError(ValueError):
    pass


@dataclass
class PoolCalibration:
    image_points: np.ndarray  # (N, 2) px of the frame this was scaled to
    pool_points: np.ndarray  # (N, 2) metres
    outline_m: np.ndarray  # (M, 2) pool outline in metres (closed polygon)
    H: np.ndarray  # image px → pool m
    H_inv: np.ndarray  # pool m → image px
    frame_wh: tuple[int, int]
    reprojection_error_m: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    # ── construction ──────────────────────────────────────────────────────────
    @classmethod
    def from_json(cls, data: dict[str, Any], frame_w: int, frame_h: int) -> PoolCalibration:
        """
        data = {"image_points": [[x, y], …] normalised 0–1 (≥ 4),
                "pool_points":  [[x_m, y_m], …] same order,
                "outline_m":    optional [[x_m, y_m], …] (default: hull of pool_points)}
        """
        try:
            img = np.asarray(data["image_points"], np.float64).reshape(-1, 2)
            pool = np.asarray(data["pool_points"], np.float64).reshape(-1, 2)
        except Exception as exc:
            raise CalibrationError(f"image_points / pool_points must be lists of [x, y]: {exc}") from exc
        if len(img) < 4 or len(img) != len(pool):
            raise CalibrationError("need at least 4 image points and the same number of pool points")
        if np.any(img < -0.5) or np.any(img > 1.5):
            raise CalibrationError("image_points must be normalised to the frame (0–1)")
        px = img * np.array([frame_w, frame_h], np.float64)
        H, _ = cv2.findHomography(px, pool, 0)
        if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
            raise CalibrationError("the points do not define a valid mapping (are 3 of them on one line?)")
        H_inv = np.linalg.inv(H)
        proj = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = float(np.max(np.linalg.norm(proj - pool, axis=1)))
        outline = np.asarray(
            data.get("outline_m") or cv2.convexHull(pool.astype(np.float32)).reshape(-1, 2), np.float64
        ).reshape(-1, 2)
        return cls(px, pool, outline, H, H_inv, (frame_w, frame_h), err, dict(data))

    # ── mapping ───────────────────────────────────────────────────────────────
    def to_pool(self, point: Point) -> Point:
        p = cv2.perspectiveTransform(np.array([[point]], np.float64), self.H)[0, 0]
        return float(p[0]), float(p[1])

    def to_image(self, point_m: Point) -> Point:
        p = cv2.perspectiveTransform(np.array([[point_m]], np.float64), self.H_inv)[0, 0]
        return float(p[0]), float(p[1])

    def pool_signed_distance(self, point_m: Point) -> float:
        """Metres inside (+) or outside (−) the pool outline."""
        return float(
            cv2.pointPolygonTest(
                self.outline_m.astype(np.float32).reshape(-1, 1, 2), (float(point_m[0]), float(point_m[1])), True
            )
        )

    def water_contour(self, frame_w: int, frame_h: int, samples_per_edge: int = 24) -> np.ndarray | None:
        """
        The pool outline in this camera's image (int32 contour, clipped to the
        frame), or None if it is not visible. Edges are sampled densely so the
        part of the pool in front of the camera projects correctly even when a
        far corner lies outside the view.
        """
        sx = frame_w / self.frame_wh[0]
        sy = frame_h / self.frame_wh[1]
        pts_m = []
        n = len(self.outline_m)
        for i in range(n):
            a, b = self.outline_m[i], self.outline_m[(i + 1) % n]
            for k in range(samples_per_edge):
                pts_m.append(a + (b - a) * k / samples_per_edge)
        pts_m = np.asarray(pts_m, np.float64)
        # keep only points in front of the camera (positive homogeneous w)
        hom = np.hstack([pts_m, np.ones((len(pts_m), 1))]) @ self.H_inv.T
        # [v3 Phase 11] a homography is defined up to SIGN: when the frame's top-left pixel lies above the
        # horizon (walls / ceiling in view - typical for a corner camera) every visible point has w < 0. "In
        # front" is the side the calibration points are on - the installer saw them in this view.
        ref = np.hstack([self.pool_points, np.ones((len(self.pool_points), 1))]) @ self.H_inv.T
        front = hom[:, 2] * (1.0 if np.median(ref[:, 2]) > 0 else -1.0) > 1e-9
        if front.sum() < 3:
            return None
        img = hom[front, :2] / hom[front, 2:3]
        img[:, 0] *= sx
        img[:, 1] *= sy
        poly = _clip_to_frame(img, frame_w, frame_h)
        if poly is None or len(poly) < 3 or cv2.contourArea(poly.astype(np.float32)) < 0.002 * frame_w * frame_h:
            return None
        return np.round(poly).astype(np.int32).reshape(-1, 1, 2)

    def summary(self) -> dict[str, Any]:
        return {
            "points": len(self.pool_points),
            "reprojection_error_m": round(self.reprojection_error_m, 3),
            "pool_size_m": [
                round(float(self.outline_m[:, 0].max() - self.outline_m[:, 0].min()), 2),
                round(float(self.outline_m[:, 1].max() - self.outline_m[:, 1].min()), 2),
            ],
        }


def _clip_to_frame(poly: np.ndarray, w: int, h: int) -> np.ndarray | None:
    """Sutherland–Hodgman clip of a polygon (N, 2) to the frame rectangle."""

    def clip(pts: list[np.ndarray], inside, inter) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for i in range(len(pts)):
            cur, prev = pts[i], pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(inter(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(inter(prev, cur))
        return out

    def ix(a, b, x):
        t = (x - a[0]) / ((b[0] - a[0]) or 1e-12)
        return np.array([x, a[1] + t * (b[1] - a[1])])

    def iy(a, b, y):
        t = (y - a[1]) / ((b[1] - a[1]) or 1e-12)
        return np.array([a[0] + t * (b[0] - a[0]), y])

    pts = [p for p in poly]
    for inside, inter in (
        (lambda p: p[0] >= 0, lambda a, b: ix(a, b, 0.0)),
        (lambda p: p[0] <= w, lambda a, b: ix(a, b, float(w))),
        (lambda p: p[1] >= 0, lambda a, b: iy(a, b, 0.0)),
        (lambda p: p[1] <= h, lambda a, b: iy(a, b, float(h))),
    ):
        pts = clip(pts, inside, inter)
        if not pts:
            return None
    return np.asarray(pts)


def default_pool_points(length_m: float, width_m: float) -> list[list[float]]:
    """Corner order A, B, C, D = (0,0), (L,0), (L,W), (0,W)."""
    return [[0.0, 0.0], [float(length_m), 0.0], [float(length_m), float(width_m)], [0.0, float(width_m)]]
