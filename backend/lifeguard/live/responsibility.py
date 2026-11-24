"""Per-camera responsibility geometry (Phase 11 WP2): which calibrated camera resolves a spot of the pool best.

[Phase 5] Cameras are independent (owner directive 2026-11-13): the live runtime no longer defers, mutes or tiles
anything by responsibility and there is no shared pool identity. What remains here is calibration geometry used by
the site coverage check (site/coverage.py) and offline site tooling; `deferral_target` / `TileRegion` are kept for
the recorded Phase 11 site replays only.

Resolving power decides it, straight from the calibration (no extra configuration): at pool point p a camera's
density is the image pixels per metre of pool there - sqrt(|det J|) of its pool -> image mapping, at the size the
frames are analysed. A camera is RESPONSIBLE for p when p is in front of it, inside its frame, its density there is
at least RESP_MIN_PX_PER_M (a person is big enough to analyse) and at least 1 / RESP_DENSITY_RATIO of the best
camera's density at p. For corner cameras that is each camera's near half, plus an overlap band where two cameras
resolve the pool about equally well - both are responsible there.

Gating (switch RESPONSIBILITY_GATING, thresholds.json; OFF until the pre-registered site replay decides): a behaviour
alert from a camera NOT responsible for the person's spot is still recorded and linked to the person's incident, but
does not sound ("deferred to <camera>") IF a responsible camera is MONITORED and sees the same person (global id)
right now. Never deferred: SUBMERSION and FAULT (live/policy.NEVER_MUTED), submersion ghosts, people without a global
id, and anybody no responsible camera currently sees - the alert then sounds as before (fail-safe).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lifeguard.vision.scene.pool_zones import contains_m, front_halfplane

Point = tuple[float, float]
STEP_M = 0.05  # finite-difference step for the Jacobian


@dataclass(frozen=True)
class CameraView:
    """A calibrated camera as responsibility sees it: its calibration and the size its frames are analysed at."""

    camera_id: str
    calibration: Any
    frame_w: int
    frame_h: int


def _image(view: CameraView, p: Point) -> tuple[float, float]:
    u, v = view.calibration.to_image(p)
    return u * view.frame_w / view.calibration.frame_wh[0], v * view.frame_h / view.calibration.frame_wh[1]


def in_view(view: CameraView, p: Point) -> bool:
    a, b, c = front_halfplane(view.calibration)
    if a * p[0] + b * p[1] + c <= 0:
        return False
    u, v = _image(view, p)
    return 0.0 <= u < view.frame_w and 0.0 <= v < view.frame_h


def density(view: CameraView, p: Point) -> float:
    """Image pixels per metre of pool at p (0 when p is not in view)."""
    if not in_view(view, p):
        return 0.0
    u0, v0 = _image(view, p)
    ux, vx = _image(view, (p[0] + STEP_M, p[1]))
    uy, vy = _image(view, (p[0], p[1] + STEP_M))
    det = ((ux - u0) * (vy - v0) - (uy - u0) * (vx - v0)) / (STEP_M * STEP_M)
    return math.sqrt(abs(det))


def responsible(views: Sequence[CameraView], p: Point, *, ratio: float, min_px_per_m: float) -> set[str]:
    dens = {v.camera_id: density(v, p) for v in views}
    usable = {c: d for c, d in dens.items() if d >= min_px_per_m}
    if not usable:
        return set()
    best = max(usable.values())
    return {c for c, d in usable.items() if d >= best / ratio}


def coverage(
    views: Sequence[CameraView],
    outline_m: Sequence[Sequence[float]],
    *,
    ratio: float,
    min_px_per_m: float,
    step_m: float,
) -> dict[str, Any]:
    """Responsibility over the pool on a `step_m` grid: cells without a responsible camera are coverage holes."""
    xs = [p[0] for p in outline_m]
    ys = [p[1] for p in outline_m]
    cells: list[dict[str, Any]] = []
    x = min(xs) + step_m / 2
    while x < max(xs):
        y = min(ys) + step_m / 2
        while y < max(ys):
            if contains_m(outline_m, (x, y)):
                r = responsible(views, (x, y), ratio=ratio, min_px_per_m=min_px_per_m)
                cells.append({"x": round(x, 2), "y": round(y, 2), "responsible": sorted(r)})
            y += step_m
        x += step_m
    holes = [c for c in cells if not c["responsible"]]
    per_camera = {v.camera_id: sum(1 for c in cells if v.camera_id in c["responsible"]) for v in views}
    shared = sum(1 for c in cells if len(c["responsible"]) > 1)
    return {
        "cells": len(cells),
        "holes": len(holes),
        "hole_cells": holes[:20],
        "cells_per_camera": per_camera,
        "overlap_cells": shared,
        "step_m": step_m,
    }


def deferral_target(
    camera_id: str,
    point_m: Point,
    views: Sequence[CameraView],
    sees_person_now: Mapping[str, bool],
    monitored: Mapping[str, bool],
    *,
    ratio: float,
    min_px_per_m: float,
) -> str | None:
    """The responsible camera this alert defers to, or None (this camera alerts itself - the fail-safe default)."""
    resp = responsible(views, point_m, ratio=ratio, min_px_per_m=min_px_per_m)
    if not resp or camera_id in resp:
        return None
    for cid in sorted(resp):
        if monitored.get(cid, False) and sees_person_now.get(cid, False):
            return cid
    return None


class TileRegion:
    """[WP3] DET_TILING_REGION="responsible": keep a tile window only if it contains a pool cell this camera is
    responsible for (cell centres on a RESP_CELL_M grid, projected into the camera's frame)."""

    CELL_M = 0.5

    def __init__(self, points_px: Sequence[tuple[float, float]]) -> None:
        self.points = list(points_px)

    @classmethod
    def for_camera(cls, view: CameraView, pool: Sequence[CameraView], ratio: float, min_px_per_m: float) -> TileRegion:
        outline = view.calibration.outline_m.tolist()
        xs, ys = [p[0] for p in outline], [p[1] for p in outline]
        pts: list[tuple[float, float]] = []
        x = min(xs) + cls.CELL_M / 2
        while x < max(xs):
            y = min(ys) + cls.CELL_M / 2
            while y < max(ys):
                p = (x, y)
                if contains_m(outline, p) and view.camera_id in responsible(
                    pool, p, ratio=ratio, min_px_per_m=min_px_per_m
                ):
                    pts.append(_image(view, p))
                y += cls.CELL_M
            x += cls.CELL_M
        return cls(pts)

    def __call__(self, window: Sequence[float]) -> bool:
        x1, y1, x2, y2 = (float(v) for v in window)
        return any(x1 <= u < x2 and y1 <= v < y2 for u, v in self.points)


def bbox_pool_point(cal: Any, bbox: Sequence[float], frame_w: int, frame_h: int) -> Point:
    """Pool position of a box: its bottom-centre (pool_geometry convention), in calibration pixel scale."""
    x1, _y1, x2, y2 = (float(v) for v in bbox)
    sx, sy = cal.frame_wh[0] / frame_w, cal.frame_wh[1] / frame_h
    px, py = cal.to_pool(((x1 + x2) / 2.0 * sx, y2 * sy))
    return float(px), float(py)


__all__ = [
    "CameraView",
    "TileRegion",
    "bbox_pool_point",
    "coverage",
    "deferral_target",
    "density",
    "in_view",
    "responsible",
]
