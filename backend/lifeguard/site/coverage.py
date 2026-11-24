"""Calibration + coverage check (Phase 11 WP6, gates_wp6.session_checks.coverage).

Every enabled camera must be calibrated to ONE pool (same pool key, outlines agreeing within OUTLINE_TOL_M), and every
0.5 m cell of that pool must have at least one RESPONSIBLE camera (live/responsibility.py: in view, enough pixels per
metre, within RESP_DENSITY_RATIO of the best camera) - a cell with none is a coverage hole. The overlap band (cells
with >= 2 responsible cameras) is reported. A PNG map shows holes / single / overlap per cell.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

STEP_M = 0.5
OUTLINE_TOL_M = 0.25


def check_coverage(cameras: list[dict[str, Any]], *, step_m: float = STEP_M) -> dict[str, Any]:
    """`cameras`: [{"camera_id", "name", "calibration": dict | None, "frame_size": [w, h]}] - the ENABLED cameras."""
    from lifeguard.live.responsibility import CameraView, coverage
    from lifeguard.vision import constants as C
    from lifeguard.vision.scene.pool_geometry import CalibrationError, PoolCalibration
    from lifeguard.vision.scene.pool_zones import pool_key

    problems: list[str] = []
    views: list[CameraView] = []
    keys: dict[str, str] = {}
    outlines: dict[str, np.ndarray] = {}
    for cam in cameras:
        cal_json = cam.get("calibration")
        if not cal_json:
            problems.append(f"{cam['name']}: not calibrated")
            continue
        w, h = cam["frame_size"]
        try:
            cal = PoolCalibration.from_json(cal_json, int(w), int(h))
        except CalibrationError as exc:
            problems.append(f"{cam['name']}: calibration invalid ({exc})")
            continue
        views.append(CameraView(cam["camera_id"], cal, int(w), int(h)))
        keys[cam["camera_id"]] = pool_key(cal)
        outlines[cam["camera_id"]] = np.asarray(cal.outline_m, np.float64)
    if len(set(keys.values())) > 1:
        problems.append(
            f"cameras calibrated to {len(set(keys.values()))} different pools: {sorted(set(keys.values()))}"
        )
    if outlines:
        ref_id, ref = next(iter(outlines.items()))
        for cid, out in outlines.items():
            if out.shape != ref.shape or float(np.max(np.abs(out - ref))) > OUTLINE_TOL_M:
                problems.append(f"camera {cid}: pool outline differs from camera {ref_id}'s (> {OUTLINE_TOL_M} m)")
    if not views:
        return {"verdict": "FAIL", "problems": problems or ["no calibrated camera"], "coverage": None}
    outline = next(iter(outlines.values())).tolist()
    cov = coverage(views, outline, ratio=C.RESP_DENSITY_RATIO, min_px_per_m=C.RESP_MIN_PX_PER_M, step_m=step_m)
    names = {c["camera_id"]: c["name"] for c in cameras}
    cov["cells_per_camera"] = {names.get(k, k): v for k, v in cov["cells_per_camera"].items()}
    if cov["holes"]:
        problems.append(f"{cov['holes']} of {cov['cells']} pool cells ({step_m} m) have no responsible camera")
    return {
        "verdict": "FAIL" if problems else "PASS",
        "problems": problems,
        "pool_key": next(iter(keys.values())) if keys else None,
        "outline_m": outline,
        "coverage": cov,
        "_views": views,  # for the map (dropped from the JSON report)
    }


def draw_map(result: dict[str, Any], path: Path, *, px_per_m: int = 40) -> Path | None:
    """PNG: each cell red (hole), green (1 responsible camera) or blue (overlap); camera corners labelled."""
    import cv2

    from lifeguard.live.responsibility import responsible
    from lifeguard.vision import constants as C

    views = result.get("_views") or []
    outline = result.get("outline_m")
    if not views or not outline:
        return None
    ol = np.asarray(outline, np.float64)
    x0, y0 = ol.min(axis=0)
    x1, y1 = ol.max(axis=0)
    step = STEP_M
    margin = 30
    img = np.full((int((y1 - y0) * px_per_m) + 2 * margin, int((x1 - x0) * px_per_m) + 2 * margin, 3), 255, np.uint8)

    def px(x: float, y: float) -> tuple[int, int]:
        return int(margin + (x - x0) * px_per_m), int(margin + (y - y0) * px_per_m)

    x = x0 + step / 2
    while x < x1:
        y = y0 + step / 2
        while y < y1:
            n = len(responsible(views, (x, y), ratio=C.RESP_DENSITY_RATIO, min_px_per_m=C.RESP_MIN_PX_PER_M))
            colour = (60, 60, 230) if n == 0 else (110, 200, 110) if n == 1 else (220, 150, 60)
            cv2.rectangle(img, px(x - step / 2, y - step / 2), px(x + step / 2, y + step / 2), colour, -1)
            y += step
        x += step
    cv2.polylines(img, [np.array([px(*p) for p in ol], np.int32)], True, (0, 0, 0), 2)
    cv2.putText(
        img,
        "red: hole  green: 1 camera  blue: overlap",
        (margin, margin - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def public(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if not k.startswith("_")}
