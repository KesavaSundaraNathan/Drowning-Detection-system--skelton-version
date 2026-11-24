"""Pool calibration per camera (Phase 7). Read: any role. Write: supervisor+ (it decides where the water is).

Every camera of one pool is calibrated against the same pool plan in metres, so a swimmer's position is
comparable across cameras. A calibrated camera with no WATER zone drawn monitors the projected pool outline
(vision/scene/zone_state.py, app/live_startup.py) and the edge band (``edge_band_m``) splits
in_water / edge / deck. A running camera picks the change up before its next analysed frame.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.orm import Session

from lifeguard.app.api.v1.schemas import CalibrationIn, CalibrationOut
from lifeguard.app.api.v1.zones import zones_changed
from lifeguard.app.deps import AnyUser, Ctx, Supervisor, client_ip
from lifeguard.core import audit
from lifeguard.core.clock import utcnow
from lifeguard.db.models import Camera
from lifeguard.vision.scene.pool_geometry import CalibrationError, PoolCalibration

router = APIRouter(tags=["calibration"])

# the outline preview is computed on this reference frame; points are normalised, so any size of the view works
_REF_W, _REF_H = 1920, 1080


def describe(
    camera_id: str, data: dict[str, Any] | None, frame_w: int = _REF_W, frame_h: int = _REF_H
) -> CalibrationOut:
    if not data:
        return CalibrationOut(camera_id=camera_id, calibration=None)
    try:
        cal = PoolCalibration.from_json(data, frame_w, frame_h)
    except CalibrationError as exc:  # stored before a stricter check: show it, flag it
        return CalibrationOut(camera_id=camera_id, calibration={**data, "error": str(exc)})
    contour = cal.water_contour(frame_w, frame_h)
    outline = (
        [[round(float(x) / frame_w, 4), round(float(y) / frame_h, 4)] for x, y in contour.reshape(-1, 2)]
        if contour is not None
        else None
    )
    return CalibrationOut(camera_id=camera_id, calibration=data, **cal.summary(), water_outline_norm=outline)


@router.get("/cameras/{camera_id}/calibration", response_model=CalibrationOut)
def get_calibration(camera_id: str, ctx: Ctx, _: AnyUser) -> CalibrationOut:
    with ctx.db.read_sessions() as rs:
        cam = rs.get(Camera, camera_id)
        if cam is None:
            raise HTTPException(404, "camera_not_found")
        return describe(camera_id, cam.calibration)


@router.put("/cameras/{camera_id}/calibration", response_model=CalibrationOut)
def put_calibration(camera_id: str, body: CalibrationIn, request: Request, ctx: Ctx, me: Supervisor) -> CalibrationOut:
    data: dict[str, Any] = {
        "image_points": [[float(x), float(y)] for x, y in body.image_points],
        "pool_points": [[float(x), float(y)] for x, y in body.pool_points],
    }
    if body.outline_m:
        data["outline_m"] = [[float(x), float(y)] for x, y in body.outline_m]
    if body.edge_band_m is not None:
        data["edge_band_m"] = float(body.edge_band_m)
    try:
        cal = PoolCalibration.from_json(data, body.frame_width, body.frame_height)
    except CalibrationError as exc:
        raise HTTPException(422, f"invalid_calibration: {exc}") from None
    if cal.water_contour(body.frame_width, body.frame_height) is None:
        raise HTTPException(422, "invalid_calibration: the pool outline does not project into this view")
    now, ip = utcnow(), client_ip(request)

    def work(db: Session) -> None:
        cam = db.get(Camera, camera_id)
        if cam is None:
            raise HTTPException(404, "camera_not_found")
        had = cam.calibration is not None
        cam.calibration, cam.updated_at = data, now
        audit.record(
            db,
            action="camera.calibration.set",
            actor=me,
            ip=ip,
            target_type="camera",
            target_id=camera_id,
            details={"replaced": had, **cal.summary(), "edge_band_m": data.get("edge_band_m")},
        )

    ctx.writer.run(work)
    zones_changed(ctx, camera_id)
    return describe(camera_id, data, body.frame_width, body.frame_height)


@router.delete("/cameras/{camera_id}/calibration", response_model=CalibrationOut)
def delete_calibration(camera_id: str, request: Request, ctx: Ctx, me: Supervisor) -> CalibrationOut:
    ip = client_ip(request)

    def work(db: Session) -> None:
        cam = db.get(Camera, camera_id)
        if cam is None:
            raise HTTPException(404, "camera_not_found")
        if cam.calibration is not None:
            audit.record(
                db, action="camera.calibration.delete", actor=me, ip=ip, target_type="camera", target_id=camera_id
            )
        cam.calibration, cam.updated_at = None, utcnow()

    ctx.writer.run(work)
    zones_changed(ctx, camera_id)
    return CalibrationOut(camera_id=camera_id, calibration=None)
