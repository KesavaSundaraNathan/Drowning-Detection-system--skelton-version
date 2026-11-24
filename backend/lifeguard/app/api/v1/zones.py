"""Zones per camera, and [Phase 11 WP2] pool zones. Read: any role. Write: supervisor+ (zones decide what is
monitored).

A pool zone is drawn once on the pool plan in metres (`/pool-zones`); every calibrated camera of that pool projects
it into its own view when its geometry loads (app/live_startup.py, vision/scene/pool_zones.py), so one zone gives
the same membership on every camera. `/pool-zones/{id}/projections` previews it per camera. Camera zones (image
polygons) are unchanged; `PUT /zones/{id}` edits camera zones only.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from lifeguard.app.api.v1.schemas import PoolZoneIn, PoolZonePreviewIn, PoolZoneProjection, ZoneIn, ZoneOut
from lifeguard.app.context import AppContext
from lifeguard.app.deps import AnyUser, Ctx, Supervisor, client_ip
from lifeguard.core import audit
from lifeguard.core.clock import utcnow
from lifeguard.db.models import Camera, Schedule, Zone

router = APIRouter(tags=["zones"])


def _check_refs(db: Session, camera_id: str | None, schedule_id: int | None) -> None:
    if camera_id is not None and db.get(Camera, camera_id) is None:
        raise HTTPException(404, "camera_not_found")
    if schedule_id is not None and db.get(Schedule, schedule_id) is None:
        raise HTTPException(422, "unknown_schedule")


def _summary(z: Zone) -> dict[str, object]:
    return {
        "name": z.name,
        "zone_type": z.zone_type,
        "sensitivity": z.sensitivity,
        "schedule_id": z.schedule_id,
        "is_active": z.is_active,
        "points": len(z.polygon or z.pool_polygon_m or []),
        **({"pool_id": z.pool_id} if z.pool_id is not None else {}),
    }


@router.get("/cameras/{camera_id}/zones", response_model=list[ZoneOut])
def list_zones(camera_id: str, ctx: Ctx, _: AnyUser) -> list[ZoneOut]:
    with ctx.db.read_sessions() as rs:
        if rs.get(Camera, camera_id) is None:
            raise HTTPException(404, "camera_not_found")
        rows = rs.execute(select(Zone).where(Zone.camera_id == camera_id).order_by(Zone.name)).scalars()
        return [ZoneOut.model_validate(z) for z in rows]


@router.post("/cameras/{camera_id}/zones", response_model=ZoneOut, status_code=201)
def create_zone(camera_id: str, body: ZoneIn, request: Request, ctx: Ctx, me: Supervisor) -> ZoneOut:
    now, ip = utcnow(), client_ip(request)

    def work(db: Session) -> ZoneOut:
        _check_refs(db, camera_id, body.schedule_id)
        zone = Zone(
            camera_id=camera_id,
            name=body.name,
            zone_type=body.zone_type,
            polygon=[list(p) for p in body.polygon],
            sensitivity=body.sensitivity,
            schedule_id=body.schedule_id,
            is_active=body.is_active,
            created_at=now,
            updated_at=now,
        )
        db.add(zone)
        db.flush()
        audit.record(
            db,
            action="zone.create",
            actor=me,
            ip=ip,
            target_type="zone",
            target_id=zone.id,
            details={"camera_id": camera_id, **_summary(zone)},
        )
        return ZoneOut.model_validate(zone)

    try:
        out: ZoneOut = ctx.writer.run(work)
    except IntegrityError:
        raise HTTPException(409, "zone_name_taken") from None
    zones_changed(ctx, out.camera_id)
    return out


@router.get("/zones/{zone_id}", response_model=ZoneOut)
def get_zone(zone_id: int, ctx: Ctx, _: AnyUser) -> ZoneOut:
    with ctx.db.read_sessions() as rs:
        zone = rs.get(Zone, zone_id)
        if zone is None:
            raise HTTPException(404, "not_found")
        return ZoneOut.model_validate(zone)


@router.put("/zones/{zone_id}", response_model=ZoneOut)
def replace_zone(zone_id: int, body: ZoneIn, request: Request, ctx: Ctx, me: Supervisor) -> ZoneOut:
    now, ip = utcnow(), client_ip(request)

    def work(db: Session) -> ZoneOut:
        zone = db.get(Zone, zone_id)
        if zone is None:
            raise HTTPException(404, "not_found")
        if zone.camera_id is None:
            raise HTTPException(409, "pool_zone: edit it with PUT /pool-zones/{id}")
        _check_refs(db, zone.camera_id, body.schedule_id)
        before = _summary(zone)
        zone.name, zone.zone_type, zone.polygon = body.name, body.zone_type, [list(p) for p in body.polygon]
        zone.sensitivity, zone.schedule_id, zone.is_active = body.sensitivity, body.schedule_id, body.is_active
        zone.updated_at = now
        audit.record(
            db,
            action="zone.update",
            actor=me,
            ip=ip,
            target_type="zone",
            target_id=zone.id,
            details={"before": before, "after": _summary(zone)},
        )
        return ZoneOut.model_validate(zone)

    try:
        out: ZoneOut = ctx.writer.run(work)
    except IntegrityError:
        raise HTTPException(409, "zone_name_taken") from None
    zones_changed(ctx, out.camera_id)
    return out


@router.delete("/zones/{zone_id}", status_code=204)
def delete_zone(zone_id: int, request: Request, ctx: Ctx, me: Supervisor) -> Response:
    ip = client_ip(request)

    def work(db: Session) -> str | None:
        zone = db.get(Zone, zone_id)
        if zone is None:
            raise HTTPException(404, "not_found")
        camera_id = zone.camera_id
        audit.record(
            db,
            action="zone.delete",
            actor=me,
            ip=ip,
            target_type="zone",
            target_id=zone.id,
            details={"camera_id": zone.camera_id, **_summary(zone)},
        )
        db.delete(zone)
        return camera_id

    zones_changed(ctx, ctx.writer.run(work))
    return Response(status_code=204)


def zones_changed(ctx: AppContext, camera_id: str | None) -> None:
    """Running cameras pick up zone / schedule edits before their next analysed frame (Phase 6)."""
    if ctx.live is not None:
        ctx.live.zones_changed(camera_id)


# ── [Phase 11 WP2] pool zones ─────────────────────────────────────────────────
@router.get("/pool-zones", response_model=list[ZoneOut])
def list_pool_zones(ctx: Ctx, _: AnyUser, pool_id: str | None = None) -> list[ZoneOut]:
    with ctx.db.read_sessions() as rs:
        q = select(Zone).where(Zone.camera_id.is_(None)).order_by(Zone.name)
        if pool_id is not None:
            q = q.where(Zone.pool_id == pool_id)
        return [ZoneOut.model_validate(z) for z in rs.execute(q).scalars()]


@router.post("/pool-zones", response_model=ZoneOut, status_code=201)
def create_pool_zone(body: PoolZoneIn, request: Request, ctx: Ctx, me: Supervisor) -> ZoneOut:
    now, ip = utcnow(), client_ip(request)

    def work(db: Session) -> ZoneOut:
        _check_refs(db, None, body.schedule_id)
        zone = Zone(
            camera_id=None,
            name=body.name,
            zone_type=body.zone_type,
            polygon=None,
            pool_id=body.pool_id,
            pool_polygon_m=[list(p) for p in body.pool_polygon_m],
            sensitivity=body.sensitivity,
            schedule_id=body.schedule_id,
            is_active=body.is_active,
            created_at=now,
            updated_at=now,
        )
        db.add(zone)
        db.flush()
        audit.record(
            db, action="zone.create", actor=me, ip=ip, target_type="zone", target_id=zone.id, details=_summary(zone)
        )
        return ZoneOut.model_validate(zone)

    try:
        out: ZoneOut = ctx.writer.run(work)
    except IntegrityError:
        raise HTTPException(409, "zone_name_taken") from None
    zones_changed(ctx, None)  # every camera of the pool reloads its geometry
    return out


@router.put("/pool-zones/{zone_id}", response_model=ZoneOut)
def replace_pool_zone(zone_id: int, body: PoolZoneIn, request: Request, ctx: Ctx, me: Supervisor) -> ZoneOut:
    now, ip = utcnow(), client_ip(request)

    def work(db: Session) -> ZoneOut:
        zone = db.get(Zone, zone_id)
        if zone is None or zone.camera_id is not None:
            raise HTTPException(404, "not_found")
        _check_refs(db, None, body.schedule_id)
        before = _summary(zone)
        zone.name, zone.zone_type, zone.pool_id = body.name, body.zone_type, body.pool_id
        zone.pool_polygon_m = [list(p) for p in body.pool_polygon_m]
        zone.sensitivity, zone.schedule_id, zone.is_active = body.sensitivity, body.schedule_id, body.is_active
        zone.updated_at = now
        audit.record(
            db,
            action="zone.update",
            actor=me,
            ip=ip,
            target_type="zone",
            target_id=zone.id,
            details={"before": before, "after": _summary(zone)},
        )
        return ZoneOut.model_validate(zone)

    try:
        out: ZoneOut = ctx.writer.run(work)
    except IntegrityError:
        raise HTTPException(409, "zone_name_taken") from None
    zones_changed(ctx, None)
    return out


@router.get("/pool-zones/{zone_id}/projections", response_model=list[PoolZoneProjection])
def pool_zone_projections(zone_id: int, ctx: Ctx, _: AnyUser) -> list[PoolZoneProjection]:
    """Per calibrated camera of the zone's pool: where the zone lands in its view."""
    with ctx.db.read_sessions() as rs:
        zone = rs.get(Zone, zone_id)
        if zone is None or zone.camera_id is not None or not zone.pool_polygon_m or zone.pool_id is None:
            raise HTTPException(404, "not_found")
        poly_m, pool = zone.pool_polygon_m, zone.pool_id
    return _projections(ctx, poly_m, pool)


@router.post("/pool-zones/preview", response_model=list[PoolZoneProjection])
def pool_zone_preview(body: PoolZonePreviewIn, ctx: Ctx, _: Supervisor) -> list[PoolZoneProjection]:
    """The editor's live preview: where an UNSAVED pool polygon would land in every calibrated camera of the pool.
    Nothing is stored or audited (read-only; supervisor+ because only the zone editor uses it)."""
    return _projections(ctx, [list(p) for p in body.pool_polygon_m], body.pool_id)


def _projections(ctx: AppContext, poly_m: list[list[float]], pool: str) -> list[PoolZoneProjection]:
    from lifeguard.vision.scene.pool_geometry import CalibrationError, PoolCalibration
    from lifeguard.vision.scene.pool_zones import pool_key, project_polygon

    w, h = 1920, 1080  # the homography is scale-free: the normalised output does not depend on the size used
    with ctx.db.read_sessions() as rs:
        cams = [(c.id, c.name, c.calibration) for c in rs.execute(select(Camera).order_by(Camera.name)).scalars()]
    out = []
    for cid, name, data in cams:
        if not data:
            continue
        try:
            cal = PoolCalibration.from_json(data, w, h)
        except CalibrationError:
            continue
        if pool_key(cal) != pool:
            continue
        contour = project_polygon(cal, poly_m, w, h)
        norm = (
            None
            if contour is None
            else [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in contour.reshape(-1, 2)]
        )
        out.append(PoolZoneProjection(camera_id=cid, camera_name=name, polygon=norm))
    return out
