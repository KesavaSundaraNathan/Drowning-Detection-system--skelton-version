"""/api/v1/cameras: camera registry. Read: any role. Write: admin.

Sources pass the allowlist (core.security.sources), are stored Fernet-encrypted, and are only ever
returned masked. No endpoint returns or logs a decrypted source; the capture worker (Phase 4) is
the only consumer of `decrypt_source`. Audit details carry the masked form only.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from lifeguard.app.api.v1.schemas import CameraIn, CameraOut, CameraPatchIn
from lifeguard.app.deps import Admin, AnyUser, Ctx, client_ip
from lifeguard.core import audit
from lifeguard.core.clock import utcnow
from lifeguard.core.security.sources import (
    SourceError,
    ValidatedSource,
    encrypt_source,
    validate_rtsp,
)
from lifeguard.db.models import Camera

router = APIRouter(prefix="/cameras", tags=["cameras"])


def _validate(source_url: str) -> ValidatedSource:
    try:
        return validate_rtsp(source_url)
    except SourceError as exc:
        raise HTTPException(422, f"invalid_source: {exc}") from None


@router.get("", response_model=list[CameraOut])
def list_cameras(ctx: Ctx, _: AnyUser) -> list[CameraOut]:
    with ctx.db.read_sessions() as rs:
        return [CameraOut.model_validate(c) for c in rs.execute(select(Camera).order_by(Camera.name)).scalars()]


@router.get("/{camera_id}", response_model=CameraOut)
def get_camera(camera_id: str, ctx: Ctx, _: AnyUser) -> CameraOut:
    with ctx.db.read_sessions() as rs:
        cam = rs.get(Camera, camera_id)
        if cam is None:
            raise HTTPException(404, "not_found")
        return CameraOut.model_validate(cam)


@router.post("", response_model=CameraOut, status_code=201)
def create_camera(body: CameraIn, request: Request, ctx: Ctx, me: Admin) -> CameraOut:
    src = _validate(body.source_url.get_secret_value())
    cipher, now, ip = encrypt_source(ctx.keys.fernet, src.value), utcnow(), client_ip(request)

    def work(db: Session) -> CameraOut:
        cam = Camera(
            id=str(uuid.uuid4()),
            name=body.name,
            source_type=src.source_type,
            source_ciphertext=cipher,
            source_display=src.display,
            default_mode=body.default_mode,
            is_enabled=body.is_enabled,
            scene_type=body.scene_type,
            scene_type_set_by="admin" if body.scene_type != "pool" else "default",
            created_at=now,
            updated_at=now,
        )
        db.add(cam)
        db.flush()
        audit.record(
            db,
            action="camera.create",
            actor=me,
            ip=ip,
            target_type="camera",
            target_id=cam.id,
            details={
                "name": cam.name,
                "source_display": src.display,
                "is_enabled": cam.is_enabled,
                "scene_type": cam.scene_type,
            },
        )
        return CameraOut.model_validate(cam)

    try:
        return ctx.writer.run(work)
    except IntegrityError:
        raise HTTPException(409, "camera_name_taken") from None


@router.patch("/{camera_id}", response_model=CameraOut)
def update_camera(camera_id: str, body: CameraPatchIn, request: Request, ctx: Ctx, me: Admin) -> CameraOut:
    src = None
    if body.source_url is not None:
        src = _validate(body.source_url.get_secret_value())
    cipher = encrypt_source(ctx.keys.fernet, src.value) if src else None
    now, ip = utcnow(), client_ip(request)

    def work(db: Session) -> CameraOut:
        cam = db.get(Camera, camera_id)
        if cam is None:
            raise HTTPException(404, "not_found")
        changed: dict[str, object] = {}
        for field in ("name", "default_mode", "is_enabled", "scene_type"):
            value = getattr(body, field)
            if value is not None and value != getattr(cam, field):
                changed[field] = value
                setattr(cam, field, value)
        if "scene_type" in changed:  # [v4] an admin decided: drowning detection on (pool) or off (standard CCTV)
            cam.scene_type_set_by = "admin"
        if src is not None and cipher is not None:
            cam.source_type, cam.source_ciphertext, cam.source_display = src.source_type, cipher, src.display
            changed["source_display"] = src.display
        cam.updated_at = now
        audit.record(
            db,
            action="camera.update",
            actor=me,
            ip=ip,
            target_type="camera",
            target_id=cam.id,
            details={"changed": changed},
        )
        return CameraOut.model_validate(cam)

    try:
        out = ctx.writer.run(work)
    except IntegrityError:
        raise HTTPException(409, "camera_name_taken") from None
    if body.scene_type is not None and ctx.live is not None:
        ctx.live.scene_type_changed(camera_id, out.scene_type)
    return out


@router.delete("/{camera_id}", status_code=204)
def delete_camera(camera_id: str, request: Request, ctx: Ctx, me: Admin) -> Response:
    ip = client_ip(request)
    if ctx.live is not None:  # a removed camera must not keep being analysed (its alerts would have no camera)
        ctx.live.forget_camera(camera_id)

    def work(db: Session) -> None:
        cam = db.get(Camera, camera_id)
        if cam is None:
            raise HTTPException(404, "not_found")
        audit.record(
            db,
            action="camera.delete",
            actor=me,
            ip=ip,
            target_type="camera",
            target_id=cam.id,
            details={"name": cam.name, "source_display": cam.source_display},
        )
        db.delete(cam)

    ctx.writer.run(work)
    return Response(status_code=204)
