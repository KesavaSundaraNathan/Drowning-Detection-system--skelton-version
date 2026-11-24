"""/api/v1/live: live monitoring status and control (plan Phase 4).

GET  /live/status                       any role - per camera: MONITORED / NOT_MONITORED, stalled,
                                        lagging, analysed fraction, zone MUTED states; scheduler and
                                        capacity figures. Never contains a camera source.
POST /live/cameras/{id}/start           supervisor - admission-controlled: 409 with the computed
                                        camera limit when the server cannot analyse one more camera
                                        at TARGET_ANALYSIS_FPS (cameras never silently lag).
POST /live/cameras/{id}/stop            supervisor - stopping monitoring is audited.
GET  /live/cameras/{id}/frame.jpg       any role - newest frame with the Model View boxes (Phase 6)
GET  /live/cameras/{id}/mjpeg           any role - multipart MJPEG of the same, <= mjpeg_fps, at most
                                        mjpeg_max_streams viewers per server (503 beyond: CPU guard).
                                        Same-origin <img>: the session cookie authenticates it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Literal

from anyio import to_thread
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from lifeguard.app.context import AppContext
from lifeguard.app.deps import AnyUser, Ctx, Supervisor, client_ip
from lifeguard.core import audit
from lifeguard.core.security.rbac import Principal
from lifeguard.core.security.sources import SourceError
from lifeguard.live.capacity import CapacityExceededError
from lifeguard.live.view import BOXES_HEADER, render, render_client

router = APIRouter(prefix="/live", tags=["live"])


def _service(ctx: AppContext) -> Any:
    if ctx.live is None or ctx.live.runtime is None:
        raise HTTPException(503, "live_runtime_disabled")
    return ctx.live


def _audit(
    ctx: AppContext, request: Request, me: Principal, action: str, camera_id: str, outcome: str, **details: Any
) -> None:
    def work(db: Session) -> None:
        audit.record(
            db,
            action=action,
            actor=me,
            ip=client_ip(request),
            target_type="camera",
            target_id=camera_id,
            outcome=outcome,
            details=details,
        )

    ctx.writer.run(work)


@router.get("/status")
def live_status(ctx: Ctx, _: AnyUser) -> dict[str, Any]:
    if ctx.live is None:
        return {"enabled": False, "cameras": []}
    status: dict[str, Any] = ctx.live.status()
    return status


@router.post("/cameras/{camera_id}/start", response_model=None)
def start_camera(camera_id: str, request: Request, ctx: Ctx, me: Supervisor) -> dict[str, Any] | JSONResponse:
    live = _service(ctx)
    try:
        live.start_camera(camera_id)
    except KeyError:
        raise HTTPException(404, "not_found") from None
    except SourceError as exc:
        raise HTTPException(422, f"invalid_source: {exc}") from None
    except CapacityExceededError as exc:
        _audit(ctx, request, me, "live.camera.start", camera_id, "denied", **exc.detail())
        return JSONResponse({"detail": exc.detail(), "message": str(exc)}, status_code=409)
    _audit(ctx, request, me, "live.camera.start", camera_id, "success")
    return {"camera_id": camera_id, "monitoring": "started"}


@router.post("/cameras/{camera_id}/stop")
def stop_camera(camera_id: str, request: Request, ctx: Ctx, me: Supervisor) -> dict[str, Any]:
    live = _service(ctx)
    if not live.stop_camera(camera_id):
        raise HTTPException(404, "not_running")
    _audit(ctx, request, me, "live.camera.stop", camera_id, "success")
    return {"camera_id": camera_id, "monitoring": "stopped"}


_NO_STORE = {"Cache-Control": "no-store, private"}
_streams = {"open": 0, "hq": 0}
_BOUNDARY = "lifeguardframe"


def _runtime(ctx: AppContext) -> Any:
    if ctx.live is None or ctx.live.runtime is None:
        raise HTTPException(503, "live_runtime_disabled")
    return ctx.live.runtime


@router.get("/cameras/{camera_id}/frame.jpg", response_class=Response)
async def camera_frame(
    camera_id: str,
    ctx: Ctx,
    _: AnyUser,
    quality: Literal["normal", "thumb"] = "normal",
    overlay: Literal["burned", "client"] = "burned",
    skeleton: bool = True,  # [rtmpose clone] draw / send the RTMPose keypoints (display only)
) -> Response:
    """[v4] quality=thumb: mjpeg_thumb_width (the Cameras page previews poll this instead of holding a stream).
    overlay=client: the frame without boxes, its boxes in the X-Lifeguard-Boxes header (the browser draws them)."""
    rt = _runtime(ctx)
    width = ctx.settings.mjpeg_thumb_width if quality == "thumb" else ctx.settings.mjpeg_width
    if overlay == "client":
        out = await to_thread.run_sync(render_client, rt.latest_view(camera_id), width, camera_id, skeleton)
        if out is None:
            raise HTTPException(404, "no_frame")
        return Response(out[0], media_type="image/jpeg", headers={**_NO_STORE, BOXES_HEADER: out[1]})
    jpg = await to_thread.run_sync(render, rt.latest_view(camera_id), width, camera_id, skeleton)
    if jpg is None:
        raise HTTPException(404, "no_frame")
    return Response(jpg, media_type="image/jpeg", headers=_NO_STORE)


@router.get("/cameras/{camera_id}/mjpeg", response_class=StreamingResponse)
async def camera_mjpeg(
    camera_id: str,
    request: Request,
    ctx: Ctx,
    _: AnyUser,
    quality: Literal["normal", "high"] = "normal",
    overlay: Literal["burned", "client"] = "burned",
    skeleton: bool = True,  # [rtmpose clone] draw / send the RTMPose keypoints (display only)
) -> StreamingResponse:
    """[v4] quality=high: mjpeg_hq_width / mjpeg_hq_fps (the Cameras page's large main view) while fewer than
    mjpeg_hq_max_streams high-quality streams are open server-wide; normal otherwise. Viewing must never take the CPU
    the drowning analysis needs. overlay=client: frames without boxes, each part carrying its own boxes in an
    X-Lifeguard-Boxes part header (view modes: the browser draws, so switching mode never reconnects)."""
    rt = _runtime(ctx)
    s = ctx.settings
    hq = quality == "high" and _streams["hq"] < s.mjpeg_hq_max_streams
    width, fps = (s.mjpeg_hq_width, s.mjpeg_hq_fps) if hq else (s.mjpeg_width, s.mjpeg_fps)
    if camera_id not in rt.camera_ids():
        raise HTTPException(404, "not_running")
    if _streams["open"] >= s.mjpeg_max_streams:
        raise HTTPException(503, "too_many_viewers")
    period = 1.0 / fps

    async def frames() -> AsyncIterator[bytes]:
        # counted INSIDE the stream, so every increment has its decrement: a viewer that leaves before the first frame
        # (fast camera switching) never started the generator and was never counted
        _streams["open"] += 1
        if hq:
            _streams["hq"] += 1
        try:
            # ends when the viewer leaves or the camera stops (the UI then shows it as not running)
            while not await request.is_disconnected() and camera_id in rt.camera_ids():
                extra = ""
                if overlay == "client":
                    out = await to_thread.run_sync(render_client, rt.latest_view(camera_id), width, camera_id, skeleton)
                    jpg, extra = (out[0], f"{BOXES_HEADER}: {out[1]}\r\n") if out is not None else (None, "")
                else:
                    jpg = await to_thread.run_sync(render, rt.latest_view(camera_id), width, camera_id, skeleton)
                if jpg is not None:
                    head = f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n{extra}\r\n"
                    yield head.encode() + jpg + b"\r\n"
                await asyncio.sleep(period)
        finally:
            _streams["open"] -= 1
            if hq:
                _streams["hq"] -= 1

    return StreamingResponse(frames(), media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}", headers=_NO_STORE)
