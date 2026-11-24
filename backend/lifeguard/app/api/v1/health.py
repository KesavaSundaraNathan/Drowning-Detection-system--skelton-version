"""Unauthenticated probes (both allowlisted in PUBLIC_ROUTES). They reveal nothing but a status word.

/api/v1/health/live   the process answers HTTP.
/api/v1/health/ready  200 {"status": "ready"} when the DB writer is alive and, if the live runtime is
                      enabled, its GPU scheduler runs and is not stalled; otherwise 503 {"status":
                      "not_ready"}. The supervisor service polls it (plan Phase 4). Per-camera detail
                      is NOT here: it is in the local heartbeat file and the authenticated /live/status.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from lifeguard import __version__

public = APIRouter(prefix="/health", tags=["health"])


@public.get("/live")
def live() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@public.get("/ready")
def ready(request: Request) -> JSONResponse:
    ctx = getattr(request.app.state, "ctx", None)
    ok = ctx is not None and ctx.writer.alive and (ctx.live is None or ctx.live.ready())
    return JSONResponse({"status": "ready" if ok else "not_ready"}, status_code=200 if ok else 503)
