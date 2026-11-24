"""API v1 assembly.

Exactly three routes are reachable without a session (`PUBLIC_ROUTES`). Every other router is mounted
under one APIRouter whose dependencies include `require_session`, and every route on it declares
its minimum role with `RequireRole` - tests/api/test_auth_matrix.py enforces both for every route.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from lifeguard.app import ws
from lifeguard.app.api.v1 import (
    alerts,
    audit_log,
    auth,
    calibration,
    cameras,
    events,
    health,
    live,
    schedules,
    system,
    users,
    zones,
)
from lifeguard.app.deps import require_session

PREFIX = "/api/v1"
PUBLIC_ROUTES = frozenset(
    {("POST", f"{PREFIX}/auth/login"), ("GET", f"{PREFIX}/health/live"), ("GET", f"{PREFIX}/health/ready")}
)


def build_router() -> APIRouter:
    api = APIRouter(prefix=PREFIX)
    api.include_router(health.public)
    api.include_router(auth.public)
    api.include_router(ws.public)  # authenticates inside the handler (session cookie + Origin)
    protected = APIRouter(dependencies=[Depends(require_session)])
    for module in (auth, users, cameras, calibration, zones, schedules, alerts, events, audit_log, live, system):
        protected.include_router(module.router)
    api.include_router(protected)
    return api
