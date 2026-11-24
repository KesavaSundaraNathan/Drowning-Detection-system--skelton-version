"""Pure-ASGI middleware (no BaseHTTPMiddleware: streaming request bodies pass through untouched).

RequestContextMiddleware
    request id (accepts a well-formed incoming X-Request-ID, else generates one) on the log context
    and the response; one access-log line per request: method, PATH ONLY (query strings may carry
    secrets and are never logged), status, duration, user id.
SecurityHeadersMiddleware
    nosniff, frame denial, no referrer, a locked-down CSP for API responses, no caching of API
    responses, cross-origin isolation, and HSTS when cookies are Secure (TLS terminates at Caddy).
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from lifeguard.core.logging import request_id_var

log = logging.getLogger("lifeguard.access")
_RID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, observe: Callable[[Scope, int, float], None] | None = None) -> None:
        self.app = app
        self.observe = observe  # [Phase 12] Prometheus request metrics (app/metrics.py), by route template

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        rid = incoming if _RID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(rid)
        started = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message.setdefault("headers", [])
                message["headers"].append((b"x-request-id", rid.encode("ascii")))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            state: dict[str, Any] = scope.get("state") or {}
            principal = state.get("principal")
            elapsed = time.perf_counter() - started
            if self.observe is not None:
                try:
                    self.observe(scope, status["code"], elapsed)
                except Exception:
                    log.debug("request metrics failed", exc_info=True)
            log.info(
                "%s %s -> %s",
                scope.get("method"),
                scope.get("path"),
                status["code"],
                extra={
                    "event": "http_request",
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status": status["code"],
                    "duration_ms": round(elapsed * 1000, 1),
                    "user_id": getattr(principal, "user_id", None),
                },
            )
            request_id_var.reset(token)


class CatchAllMiddleware:
    """Innermost safety net: an unhandled exception becomes a generic 500 JSON (with the request id)
    that still flows out through SecurityHeaders and RequestContext - Starlette's own 500 handler
    sits outside user middleware and would skip both. The stack trace goes to the log only."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = {"value": False}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                started["value"] = True
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            logging.getLogger("lifeguard.app").error("unhandled exception: %s", type(exc).__name__, exc_info=exc)
            if started["value"]:
                raise
            body = json.dumps({"detail": "internal_error", "request_id": request_id_var.get()}).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
                }
            )
            await send({"type": "http.response.body", "body": body})


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp, hsts: bool) -> None:
        self.app = app
        self.headers: list[tuple[bytes, bytes]] = [
            (b"x-content-type-options", b"nosniff"),
            (b"x-frame-options", b"DENY"),
            (b"referrer-policy", b"no-referrer"),
            (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'; base-uri 'none'"),
            (b"cross-origin-opener-policy", b"same-origin"),
            (b"cross-origin-resource-policy", b"same-origin"),
            (b"permissions-policy", b"camera=(), microphone=(), geolocation=(), payment=()"),
            (b"cache-control", b"no-store"),
        ]
        if hsts:
            self.headers.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                existing = {k.lower() for k, _ in message.get("headers", [])}
                message.setdefault("headers", [])
                message["headers"].extend(h for h in self.headers if h[0] not in existing)
            await send(message)

        await self.app(scope, receive, send_wrapper)
