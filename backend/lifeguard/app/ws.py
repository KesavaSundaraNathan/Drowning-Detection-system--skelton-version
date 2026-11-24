"""Real-time feed over WebSocket (plan Phase 5): GET /api/v1/ws

Security:
  * Origin check BEFORE the handshake is accepted: the Origin must name this server's own Host (and be
    https whenever cookies are Secure - always in production) or be one of the configured CORS
    origins. A missing Origin is refused. This is the WebSocket CSRF defence.
  * Session-cookie auth with the same check as HTTP (expiry, idle timeout, revocation); re-checked
    every ws_recheck_s, so a logged-out / revoked / expired session is disconnected (code 4401).
  * The feed never contains camera sources; every event type is listed in FORWARDED.

Isolation: each client has its OWN bounded queue (ws_queue_size) and every send is bounded by
ws_send_timeout_s. A client that falls behind is disconnected (code 1013) - it can never stall the
others or the publisher (publishing is a non-blocking put on the event loop).

Events: {"type": "hello"} on connect, then alert.created / alert.updated / alert.escalated /
fault.cleared / live.summary / event.opened / event.updated / event.closed (see alerts/outbox.alert_event,
alerts/events.event_payload, app/live_startup.py), and a
{"type": "heartbeat"} every `ws_heartbeat_s` (Phase 7; the UI's dead-feed detector). The database stays
the source of truth: a client re-reads /alerts after a reconnect.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any
from urllib.parse import urlsplit

from anyio import to_thread
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from lifeguard.app.deps import SESSION_COOKIE
from lifeguard.core.clock import utcnow
from lifeguard.core.config import Settings
from lifeguard.core.security import sessions
from lifeguard.core.security.rbac import Principal

logger = logging.getLogger(__name__)

FORWARDED = frozenset(
    {
        "alert.created",
        "alert.updated",
        "alert.escalated",
        "fault.cleared",
        "live.summary",
        "event.opened",  # [Phase 5] alerts/events.py
        "event.updated",
        "event.closed",
    }
)
CLOSE_POLICY = 1008  # bad origin / not authenticated (before accept -> HTTP 403)
CLOSE_SLOW = 1013  # "try again later": this client could not keep up
CLOSE_SESSION = 4401  # session expired / revoked while connected

public = APIRouter(tags=["ws"])
_ids = itertools.count(1)


@dataclass(eq=False)  # identity semantics: clients live in a set
class Client:
    principal: Principal
    queue: asyncio.Queue[dict[str, Any]]
    id: int = field(default_factory=lambda: next(_ids))
    kicked: asyncio.Event = field(default_factory=asyncio.Event)
    reason: str = ""
    sent: int = 0


class Hub:
    """Fan-out of bus events to WebSocket clients; `publish` is safe from ANY thread."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.clients: set[Client] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.dropped_slow = 0

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def publish(self, event: dict[str, Any]) -> None:
        if event.get("type") not in FORWARDED or self.loop is None or self.loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):  # loop shutting down
            self.loop.call_soon_threadsafe(self._fanout, event)

    def _fanout(self, event: dict[str, Any]) -> None:
        for c in list(self.clients):
            try:
                c.queue.put_nowait(event)
            except asyncio.QueueFull:
                self.kick(c, "queue overflow")

    def kick(self, c: Client, reason: str) -> None:
        if not c.kicked.is_set():
            c.reason = reason
            c.kicked.set()
            self.dropped_slow += 1
            logger.warning("websocket client %d (%s) dropped: %s", c.id, c.principal.username, reason)

    def register(self, principal: Principal) -> Client:
        c = Client(principal, asyncio.Queue(maxsize=self.settings.ws_queue_size))
        self.clients.add(c)
        return c

    def unregister(self, c: Client) -> None:
        self.clients.discard(c)

    def stats(self) -> dict[str, Any]:
        return {"clients": len(self.clients), "dropped_slow": self.dropped_slow}


def origin_allowed(origin: str | None, host: str | None, settings: Settings) -> bool:
    """Same-host Origin (https whenever cookies are Secure, i.e. always in production) or a CORS origin.

    The request scheme is not compared: behind the TLS-terminating proxy it depends on proxy headers;
    the Secure-cookie rule pins the scheme instead."""
    if not origin:
        return False
    origin = origin.rstrip("/")
    if origin in settings.cors_origins:
        return True
    parts = urlsplit(origin)
    # loopback http is a secure context in browsers (Secure cookies work there): the local dev UI
    loopback = (parts.hostname or "") in ("localhost", "127.0.0.1", "::1")
    allowed_schemes = ("https",) if settings.cookie_secure and not loopback else ("http", "https")
    if not host or parts.scheme not in allowed_schemes or parts.path or parts.query or parts.username:
        return False
    return parts.netloc.lower() == host.lower()


def _authenticate(ctx: Any, token: str | None) -> Principal | None:
    if not token:
        return None
    with ctx.db.read_sessions() as s:
        res = sessions.check(
            s,
            session_key=ctx.keys.session_key,
            token=token,
            now=utcnow(),
            idle=timedelta(minutes=ctx.settings.session_idle_minutes),
        )
    return res.principal


async def _sender(ws: WebSocket, c: Client, timeout: float) -> None:
    while True:
        get = asyncio.ensure_future(c.queue.get())
        kick = asyncio.ensure_future(c.kicked.wait())
        done, _ = await asyncio.wait({get, kick}, return_when=asyncio.FIRST_COMPLETED)
        if kick in done:
            get.cancel()
            return
        kick.cancel()
        try:
            await asyncio.wait_for(ws.send_json(get.result()), timeout)
        except TimeoutError:
            c.reason = f"send slower than {timeout:g} s"
            return
        c.sent += 1


async def _receiver(ws: WebSocket) -> None:
    with contextlib.suppress(WebSocketDisconnect):
        while True:
            await ws.receive_text()  # clients may send pings; content is ignored


async def _heartbeat(c: Client, interval: float) -> None:
    """A beat every `interval` s through the client's own queue (one writer per socket). The UI treats
    silence as a lost feed, which is the only signal it gets when the server dies behind a proxy."""
    while True:
        await asyncio.sleep(interval)
        try:
            c.queue.put_nowait({"type": "heartbeat", "server_time": utcnow().isoformat()})
        except asyncio.QueueFull:
            c.reason = "queue overflow"
            return


async def _rechecker(ws: WebSocket, ctx: Any, token: str, c: Client) -> None:
    while True:
        await asyncio.sleep(ctx.settings.ws_recheck_s)
        if await to_thread.run_sync(_authenticate, ctx, token) is None:
            c.reason = "session ended"
            return


@public.websocket("/ws")
async def feed(websocket: WebSocket) -> None:
    ctx = websocket.app.state.ctx
    hub: Hub | None = getattr(websocket.app.state, "hub", None)
    s: Settings = ctx.settings
    if hub is None or not origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host"), s):
        await websocket.close(code=CLOSE_POLICY)
        return
    token = websocket.cookies.get(SESSION_COOKIE)
    principal = await to_thread.run_sync(_authenticate, ctx, token)
    if principal is None or token is None:
        await websocket.close(code=CLOSE_POLICY)
        return
    hub.bind(asyncio.get_running_loop())
    await websocket.accept()
    client = hub.register(principal)
    try:
        await websocket.send_json(
            {
                "type": "hello",
                "user": principal.username,
                "role": principal.role.value,
                "server_time": utcnow().isoformat(),
            }
        )
        tasks = {
            asyncio.ensure_future(_sender(websocket, client, s.ws_send_timeout_s)),
            asyncio.ensure_future(_receiver(websocket)),
            asyncio.ensure_future(_rechecker(websocket, ctx, token, client)),
            asyncio.ensure_future(_heartbeat(client, s.ws_heartbeat_s)),
        }
        _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
    finally:
        hub.unregister(client)
    code = CLOSE_SESSION if client.reason == "session ended" else CLOSE_SLOW if client.reason else 1000
    with contextlib.suppress(Exception):
        await websocket.close(code=code)
