"""ASGI application factory.

    uvicorn --factory lifeguard.app.main:create_app --host 127.0.0.1 --port 8000 --workers 1 \
            --proxy-headers --forwarded-allow-ips 127.0.0.1

Startup REFUSES to proceed (StartupError, the process exits) when:
  * the session secret or the credential-encryption key is missing / unreadable (core.security.keys);
  * the database schema is not at the migration head (db.migrate.assert_at_head);
  * the configuration is insecure for its profile (validated in core.config);
  * the detector does not run on the requested execution provider (app/vision_startup.py: a FAULT
    alert is recorded and the missing CUDA/cuDNN DLLs are logged; on by default in production/demo);
  * the live runtime (Phase 4: batched GPU scheduler + capacity measurement) cannot start.
There is no default admin: the first one is created with `python -m lifeguard admin create`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lifeguard import __version__
from lifeguard.alerts import outbox
from lifeguard.app import errors, vision_startup
from lifeguard.app.api.v1 import build_router
from lifeguard.app.context import AppContext
from lifeguard.app.delivery import DeliveryService
from lifeguard.app.live_startup import LiveService
from lifeguard.app.metrics import AppCollector, AppMetrics, gate_status
from lifeguard.app.middleware import CatchAllMiddleware, RequestContextMiddleware, SecurityHeadersMiddleware
from lifeguard.app.ws import Hub
from lifeguard.core import events
from lifeguard.core.config import IGNORED_SETTINGS, Settings, load_settings
from lifeguard.core.logging import configure_logging
from lifeguard.core.security.keys import load_keys
from lifeguard.core.security.passwords import DummyHasher
from lifeguard.core.startup import TIMELINE
from lifeguard.db.migrate import assert_at_head
from lifeguard.db.session import make_database
from lifeguard.db.writer import DBWriter
from lifeguard.live.preopen import PreOpener

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    metrics = AppMetrics(cfg)  # [Phase 12] Prometheus: request instruments + loopback listener

    def _ctx_attr(name: str) -> Any:
        ctx = getattr(app.state, "ctx", None)
        return getattr(ctx, name, None) if ctx is not None else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg.ensure_dirs()
        configure_logging(cfg.log_dir, cfg.log_level, cfg.log_max_bytes, cfg.log_backups)
        TIMELINE.mark("lifespan")  # [Phase 11 WP5] process created -> interpreter + imports + app built
        metrics.attach(
            AppCollector(
                cfg, live=lambda: _ctx_attr("live"), delivery=lambda: _ctx_attr("delivery"), gate=gate_status(cfg)
            )
        )
        app.state.metrics = metrics
        app.state.metrics_port = None
        if cfg.metrics_is_enabled:
            try:  # metrics are observability, not safety: a taken port is logged, the API still starts
                app.state.metrics_port = metrics.start()
            except OSError as exc:
                log.error("Prometheus metrics listener not started on %s: %s", cfg.metrics_bind, exc)
        keys = load_keys(cfg)
        db = make_database(cfg.effective_database_url)
        writer: DBWriter | None = None
        live: LiveService | None = None
        preopener: PreOpener | None = None
        delivery: DeliveryService | None = None
        hub = Hub(cfg)
        hub.bind(asyncio.get_running_loop())
        app.state.hub = hub
        unsubscribe = events.subscribe(hub.publish)
        try:
            assert_at_head(db.write_engine)
            TIMELINE.mark("database_ready")
            writer = DBWriter(db.write_sessions)
            writer.start()
            app.state.ctx = AppContext(
                settings=cfg, keys=keys, db=db, writer=writer, dummy_hasher=DummyHasher(cfg.bcrypt_rounds)
            )
            # Phase 5: every FAULT / audible alert from here on is queued in the outbox and delivered
            outbox.configure(cfg)
            delivery = DeliveryService(cfg, db, writer)
            delivery.start()
            # [Phase 11 WP7] open the RTSP cameras NOW, in parallel with the guard + capacity phase; frames are
            # discarded until admission and every un-admitted stream is closed on every path (finally below)
            preopener = vision_startup.start_preopen(cfg, keys, db)
            # Phase 3: thresholds audited; detector proven on the requested provider (else FAULT + refuse)
            vision_startup.audit_thresholds(writer)
            vision_startup.check_vision(cfg, writer)
            TIMELINE.mark("vision_guard", provider=cfg.inference_provider)
            # Phase 4: live runtime (batched GPU scheduler, admission control), API heartbeat and the
            # API side of the supervisor watchdog
            live = vision_startup.start_live(cfg, keys, db, writer, preopener)
            live.delivery_status = delivery.status
            app.state.ctx = replace(app.state.ctx, live=live, delivery=delivery)
            vision_startup.clear_startup_faults(writer)
            log.info("Life Guard %s started (env=%s)", __version__, cfg.env)
            if IGNORED_SETTINGS:
                log.warning(
                    "ignored settings of removed features (delete them from the env file): %s",
                    ", ".join(IGNORED_SETTINGS),
                )
            TIMELINE.mark("ready")
            yield
        finally:
            metrics.stop()
            unsubscribe()
            if preopener is not None:
                preopener.close()  # idempotent: startup refused before / during admission must not leak a stream
            if live is not None:
                live.stop()
            if delivery is not None:
                delivery.stop()
            if writer is not None:
                writer.stop()
            db.dispose()
            log.info("Life Guard stopped")

    app = FastAPI(
        title="Life Guard",
        version=__version__,
        lifespan=lifespan,
        description="Pool drowning-detection assist system. ASSIST / PILOT: not a certified life-saving system.",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    errors.install(app)
    app.include_router(build_router())
    # Starlette: the LAST added middleware is the OUTERMOST. Request flow:
    # RequestContext -> SecurityHeaders -> [CORS] -> CatchAll -> routes.
    app.add_middleware(CatchAllMiddleware)
    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
            allow_headers=["Content-Type", "X-CSRF-Token", "X-Request-ID"],
            max_age=600,
        )
    app.add_middleware(SecurityHeadersMiddleware, hsts=cfg.cookie_secure)
    app.add_middleware(RequestContextMiddleware, observe=metrics.observe_request)
    return app
