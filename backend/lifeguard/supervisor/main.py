"""Supervisor service - an independent process / Windows service (plan Phase 4, fixes audit #3).

    python -m lifeguard.supervisor            (NSSM service "LifeGuardSupervisor")

Every ``supervisor_poll_s`` it checks, WITHOUT going through the API process:

  API_DOWN               GET {supervisor_api_url}/api/v1/health/ready is not 200, or the API heartbeat
                         file is stale;
  CAMERA_STALLED:<id>    a camera's last frame is older than STALL_S (from the API heartbeat);
  CAMERA_NOT_MONITORED:<id>  a camera is NOT_MONITORED (dead detector, refused by capacity, bad source);
  CAMERA_LAGGING:<id>    a camera's analysis has been behind real time for LAG_POLLS polls in a row;
  GPU_SCHEDULER_STALLED  the batched scheduler is stalled or dead while cameras are running;
  DELIVERY_STALLED       the API's outbox dispatcher or escalator is not running (Phase 5);
  OUTBOX_BACKLOG         an alert / FAULT has waited more than OUTBOX_BACKLOG_S for delivery;
  DISK_LOW               free space on the data drive < disk_min_free_gb;
  BACKUP_MISSED          the newest backup is older than backup_max_age_h.

API_DOWN is not raised during the first ``supervisor_startup_grace_s``
(services start in any order); every other check runs from the first poll.

A new problem sends a SYSTEM FAULT on EVERY configured channel (ntfy, SMTP, webhook) directly; an
unresolved one is repeated every ``supervisor_renotify_s``; a cleared one sends a RESOLVED notice.
If every channel fails, the notification is retried on the next poll. The supervisor writes its own
heartbeat, which the API watches (mutual watchdog: the API raises SUPERVISOR_DOWN out of band).

In production/demo the supervisor refuses to start with no channel configured.
"""

from __future__ import annotations

import logging
import shutil
import signal
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from lifeguard.alerts.channels import Notification, send_all
from lifeguard.core import heartbeat
from lifeguard.core.config import Settings, load_settings
from lifeguard.core.logging import configure_logging

logger = logging.getLogger(__name__)

READY_PATH = "/api/v1/health/ready"
OUTBOX_BACKLOG_S = 120.0  # an undelivered alarm older than this is a SYSTEM FAULT of its own
LAG_POLLS = 3  # consecutive polls a camera must lag before CAMERA_LAGGING (startup jitter is not a fault)
EXIT_NO_CHANNEL = 2

HttpGet = Callable[[str, float], int]
Notifier = Callable[[Notification], dict[str, str | None]]


def http_get_status(url: str, timeout: float) -> int:
    """Status code of a GET (http(s) only); raises OSError on connection failure."""
    if urlsplit(url).scheme not in ("http", "https"):
        raise ValueError(f"refusing non-http(s) URL {url!r}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310  # noqa: S310 - scheme checked above
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


@dataclass(frozen=True)
class Problem:
    key: str  # e.g. "CAMERA_STALLED:cam-1"
    code: str
    message: str
    detail: dict[str, Any]


@dataclass
class _Active:
    problem: Problem
    first_seen: float
    last_notified: float | None = None
    notified_ok: bool = False


class Supervisor:
    def __init__(
        self,
        settings: Settings,
        *,
        http_get: HttpGet | None = None,
        notifier: Notifier | None = None,
        clock: Callable[[], float] = time.time,
        disk_free: Callable[[], int] | None = None,
    ) -> None:
        self.settings = settings
        self.http_get = http_get or http_get_status
        self.notifier: Notifier = notifier or (lambda n: send_all(settings, n))
        self.clock = clock
        self.disk_free = disk_free or (lambda: shutil.disk_usage(settings.data_dir).free)
        self.active: dict[str, _Active] = {}
        self.started_at = clock()
        self.polls = 0
        self.sent: list[tuple[str, str, dict[str, str | None]]] = []  # (kind, key, channel results)
        self._lag_polls: dict[str, int] = {}
        self._stop = threading.Event()

    # ── checks ────────────────────────────────────────────────────────────────
    @property
    def api_timeout_s(self) -> float:
        return max(3 * self.settings.heartbeat_interval_s, 5.0)

    def in_startup_grace(self, now: float) -> bool:
        return now - self.started_at < self.settings.supervisor_startup_grace_s

    def check_api(self, now: float) -> list[Problem]:
        s = self.settings
        url = s.supervisor_api_url.rstrip("/") + READY_PATH
        try:
            status: int | str = self.http_get(url, min(5.0, max(1.0, s.supervisor_poll_s)))
        except Exception as exc:  # noqa: BLE001 - any probe failure means API_DOWN (reported below)
            status = f"{type(exc).__name__}"
        hb = heartbeat.read(s.heartbeat_dir, heartbeat.API, now)
        hb_age = "missing" if hb.age_s is None else f"{hb.age_s:.1f} s"
        if status != 200 or not hb.fresh(self.api_timeout_s):
            if self.in_startup_grace(now):
                return []  # the API may still be loading the model / measuring capacity
            return [
                Problem(
                    "API_DOWN",
                    "API_DOWN",
                    f"Life Guard API not ready (probe: {status}, heartbeat age: {hb_age}) - "
                    "live monitoring is NOT guaranteed",
                    {"probe": str(status), "heartbeat_age": hb_age},
                )
            ]
        return self.check_live(hb.payload)

    def check_live(self, payload: dict[str, Any]) -> list[Problem]:
        s = self.settings
        out: list[Problem] = []
        live = payload.get("live")
        for cid, reason in sorted((payload.get("not_started") or {}).items()):
            out.append(
                Problem(
                    f"CAMERA_NOT_MONITORED:{cid}",
                    "CAMERA_NOT_MONITORED",
                    f"camera {cid} is NOT MONITORED ({reason})",
                    {"camera": cid, "reason": reason},
                )
            )
        out.extend(self.check_delivery(payload.get("delivery")))
        if not live:
            return out
        cams: dict[str, dict[str, Any]] = live.get("cameras") or {}
        for cid, cam in sorted(cams.items()):
            age = float(cam.get("last_frame_age_s") or 0.0)
            if cam.get("stalled") or age > s.stall_s:
                out.append(
                    Problem(
                        f"CAMERA_STALLED:{cid}",
                        "CAMERA_STALLED",
                        f"camera {cid}: no frame for {age:.1f} s (STALL_S={s.stall_s:g}) - NOT MONITORED",
                        {"camera": cid, "last_frame_age_s": round(age, 1)},
                    )
                )
            elif self._lagging(cid, bool(cam.get("lagging"))):
                out.append(
                    Problem(
                        f"CAMERA_LAGGING:{cid}",
                        "CAMERA_LAGGING",
                        f"camera {cid}: analysis is behind real time - frames are being dropped",
                        {"camera": cid},
                    )
                )
            elif cam.get("monitoring") == "NOT_MONITORED":
                out.append(
                    Problem(
                        f"CAMERA_NOT_MONITORED:{cid}",
                        "CAMERA_NOT_MONITORED",
                        f"camera {cid} is NOT MONITORED (analysis failing)",
                        {"camera": cid},
                    )
                )
        round_age = live.get("scheduler_last_round_age_s")
        busy = bool(cams) and round_age is not None and float(round_age) > s.scheduler_stall_s
        if live.get("scheduler_stalled") or not live.get("scheduler_alive", True) or busy:
            out.append(
                Problem(
                    "GPU_SCHEDULER_STALLED",
                    "GPU_SCHEDULER_STALLED",
                    "the GPU scheduler is stalled or dead - NO camera is being analysed",
                    {"last_round_age_s": round_age},
                )
            )
        return out

    def check_delivery(self, delivery: dict[str, Any] | None) -> list[Problem]:
        """The API's outbox dispatcher / escalator are alive and not hopelessly behind (Phase 5)."""
        if not delivery:
            return []
        if (
            delivery.get("error")
            or not delivery.get("dispatcher_alive", True)
            or not delivery.get("escalator_alive", True)
        ):
            return [
                Problem(
                    "DELIVERY_STALLED",
                    "DELIVERY_STALLED",
                    "the API's alert delivery (outbox dispatcher / escalation) is not running - "
                    "alerts are recorded but NOT sent",
                    {k: delivery.get(k) for k in ("dispatcher_alive", "escalator_alive", "error")},
                )
            ]
        oldest = delivery.get("oldest_pending_s")
        if oldest is not None and float(oldest) > OUTBOX_BACKLOG_S:
            return [
                Problem(
                    "OUTBOX_BACKLOG",
                    "OUTBOX_BACKLOG",
                    f"an alert has waited {float(oldest):.0f} s for delivery "
                    f"(channels down: {', '.join(delivery.get('channels_down') or []) or 'none'})",
                    {"oldest_pending_s": oldest, "channels_down": delivery.get("channels_down")},
                )
            ]
        return []

    def _lagging(self, camera_id: str, lagging: bool) -> bool:
        n = self._lag_polls.get(camera_id, 0) + 1 if lagging else 0
        self._lag_polls[camera_id] = n
        return n >= LAG_POLLS

    def check_disk(self) -> list[Problem]:
        free_gb = self.disk_free() / 1024**3
        if free_gb >= self.settings.disk_min_free_gb:
            return []
        return [
            Problem(
                "DISK_LOW",
                "DISK_LOW",
                f"only {free_gb:.1f} GB free on the data drive (minimum {self.settings.disk_min_free_gb:g} GB)",
                {"free_gb": round(free_gb, 2)},
            )
        ]

    def check_backup(self, now: float) -> list[Problem]:
        s = self.settings
        limit = s.backup_max_age_h * 3600.0
        newest = max((p.stat().st_mtime for p in s.backups_dir.glob("*") if p.is_file()), default=None)
        if newest is None:
            if now - self.started_at < limit:
                return []  # fresh install: the first nightly backup has not run yet
            age_txt = "no backup found"
        elif now - newest <= limit:
            return []
        else:
            age_txt = f"newest backup is {(now - newest) / 3600:.1f} h old"
        return [
            Problem(
                "BACKUP_MISSED",
                "BACKUP_MISSED",
                f"{age_txt} (limit {s.backup_max_age_h:g} h)",
                {"limit_h": s.backup_max_age_h},
            )
        ]

    def checks(self, now: float) -> list[Problem]:
        out: list[Problem] = []
        checks: list[Callable[[], list[Problem]]] = [
            lambda: self.check_api(now),
            self.check_disk,
            lambda: self.check_backup(now),
        ]
        for fn in checks:
            try:
                out.extend(fn())
            except Exception as exc:  # a broken check is itself a fault, never silence
                logger.exception("supervisor check failed")
                out.append(
                    Problem("SUPERVISOR_CHECK_FAILED", "SUPERVISOR_CHECK_FAILED", f"{type(exc).__name__}: {exc}", {})
                )
        return out

    # ── notification state machine ────────────────────────────────────────────
    def _notify(self, kind: str, p: Problem, severity: str, now: float) -> bool:
        note = Notification(
            kind=kind,
            code=p.code,
            title=f"{p.code}: {p.message}"[:180],
            message=p.message,
            severity=severity,
            source="supervisor",
            at=datetime.fromtimestamp(now, UTC).isoformat(timespec="seconds"),
            detail=p.detail,
            event_id=f"{p.key}@{int(self.active[p.key].first_seen) if p.key in self.active else int(now)}",
        )
        results = self.notifier(note)
        self.sent.append((kind, p.key, results))
        ok = bool(results) and any(err is None for err in results.values())
        if not results:
            logger.critical("SYSTEM FAULT %s NOT DELIVERED: no out-of-band channel is configured", p.key)
        elif not ok:
            logger.critical("SYSTEM FAULT %s NOT DELIVERED on any channel: %s", p.key, results)
        else:
            logger.error("%s %s delivered: %s", kind, p.key, results)
        return ok

    def step(self, now: float | None = None) -> list[Problem]:
        now = self.clock() if now is None else now
        self.polls += 1
        problems = {p.key: p for p in self.checks(now)}
        for key, p in problems.items():
            act = self.active.get(key)
            if act is None:
                act = self.active[key] = _Active(p, now)
            act.problem = p
            due = (
                act.last_notified is None
                or not act.notified_ok
                or now - act.last_notified >= self.settings.supervisor_renotify_s
            )
            if due:
                act.notified_ok = self._notify("SYSTEM_FAULT", p, "critical", now)
                act.last_notified = now
        for key in [k for k in self.active if k not in problems]:
            act = self.active.pop(key)
            if act.notified_ok:
                self._notify("RESOLVED", act.problem, "info", now)
        return list(problems.values())

    def heartbeat_payload(self) -> dict[str, Any]:
        return {"polls": self.polls, "active": sorted(self.active), "channels": self.settings.channels_configured}

    # ── service loop ──────────────────────────────────────────────────────────
    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:
                logger.exception("supervisor step failed")
            self._stop.wait(self.settings.supervisor_poll_s)


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    del argv
    cfg = settings or load_settings()
    cfg.ensure_dirs()
    configure_logging(cfg.log_dir, cfg.log_level, cfg.log_max_bytes, cfg.log_backups, name="supervisor")
    if not cfg.channels_configured and cfg.env not in ("development", "test"):
        logger.critical("supervisor refuses to start: no out-of-band channel (ntfy / SMTP / webhook) configured")
        return EXIT_NO_CHANNEL
    sup = Supervisor(cfg)
    hb = heartbeat.HeartbeatWriter(
        cfg.heartbeat_dir, heartbeat.SUPERVISOR, sup.heartbeat_payload, cfg.heartbeat_interval_s
    )
    hb.start()
    for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGBREAK", signal.SIGINT)):
        signal.signal(sig, lambda *_: sup.stop())
    logger.info("supervisor started; channels: %s", cfg.channels_configured or "NONE")
    try:
        sup.run()
    finally:
        hb.stop()
        logger.info("supervisor stopped")
    return 0
