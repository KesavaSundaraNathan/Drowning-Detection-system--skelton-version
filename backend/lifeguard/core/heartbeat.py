"""Cross-process heartbeats (plan Phase 4: supervisor <-> API mutual watchdog).

Each process writes ``<data_dir>/run/heartbeats/<component>.json`` atomically (temp file +
``os.replace``) every ``heartbeat_interval_s``. A heartbeat is a small JSON object:

    {"component": "api", "pid": 1234, "at": <unix wall-clock seconds>, "seq": 17, ...payload}

Files - not the database - on purpose: the supervisor must keep working when the database is
locked, corrupt or the disk is full of WAL, and a reader never blocks a writer. Readers judge
freshness by ``at`` against their own wall clock; a missing, unreadable or future-dated file is
"stale" (never "healthy by default").
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

API = "api"
SUPERVISOR = "supervisor"
COMPONENTS = (API, SUPERVISOR)
_CLOCK_SKEW_S = 5.0  # a heartbeat from further in the future than this is not trusted
# [v7] Windows refuses to replace a file another handle has open (Python opens without FILE_SHARE_DELETE), and to open
# a file while it is being replaced: a reader (metrics, supervisor) and the writer collide now and then
# ("Access is denied", WinError 5 / errno 13). Both sides retry briefly; the collision lasts well under a millisecond.
_SHARING_RETRIES = 20
_SHARING_RETRY_S = 0.005


def heartbeat_dir(data_dir: Path) -> Path:
    return data_dir / "run" / "heartbeats"


def write(directory: Path, component: str, payload: dict[str, Any], seq: int = 0) -> None:
    """Atomically replace `component`'s heartbeat file."""
    if component not in COMPONENTS:
        raise ValueError(f"unknown heartbeat component {component!r}")
    directory.mkdir(parents=True, exist_ok=True)
    body = {**payload, "component": component, "pid": os.getpid(), "at": time.time(), "seq": seq}
    tmp = directory / f".{component}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(body, separators=(",", ":"), default=str), encoding="utf-8")
    _retry_sharing(lambda: tmp.replace(directory / f"{component}.json"))  # atomic on NTFS and POSIX


def _retry_sharing(fn: Callable[[], Any]) -> Any:
    """Run fn, retrying a PermissionError from a momentary Windows sharing collision (see _SHARING_RETRIES)."""
    for attempt in range(_SHARING_RETRIES):
        try:
            return fn()
        except PermissionError:
            if attempt == _SHARING_RETRIES - 1:
                raise
            time.sleep(_SHARING_RETRY_S)
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class Reading:
    component: str
    age_s: float | None  # None = missing / unreadable / not trustworthy
    payload: dict[str, Any]

    def fresh(self, max_age_s: float) -> bool:
        return self.age_s is not None and self.age_s <= max_age_s


def read(directory: Path, component: str, now: float | None = None) -> Reading:
    path = directory / f"{component}.json"
    now = time.time() if now is None else now
    try:
        data = json.loads(_retry_sharing(lambda: path.read_text(encoding="utf-8")))
        at = float(data["at"])
    except FileNotFoundError:
        return Reading(component, None, {})
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("heartbeat %s unreadable", path, exc_info=True)
        return Reading(component, None, {})
    if not isinstance(data, dict):
        return Reading(component, None, {})
    age = now - at
    if age < -_CLOCK_SKEW_S:
        logger.warning("heartbeat %s is %.1f s in the future - treated as stale", component, -age)
        return Reading(component, None, data)
    return Reading(component, max(age, 0.0), data)


class HeartbeatWriter:
    """Background thread that writes `component`'s heartbeat from `payload_fn()` every `interval_s`.

    A payload_fn that raises still produces a heartbeat (with ``"payload_error"``) - the process is
    alive - but the error is logged; only a dead process stops the heartbeat.
    """

    def __init__(
        self, directory: Path, component: str, payload_fn: Callable[[], dict[str, Any]], interval_s: float = 1.0
    ) -> None:
        self.directory = directory
        self.component = component
        self.payload_fn = payload_fn
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.seq = 0

    def beat(self) -> None:
        try:
            payload = self.payload_fn()
        except Exception as exc:
            logger.exception("heartbeat payload for %s failed", self.component)
            payload = {"payload_error": f"{type(exc).__name__}: {exc}"}
        self.seq += 1
        try:
            write(self.directory, self.component, payload, self.seq)
        except OSError:
            logger.exception("heartbeat write for %s failed", self.component)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.beat()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self.beat()
        self._thread = threading.Thread(target=self._loop, name=f"heartbeat-{self.component}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
