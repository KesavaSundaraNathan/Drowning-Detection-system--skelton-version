"""Startup timeline (plan Phase 11 WP5): when each phase of this process's start finished, in seconds since the
process was CREATED (Windows: GetProcessTimes; elsewhere: when lifeguard was first imported).

    TIMELINE.mark("vision_guard", provider="cuda")       # logged at once
    TIMELINE.camera("cam-1", "admitted")
    TIMELINE.as_dict()                                    # GET /live/status -> "startup"

Process-level marks are fixed once (a mark is kept from its first call). Cameras get `admitted`,
`first_frame` and `monitored` (first analysed frame). When every camera expected at startup is monitored
(or has been given up on) one summary line is logged.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_IMPORTED_MONO = time.monotonic()


def process_start_monotonic() -> float:
    """time.monotonic() value of this process's creation (import time where it cannot be read)."""
    if sys.platform == "win32":
        try:
            from ctypes import wintypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
            k32.GetProcessTimes.restype = wintypes.BOOL
            ft = [wintypes.FILETIME() for _ in range(4)]
            if k32.GetProcessTimes(k32.GetCurrentProcess(), *(ctypes.byref(f) for f in ft)):
                created_100ns = (ft[0].dwHighDateTime << 32) | ft[0].dwLowDateTime
                created_unix = created_100ns / 1e7 - 11_644_473_600  # FILETIME epoch 1601 -> Unix epoch
                age = time.time() - created_unix
                if 0 <= age < 86_400 * 365:
                    return time.monotonic() - age
        except (OSError, AttributeError, ValueError):
            logger.debug("process creation time unavailable", exc_info=True)
    return _IMPORTED_MONO


class StartupTimeline:
    def __init__(self, origin: float | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.origin = process_start_monotonic() if origin is None else origin
        self.clock = clock
        self._lock = threading.Lock()
        self.marks: dict[str, dict[str, Any]] = {}
        self.cameras: dict[str, dict[str, float]] = {}
        self.expected: set[str] = set()
        self.summary_logged = False

    def since_start(self, mono: float | None = None) -> float:
        return round((self.clock() if mono is None else mono) - self.origin, 3)

    def mark(self, name: str, mono: float | None = None, **detail: Any) -> float:
        """Record phase `name` (first call wins) and log it. Returns seconds since process creation."""
        t = self.since_start(mono)
        with self._lock:
            if name in self.marks:
                return float(self.marks[name]["t"])
            self.marks[name] = {"t": t, **detail}
        logger.info("startup: %s at %.2f s %s", name, t, detail or "")
        return t

    def camera(self, camera_id: str, event: str, mono: float | None = None) -> None:
        t = self.since_start(mono)
        with self._lock:
            self.cameras.setdefault(camera_id, {}).setdefault(event, t)

    def expect(self, camera_ids: list[str]) -> None:
        """The cameras admitted at startup: the summary waits for them."""
        with self._lock:
            self.expected = set(camera_ids)

    def all_monitored_s(self) -> float | None:
        with self._lock:
            if not self.expected:
                return None
            times = [self.cameras.get(c, {}).get("monitored") for c in self.expected]
        return None if any(t is None for t in times) else max(t for t in times if t is not None)

    def maybe_log_summary(self, give_up_s: float = 120.0) -> bool:
        """One INFO line once every expected camera is monitored, or after `give_up_s` (pending ones listed)."""
        if self.summary_logged or not self.expected:
            return False
        done = self.all_monitored_s()
        if done is None and self.since_start() < give_up_s:
            return False
        self.summary_logged = True
        d = self.as_dict()
        pending = sorted(c for c in self.expected if "monitored" not in d["cameras"].get(c, {}))
        logger.info(
            "startup timeline: all %d camera(s) monitored at %s s; phases %s; pending %s",
            len(self.expected),
            done,
            {k: v["t"] for k, v in d["marks"].items()},
            pending,
        )
        return True

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            marks = {k: dict(v) for k, v in self.marks.items()}
            cams = {k: dict(v) for k, v in self.cameras.items()}
            expected = sorted(self.expected)
        return {"marks": marks, "cameras": cams, "expected": expected, "all_monitored_s": self.all_monitored_s()}


TIMELINE = StartupTimeline()
