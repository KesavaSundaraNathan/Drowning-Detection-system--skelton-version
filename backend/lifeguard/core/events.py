"""In-process event bus (Phase 5): alert / FAULT / escalation events -> WebSocket hub, dispatcher wake-up.

    events.publish({"type": "alert.created", ...})       # from any thread, AFTER the DB commit
    unsubscribe = events.subscribe(callback)              # callback(event) runs on the publisher's thread

Publishing never raises: a failing subscriber is logged and the others still run. Events are
notifications only - the database is the source of truth (a client that missed an event re-reads it).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

Event = dict[str, Any]
Subscriber = Callable[[Event], None]

_lock = threading.Lock()
_subscribers: list[Subscriber] = []


def subscribe(fn: Subscriber) -> Callable[[], None]:
    with _lock:
        _subscribers.append(fn)

    def unsubscribe() -> None:
        with _lock:
            if fn in _subscribers:
                _subscribers.remove(fn)

    return unsubscribe


def publish(event: Event) -> None:
    with _lock:
        subs = list(_subscribers)
    for fn in subs:
        try:
            fn(event)
        except Exception:
            logger.exception("event subscriber failed for %s", event.get("type"))


def clear() -> None:
    """Drop every subscriber (tests, process shutdown)."""
    with _lock:
        _subscribers.clear()
