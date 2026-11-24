"""Out-of-band channels: signed webhook, ntfy, SMTP (stdlib only - no new runtime dependency).

Phase 4 uses them DIRECTLY from the supervisor service, so a SYSTEM FAULT is delivered even when
the API process, its database or its outbox is down. Phase 5 puts the outbox (retry, at-least-once,
escalation) in front of the same senders for every alert.

    results = send_all(settings, Notification(...))   # {"webhook": None, "ntfy": "timeout", ...}

`send_all` never raises: each channel is tried independently and its error (or None) is returned,
so one dead channel can never stop the others.
"""

from __future__ import annotations

import logging

from lifeguard.alerts.channels.base import ChannelError, Notification
from lifeguard.alerts.channels.ntfy import send_ntfy
from lifeguard.alerts.channels.smtp import send_smtp
from lifeguard.alerts.channels.webhook import send_webhook
from lifeguard.core.config import Settings

logger = logging.getLogger(__name__)

__all__ = ["ChannelError", "Notification", "send_all", "send_ntfy", "send_one", "send_smtp", "send_webhook"]


def send_one(settings: Settings, channel: str, note: Notification) -> None:
    """Send on one channel; raises ChannelError (or OSError) on failure."""
    if channel == "webhook":
        send_webhook(settings, note)
    elif channel == "ntfy":
        send_ntfy(settings, note)
    elif channel == "smtp":
        send_smtp(settings, note)
    else:
        raise ChannelError(f"unknown channel {channel!r}")


def send_all(settings: Settings, note: Notification) -> dict[str, str | None]:
    """Send `note` on every configured channel. Returns {channel: None on success | error text}."""
    results: dict[str, str | None] = {}
    for name in settings.channels_configured:
        try:
            send_one(settings, name, note)
            results[name] = None
        except Exception as exc:  # noqa: BLE001 - every channel is isolated from the others; error returned + logged
            results[name] = f"{type(exc).__name__}: {exc}"
            logger.error("channel %s failed for %s: %s", name, note.code, results[name])
    return results
