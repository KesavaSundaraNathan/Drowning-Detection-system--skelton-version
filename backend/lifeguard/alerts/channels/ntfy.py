"""ntfy push (self-hosted ntfy server, Apache-2.0). Every drowning alarm, escalation and fault uses priority 5
("max/urgent"; [Phase 6] owner decision 2026-11-15: alerts/outbox.note_for_alert); a cleared fault uses 3."""

from __future__ import annotations

from lifeguard.alerts.channels.base import ChannelError, Notification, http_post
from lifeguard.core.config import Settings

PRIORITY = {"critical": "5", "high": "4", "info": "3"}


def _header_safe(text: str) -> str:
    # HTTP header values must be latin-1 and single-line
    return text.replace("\r", " ").replace("\n", " ").encode("latin-1", "replace").decode("latin-1")[:250]


def build_request(settings: Settings, note: Notification) -> tuple[str, bytes, dict[str, str]]:
    if not settings.ntfy_url or not settings.ntfy_topic:
        raise ChannelError("ntfy not configured")
    topic = settings.ntfy_topic_for(note.tier)
    url = f"{settings.ntfy_url.rstrip('/')}/{topic}"
    headers = {
        "Title": _header_safe(note.subject()),
        "Priority": PRIORITY.get(note.severity, "5"),
        "Tags": _header_safe(f"rotating_light,{note.kind.lower()},{note.code.lower()}"),
        "Content-Type": "text/plain; charset=utf-8",
    }
    if settings.ntfy_token is not None:
        headers["Authorization"] = f"Bearer {settings.ntfy_token.get_secret_value()}"
    body = f"{note.message}\n\nsource: {note.source}\nat: {note.at}\nid: {note.event_id}".encode()
    return url, body, headers


def send_ntfy(settings: Settings, note: Notification) -> None:
    url, body, headers = build_request(settings, note)
    http_post(url, body, headers, settings.channel_timeout_s)


def verify_ntfy(settings: Settings, event_id: str, since_unix: int, tier: int = 0) -> bool:
    """Delivery receipt: poll the topic (ntfy JSON stream API) and look for `event_id`."""
    import json
    import urllib.request
    from urllib.parse import urlsplit

    if not settings.ntfy_url:
        raise ChannelError("ntfy not configured")
    url = f"{settings.ntfy_url.rstrip('/')}/{settings.ntfy_topic_for(tier)}/json?poll=1&since={int(since_unix)}"
    if urlsplit(url).scheme not in ("http", "https"):
        raise ChannelError("refusing non-http(s) ntfy URL")
    req = urllib.request.Request(url, method="GET")  # noqa: S310 - scheme checked above
    if settings.ntfy_token is not None:
        req.add_header("Authorization", f"Bearer {settings.ntfy_token.get_secret_value()}")
    try:
        with urllib.request.urlopen(req, timeout=settings.channel_timeout_s) as resp:  # nosec B310  # noqa: S310
            raw = resp.read().decode("utf-8", "replace")
    except OSError as exc:
        raise ChannelError(f"{type(exc).__name__}: {exc}") from exc
    for line in raw.splitlines():
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if isinstance(msg, dict) and event_id in str(msg.get("message", "")):
            return True
    return False
