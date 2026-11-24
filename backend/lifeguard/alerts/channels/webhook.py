"""Signed webhook (plan Phase 5 spec, needed now by the Phase 4 supervisor).

Headers on every POST (JSON body = Notification):

    X-LifeGuard-Timestamp: <unix seconds>
    X-LifeGuard-Nonce:     <32 hex chars, unique per delivery>
    X-LifeGuard-Signature: v1=<hex HMAC-SHA256(secret, f"{timestamp}.{nonce}." + body)>

A receiver recomputes the HMAC with `verify_signature`, rejects a timestamp outside its tolerance
(default 300 s) and remembers nonces for that window, so a captured request cannot be replayed.

Rename shim: for one release every request also carries the old `X-AquaGuard-*` spelling of each  # legacy-name
header (same values), so a receiver still configured for the old names keeps working. Drop LEGACY_HEADERS
and the duplication in `build_headers` in the release after this one.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

from lifeguard.alerts.channels.base import ChannelError, Notification, http_post
from lifeguard.core.config import Settings

SIG_HEADER = "X-LifeGuard-Signature"
TS_HEADER = "X-LifeGuard-Timestamp"
TIER_HEADER = "X-LifeGuard-Tier"
EVENT_HEADER = "X-LifeGuard-Event"
NONCE_HEADER = "X-LifeGuard-Nonce"
SCHEME = "v1"

# Rename shim (one release): old header name -> current header name, duplicated by build_headers.
LEGACY_HEADERS = {
    "X-AquaGuard-Signature": SIG_HEADER,  # legacy-name
    "X-AquaGuard-Timestamp": TS_HEADER,  # legacy-name
    "X-AquaGuard-Tier": TIER_HEADER,  # legacy-name
    "X-AquaGuard-Event": EVENT_HEADER,  # legacy-name
    "X-AquaGuard-Nonce": NONCE_HEADER,  # legacy-name
}


def sign(secret: bytes, timestamp: str, nonce: str, body: bytes) -> str:
    mac = hmac.new(secret, f"{timestamp}.{nonce}.".encode() + body, hashlib.sha256).hexdigest()
    return f"{SCHEME}={mac}"


def verify_signature(
    secret: bytes, headers: dict[str, str], body: bytes, now: float | None = None, tolerance_s: float = 300.0
) -> bool:
    """Receiver side. Constant-time compare; stale or future timestamps fail."""
    try:
        ts, nonce, sig = headers[TS_HEADER], headers[NONCE_HEADER], headers[SIG_HEADER]
        age = (time.time() if now is None else now) - int(ts)
    except (KeyError, ValueError):
        return False
    if abs(age) > tolerance_s:
        return False
    return hmac.compare_digest(sign(secret, ts, nonce, body), sig)


def _with_legacy_headers(headers: dict[str, str]) -> dict[str, str]:
    """Rename shim (one release): duplicate each current header under its old X-AquaGuard-* name too."""
    out = dict(headers)
    for legacy, current in LEGACY_HEADERS.items():
        if current in headers:
            out[legacy] = headers[current]
    return out


def build_headers(secret: bytes, body: bytes, now: float | None = None) -> dict[str, str]:
    ts = str(int(time.time() if now is None else now))
    nonce = secrets.token_hex(16)
    return _with_legacy_headers(
        {
            "Content-Type": "application/json",
            "User-Agent": "Life Guard",
            TS_HEADER: ts,
            NONCE_HEADER: nonce,
            SIG_HEADER: sign(secret, ts, nonce, body),
        }
    )


def send_webhook(settings: Settings, note: Notification) -> None:
    if not settings.webhook_url or settings.webhook_secret is None:
        raise ChannelError("webhook not configured")
    body = note.to_json()
    headers = build_headers(settings.webhook_secret.get_secret_value().encode("utf-8"), body)
    headers[TIER_HEADER] = str(note.tier)  # informational; the signed body carries the same tier
    headers[EVENT_HEADER] = note.event_id
    headers["X-AquaGuard-Tier"] = headers[TIER_HEADER]  # legacy-name
    headers["X-AquaGuard-Event"] = headers[EVENT_HEADER]  # legacy-name
    http_post(settings.webhook_url, body, headers, settings.channel_timeout_s)
