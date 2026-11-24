"""Camera source policy: allowlist, masking, and encryption at rest (audit #2 and #7).

* Only ``rtsp://`` and ``rtsps://`` URLs with a real host are accepted - never ``http``, ``file``,
  ``concat``, ``pipe`` or any other FFmpeg protocol (v2 handed arbitrary strings to FFmpeg: SSRF
  and local-file read). Whitespace/control characters are rejected outright.
* Life Guard is RTSP-only: there are no file cameras (a database row of the old demo-profile type
  ``file`` is refused when started, with a SOURCE_INVALID fault naming the camera).
* The full source is stored Fernet-encrypted; responses carry only `mask_source()`:
  ``rtsp://***@host:554/path?***`` (userinfo and query - where some cameras put credentials -
  are always masked).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from cryptography.fernet import Fernet

SourceType = Literal["rtsp"]
_ALLOWED_SCHEMES = ("rtsp", "rtsps")
_CONTROL = re.compile(r"[\x00-\x20\x7f]")
# RFC 1123 labels: alphanumeric at both ends, hyphens only inside, 1-63 chars.
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_HOST = re.compile(rf"^(?:{_LABEL}\.)*{_LABEL}$|^\[[0-9A-Fa-f:.]+\]$|^[0-9.]+$")
MAX_URL_LEN = 1024


class SourceError(ValueError):
    """Rejected camera source; the message never contains the submitted secret."""


@dataclass(frozen=True)
class ValidatedSource:
    source_type: SourceType
    value: str  # the full source (secret) - only ever stored encrypted
    display: str  # safe to show / log


def mask_source(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in _ALLOWED_SCHEMES:
        return "***"
    host = parts.hostname or "?"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parts.port}" if parts.port else ""
    userinfo = "***@" if (parts.username or parts.password or "@" in parts.netloc) else ""
    query = "?***" if parts.query else ""
    return f"{parts.scheme}://{userinfo}{host}{port}{parts.path}{query}"


def validate_rtsp(url: str) -> ValidatedSource:
    if not url or len(url) > MAX_URL_LEN:
        raise SourceError(f"source URL must be 1-{MAX_URL_LEN} characters")
    if _CONTROL.search(url):
        raise SourceError("source URL must not contain whitespace or control characters")
    parts = urlsplit(url)
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        raise SourceError("only rtsp:// and rtsps:// camera URLs are allowed")
    try:
        port = parts.port
    except ValueError as exc:
        raise SourceError("invalid port in camera URL") from exc
    host = parts.hostname or ""
    if not host or not _HOST.match(host if ":" not in host else f"[{host}]"):
        raise SourceError("camera URL must contain a valid host")
    if port is not None and not (1 <= port <= 65535):
        raise SourceError("invalid port in camera URL")
    if parts.fragment:
        raise SourceError("camera URL must not contain a fragment")
    return ValidatedSource("rtsp", url, mask_source(url))


def encrypt_source(fernet: Fernet, value: str) -> str:
    return fernet.encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_source(fernet: Fernet, ciphertext: str) -> str:
    return fernet.decrypt(ciphertext.encode("ascii")).decode("utf-8")
