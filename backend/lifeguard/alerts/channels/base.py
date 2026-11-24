"""The message every channel sends, and the shared HTTP POST helper."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit

SITE_PREFIX = "Life Guard"


class ChannelError(RuntimeError):
    pass


@dataclass(frozen=True)
class Notification:
    """One out-of-band message. `kind` is "SYSTEM_FAULT" (supervisor), "FAULT" or "ALERT" (Phase 5)."""

    kind: str
    code: str  # e.g. API_DOWN, CAMERA_STALLED, SUBMERSION
    title: str
    message: str
    severity: str = "critical"  # critical | high | info
    source: str = "system"
    at: str = ""  # ISO-8601 UTC
    detail: dict[str, Any] = field(default_factory=dict)
    event_id: str = ""  # stable id: receivers de-duplicate on it
    tier: int = 0  # contact tier (0 = primary; escalation raises it)
    alert_id: int | None = None

    def subject(self) -> str:
        tier = f" T{self.tier}" if self.tier else ""
        return f"[{SITE_PREFIX} {self.kind}{tier}] {self.title}"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Notification:
        fields = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = json.loads(self.to_json())  # JSON-safe copy (datetimes etc. as strings)
        return out

    def to_json(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def http_post(url: str, body: bytes, headers: dict[str, str], timeout: float) -> int:
    """POST `body`; returns the status (2xx) or raises ChannelError. http(s) only (validated in config)."""
    if urlsplit(url).scheme not in ("http", "https"):
        raise ChannelError(f"refusing non-http(s) URL scheme: {url!r}")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")  # noqa: S310 - scheme checked above
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310  # noqa: S310 - scheme checked above
            status = int(resp.status)
    except urllib.error.HTTPError as exc:
        raise ChannelError(f"HTTP {exc.code} from {urlsplit(url).netloc}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ChannelError(f"{type(exc).__name__}: {exc}") from exc
    if not 200 <= status < 300:
        raise ChannelError(f"HTTP {status} from {urlsplit(url).netloc}")
    return status
