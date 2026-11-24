"""Camera sources for the site kit, read LOCALLY (Phase 11 WP6).

The kit runs elevated on the server: it reads the cameras from the database and decrypts their sources with the
server's own keys (env file + DPAPI), exactly as the API's live service does. No API endpoint ever returns a source.
A `SiteCamera` keeps its URL private: `repr` / `str` / `public()` show only the masked form, and `Redactor`
scrubs every known secret (the URL, its userinfo, password and query) from any text before it is logged or
written - FFmpeg echoes its input URL in error messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, urlsplit

from lifeguard.core.logging import redact as redact_generic
from lifeguard.core.security.sources import mask_source


@dataclass(frozen=True)
class SiteCamera:
    camera_id: str
    name: str
    source_type: str
    calibration: dict[str, Any] | None
    is_enabled: bool
    _url: str = field(repr=False)

    @property
    def url(self) -> str:
        return self._url

    @property
    def masked(self) -> str:
        return mask_source(self._url) if self.source_type == "rtsp" else "<file>"

    def public(self) -> dict[str, Any]:
        """Everything a report may contain about the camera."""
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "source": self.masked,
            "calibrated": self.calibration is not None,
            "enabled": self.is_enabled,
        }

    def __str__(self) -> str:
        return f"{self.name} ({self.masked})"


def load_cameras(settings: Any | None = None) -> list[SiteCamera]:
    """Every camera of this installation with its decrypted source (needs the env file and the server keys)."""
    from sqlalchemy import select

    from lifeguard.core.config import load_settings
    from lifeguard.core.security.keys import load_keys
    from lifeguard.core.security.sources import decrypt_source
    from lifeguard.db.models import Camera
    from lifeguard.db.session import make_database

    cfg = settings or load_settings()
    keys = load_keys(cfg)
    db = make_database(cfg.effective_database_url)
    try:
        with db.read_sessions() as rs:
            rows = list(rs.execute(select(Camera).order_by(Camera.name)).scalars())
            return [
                SiteCamera(
                    c.id,
                    c.name,
                    c.source_type,
                    dict(c.calibration) if c.calibration else None,
                    bool(c.is_enabled),
                    decrypt_source(keys.fernet, c.source_ciphertext),
                )
                for c in rows
            ]
    finally:
        db.dispose()


class Redactor:
    """Removes every known secret, then anything URL-credential / key=value shaped (core.logging.redact)."""

    def __init__(self, urls: list[str] | None = None) -> None:
        self._secrets: list[str] = []
        for u in urls or []:
            self.add(u)

    def add(self, url: str) -> None:
        parts = urlsplit(url)
        found = {url}
        if parts.password:
            found |= {parts.password, unquote(parts.password)}
        if parts.username and parts.password:
            found.add(parts.netloc.rsplit("@", 1)[0])
        if parts.query:
            found |= {parts.query, unquote(parts.query)}
        # longest first: the full URL is replaced before its parts
        self._secrets = sorted({*self._secrets, *(s for s in found if len(s) >= 3)}, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for s in self._secrets:
            if s in text:
                text = text.replace(s, mask_source(s) if "://" in s else "***")
        return redact_generic(text)
