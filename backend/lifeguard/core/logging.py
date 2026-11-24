"""Structured logging: JSON lines to a rotating file (+ stderr), with secret redaction.

* One configuration for the whole process (uvicorn's loggers propagate into it).
* Every record carries the current request id (set by the request-id middleware).
* A redaction filter masks credentials embedded in URLs (``rtsp://user:pass@host`` ->
  ``rtsp://***@host``) and ``password=`` / ``token=`` style values BEFORE anything is
  formatted, so a secret that reaches a log call still never reaches disk.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_URL_CREDS = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/@\s:]+(?::[^/@\s]*)?@")
_KV_SECRET = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|token|csrf|api[_-]?key|credential)(\s*[=:]\s*)([^\s&,;\"']+)"
)
_LOG_NAME = "lifeguard.log"


def redact(text: str) -> str:
    text = _URL_CREDS.sub(r"\1***@", text)
    return _KV_SECRET.sub(r"\1\2***", text)


class RedactingFilter(logging.Filter):
    """Masks secrets in the message and its arguments; also stamps the request id."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        if record.args:
            record.msg = record.getMessage()
            record.args = None
        record.msg = redact(str(record.msg))
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            doc["exc"] = redact(self.formatException(record.exc_info))
        for key in ("event", "user_id", "path", "method", "status", "duration_ms", "camera_id"):
            if hasattr(record, key):
                doc[key] = getattr(record, key)
        return json.dumps(doc, ensure_ascii=False, default=str)


_configured_file: Path | None = None


def configure_logging(
    log_dir: Path,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backups: int = 10,
    to_stderr: bool = True,
    name: str | None = None,
) -> Path:
    """(Re)configure the root logger. Idempotent; returns the log file path.

    `name` gives a separate process (supervisor) its own file: rotating one file from
    several processes is not safe on Windows."""
    global _configured_file
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / (f"{name}.log" if name else _LOG_NAME)
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_lifeguard", False):
            root.removeHandler(h)
            h.close()
    fmt = JsonFormatter()
    redactor = RedactingFilter()
    file_handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8", delay=False
    )
    handlers: list[logging.Handler] = [file_handler]
    if to_stderr:
        handlers.append(logging.StreamHandler(sys.stderr))
    for h in handlers:
        h.setFormatter(fmt)
        h.addFilter(redactor)
        h._lifeguard = True  # type: ignore[attr-defined]
        root.addHandler(h)
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # our access log replaces it
    _configured_file = path
    return path


def log_file() -> Path | None:
    return _configured_file
