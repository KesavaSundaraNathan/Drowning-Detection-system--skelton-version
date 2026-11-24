"""Exceptions shared across the server."""

from __future__ import annotations


class StartupError(RuntimeError):
    """The server must not start (missing secret, unmigrated DB, insecure config, ...).

    Raised during application startup; the message says exactly what to fix and
    never contains a secret value.
    """
