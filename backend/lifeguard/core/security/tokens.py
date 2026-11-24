"""Opaque random tokens and their keyed hashes.

Session and CSRF tokens are 256-bit random values handed to the browser once. The database
stores only HMAC-SHA256(session_key, token): a leaked database (or backup) cannot be used to
hijack a session, and the server never needs the raw token again.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

TOKEN_BYTES = 32  # 256 bits


def new_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def token_hash(session_key: bytes, token: str) -> str:
    return hmac.new(session_key, token.encode("utf-8"), hashlib.sha256).hexdigest()


def constant_time_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
