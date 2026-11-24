"""Password hashing (bcrypt) and policy.

Policy (NIST SP 800-63B style: length over composition rules):
  * at least ``password_min_length`` characters (default 12, cannot be configured below 12);
  * at most 72 bytes UTF-8 - bcrypt ignores everything after byte 72, so longer input would
    give a false sense of strength (bcrypt 5 raises instead; we reject with a clear message);
  * must not contain the username, must not be a well-known common password, and must not be
    a single repeated character.

`verify_unknown_user()` burns the same bcrypt time for a username that does not exist, so
response timing does not reveal which accounts exist.
"""

from __future__ import annotations

import secrets

import bcrypt

_BCRYPT_MAX_BYTES = 72

# Common passwords / patterns (lower-case; compared after lower-casing). Kept small and local:
# the policy's job is to stop the obvious, the lockout stops online guessing.
_COMMON = frozenset(
    {
        "password",
        "password1",
        "password12",
        "password123",
        "password1234",
        "passw0rd",
        "p@ssw0rd",
        "123456789012",
        "1234567890123",
        "qwertyuiop12",
        "qwerty123456",
        "abc123456789",
        "letmein12345",
        "welcome12345",
        "administrator",
        "admin1234567",
        "aquaguard123",  # legacy-name: block guesses of the old product name too
        "aquaguard1234",  # legacy-name
        "lifeguard123",
        "lifeguard1234",
        "swimmingpool",
        "swimmingpool1",
        "changeme1234",
        "iloveyou1234",
        "trustno11234",
        "111111111111",
        "000000000000",
        "123123123123",
        "monkey123456",
        "football1234",
        "baseball1234",
        "superman1234",
        "batman123456",
        "sunshine1234",
        "princess1234",
        "dragon123456",
        "master123456",
    }
)


class PasswordPolicyError(ValueError):
    """The password does not meet the policy; the message is safe to show to the user."""


def check_policy(password: str, username: str, min_length: int) -> None:
    if len(password) < min_length:
        raise PasswordPolicyError(f"password must be at least {min_length} characters")
    if len(password.encode("utf-8")) > _BCRYPT_MAX_BYTES:
        raise PasswordPolicyError(f"password must be at most {_BCRYPT_MAX_BYTES} bytes")
    low = password.lower()
    if username and username.lower() in low:
        raise PasswordPolicyError("password must not contain the username")
    if low in _COMMON:
        raise PasswordPolicyError("password is too common")
    if len(set(password)) == 1:
        raise PasswordPolicyError("password must not be a single repeated character")


def hash_password(password: str, rounds: int) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=rounds)).decode("ascii")


def verify_password(password: str, stored_hash: str) -> bool:
    data = password.encode("utf-8")
    if len(data) > _BCRYPT_MAX_BYTES or not stored_hash.startswith(("$2a$", "$2b$", "$2y$")):
        return False
    try:
        return bcrypt.checkpw(data, stored_hash.encode("ascii"))
    except ValueError:
        return False


class DummyHasher:
    """Holds one throw-away hash so unknown users cost the same bcrypt time as known ones."""

    def __init__(self, rounds: int) -> None:
        self._hash = hash_password(secrets.token_urlsafe(24), rounds)

    def burn(self, password: str) -> bool:
        verify_password(password, self._hash)
        return False
