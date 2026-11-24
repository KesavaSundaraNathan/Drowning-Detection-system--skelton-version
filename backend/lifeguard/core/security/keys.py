"""Server secrets: the session HMAC key and the camera-credential Fernet key.

Production (env=production/demo): both live in protected files under ``data_dir/keys`` created once by
``lifeguard keys init``.

* Windows: DPAPI blobs (``session.key.dpapi``, ``credentials.key.dpapi``). DPAPI binds the ciphertext to the
  Windows account that created it (or, with ``--machine``, to the machine), so a copied file is useless elsewhere.
* Linux / WSL / other POSIX: owner-only files (``session.key``, ``credentials.key``), created exclusively with
  mode 0600 like an SSH private key. The service account owns them; a file any other account can read or write
  is refused at startup (``chmod 600`` it). Protection at rest beyond that is the disk's (e.g. LUKS).

``LIFEGUARD_SESSION_SECRET`` may be supplied instead of the session file (env file with an
ACL); ``LIFEGUARD_CREDENTIAL_KEY`` is accepted only in development/test (enforced by Settings).
`load_keys()` raises StartupError - the server refuses to start - if either secret is missing
or unreadable. Secret values never appear in messages or logs.
"""

from __future__ import annotations

import base64
import os
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from lifeguard.core.config import Settings
from lifeguard.core.errors import StartupError

WINDOWS = sys.platform == "win32"
SESSION_FILE = "session.key.dpapi" if WINDOWS else "session.key"
CREDENTIAL_FILE = "credentials.key.dpapi" if WINDOWS else "credentials.key"
_ENTROPY = {SESSION_FILE: b"AquaGuard v3 session key", CREDENTIAL_FILE: b"AquaGuard v3 credential key"}
_CRYPTPROTECT_UI_FORBIDDEN = 0x1
_CRYPTPROTECT_LOCAL_MACHINE = 0x4


@dataclass(frozen=True)
class ServerKeys:
    session_key: bytes  # HMAC-SHA256 key for session and CSRF token hashes
    fernet: Fernet  # encrypts camera source URLs at rest

    def __repr__(self) -> str:  # never leak key material through repr/logging
        return "ServerKeys(<redacted>)"


def _require_windows() -> None:
    if sys.platform != "win32":
        raise StartupError("DPAPI-protected keys require Windows (this product is Windows-native)")


def dpapi_protect(data: bytes, entropy: bytes, machine_scope: bool = False) -> bytes:
    _require_windows()
    import win32crypt

    flags = _CRYPTPROTECT_UI_FORBIDDEN | (_CRYPTPROTECT_LOCAL_MACHINE if machine_scope else 0)
    blob: bytes = win32crypt.CryptProtectData(data, "Life Guard", entropy, None, None, flags)
    return blob


def dpapi_unprotect(blob: bytes, entropy: bytes) -> bytes:
    _require_windows()
    import win32crypt

    _desc, data = win32crypt.CryptUnprotectData(blob, entropy, None, None, _CRYPTPROTECT_UI_FORBIDDEN)
    return bytes(data)


def write_protected(path: Path, data: bytes, entropy: bytes, machine_scope: bool = False) -> None:
    """Write a secret file that only this account can use: DPAPI on Windows, an exclusive 0600 file on POSIX."""
    if WINDOWS:
        path.write_bytes(dpapi_protect(data, entropy, machine_scope))
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def read_protected(path: Path, entropy: bytes) -> bytes:
    """Read a file written by write_protected; StartupError if it is unreadable or (POSIX) not owner-only."""
    if WINDOWS:
        return dpapi_unprotect(path.read_bytes(), entropy)
    st = path.stat()
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise StartupError(f"{path.name} is readable or writable by other accounts - chmod 600 {path}")
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise StartupError(f"{path.name} belongs to another account (uid {st.st_uid}) - chown it to the service user")
    return path.read_bytes()


def init_key_files(keys_dir: Path, machine_scope: bool = False) -> list[Path]:
    """Create both DPAPI key files. Refuses to overwrite: a new credential key would orphan every
    encrypted camera source, and a new session key would silently log everyone out."""
    keys_dir.mkdir(parents=True, exist_ok=True)
    if not WINDOWS:
        keys_dir.chmod(0o700)
    targets = {SESSION_FILE: base64.urlsafe_b64encode(secrets.token_bytes(32)), CREDENTIAL_FILE: Fernet.generate_key()}
    existing = [keys_dir / n for n in targets if (keys_dir / n).exists()]
    if existing:
        raise StartupError(f"refusing to overwrite existing key file(s): {', '.join(p.name for p in existing)}")
    written = []
    for name, material in targets.items():
        path = keys_dir / name
        write_protected(path, material, _ENTROPY[name], machine_scope)
        written.append(path)
    return written


def _read_dpapi(path: Path) -> bytes:
    if not path.exists():
        raise StartupError(f"missing key file {path} - run `python -m lifeguard keys init` once")
    try:
        return read_protected(path, _ENTROPY[path.name])
    except StartupError:
        raise
    except Exception as exc:
        # DPAPI failure: wrong Windows account/machine, or a damaged file. Never echo the bytes.
        raise StartupError(
            f"cannot read {path.name} ({type(exc).__name__}); it was created by a different account "
            "or machine, or the file is damaged"
        ) from exc


def load_keys(settings: Settings) -> ServerKeys:
    if settings.session_secret is not None:
        session_key = settings.session_secret.get_secret_value().encode("utf-8")
    else:
        session_key = _read_dpapi(settings.keys_dir / SESSION_FILE)
    if len(session_key) < 32:
        raise StartupError("session secret is shorter than 32 bytes")

    if settings.credential_key is not None:
        raw = settings.credential_key.get_secret_value().encode("ascii")
    else:
        raw = _read_dpapi(settings.keys_dir / CREDENTIAL_FILE)
    try:
        fernet = Fernet(raw)
        fernet.decrypt(fernet.encrypt(b"self-test"))
    except (ValueError, InvalidToken) as exc:
        raise StartupError("credential key is not a valid Fernet key") from exc
    return ServerKeys(session_key=session_key, fernet=fernet)
