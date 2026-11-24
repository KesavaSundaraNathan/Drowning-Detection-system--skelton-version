"""Old AquaGuard names still honoured after the rename to Life Guard. One place, so the final name grep  # legacy-name
lists it, and so every fallback is documented together instead of scattered.

Kept, because changing them would break a live install:
  * environment variables: ``AQUAGUARD_<NAME>`` is copied to ``LIFEGUARD_<NAME>`` when the new name is not
    already set (process environment, read once at import time here, before ``Settings`` is defined);
  * the data directory of an existing install (``C:\\ProgramData\\AquaGuard`` / ``/var/lib/aquaguard``): used
    when only the old one exists, and logged once;
  * the SQLite file of an existing data directory (``aquaguard.db``): used when only it exists there.

Everything else in the rename has no shim (cookies, browser storage keys, webhook header pairing, DPAPI
entropy, data-format ids) - see the rebuild plan for the full list and why.

Every line that spells an old name carries the marker ``legacy-name`` so the rename tooling leaves it alone.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("lifeguard.legacy")

OLD_PREFIX = "AQUAGUARD_"  # legacy-name
NEW_PREFIX = "LIFEGUARD_"
OLD_DATA_DIR = (  # legacy-name
    Path(r"C:\ProgramData\AquaGuard") if sys.platform == "win32" else Path("/var/lib/aquaguard")  # legacy-name
)
NEW_DATA_DIR = Path(r"C:\ProgramData\LifeGuard") if sys.platform == "win32" else Path("/var/lib/lifeguard")
OLD_DB_FILE = "aquaguard.db"  # legacy-name
NEW_DB_FILE = "lifeguard.db"

ENV_COPIED: list[str] = []  # old variable names copied to the new prefix in this process (logged once at startup)


def migrate_environment(environ: os._Environ[str] | dict[str, str] | None = None) -> list[str]:
    """Copy every ``AQUAGUARD_X`` in the environment to ``LIFEGUARD_X`` unless ``LIFEGUARD_X`` is already set.

    Idempotent: safe to call more than once (a key already copied is simply seen as already set next time).
    """
    env = os.environ if environ is None else environ
    for key in list(env):
        if key.upper().startswith(OLD_PREFIX):
            new = NEW_PREFIX + key[len(OLD_PREFIX) :]
            if new not in env:
                env[new] = env[key]
                if key not in ENV_COPIED:
                    ENV_COPIED.append(key)
    return ENV_COPIED


def default_data_dir() -> Path:
    """The new default data directory, unless only an existing install's old one is there."""
    if not NEW_DATA_DIR.exists() and OLD_DATA_DIR.exists():
        logger.warning(
            "using the existing data directory %s (move it to %s when convenient)", OLD_DATA_DIR, NEW_DATA_DIR
        )
        return OLD_DATA_DIR
    return NEW_DATA_DIR


def sqlite_file(db_dir: Path) -> Path:
    """``lifeguard.db`` in `db_dir`, unless only an existing install's ``aquaguard.db`` is there."""
    new, old = db_dir / NEW_DB_FILE, db_dir / OLD_DB_FILE
    if not new.exists() and old.exists():
        logger.warning("using the existing database file %s", old)
        return old
    return new


# Run once, as early as possible: before core.config.Settings (env_prefix="LIFEGUARD_") reads the environment,
# and before LIFEGUARD_ENV_FILE is looked up to find the env file itself.
migrate_environment()
