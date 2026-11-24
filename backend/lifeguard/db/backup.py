"""Online SQLite backup (sqlite3 backup API) - consistent copy while the server keeps running.

Run from a Windows scheduled task (Phase 8) via ``python -m lifeguard db backup``. Each copy is
checked with ``PRAGMA integrity_check`` before it counts, the newest `keep` copies are retained,
the run is audited, and ``backups/last_backup.json`` is stamped for the supervisor's
missed-backup check (Phase 4).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from lifeguard.core.clock import utcnow

MARKER = "last_backup.json"


@dataclass(frozen=True)
class BackupResult:
    path: Path
    size_bytes: int
    integrity: str
    pruned: int


def backup_sqlite(db_path: Path, dest_dir: Path, keep: int) -> BackupResult:
    if not db_path.exists():
        raise FileNotFoundError(f"database not found: {db_path}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    target = dest_dir / f"lifeguard-{stamp}.db"
    src = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    dst = sqlite3.connect(str(target))
    try:
        src.backup(dst, pages=1024)
    finally:
        dst.close()
        src.close()
    check = sqlite3.connect(f"{target.resolve().as_uri()}?mode=ro", uri=True)
    try:
        integrity = str(check.execute("PRAGMA integrity_check").fetchone()[0])
    finally:
        check.close()
    if integrity != "ok":
        target.unlink(missing_ok=True)
        raise RuntimeError(f"backup failed integrity_check: {integrity}")
    # prune both spellings together: an AquaGuard install's old backups count toward `keep` too  # legacy-name
    copies = sorted({*dest_dir.glob("lifeguard-*.db"), *dest_dir.glob("aquaguard-*.db")})  # legacy-name
    stale = copies[:-keep] if len(copies) > keep else []
    for old in stale:
        old.unlink(missing_ok=True)
    (dest_dir / MARKER).write_text(
        json.dumps({"at": utcnow().isoformat(), "file": target.name, "size_bytes": target.stat().st_size}),
        encoding="utf-8",
    )
    return BackupResult(target, target.stat().st_size, integrity, len(stale))
