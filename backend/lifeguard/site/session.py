"""The site-session folder (Phase 11 WP6).

<DataDir>\\site_session\\<session_id>\\        created by site_session.ps1 with SYSTEM + Administrators only
    session.json        header (legal references), cameras (public facts), kit + thresholds versions
    geometry.sqlite     cameras (NO source), schedules, zones - what the replay builds geometry from
    checks\\            config_check / camera_check / coverage (+ png) / capacity / live_run
    recordings\\<block>\\*.mkv + recordings.json (per block: files with absolute start / end)
    drill_log.json/.csv, sync.json, RIGHTS.csv
    replay\\<block>\\<arm>\\   site_replay output;  report\\  site_report output
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import insert, select

SNAPSHOT_TABLES = ("cameras", "schedules", "zones")


class SessionFolder:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, name: str, data: Any) -> Path:
        p = self.path(*name.split("/"))
        p.write_bytes((json.dumps(data, indent=1, default=str) + "\n").encode("utf-8"))
        return p

    def read_json(self, name: str, default: Any = None) -> Any:
        p = self.root.joinpath(*name.split("/"))
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    def add_recordings(self, block: str, recordings: list[dict[str, Any]]) -> None:
        reg = self.read_json("recordings.json", {"blocks": {}})
        reg["blocks"].setdefault(block, []).extend(recordings)
        self.write_json("recordings.json", reg)


def snapshot_geometry(source_url: str, target: Path) -> dict[str, int]:
    """A fresh, migrated SQLite with the cameras (source emptied), schedules and zones of the live database - all the
    replay's geometry loader reads. No users, no sessions, no camera source, no alerts."""
    from lifeguard.db import migrate
    from lifeguard.db.models import Base
    from lifeguard.db.session import make_database

    if target.exists():
        target.unlink()
    src = make_database(source_url)
    dst = make_database(f"sqlite:///{target.as_posix()}")
    counts: dict[str, int] = {}
    try:
        migrate.upgrade(dst.write_engine)
        with src.read_sessions() as rs, dst.write_engine.begin() as conn:
            for name in SNAPSHOT_TABLES:
                table = Base.metadata.tables[name]
                rows = [dict(r._mapping) for r in rs.execute(select(table))]
                if name == "cameras":
                    for r in rows:
                        r["source_ciphertext"] = ""  # never leaves the live database
                if rows:
                    conn.execute(insert(table), rows)
                counts[name] = len(rows)
    finally:
        src.dispose()
        dst.dispose()
    return counts
