"""Database engines.

Two engines per database:
  * the WRITE engine - used only by the single DB-writer thread (db.writer);
  * the READ engine - every request handler reads through it, and it is read-only at the
    driver level (SQLite: ``mode=ro`` URI + ``PRAGMA query_only``; PostgreSQL:
    ``default_transaction_read_only``). A handler that tries to write outside the writer fails
    loudly instead of racing it.

SQLite runs in WAL mode (readers never block the writer), with ``busy_timeout``, foreign keys ON
and ``synchronous=FULL`` (an acknowledged alert/audit row survives a power cut).
PostgreSQL is a configuration switch (``LIFEGUARD_DATABASE_URL=postgresql+psycopg://...``); its
driver is not in serving.lock yet and must be added (and locked) before that switch is used.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

BUSY_TIMEOUT_MS = 30_000


@dataclass(frozen=True)
class Database:
    write_engine: Engine
    read_engine: Engine
    write_sessions: sessionmaker[Session]
    read_sessions: sessionmaker[Session]

    def dispose(self) -> None:
        self.write_engine.dispose()
        self.read_engine.dispose()


def _sqlite_pragmas(dbapi_conn: sqlite3.Connection, read_only: bool) -> None:
    cur = dbapi_conn.cursor()
    try:
        cur.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        cur.execute("PRAGMA foreign_keys = ON")
        if read_only:
            cur.execute("PRAGMA query_only = ON")
        else:
            cur.execute("PRAGMA journal_mode = WAL")
            cur.execute("PRAGMA synchronous = FULL")
    finally:
        cur.close()


def _sqlite_engines(path: Path) -> tuple[Engine, Engine]:
    path.parent.mkdir(parents=True, exist_ok=True)

    def write_conn() -> sqlite3.Connection:
        conn = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_MS / 1000, check_same_thread=False, isolation_level=None)
        _sqlite_pragmas(conn, read_only=False)
        return conn

    def read_conn() -> sqlite3.Connection:
        conn = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro",
            uri=True,
            timeout=BUSY_TIMEOUT_MS / 1000,
            check_same_thread=False,
            isolation_level=None,
        )
        _sqlite_pragmas(conn, read_only=True)
        return conn

    # The writer must create the file (and the WAL) before any read-only connection opens it.
    write = create_engine("sqlite://", creator=write_conn, poolclass=NullPool)
    with write.connect():
        pass
    read = create_engine("sqlite://", creator=read_conn, poolclass=NullPool)
    for eng in (write, read):
        # pysqlite autocommit mode + explicit BEGIN: SQLAlchemy owns the transaction boundaries.
        event.listen(eng, "begin", lambda conn: conn.exec_driver_sql("BEGIN"))
    return write, read


def _sqlite_path(url: str) -> Path:
    prefix = "sqlite:///"
    if not url.startswith(prefix) or url == prefix:
        raise ValueError("SQLite URL must be sqlite:///<absolute path>")
    return Path(url[len(prefix) :])


def make_database(url: str) -> Database:
    if url.startswith("sqlite:"):
        write, read = _sqlite_engines(_sqlite_path(url))
    else:
        connect_args: dict[str, Any] = {"options": "-c default_transaction_read_only=on"}
        write = create_engine(url, pool_pre_ping=True)
        read = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
    return Database(
        write_engine=write,
        read_engine=read,
        write_sessions=sessionmaker(write, expire_on_commit=False),
        read_sessions=sessionmaker(read, expire_on_commit=False),
    )
