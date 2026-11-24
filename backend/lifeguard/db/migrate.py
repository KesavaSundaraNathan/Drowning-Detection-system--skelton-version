"""Alembic, driven programmatically (no alembic.ini): `upgrade()`, `current()`, `head()`.

The server refuses to start unless the database is at the head revision (`assert_at_head`),
so a deployment can never run new code against an old schema. `lifeguard db upgrade` (CLI)
is the only way the schema changes.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine

from lifeguard.core.errors import StartupError

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config(engine: Engine) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.attributes["engine"] = engine
    return cfg


def head() -> str:
    heads = ScriptDirectory.from_config(alembic_config_for_scripts()).get_heads()
    if len(heads) != 1:
        raise StartupError(f"expected exactly one migration head, found {heads}")
    return heads[0]


def alembic_config_for_scripts() -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    return cfg


def current(engine: Engine) -> str | None:
    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def upgrade(engine: Engine, revision: str = "head") -> None:
    command.upgrade(alembic_config(engine), revision)


def assert_at_head(engine: Engine) -> None:
    cur, want = current(engine), head()
    if cur != want:
        raise StartupError(
            f"database schema is at {cur or 'nothing'} but this build needs {want} - "
            "run `python -m lifeguard db upgrade` (after `db backup`)"
        )
