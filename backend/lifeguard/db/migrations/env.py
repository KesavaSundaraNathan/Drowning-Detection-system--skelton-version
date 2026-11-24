"""Alembic environment: always runs online against the engine handed in by db.migrate."""

from __future__ import annotations

from alembic import context

from lifeguard.db.models import Base

config = context.config
engine = config.attributes.get("engine")
if engine is None:
    raise RuntimeError("run migrations through lifeguard.db.migrate (python -m lifeguard db upgrade)")

with engine.connect() as connection:
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        render_as_batch=connection.dialect.name == "sqlite",   # SQLite ALTERs via table rebuild
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()
