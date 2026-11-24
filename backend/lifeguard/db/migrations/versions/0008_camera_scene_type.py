"""camera scene type (v4: pool cameras run drowning detection, standard CCTV cameras only person presence)

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-15 12:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = '0008'
down_revision = '0007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Every existing camera stays a pool camera (drowning detection on) until an admin says otherwise.
    # Columns only, no CHECK constraint: on SQLite a new constraint makes batch mode RECREATE `cameras`, and dropping
    # the old table cascades into zones / alerts (tests/unit/test_pool_zone_migration.py lost its zone that way).
    # The values are validated by the API schema (SceneType).
    with op.batch_alter_table('cameras', schema=None) as batch_op:
        batch_op.add_column(sa.Column('scene_type', sa.String(length=16), nullable=False, server_default='pool'))
        batch_op.add_column(
            sa.Column('scene_type_set_by', sa.String(length=8), nullable=False, server_default='default')
        )


def downgrade() -> None:
    # Not batch mode: on SQLite a batch drop_column RECREATES `cameras` and the drop of the old table cascades into
    # zones / alerts. SQLite >= 3.35 drops a column in place (no rebuild, no cascade).
    if op.get_bind().dialect.name == 'sqlite':
        op.execute('ALTER TABLE cameras DROP COLUMN scene_type_set_by')
        op.execute('ALTER TABLE cameras DROP COLUMN scene_type')
        return
    op.drop_column('cameras', 'scene_type_set_by')
    op.drop_column('cameras', 'scene_type')
