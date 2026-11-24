"""camera pool calibration

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-08 00:20:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Pool calibration (Phase 7): image points <-> pool metres, see vision/scene/pool_geometry.py
    with op.batch_alter_table('cameras', schema=None) as batch_op:
        batch_op.add_column(sa.Column('calibration', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('cameras', schema=None) as batch_op:
        batch_op.drop_column('calibration')
