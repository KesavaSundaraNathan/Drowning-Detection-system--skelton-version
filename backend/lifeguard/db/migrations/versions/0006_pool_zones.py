"""pool-level zones (Phase 11 WP2): zones on the pool plan in metres, projected into every calibrated camera

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-09 18:00:00.000000

The zones table is rebuilt (SQLite cannot relax NOT NULL in place): run it with the services stopped.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None

CAMERA_OR_POOL = (
    "(camera_id IS NOT NULL AND polygon IS NOT NULL AND pool_id IS NULL AND pool_polygon_m IS NULL)"
    " OR (camera_id IS NULL AND polygon IS NULL AND pool_id IS NOT NULL AND pool_polygon_m IS NOT NULL)"
)


def upgrade() -> None:
    with op.batch_alter_table('zones', schema=None, recreate='always') as batch_op:
        batch_op.add_column(sa.Column('pool_id', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('pool_polygon_m', sa.JSON(none_as_null=True), nullable=True))
        batch_op.alter_column('camera_id', existing_type=sa.String(length=36), nullable=True)
        batch_op.alter_column('polygon', existing_type=sa.JSON(), nullable=True)
        batch_op.create_unique_constraint(batch_op.f('uq_zones_pool_id_name'), ['pool_id', 'name'])
        batch_op.create_check_constraint(batch_op.f('ck_zones_camera_or_pool'), CAMERA_OR_POOL)
        batch_op.create_index(batch_op.f('ix_zones_pool_id'), ['pool_id'], unique=False)


def downgrade() -> None:
    op.execute("DELETE FROM zones WHERE camera_id IS NULL")  # pool zones cannot exist in the old schema
    with op.batch_alter_table('zones', schema=None, recreate='always') as batch_op:
        batch_op.drop_index(batch_op.f('ix_zones_pool_id'))
        batch_op.drop_constraint(batch_op.f('ck_zones_camera_or_pool'), type_='check')
        batch_op.drop_constraint(batch_op.f('uq_zones_pool_id_name'), type_='unique')
        batch_op.alter_column('polygon', existing_type=sa.JSON(), nullable=False)
        batch_op.alter_column('camera_id', existing_type=sa.String(length=36), nullable=False)
        batch_op.drop_column('pool_polygon_m')
        batch_op.drop_column('pool_id')
