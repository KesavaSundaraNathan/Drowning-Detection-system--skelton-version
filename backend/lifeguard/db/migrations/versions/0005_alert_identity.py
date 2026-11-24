"""alert global id + incident (Phase 11 WP1: live cross-camera identity)

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-09 09:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = '0005'
down_revision = '0004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # one person across cameras (global_id) and one incident per person (incident_id = first alert id)
    with op.batch_alter_table('alerts', schema=None) as batch_op:
        batch_op.add_column(sa.Column('global_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('incident_id', sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f('ix_alerts_incident_id'), ['incident_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('alerts', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_alerts_incident_id'))
        batch_op.drop_column('incident_id')
        batch_op.drop_column('global_id')
