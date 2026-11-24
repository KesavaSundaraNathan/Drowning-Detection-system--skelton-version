"""analysis job camera (upload analysis with a registered camera's zones)

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-10 20:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = '0007'
down_revision = '0006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # the upload is analysed with this camera's zones + calibration (NULL = the whole frame is water)
    with op.batch_alter_table('analysis_jobs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('camera_id', sa.String(length=36), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('analysis_jobs', schema=None) as batch_op:
        batch_op.drop_column('camera_id')
