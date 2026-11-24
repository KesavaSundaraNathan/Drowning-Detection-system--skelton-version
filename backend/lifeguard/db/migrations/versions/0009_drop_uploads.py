"""drop uploads (Life Guard rebuild: RTSP only, the upload feature is removed)

Drops `analysis_jobs`, `uploads` and `alerts.upload_id`. Alerts raised by upload analysis keep their rows (the
review history stays); they simply no longer point at an upload. Files left in `<data_dir>/uploads` are purged by
the nightly retention job.

Revision ID: 0009
Revises: 0008
Create Date: 2026-11-13 14:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

import lifeguard.db.types

revision = '0009'
down_revision = '0008'
branch_labels = None
depends_on = None


def _drop_alert_upload_column() -> None:
    bind = op.get_bind()
    if bind.dialect.name != 'sqlite':
        op.drop_constraint('fk_alerts_upload_id_uploads', 'alerts', type_='foreignkey')
        op.drop_column('alerts', 'upload_id')
        return
    # SQLite cannot drop a column that is part of a foreign key, so batch mode rebuilds `alerts`. Dropping the old
    # table runs an implicit DELETE, and with foreign_keys=ON that sets alert_outbox.alert_id to NULL
    # (ON DELETE SET NULL). Keep the links and put them back after the rebuild.
    links = bind.execute(sa.text('SELECT id, alert_id FROM alert_outbox WHERE alert_id IS NOT NULL')).all()
    with op.batch_alter_table('alerts', schema=None) as batch_op:
        batch_op.drop_constraint('fk_alerts_upload_id_uploads', type_='foreignkey')
        batch_op.drop_column('upload_id')
    for row_id, alert_id in links:
        bind.execute(
            sa.text('UPDATE alert_outbox SET alert_id = :a WHERE id = :i'), {'a': alert_id, 'i': row_id}
        )


def upgrade() -> None:
    _drop_alert_upload_column()
    with op.batch_alter_table('analysis_jobs', schema=None) as batch_op:
        batch_op.drop_index('ix_analysis_jobs_status_id')
        batch_op.drop_index(batch_op.f('ix_analysis_jobs_upload_id'))
        batch_op.drop_index(batch_op.f('ix_analysis_jobs_created_at'))
    op.drop_table('analysis_jobs')
    with op.batch_alter_table('uploads', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_uploads_user_id'))
        batch_op.drop_index(batch_op.f('ix_uploads_created_at'))
    op.drop_table('uploads')


def downgrade() -> None:
    # Restores the empty tables and the column (the deleted uploads themselves cannot come back).
    op.create_table(
        'uploads',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('original_name', sa.String(length=255), nullable=False),
        sa.Column('stored_name', sa.String(length=80), nullable=False),
        sa.Column('size_bytes', sa.Integer(), nullable=False),
        sa.Column('sha256', sa.String(length=64), nullable=False),
        sa.Column('width', sa.Integer(), nullable=False),
        sa.Column('height', sa.Integer(), nullable=False),
        sa.Column('fps', sa.Float(), nullable=False),
        sa.Column('frame_count', sa.Integer(), nullable=False),
        sa.Column('created_at', lifeguard.db.types.UTCDateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_uploads_user_id_users'), ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_uploads')),
        sa.UniqueConstraint('stored_name', name=op.f('uq_uploads_stored_name')),
    )
    with op.batch_alter_table('uploads', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_uploads_created_at'), ['created_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_uploads_user_id'), ['user_id'], unique=False)
    op.create_table(
        'analysis_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('upload_id', sa.Integer(), nullable=False),
        sa.Column('requested_by', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('mode', sa.String(length=8), nullable=False),
        sa.Column('camera_id', sa.String(length=36), nullable=True),
        sa.Column('attempts', sa.Integer(), nullable=False),
        sa.Column('frames_done', sa.Integer(), nullable=False),
        sa.Column('frame_count', sa.Integer(), nullable=False),
        sa.Column('alerts_raised', sa.Integer(), nullable=False),
        sa.Column('worker_pid', sa.Integer(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', lifeguard.db.types.UTCDateTime(timezone=True), nullable=False),
        sa.Column('started_at', lifeguard.db.types.UTCDateTime(timezone=True), nullable=True),
        sa.Column('finished_at', lifeguard.db.types.UTCDateTime(timezone=True), nullable=True),
        sa.Column('heartbeat_at', lifeguard.db.types.UTCDateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'processing', 'done', 'failed', 'cancelled')", name=op.f('ck_analysis_jobs_status')
        ),
        sa.ForeignKeyConstraint(
            ['requested_by'], ['users.id'], name=op.f('fk_analysis_jobs_requested_by_users'), ondelete='SET NULL'
        ),
        sa.ForeignKeyConstraint(
            ['upload_id'], ['uploads.id'], name=op.f('fk_analysis_jobs_upload_id_uploads'), ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_analysis_jobs')),
    )
    with op.batch_alter_table('analysis_jobs', schema=None) as batch_op:
        batch_op.create_index('ix_analysis_jobs_status_id', ['status', 'id'], unique=False)
        batch_op.create_index(batch_op.f('ix_analysis_jobs_upload_id'), ['upload_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_analysis_jobs_created_at'), ['created_at'], unique=False)
    with op.batch_alter_table('alerts', schema=None) as batch_op:
        batch_op.add_column(sa.Column('upload_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            batch_op.f('fk_alerts_upload_id_uploads'), 'uploads', ['upload_id'], ['id'], ondelete='SET NULL'
        )
