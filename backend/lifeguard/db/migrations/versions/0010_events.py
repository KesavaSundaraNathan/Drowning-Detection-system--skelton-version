"""events (Life Guard rebuild Phase 5: one event per person and incident on one camera)

Creates `events` and adds the nullable, indexed `alerts.event_id` (no FK, like incident_id) (alerts/events.py groups a track's alert fires into one event).
Existing alerts keep their rows and get no event.

On SQLite the column is added IN PLACE (ALTER TABLE ... ADD COLUMN), never in batch mode: a batch
rebuild of `alerts` drops the old table, and with foreign_keys=ON that sets alert_outbox.alert_id to NULL
(see 0008 / 0009).

Revision ID: 0010
Revises: 0009
Create Date: 2026-11-15 18:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

import lifeguard.db.types

revision = '0010'
down_revision = '0009'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'events',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('camera_id', sa.String(length=36), nullable=True),
        sa.Column('track_id', sa.Integer(), nullable=True),
        sa.Column('opened_at', lifeguard.db.types.UTCDateTime(timezone=True), nullable=False),
        sa.Column('updated_at', lifeguard.db.types.UTCDateTime(timezone=True), nullable=False),
        sa.Column('closed_at', lifeguard.db.types.UTCDateTime(timezone=True), nullable=True),
        sa.Column('severity', sa.String(length=16), nullable=False),
        sa.Column('scenario', sa.String(length=32), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('alert_count', sa.Integer(), nullable=False),
        sa.Column('first_alert_id', sa.Integer(), nullable=True),
        sa.Column('last_alert_id', sa.Integer(), nullable=True),
        sa.Column('acknowledged_by', sa.Integer(), nullable=True),
        sa.Column('acknowledged_at', lifeguard.db.types.UTCDateTime(timezone=True), nullable=True),
        sa.Column('resolved_by', sa.Integer(), nullable=True),
        sa.Column('resolved_at', lifeguard.db.types.UTCDateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('open', 'acknowledged', 'resolved')", name=op.f('ck_events_status')),
        sa.CheckConstraint("severity IN ('critical', 'high', 'caution')", name=op.f('ck_events_severity')),
        sa.ForeignKeyConstraint(
            ['acknowledged_by'], ['users.id'], name=op.f('fk_events_acknowledged_by_users'), ondelete='SET NULL'
        ),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], name=op.f('fk_events_camera_id_cameras'), ondelete='SET NULL'),
        sa.ForeignKeyConstraint(
            ['resolved_by'], ['users.id'], name=op.f('fk_events_resolved_by_users'), ondelete='SET NULL'
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_events')),
    )
    with op.batch_alter_table('events', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_events_camera_id'), ['camera_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_events_opened_at'), ['opened_at'], unique=False)
        batch_op.create_index('ix_events_open', ['camera_id', 'track_id', 'closed_at'], unique=False)
    op.add_column('alerts', sa.Column('event_id', sa.Integer(), nullable=True))  # SQLite: ADD COLUMN in place
    op.create_index(op.f('ix_alerts_event_id'), 'alerts', ['event_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_alerts_event_id'), table_name='alerts')
    if op.get_bind().dialect.name == 'sqlite':
        op.execute('ALTER TABLE alerts DROP COLUMN event_id')  # SQLite >= 3.35: in place, no rebuild / cascade
    else:
        op.drop_column('alerts', 'event_id')
    with op.batch_alter_table('events', schema=None) as batch_op:
        batch_op.drop_index('ix_events_open')
        batch_op.drop_index(batch_op.f('ix_events_opened_at'))
        batch_op.drop_index(batch_op.f('ix_events_camera_id'))
    op.drop_table('events')
