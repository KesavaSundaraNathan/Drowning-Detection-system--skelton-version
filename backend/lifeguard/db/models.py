"""ORM models (SQLAlchemy 2, typed). Schema changes go through Alembic (db/migrations).

Differences from v2 (audit 1.1 database rows):
  * real foreign keys with explicit ON DELETE behaviour, CHECK constraints on every enum-like
    column, indexes on the columns the API filters by;
  * timezone-aware UTC timestamps everywhere (db.types.UTCDateTime);
  * sessions store only HMAC hashes of their tokens, plus absolute expiry and last-seen;
  * camera sources are stored only encrypted (+ a masked display string), never in clear.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql.expression import true

from lifeguard.db.types import UTCDateTime

NAMING = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

ROLES = ("lifeguard", "supervisor", "admin")
MODES = ("Low", "Medium", "High")
ZONE_TYPES = ("WATER", "DECK", "SHALLOW")
SCHEDULE_STATUSES = ("Active", "Disabled")
ALERT_STATUSES = ("pending", "acknowledged", "resolved", "false_alarm")
EVENT_STATUSES = ("open", "acknowledged", "resolved")
EVENT_SEVERITIES = ("critical", "high", "caution")
SOURCE_TYPES = ("rtsp", "file")
# [v4] "pool": full drowning detection; "standard_cctv": person presence only, no drowning alerts
SCENE_TYPES = ("pool", "standard_cctv")
SCENE_TYPE_SET_BY = ("default", "admin")
OUTBOX_KINDS = ("ALERT", "FAULT", "FAULT_CLEARED", "ESCALATION", "SELF_TEST")
OUTBOX_CHANNELS = ("webhook", "ntfy", "smtp")
OUTBOX_STATUSES = ("pending", "sent", "dead")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)
    type_annotation_map: ClassVar[dict[Any, Any]] = {datetime: UTCDateTime()}


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint(_in("role", ROLES), name="role"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)  # stored lower-case
    password_hash: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(16))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
    password_changed_at: Mapped[datetime]


class UserSession(Base):
    __tablename__ = "user_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)  # HMAC-SHA256 hex
    csrf_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime]
    last_seen_at: Mapped[datetime]
    expires_at: Mapped[datetime]  # absolute expiry
    revoked_at: Mapped[datetime | None]
    ip: Mapped[str] = mapped_column(String(45))
    user_agent: Mapped[str] = mapped_column(String(256))


class LoginAttempt(Base):
    """One row per login attempt. success=None = in flight (counts as a failure until resolved)."""

    __tablename__ = "login_attempts"
    __table_args__ = (
        Index("ix_login_attempts_username_at", "username", "at"),
        Index("ix_login_attempts_ip_at", "ip", "at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    at: Mapped[datetime]
    username: Mapped[str] = mapped_column(String(64))
    ip: Mapped[str] = mapped_column(String(45))
    success: Mapped[bool | None]


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_events_action_at", "action", "at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    at: Mapped[datetime] = mapped_column(index=True)
    actor_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    actor_username: Mapped[str] = mapped_column(String(64))
    ip: Mapped[str] = mapped_column(String(45))
    action: Mapped[str] = mapped_column(String(64))
    target_type: Mapped[str] = mapped_column(String(32))
    target_id: Mapped[str] = mapped_column(String(64))
    outcome: Mapped[str] = mapped_column(String(16))
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Camera(Base):
    __tablename__ = "cameras"
    __table_args__ = (
        CheckConstraint(_in("source_type", SOURCE_TYPES), name="source_type"),
        CheckConstraint(_in("default_mode", MODES), name="default_mode"),
        # scene_type / scene_type_set_by: no CHECK (migration 0008 explains why); validated by the API schema
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # uuid4
    name: Mapped[str] = mapped_column(String(100), unique=True)
    source_type: Mapped[str] = mapped_column(String(8))
    source_ciphertext: Mapped[str] = mapped_column(Text)  # Fernet token
    source_display: Mapped[str] = mapped_column(String(1100))  # masked, safe to show
    default_mode: Mapped[str] = mapped_column(String(8), default="Medium")
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # Pool calibration (Phase 7): {"image_points": [[x, y], ...] normalised, "pool_points": [[x_m, y_m], ...],
    # "outline_m"?: [[x_m, y_m], ...], "edge_band_m"?: float} - see vision/scene/pool_geometry.py
    calibration: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True, default=None)
    # [v4] fail-safe default "pool": only an admin turns drowning detection off (the automatic scene check in
    # vision/scene/scene_type.py only SUGGESTS "standard_cctv", it never applies it)
    scene_type: Mapped[str] = mapped_column(String(16), default="pool", server_default="pool")
    scene_type_set_by: Mapped[str] = mapped_column(String(8), default="default", server_default="default")
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class Schedule(Base):
    __tablename__ = "schedules"
    __table_args__ = (
        CheckConstraint(_in("status", SCHEDULE_STATUSES), name="status"),
        CheckConstraint(_in("monitoring_mode", MODES), name="monitoring_mode"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    days: Mapped[list[str]] = mapped_column(JSON)  # ["Mon", ...]
    start_time: Mapped[str] = mapped_column(String(5))  # "HH:MM" local
    end_time: Mapped[str] = mapped_column(String(5))
    monitoring_mode: Mapped[str] = mapped_column(String(8), default="Medium")
    status: Mapped[str] = mapped_column(String(10), default="Active")
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class Zone(Base):
    """A camera zone (camera_id + image `polygon`, normalised) or [Phase 11] a POOL zone (pool_id +
    `pool_polygon_m` on the pool plan in metres, camera_id NULL) that every calibrated camera of that pool
    projects into its own view (vision/scene/pool_zones.py) - one zone, the same membership on every camera."""

    __tablename__ = "zones"
    __table_args__ = (
        UniqueConstraint("camera_id", "name"),
        UniqueConstraint("pool_id", "name"),
        CheckConstraint(_in("zone_type", ZONE_TYPES), name="zone_type"),
        CheckConstraint("sensitivity >= 0.25 AND sensitivity <= 3.0", name="sensitivity"),
        CheckConstraint(
            "(camera_id IS NOT NULL AND polygon IS NOT NULL AND pool_id IS NULL AND pool_polygon_m IS NULL)"
            " OR (camera_id IS NULL AND polygon IS NULL AND pool_id IS NOT NULL AND pool_polygon_m IS NOT NULL)",
            name="camera_or_pool",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[str | None] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    zone_type: Mapped[str] = mapped_column(String(8))
    polygon: Mapped[list[list[float]] | None] = mapped_column(JSON(none_as_null=True))  # [[x, y], ...] in [0, 1]
    pool_id: Mapped[str | None] = mapped_column(String(64), index=True)
    pool_polygon_m: Mapped[list[list[float]] | None] = mapped_column(JSON(none_as_null=True))  # pool plan, metres
    sensitivity: Mapped[float] = mapped_column(Float, default=1.0)
    schedule_id: Mapped[int | None] = mapped_column(ForeignKey("schedules.id", ondelete="SET NULL"))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class Alert(Base):
    """Alert rows are produced by the live pipeline (Phase 4/5); Phase 2 owns the
    review workflow (acknowledge / resolve / false alarm) and its audit trail."""

    __tablename__ = "alerts"
    __table_args__ = (
        CheckConstraint(_in("status", ALERT_STATUSES), name="status"),
        Index("ix_alerts_status_created_at", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(index=True)
    camera_id: Mapped[str | None] = mapped_column(ForeignKey("cameras.id", ondelete="SET NULL"), index=True)
    track_id: Mapped[int | None]
    scenario: Mapped[str] = mapped_column(String(32))
    priority: Mapped[str] = mapped_column(String(16))
    risk_score: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float)
    zone_name: Mapped[str | None] = mapped_column(String(100))
    reason: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="pending")
    acknowledged_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    acknowledged_at: Mapped[datetime | None]
    closed_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    closed_at: Mapped[datetime | None]
    close_note: Mapped[str | None] = mapped_column(Text)
    # ── Phase 5 ──
    # sounds / escalates? False = muted by a zone schedule (never for SUBMERSION / FAULT)
    audible: Mapped[bool] = mapped_column(Boolean, default=True, server_default=true())
    fault_code: Mapped[str | None] = mapped_column(String(48))  # FAULT rows only
    fault_source: Mapped[str | None] = mapped_column(String(64))  # camera id / "system"
    cleared_at: Mapped[datetime | None]  # FAULT condition ended (a FAULT_CLEARED notice was queued)
    snapshot_path: Mapped[str | None] = mapped_column(String(255))  # name inside snapshots_dir
    recording_path: Mapped[str | None] = mapped_column(String(255))  # name inside recordings_dir
    escalation_level: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    escalated_at: Mapped[datetime | None]
    # ── Phase 11 ──
    # the person's pool-wide id and cross-camera incident (removed in Phase 5: cameras are independent; the columns
    # stay for old rows and are NULL for new ones)
    global_id: Mapped[int | None]
    incident_id: Mapped[int | None] = mapped_column(Integer, index=True)
    # ── Phase 5 (rebuild) ── the event this behaviour alert belongs to (alerts/events.py; NULL for FAULT rows)
    # a plain indexed id like incident_id (no FK: on SQLite adding one in place is not reflected by name, and a batch
    # rebuild of `alerts` would cascade into alert_outbox); events are never deleted
    event_id: Mapped[int | None] = mapped_column(Integer, index=True)


class Event(Base):
    """[Phase 5] One event per person and incident on ONE camera (cameras are independent): the alert fires of one
    track, from its first alert until it has been quiet for EVENT_IDLE_CLOSE_S (alerts/events.py). Alerts stay rows of
    their own; an event groups them for the AlertCenter, one acknowledge, and the alerts-per-event metric."""

    __tablename__ = "events"
    __table_args__ = (
        CheckConstraint(_in("status", EVENT_STATUSES), name="status"),
        CheckConstraint(_in("severity", EVENT_SEVERITIES), name="severity"),
        Index("ix_events_open", "camera_id", "track_id", "closed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[str | None] = mapped_column(ForeignKey("cameras.id", ondelete="SET NULL"), index=True)
    track_id: Mapped[int | None]
    opened_at: Mapped[datetime] = mapped_column(index=True)
    updated_at: Mapped[datetime] = mapped_column()
    closed_at: Mapped[datetime | None] = mapped_column()
    severity: Mapped[str] = mapped_column(String(16))
    scenario: Mapped[str] = mapped_column(String(32))  # the worst scenario seen so far
    status: Mapped[str] = mapped_column(String(16), default="open")
    alert_count: Mapped[int] = mapped_column(Integer, default=0)
    first_alert_id: Mapped[int | None] = mapped_column(Integer)
    last_alert_id: Mapped[int | None] = mapped_column(Integer)
    acknowledged_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    acknowledged_at: Mapped[datetime | None] = mapped_column()
    resolved_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    resolved_at: Mapped[datetime | None] = mapped_column()


class AlertOutbox(Base):
    """Out-of-band deliveries (Phase 5). One row per (event, channel), written in the SAME transaction as
    the alert / FAULT / escalation it announces, so nothing is lost between "recorded" and "sent".
    The dispatcher retries with backoff until the row is sent (at-least-once; receivers de-duplicate
    on event_id) or older than outbox_max_age_h ("dead", which raises a FAULT)."""

    __tablename__ = "alert_outbox"
    __table_args__ = (
        CheckConstraint(_in("kind", OUTBOX_KINDS), name="kind"),
        CheckConstraint(_in("channel", OUTBOX_CHANNELS), name="channel"),
        CheckConstraint(_in("status", OUTBOX_STATUSES), name="status"),
        UniqueConstraint("event_id", "channel"),
        Index("ix_alert_outbox_due", "status", "channel", "next_attempt_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alert_id: Mapped[int | None] = mapped_column(ForeignKey("alerts.id", ondelete="SET NULL"), index=True)
    event_id: Mapped[str] = mapped_column(String(96))
    kind: Mapped[str] = mapped_column(String(16))
    channel: Mapped[str] = mapped_column(String(8))
    tier: Mapped[int] = mapped_column(Integer, default=0)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(8), default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column()
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(index=True)
    sent_at: Mapped[datetime | None] = mapped_column()
