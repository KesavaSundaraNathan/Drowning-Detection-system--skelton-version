"""API v1 request/response models. Request bodies forbid unknown fields."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)

from lifeguard.core.schedule import WEEKDAYS, parse_hhmm
from lifeguard.core.security.rbac import Role

Mode = Literal["Low", "Medium", "High"]
ZoneType = Literal["WATER", "DECK", "SHALLOW"]
SceneType = Literal["pool", "standard_cctv"]  # [v4] db.models.SCENE_TYPES
Weekday = Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
# The pattern is checked BEFORE to_lower, so it accepts both cases; the stored value is lower-case.
Username = Annotated[
    str,
    StringConstraints(strip_whitespace=True, to_lower=True, min_length=3, max_length=64, pattern=r"^[A-Za-z0-9._-]+$"),
]
Password = Annotated[str, StringConstraints(min_length=1, max_length=256)]
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


class In(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Out(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ── auth / users ──────────────────────────────────────────────────────────────
class LoginIn(In):
    username: Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True, min_length=1, max_length=64)]
    password: Password


class UserOut(Out):
    id: int
    username: str
    role: Role
    is_active: bool
    created_at: datetime


class LoginOut(BaseModel):
    user: UserOut
    csrf_token: str
    expires_at: datetime


class PasswordChangeIn(In):
    current_password: Password
    new_password: Password


class UserCreateIn(In):
    username: Username
    role: Role
    password: Password


class UserPatchIn(In):
    role: Role | None = None
    is_active: bool | None = None


class PasswordResetIn(In):
    new_password: Password


# ── cameras ───────────────────────────────────────────────────────────────────
class CameraIn(In):
    name: Name
    source_url: SecretStr  # rtsp:// or rtsps:// (never echoed back); RTSP only
    default_mode: Mode = "Medium"
    is_enabled: bool = True
    scene_type: SceneType = "pool"


class CameraPatchIn(In):
    name: Name | None = None
    source_url: SecretStr | None = None
    default_mode: Mode | None = None
    is_enabled: bool | None = None
    scene_type: SceneType | None = None


class CameraOut(Out):
    id: str
    name: str
    source_type: Literal["rtsp", "file"]  # "file": a legacy demo row, refused at start until given an RTSP URL
    source: str = Field(validation_alias="source_display")  # masked: rtsp://***@host/path
    default_mode: Mode
    is_enabled: bool
    scene_type: SceneType
    scene_type_set_by: Literal["default", "admin"]
    created_at: datetime
    updated_at: datetime


# ── zones ─────────────────────────────────────────────────────────────────────
class ZoneIn(In):
    name: Name
    zone_type: ZoneType = "WATER"
    polygon: list[tuple[float, float]] = Field(min_length=3, max_length=512)  # 512: payload guard, not a UI limit
    sensitivity: float = Field(1.0, ge=0.25, le=3.0)
    schedule_id: int | None = None
    is_active: bool = True

    @field_validator("polygon")
    @classmethod
    def _polygon(cls, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for x, y in pts:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("polygon points must be normalised to [0, 1]")
        area2 = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1], strict=True))
        if abs(area2) / 2 < 1e-4:
            raise ValueError("polygon is degenerate (zero area)")
        return pts


class PoolZoneIn(In):
    """[Phase 11 WP2] A zone drawn on the POOL PLAN in metres; every calibrated camera of `pool_id` projects it."""

    name: Name
    zone_type: ZoneType = "WATER"
    pool_id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]{1,64}$")] = "pool"
    pool_polygon_m: list[tuple[float, float]] = Field(min_length=3, max_length=64)
    sensitivity: float = Field(1.0, ge=0.25, le=3.0)
    schedule_id: int | None = None
    is_active: bool = True

    @field_validator("pool_polygon_m")
    @classmethod
    def _polygon_m(cls, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for x, y in pts:
            if not (-10.0 <= x <= 200.0 and -10.0 <= y <= 200.0):
                raise ValueError("pool coordinates must be metres on the pool plan (-10..200)")
        area2 = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1], strict=True))
        if abs(area2) / 2 < 0.25:
            raise ValueError("pool zone is degenerate (area under 0.25 m2)")
        return pts


class ZoneOut(Out):
    id: int
    camera_id: str | None  # None: a pool zone
    name: str
    zone_type: ZoneType
    polygon: list[list[float]] | None  # camera zone, normalised image coordinates
    pool_id: str | None = None  # [Phase 11] pool zone
    pool_polygon_m: list[list[float]] | None = None  # [Phase 11] pool zone, metres on the pool plan
    sensitivity: float
    schedule_id: int | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


# ── schedules ─────────────────────────────────────────────────────────────────
class ScheduleIn(In):
    name: Name
    days: list[Weekday] = Field(min_length=1, max_length=7)
    start_time: str
    end_time: str
    monitoring_mode: Mode = "Medium"
    status: Literal["Active", "Disabled"] = "Active"

    @field_validator("days")
    @classmethod
    def _unique_days(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("days must be unique")
        return sorted(v, key=WEEKDAYS.index)

    @field_validator("start_time", "end_time")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        parse_hhmm(v)
        return v

    @model_validator(mode="after")
    def _differ(self) -> ScheduleIn:
        if self.start_time == self.end_time:
            raise ValueError("start_time and end_time must differ (an overnight window has end < start)")
        return self


class ScheduleOut(Out):
    id: int
    name: str
    days: list[Weekday]
    start_time: str
    end_time: str
    monitoring_mode: Mode
    status: Literal["Active", "Disabled"]
    created_at: datetime
    updated_at: datetime


# ── alerts ────────────────────────────────────────────────────────────────────
class AlertOut(Out):
    id: int
    created_at: datetime
    camera_id: str | None
    track_id: int | None
    scenario: str
    priority: str
    risk_score: int
    confidence: float
    zone_name: str | None
    reason: str
    status: Literal["pending", "acknowledged", "resolved", "false_alarm"]
    acknowledged_by: int | None
    acknowledged_at: datetime | None
    closed_by: int | None
    closed_at: datetime | None
    close_note: str | None
    audible: bool
    fault_code: str | None
    fault_source: str | None
    cleared_at: datetime | None
    escalation_level: int
    escalated_at: datetime | None
    global_id: int | None = None  # [Phase 11, removed in Phase 5] always null for new alerts
    incident_id: int | None = None  # [Phase 11, removed in Phase 5] always null for new alerts
    event_id: int | None = None  # [Phase 5] the person's event on this camera (alerts/events.py)
    snapshot_available: bool = False
    recording_available: bool = False
    title: str | None = None  # [Stage 4] "Drowning Alert" (None for a FAULT: its fault code names it)
    alert_reason: str | None = None  # [Stage 4] "Submersion / sudden blackout" | "Active drowning" | "Drowning"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def severity(self) -> str | None:
        """[Phase 6] The alert's tier (alerts/events.severity_of; None for a FAULT)."""
        from lifeguard.alerts.events import severity_of

        return None if self.scenario == "FAULT" else severity_of(self.scenario, self.risk_score)

    @model_validator(mode="before")
    @classmethod
    def _media_flags(cls, data: Any) -> Any:
        # file NAMES are never exposed - only whether the authenticated download exists
        if hasattr(data, "snapshot_path"):
            data = {
                **{k: getattr(data, k) for k in cls.model_fields if hasattr(data, k)},
                "snapshot_available": data.snapshot_path is not None,
                "recording_available": data.recording_path is not None,
            }
        return _one_alert(data)


class AlertCloseIn(In):
    note: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


def _one_alert(data: Any, fields: Any = ()) -> Any:
    """[Stage 4] A stored scenario (a reason code, or an old type name: accepted, never rewritten) is reported as
    scenario DROWNING + title "Drowning Alert" + its reason text; an old row's reason loses its type label."""
    from lifeguard.vision.scene import alert_reason as AR

    if not isinstance(data, dict) and hasattr(data, "scenario"):  # an ORM row
        data = {k: getattr(data, k) for k in fields if hasattr(data, k)}
    if not isinstance(data, dict) or "scenario" not in data:
        return data
    stored = data["scenario"]
    out = {
        **data,
        "scenario": AR.api_scenario(stored),
        "title": AR.title_of(stored),
        "alert_reason": AR.reason_text(stored),
    }
    if "reason" in data:
        out["reason"] = AR.clean_reason(data["reason"], stored)
    return out


class EventOut(Out):
    """[Phase 5] One person's incident on one camera (alerts/events.py)."""

    @model_validator(mode="before")
    @classmethod
    def _one_alert(cls, data: Any) -> Any:
        return _one_alert(data, cls.model_fields)

    id: int
    camera_id: str | None
    track_id: int | None
    opened_at: datetime
    updated_at: datetime
    closed_at: datetime | None
    severity: Literal["critical", "high", "caution"]
    scenario: str
    status: Literal["open", "acknowledged", "resolved"]
    alert_count: int
    title: str | None = None  # [Stage 4] "Drowning Alert"
    alert_reason: str | None = None  # [Stage 4] the reason of the event's alerts
    first_alert_id: int | None
    last_alert_id: int | None
    acknowledged_by: int | None
    acknowledged_at: datetime | None
    resolved_by: int | None
    resolved_at: datetime | None


# ── audit ───────────────────────────────────────────────────────────
class AuditOut(Out):
    id: int
    at: datetime
    actor_user_id: int | None
    actor_username: str
    ip: str
    action: str
    target_type: str
    target_id: str
    outcome: str
    details: dict[str, object]


def safe_filename(name: str) -> str:
    """Basename only, no control characters, max 255 chars."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not base or _CTRL.search(base) or len(base) > 255 or base in (".", ".."):
        raise ValueError("invalid file name")
    return base


# ── pool calibration (Phase 7) ───────────────────────────────────────────────
Point2 = tuple[float, float]


class CalibrationIn(In):
    """>= 4 points clicked on the camera image (normalised 0-1) and the same points on the pool plan in
    metres, same order (corners A=(0,0), B=(L,0), C=(L,W), D=(0,W)). The frame size is the size of the
    image the points were clicked on; the homography is scale-free, so any size of the same view works."""

    image_points: list[Point2] = Field(min_length=4, max_length=16)
    pool_points: list[Point2] = Field(min_length=4, max_length=16)
    outline_m: list[Point2] | None = Field(None, min_length=3, max_length=64)
    edge_band_m: float | None = Field(None, ge=0.0, le=5.0)
    frame_width: int = Field(1920, ge=16, le=8192)
    frame_height: int = Field(1080, ge=16, le=8192)

    @field_validator("image_points")
    @classmethod
    def _image(cls, pts: list[Point2]) -> list[Point2]:
        for x, y in pts:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("image points must be normalised to [0, 1]")
        return pts

    @field_validator("pool_points", "outline_m")
    @classmethod
    def _metres(cls, pts: list[Point2] | None) -> list[Point2] | None:
        for x, y in pts or []:
            if not (-10.0 <= x <= 200.0 and -10.0 <= y <= 200.0):
                raise ValueError("pool coordinates must be metres on the pool plan (-10..200)")
        return pts

    @model_validator(mode="after")
    def _same_count(self) -> CalibrationIn:
        if len(self.image_points) != len(self.pool_points):
            raise ValueError("image_points and pool_points must have the same number of points")
        return self


class PoolZonePreviewIn(In):
    """[Phase 11 WP2] An unsaved pool polygon to preview in every calibrated camera of `pool_id`."""

    pool_id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]{1,64}$")] = "pool"
    pool_polygon_m: list[tuple[float, float]] = Field(min_length=3, max_length=64)

    @field_validator("pool_polygon_m")
    @classmethod
    def _polygon_m(cls, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return PoolZoneIn._polygon_m(pts)


class PoolZoneProjection(Out):
    """[Phase 11 WP2] Where a pool zone lands in one calibrated camera's view (normalised; None = not in view)."""

    camera_id: str
    camera_name: str
    polygon: list[list[float]] | None


class CalibrationOut(Out):
    camera_id: str
    calibration: dict[str, Any] | None
    points: int | None = None
    reprojection_error_m: float | None = None
    pool_size_m: list[float] | None = None
    water_outline_norm: list[list[float]] | None = None
