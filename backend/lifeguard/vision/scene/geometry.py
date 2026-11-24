"""services/zone_service.py — Zone CRUD + fast geometry for the live pipeline.

PHASE 2
───────
Zones are polygons drawn over a REAL frame of a specific camera and stored as
normalised coordinates (x, y ∈ [0, 1]). This module provides:

  * `ZoneService`     — CRUD, camera snapshot passthrough, DB → geometry load
  * `ZonePolygon`     — one zone scaled to a concrete frame size (pixel ints)
  * `ZoneGeometry`    — `cv2.pointPolygonTest`-based classification of person
                        points (foot point / ankles) against WATER / DECK /
                        SHALLOW zones, with sensitivity & schedule resolution

The pipeline calls `ZoneService.load_geometry(camera_id, frame_shape)` every
few seconds and `geometry.classify(point)` per person per frame.

[v3] The geometry half of the v2 baseline services/zone_service.py (ZonePolygon, ZoneHit,
ZoneGeometry verbatim). Zone CRUD is the Phase 2 API; schedules use core/schedule.py.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from lifeguard.core.schedule import window_contains
from lifeguard.vision.constants import DEFAULT_ZONE_SENSITIVITY, MONITORED_ZONE_TYPES

logger = logging.getLogger(__name__)


Point = tuple[float, float]


# ═════════════════════════════════════════════════════════════════════════════
# Geometry
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class ZonePolygon:
    """A zone scaled to pixel coordinates of a specific frame size."""

    zone_id: int
    name: str
    zone_type: str  # WATER / DECK / SHALLOW
    contour: np.ndarray  # (N, 1, 2) int32 — cv2 contour layout
    sensitivity_multiplier: float
    schedule_id: str | None
    priority: str = "Medium"

    @property
    def monitored(self) -> bool:
        return self.zone_type in MONITORED_ZONE_TYPES

    def contains(self, point: Point) -> bool:
        # measureDist=False → +1 inside, 0 on edge, -1 outside (fast integer path)
        return cv2.pointPolygonTest(self.contour, (float(point[0]), float(point[1])), False) >= 0

    def signed_distance(self, point: Point) -> float:
        return float(cv2.pointPolygonTest(self.contour, (float(point[0]), float(point[1])), True))

    def to_overlay_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "zone_type": self.zone_type,
            "points": self.contour.reshape(-1, 2).tolist(),
        }


@dataclass
class ZoneHit:
    """Result of classifying one person point."""

    zone: ZonePolygon | None
    in_water: bool  # inside a WATER/SHALLOW zone (or no zones defined)
    monitored: bool  # should this person be risk-scored?
    zone_name: str
    zone_type: str  # WATER / DECK / SHALLOW / NONE / ALL
    sensitivity_multiplier: float
    schedule_id: str | None


def normalized_to_contour(points: Sequence[Sequence[float]], frame_w: int, frame_h: int) -> np.ndarray:
    """[[x_norm, y_norm], …] → int32 contour in pixel space."""
    pts = np.array(
        [[int(round(float(x) * frame_w)), int(round(float(y) * frame_h))] for x, y in points],
        dtype=np.int32,
    )
    return pts.reshape(-1, 1, 2)


def _coords_to_pairs(coords: Any) -> list[tuple[float, float]]:
    """Accept both [{x,y}] (legacy) and [[x,y]] shapes."""
    out: list[tuple[float, float]] = []
    for c in coords or []:
        if isinstance(c, dict):
            out.append((float(c["x"]), float(c["y"])))
        else:
            out.append((float(c[0]), float(c[1])))
    return out


@dataclass
class ScheduleSetting:
    """A zone's monitoring schedule (v3 shape: several weekdays per schedule)."""

    days: list[str]
    start_time: str  # "HH:MM" local
    end_time: str
    monitoring_mode: str = "Medium"
    status: str = "Active"


def schedule_is_now(sched: ScheduleSetting, now: datetime | None = None) -> bool:
    """True when `now` (local time; default: now) falls inside the schedule's window.

    [v3] Replaces v2's services/zone_service.schedule_is_now, which assigned the tail of an
    overnight window to the wrong weekday (fixed and tested in Phase 2: core/schedule.py).
    """
    at = now or datetime.now().astimezone()
    try:
        return window_contains(list(sched.days), sched.start_time, sched.end_time, at)
    except ValueError:
        return False


class ZoneGeometry:
    """
    All zones of one camera at one frame size, plus schedule lookups.

    Classification rules (Phase 2):
      * No zones defined for the camera → the whole frame is treated as WATER
        (backwards compatible: everybody is monitored at the default mode).
      * Point inside a WATER or SHALLOW zone → monitored; the zone's
        sensitivity multiplier and schedule apply. Smaller zones win ties so a
        SHALLOW island inside a big WATER zone is honoured.
      * Point inside a DECK zone, or outside every zone → NOT monitored
        (non-swimmer). The detector still tracks and draws them.
    """

    def __init__(
        self,
        camera_id: str,
        frame_w: int,
        frame_h: int,
        zones: Iterable[ZonePolygon],
        schedules: dict[str, ScheduleSetting] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.frame_w = frame_w
        self.frame_h = frame_h
        # smaller zones first so nested zones take precedence
        self.zones: list[ZonePolygon] = sorted(zones, key=lambda z: cv2.contourArea(z.contour))
        self.schedules: dict[str, ScheduleSetting] = schedules or {}
        # Multi-camera: pool calibration of this camera (detection/pool_geometry.PoolCalibration)
        self.calibration: Any = None
        # True when the camera has its own drawn WATER / SHALLOW polygon (any number of points, no metres): that
        # polygon, not the calibration outline, decides in-water vs on-land (vision/scene/zone_state.py).
        self.drawn_water: bool = False

    @property
    def has_zones(self) -> bool:
        return len(self.zones) > 0

    def classify(self, point: Point) -> ZoneHit:
        if not self.zones:
            return ZoneHit(None, True, True, "Whole Frame", "ALL", DEFAULT_ZONE_SENSITIVITY, None)
        for z in self.zones:
            if z.contains(point):
                return ZoneHit(
                    z, z.monitored, z.monitored, z.name, z.zone_type, z.sensitivity_multiplier, z.schedule_id
                )
        return ZoneHit(None, False, False, "Outside Zones", "NONE", DEFAULT_ZONE_SENSITIVITY, None)

    def classify_any(self, points: Iterable[Point]) -> ZoneHit:
        """
        Classify a set of candidate points (e.g. both ankles + bbox bottom).
        Returns the first monitored hit, else the last non-monitored hit.
        """
        last: ZoneHit | None = None
        for p in points:
            hit = self.classify(p)
            if hit.monitored:
                return hit
            last = hit
        return last or self.classify((0.0, 0.0))

    def edge_distance(self, point: Point) -> float:
        """
        Signed pixel distance from `point` to the water boundary (Phase 4):
        positive inside the water (distance to the nearest pool edge), negative
        outside (distance to the nearest water zone). With no zones the whole
        frame is water, so the frame border acts as the edge.
        """
        x, y = float(point[0]), float(point[1])
        if not self.zones:
            return float(min(x, y, self.frame_w - x, self.frame_h - y))
        water = [z for z in self.zones if z.monitored]
        if not water:
            return float("-inf")
        return max(z.signed_distance((x, y)) for z in water)

    def mode_for(self, hit: ZoneHit, default_mode: str, now: datetime | None = None) -> str:
        """
        Resolve the monitoring mode for a zone hit:
          schedule attached and currently inside its window → schedule's mode
          otherwise                                          → camera default
        Monitoring never switches off outside a window — the default mode applies.
        """
        if not hit.schedule_id:
            return default_mode
        sched = self.schedules.get(str(hit.schedule_id))
        if sched is None or (sched.status or "Active") != "Active":
            return default_mode
        if schedule_is_now(sched, now):
            return sched.monitoring_mode or default_mode
        return default_mode

    def monitoring_active(self, hit: ZoneHit, now: datetime | None = None) -> bool:
        """
        Phase 1.2 spec — is this zone inside an active ScheduleMonitoring window?

          * no schedule attached           → always active (24/7 monitoring)
          * schedule attached but Disabled → always active (the schedule is off,
                                             not the monitoring)
          * schedule attached and Active   → only inside its day + time window

        A False result does NOT stop detection. The pipeline keeps tracking,
        classifying and logging telemetry; it only suppresses the siren / live
        emergency escalation, so out-of-hours footage is still reviewable.
        """
        if not hit.schedule_id:
            return True
        sched = self.schedules.get(str(hit.schedule_id))
        if sched is None or (sched.status or "Active") != "Active":
            return True
        # [v3, Phase 4] schedule_is_now() answers False for a MALFORMED schedule, which muted the siren
        # (fail-unsafe). A schedule that cannot be evaluated never mutes.
        try:
            window_contains(list(sched.days), sched.start_time, sched.end_time, now or datetime.now().astimezone())
        except ValueError:
            logger.warning("zone %s: schedule %s is malformed - alerts stay audible", hit.zone_name, hit.schedule_id)
            return True
        return schedule_is_now(sched, now)

    def schedule_for(self, hit: ZoneHit) -> ScheduleSetting | None:
        """The ScheduleSetting attached to this zone hit, if any."""
        if not hit.schedule_id:
            return None
        return self.schedules.get(str(hit.schedule_id))

    def overlay(self) -> list[dict[str, Any]]:
        return [z.to_overlay_dict() for z in self.zones]
