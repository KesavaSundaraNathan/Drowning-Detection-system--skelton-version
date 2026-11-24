"""[2.20.0] Per-camera polygon ROIs from a JSON file: zones for sources that have none in the database.

Live cameras already get zones drawn in the UI (db `zones` table -> `geometry.ZoneGeometry`). Uploads, the eval
harness and ad-hoc RTSP sources (MediaMTX test feeds) have no database zones, so until now the whole frame was
water. This module reads the same zone semantics from a file and builds the same `ZoneGeometry`, so the scene
tracker needs no new code path:

    cfg = RoiConfig.load(Path("config/roi_zones.json"))      # validated; raises RoiConfigError on bad input
    geometry = cfg.geometry_for("Test Feed", frame_w, frame_h)   # ZoneGeometry or None (no entry)
    masker = cfg.masker_for("Test Feed")                          # PixelMasker or None

Zone types (same as the database):
  WATER    monitored at the camera mode x `sensitivity`;
  SHALLOW  monitored, typically with `sensitivity` < 1 (entry steps, a crowded shallow end);
  DECK     a person's foot point here = not in the water, not risk-scored (still tracked and drawn).
  MASK     pixels blacked out BEFORE detection: only for areas that can never hold a person in the water -
           burned-in watermarks and timestamps, a TV screen, sky. Refused if it overlaps a WATER/SHALLOW zone.

Safety: there is deliberately no "ignore this part of the water" type. A crowded area of the pool is where a
victim is hardest to see, not where one cannot drown: give it SHALLOW with a sensitivity multiplier instead, so
it is still monitored. Pixel masking is limited to MASK zones for the same reason - a blacked-out area hides a
victim from every later stage.

File format (`aquaguard-roi/1`), coordinates normalised to [0, 1] of the frame:

    {"format": "aquaguard-roi/1",
     "cameras": {
       "Test Feed": {"zones": [
         {"name": "Deep end", "type": "WATER", "polygon": [[0.1, 0.4], [0.9, 0.4], [0.9, 1.0], [0.1, 1.0]]},
         {"name": "Steps", "type": "SHALLOW", "sensitivity": 0.8, "polygon": [[0.1, 0.4], [0.25, 0.4], ...]},
         {"name": "Watermark", "type": "MASK", "polygon": [[0.75, 0.9], [1.0, 0.9], [1.0, 1.0], [0.75, 1.0]]}
       ]}}}

A camera key is matched exactly first, then as an `fnmatch` pattern (e.g. "IMG_54*" for a set of uploads).
"""

from __future__ import annotations

import fnmatch
import functools
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lifeguard.vision.scene.geometry import ZoneGeometry, ZonePolygon, normalized_to_contour

logger = logging.getLogger(__name__)

FORMAT = "aquaguard-roi/1"
ZONE_TYPES = ("WATER", "SHALLOW", "DECK", "MASK")
_MONITORED = ("WATER", "SHALLOW")


class RoiConfigError(ValueError):
    """The ROI file is malformed or unsafe (message names the camera and zone)."""


@dataclass(frozen=True)
class RoiZone:
    """One polygon of one camera, normalised coordinates."""

    name: str
    zone_type: str
    polygon: tuple[tuple[float, float], ...]
    sensitivity: float = 1.0
    priority: str = "Medium"

    def contour(self, frame_w: int, frame_h: int) -> np.ndarray:
        return normalized_to_contour(self.polygon, frame_w, frame_h)


@dataclass(frozen=True)
class CameraRoi:
    zones: tuple[RoiZone, ...]

    @property
    def scene_zones(self) -> tuple[RoiZone, ...]:
        return tuple(z for z in self.zones if z.zone_type != "MASK")

    @property
    def mask_zones(self) -> tuple[RoiZone, ...]:
        return tuple(z for z in self.zones if z.zone_type == "MASK")


@dataclass(frozen=True)
class RoiConfig:
    cameras: dict[str, CameraRoi] = field(default_factory=dict)

    # ── loading ────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: Path) -> RoiConfig:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RoiConfigError(f"{path}: {exc}") from exc
        cfg = cls.from_dict(doc)
        logger.info("ROI config %s: %d camera entries", path, len(cfg.cameras))
        return cfg

    @classmethod
    def from_dict(cls, doc: Any) -> RoiConfig:
        if not isinstance(doc, dict) or doc.get("format") != FORMAT:
            raise RoiConfigError(f"expected an object with format {FORMAT!r}")
        cams = doc.get("cameras")
        if not isinstance(cams, dict):
            raise RoiConfigError("'cameras' must be an object keyed by camera id or pattern")
        return cls({str(k): _parse_camera(str(k), v) for k, v in cams.items()})

    # ── lookup ─────────────────────────────────────────────────────────────
    def entry_for(self, camera_id: str) -> CameraRoi | None:
        if camera_id in self.cameras:
            return self.cameras[camera_id]
        for pattern, roi in self.cameras.items():
            if fnmatch.fnmatchcase(camera_id, pattern):
                return roi
        return None

    def geometry_for(self, camera_id: str, frame_w: int, frame_h: int) -> ZoneGeometry | None:
        """The camera's scene zones as a ZoneGeometry (None: no entry, or MASK zones only = whole frame water)."""
        roi = self.entry_for(camera_id)
        if roi is None or not roi.scene_zones:
            return None
        zones = [
            ZonePolygon(
                zone_id=i + 1,
                name=z.name,
                zone_type=z.zone_type,
                contour=z.contour(frame_w, frame_h),
                sensitivity_multiplier=z.sensitivity,
                schedule_id=None,
                priority=z.priority,
            )
            for i, z in enumerate(roi.scene_zones)
        ]
        return ZoneGeometry(camera_id, frame_w, frame_h, zones)

    def masker_for(self, camera_id: str) -> PixelMasker | None:
        roi = self.entry_for(camera_id)
        if roi is None or not roi.mask_zones:
            return None
        return PixelMasker(roi.mask_zones)


class PixelMasker:
    """Blacks out MASK polygons in a frame (in place). The mask is built once per frame size."""

    def __init__(self, zones: Sequence[RoiZone]) -> None:
        self.zones = tuple(zones)
        self._mask: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._shape != (h, w):
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [z.contour(w, h) for z in self.zones], 255)
            self._mask, self._shape = mask.astype(bool), (h, w)
        if self._mask is not None:
            frame[self._mask] = 0
        return frame


# ── parsing / validation ───────────────────────────────────────────────────
def _parse_camera(cam: str, raw: Any) -> CameraRoi:
    if not isinstance(raw, dict) or not isinstance(raw.get("zones"), list):
        raise RoiConfigError(f"camera {cam!r}: expected {{'zones': [...]}}")
    zones = tuple(_parse_zone(cam, k, z) for k, z in enumerate(raw["zones"]))
    _refuse_masked_water(cam, zones)
    return CameraRoi(zones)


def _parse_zone(cam: str, k: int, raw: Any) -> RoiZone:
    where = f"camera {cam!r} zone {k}"
    if not isinstance(raw, dict):
        raise RoiConfigError(f"{where}: expected an object")
    ztype = str(raw.get("type", "")).upper()
    if ztype not in ZONE_TYPES:
        raise RoiConfigError(f"{where}: type must be one of {ZONE_TYPES}, got {raw.get('type')!r}")
    poly = raw.get("polygon")
    if not isinstance(poly, list) or len(poly) < 3:
        raise RoiConfigError(f"{where}: polygon needs at least 3 [x, y] points")
    pts: list[tuple[float, float]] = []
    for p in poly:
        if not (isinstance(p, list | tuple) and len(p) == 2):
            raise RoiConfigError(f"{where}: bad point {p!r}")
        x, y = float(p[0]), float(p[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise RoiConfigError(f"{where}: point {p!r} is outside [0, 1] (coordinates are normalised)")
        pts.append((x, y))
    sens = float(raw.get("sensitivity", 1.0))
    if not 0.1 <= sens <= 3.0:
        raise RoiConfigError(f"{where}: sensitivity {sens} outside [0.1, 3.0]")
    return RoiZone(
        str(raw.get("name") or f"{ztype.title()} {k + 1}"), ztype, tuple(pts), sens, str(raw.get("priority", "Medium"))
    )


def _refuse_masked_water(cam: str, zones: Sequence[RoiZone]) -> None:
    """A MASK may not cover monitored water (it would hide a victim from detection)."""
    masks = [z for z in zones if z.zone_type == "MASK"]
    water = [z for z in zones if z.zone_type in _MONITORED]
    if not masks or not water:
        return
    size = 512
    wm = np.zeros((size, size), np.uint8)
    cv2.fillPoly(wm, [z.contour(size, size) for z in water], 1)
    for m in masks:
        mm = np.zeros((size, size), np.uint8)
        cv2.fillPoly(mm, [m.contour(size, size)], 1)
        overlap = int(np.count_nonzero(wm & mm))
        if overlap > 0.01 * max(1, int(np.count_nonzero(mm))):
            raise RoiConfigError(
                f"camera {cam!r}: MASK zone {m.name!r} overlaps monitored water - masked pixels hide a victim from "
                "detection; use SHALLOW with a lower sensitivity for a busy area of the pool instead"
            )


ENV_VAR = "LIFEGUARD_ROI_FILE"


@functools.lru_cache(maxsize=4)
def _load_cached(path: str) -> RoiConfig:
    return RoiConfig.load(Path(path))


def from_env() -> RoiConfig | None:
    """The ROI file named by LIFEGUARD_ROI_FILE (parsed once per path), or None when the variable is unset.

    A bad file raises RoiConfigError: a zone file that silently failed to load would leave the operator believing
    the zones apply."""
    raw = os.environ.get(ENV_VAR, "").strip()
    return _load_cached(raw) if raw else None
