# mypy: disable-error-code="operator"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""
[v3] Ported verbatim from the v2 baseline detection/zone_state.py (imports only).

detection/zone_state.py — per-person zone state: IN_WATER / EDGE / DECK.

Client requirement: every person (15–30 at once) is classified as inside the
pool, on the pool edge, or outside on the deck; only in-water people are
scored at full drowning sensitivity.

Three signals are fused per frame, then smoothed per track by the scene
tracker's existing time hysteresis (`SceneBehaviorTracker._update_category`):

 1. GEOMETRY (primary; cameras are fixed and calibrated at install)
      * calibrated camera: the box's bottom-centre projected to pool metres,
        signed distance to the pool outline (+ inside). The outline is split
        into an INNER water area, an EDGE ring of ±ZONE_EDGE_BAND_M around the
        coping, and the DECK beyond. The band is per camera
        (calibration "edge_band_m", set in PoolCalibrationPanel).
      * zones drawn but no calibration: the same split in pixels around the
        WATER/SHALLOW zone boundary, band = ZONE_EDGE_BAND_BOX × box height.
    For a swimmer the bottom of the box is where the body meets the water
    (the legs are underwater), which lies on the water plane the homography
    describes; for someone standing on the deck it is their feet.
 2. DETECTOR VOTE — the zone-aware detector's P(in water | person)
    (classes person_in_water / person_out_of_water, TrackedPerson.water_prob),
    smoothed per track (EMA). It resolves exactly the cases geometry cannot:
    a person sitting on the coping with legs in the water vs a swimmer
    holding the wall vs someone standing next to the edge.
 3. No geometry at all → vote alone; with neither, None (the caller keeps
    the legacy colour/pose land-evidence path).

Safety bias: when the signals are missing or disagree the decision leans to
the MONITORED side (IN_WATER / EDGE), never silently to DECK — a false
"deck" hides a drowning, a false "water" costs at most a false alarm that
behaviour cue agreement still has to earn.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from lifeguard.vision.constants import (
    ZONE_EDGE_BAND_BOX,
    ZONE_EDGE_BAND_M,
    ZONE_VOTE_DECK,
    ZONE_VOTE_EMA_SECONDS,
    ZONE_VOTE_STRONG,
    ZONE_VOTE_WATER,
)

logger = logging.getLogger(__name__)

IN_WATER = "IN_WATER"
EDGE = "EDGE"
DECK = "DECK"
ZONE_STATES = (IN_WATER, EDGE, DECK)

# geometry regions
_INNER, _EDGE_IN, _EDGE_OUT, _OUTSIDE = "inner", "edge_in", "edge_out", "outside"


@dataclass
class ZoneObservation:
    state: str  # IN_WATER / EDGE / DECK
    region: str | None  # geometry region used (None = no geometry)
    signed_distance: float | None  # metres (calibrated) or px (zones), + = inside the water
    vote: float | None  # smoothed detector in-water vote used


class WaterVote:
    """Per-track exponential moving average of the detector's in-water vote (time-constant in seconds)."""

    __slots__ = ("value", "t")

    def __init__(self) -> None:
        self.value: float | None = None
        self.t: float | None = None

    def update(self, p: float | None, t: float) -> float | None:
        if p is None or not math.isfinite(p):
            return self.value
        if self.value is None or self.t is None:
            self.value, self.t = float(p), t
            return self.value
        dt = max(t - self.t, 0.0)
        a = 1.0 - math.exp(-dt / max(ZONE_VOTE_EMA_SECONDS, 1e-3))
        self.value += a * (float(p) - self.value)
        self.t = t
        return self.value


def edge_band_m(calibration: Any) -> float:
    raw = getattr(calibration, "raw", None) or {}
    try:
        v = float(raw.get("edge_band_m", ZONE_EDGE_BAND_M))
        return v if 0.0 < v <= 3.0 else ZONE_EDGE_BAND_M
    except (TypeError, ValueError):
        return ZONE_EDGE_BAND_M


def _region(d: float, band: float) -> str:
    if d > band:
        return _INNER
    if d >= 0.0:
        return _EDGE_IN
    if d >= -band:
        return _EDGE_OUT
    return _OUTSIDE


def geometry_region(bbox: tuple[float, float, float, float], geometry: Any) -> tuple[str | None, float | None]:
    """(region, signed distance) of a person box from the camera's geometry, or (None, None)."""
    if geometry is None:
        return None, None
    x1, y1, x2, y2 = bbox
    point = ((x1 + x2) / 2.0, float(y2))
    cal = getattr(geometry, "calibration", None)
    # a drawn water polygon is the water for this camera; the calibration then only serves pool-plan features
    if cal is not None and not getattr(geometry, "drawn_water", False):
        try:
            d = cal.pool_signed_distance(cal.to_pool(point))
            if math.isfinite(d):
                return _region(d, edge_band_m(cal)), d
        except Exception:
            logger.warning("pool calibration mapping failed - using zones", exc_info=True)
    if getattr(geometry, "has_zones", False):
        d = geometry.edge_distance(point)
        if d is None or not math.isfinite(d):
            return (_OUTSIDE, d) if d == -math.inf else (None, None)
        band = ZONE_EDGE_BAND_BOX * max(y2 - y1, 1.0)
        return _region(d, band), d
    return None, None


def fuse(region: str | None, vote: float | None) -> str | None:
    """
    Decision table (vote = smoothed P(in water); None = detector has no opinion):

      region \\ vote   None        < WATER      >= WATER     >= STRONG
      inner            IN_WATER    IN_WATER     IN_WATER     IN_WATER
      edge (inside)    IN_WATER    EDGE         IN_WATER     IN_WATER
      edge (outside)   DECK        DECK         EDGE         EDGE
      outside          DECK        DECK         DECK         EDGE
      no geometry      None        DECK*        IN_WATER     IN_WATER

    `inner` ignores a dry vote: standing in shallow water shows the legs, and
    the calibrated outline is the ground truth there. `outside` + a strong wet
    vote is someone falling / jumping in, or a near-wall swimmer whose box
    bottom projects past the coping — kept monitored as EDGE.

    * [v7] With no geometry the vote is the only evidence, so DECK needs it below
    ZONE_VOTE_DECK; a vote between ZONE_VOTE_DECK and ZONE_VOTE_WATER is no opinion
    (None: the caller's colour / pose path, monitored until proven on land). An
    out-of-domain upload (open water) gave a backstroke swimmer votes of 0.19-0.5.
    """
    has_vote = vote is not None and math.isfinite(vote)
    wet = has_vote and vote >= ZONE_VOTE_WATER
    strong = has_vote and vote >= ZONE_VOTE_STRONG
    if region is None:
        if not has_vote:
            return None
        if wet:
            return IN_WATER
        return DECK if vote < ZONE_VOTE_DECK else None
    if region == _INNER:
        return IN_WATER
    if region == _EDGE_IN:
        return IN_WATER if (not has_vote or wet) else EDGE
    if region == _EDGE_OUT:
        return EDGE if wet else DECK
    return EDGE if strong else DECK


def observe(bbox: tuple[float, float, float, float], geometry: Any, vote: float | None) -> ZoneObservation | None:
    region, d = geometry_region(bbox, geometry)
    state = fuse(region, vote)
    if state is None:
        return None
    return ZoneObservation(state, region, d, vote)


def lower_mode(mode: str) -> str:
    """EDGE people are monitored one sensitivity step lower (High→Medium→Low)."""
    return {"High": "Medium", "Medium": "Low"}.get(mode, mode)
