"""Floatable objects and person <-> floatable geometry (Scenario C association).

[v3] From the v2 baseline detection/floatables.py: the geometry helpers and FloatableObject
verbatim. The FloatableDetector is DROPPED (plan: it had no model); the association logic
stays because the scene tracker and the synthetic behaviour suite feed floatables to it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-Union of two xyxy boxes (0.0 when they do not overlap)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def best_iou(bbox: Sequence[float], objects: Sequence[FloatableObject]) -> float:
    """Highest IoU between a person box and any floatable box."""
    return max((bbox_iou(bbox, o.bbox) for o in objects), default=0.0)


def centroid_distance(point: tuple[float, float], obj: FloatableObject) -> float:
    """Euclidean distance from a point to a floatable's bbox centroid."""
    x1, y1, x2, y2 = obj.bbox
    return math.hypot(point[0] - (x1 + x2) / 2.0, point[1] - (y1 + y2) / 2.0)


def nearest_centroid_distance(point: tuple[float, float], objects: Sequence[FloatableObject]) -> float:
    """Distance to the NEAREST floatable centroid; inf when there are none."""
    return min((centroid_distance(point, o) for o in objects), default=math.inf)


@dataclass
class FloatableObject:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_name: str

    def to_dict(self) -> dict:
        return {"bbox": list(self.bbox), "confidence": round(self.confidence, 3), "class": self.class_name}


def distance_to_floatable(point: tuple[float, float], objects: Sequence[FloatableObject]) -> float:
    """Pixel distance from a point to the nearest floatable bbox (0 inside); inf if none."""
    best = math.inf
    px, py = point
    for o in objects:
        x1, y1, x2, y2 = o.bbox
        dx = max(x1 - px, 0.0, px - x2)
        dy = max(y1 - py, 0.0, py - y2)
        best = min(best, math.hypot(dx, dy))
    return best
