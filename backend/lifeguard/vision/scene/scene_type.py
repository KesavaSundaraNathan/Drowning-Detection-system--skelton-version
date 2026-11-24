"""[v4] Pool vs standard CCTV: an automatic SUGGESTION of what a camera looks at.

A camera's ``scene_type`` (db Camera.scene_type) decides whether it runs drowning detection ("pool") or only
person detection + tracking ("standard_cctv": no pose, no behaviour scoring, no drowning alert can fire). This
module looks at the stream and suggests which one fits; it NEVER changes the setting itself - turning drowning
detection off is always an admin decision, confirmed on the Cameras page.

Evidence, in order:

  * water zones / a pool calibration drawn for the camera -> "pool" (an admin already said where the water is);
  * the learned scene model (scene_classifier.py: zero-shot CLIP ViT-B/32, MODELS.json "clip_vit_b32_scene"), on
    a keyframe every KEYFRAME_EVERY_S for the first EVIDENCE_S, then every RECHECK_S: any keyframe with a pool
    probability >= MODEL_POOL_PROB -> "pool"; at least MODEL_MIN_KEYFRAMES keyframes, ALL below MODEL_CCTV_MAX ->
    "standard_cctv", unless a veto holds (below);
  * the colour water estimate (scene/water.py, v2's estimator) covering >= POOL_MIN_WATER_FRAC -> "pool";
  * otherwise None (undecided), with the reason.

Vetoes on "standard_cctv" (any one keeps the suggestion at "pool" or undecided): drawn water zones; colour water on
>= COLOUR_VETO_FRAC of the frame; a mostly greyscale / infrared image (no office measured under IR yet).

Measured 2026-10-15 (keyframes at t = 0, 5, ... 30 s, as above):
  * colour alone cannot separate the two: held-out pool clips (build/v2_baseline/Datasets) reached a water
    coverage of only 0.13-0.16 in three clips, the four site offices 0.09-0.25 (blue chairs, cool walls) - so
    colour only ever says "pool";
  * the scene model separates them widely: every keyframe of all 19 held-out pool clips (107 keyframes, including
    the fully infrared ones) scored a pool probability >= 0.985; the site offices (4 cameras x main + sub stream,
    56 keyframes) scored <= 0.004, except "rvv main room" (seen as a corridor) at <= 0.201. MODEL_CCTV_MAX = 0.3
    keeps a 3x margin below the lowest pool keyframe while still recognising every measured office.
  Limit of that evidence: the held-out clips are mostly close / handheld pool footage, not a wide elevated CCTV
  view of an empty pool - one more reason the server only suggests. Changing the prompts (tools/export_scene_clip.py)
  or these thresholds needs this measurement again.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

import cv2
import numpy as np

POOL = "pool"
STANDARD_CCTV = "standard_cctv"
SCENE_TYPES = (POOL, STANDARD_CCTV)

EVIDENCE_S = 30.0  # stream seconds before an undecided camera says "no pool recognised"
POOL_MIN_WATER_FRAC = 0.15  # colour: share of the frame the confident water estimate must cover for "pool"
COLOUR_VETO_FRAC = 0.30  # colour: this much water blocks a CCTV suggestion (site offices reached 0.25)
GREY_SAT_MAX = 18.0  # median HSV saturation (0-255) below which a frame counts as greyscale / infrared
GREY_SHARE_MAX = 0.5  # more greyscale samples than this: no CCTV suggestion
SAMPLE_WIDTH = 96  # px: the colour check runs on a thumbnail
# learned scene model (measurement in the module docstring)
KEYFRAME_EVERY_S = 5.0
RECHECK_S = 600.0  # after the first EVIDENCE_S: lighting / camera changes
MODEL_POOL_PROB = 0.5
MODEL_CCTV_MAX = 0.3
MODEL_MIN_KEYFRAMES = 5
MODEL_KEEP = 12  # verdicts kept (the CCTV rule looks at all of them)


class SceneTypeEstimator:
    def __init__(self, classifier: Any = None) -> None:
        """`classifier`: a scene_classifier.SceneClassifierService (or None: colour-only evidence)."""
        self.classifier = classifier
        self.first_t: float | None = None
        self.last_sample_t: float | None = None
        self.samples = 0
        self.grey_samples = 0
        self.water_frac = 0.0  # largest confident water coverage seen
        self.configured = False  # water zones / calibration drawn
        self.verdicts: deque[Any] = deque(maxlen=MODEL_KEEP)  # scene_classifier.SceneVerdict, oldest first
        self.next_keyframe_t: float | None = None
        self._pending = threading.Event()  # a keyframe is with the classifier

    def reset(self) -> None:
        self.__init__(self.classifier)  # type: ignore[misc]

    def update(self, frame: np.ndarray, t: float, water: Any, geometry: Any) -> None:
        """Call once per frame after the water estimate updated (cheap: colour samples only when the water estimate
        updated, a keyframe for the scene model every KEYFRAME_EVERY_S, handed over without waiting)."""
        self.configured = has_configured_water(geometry)
        if self.first_t is None:
            self.first_t = t
        self._maybe_keyframe(frame, t)
        if getattr(water, "last_t", None) != t or self.last_sample_t == t:
            return
        self.last_sample_t = t
        self.samples += 1
        if is_greyscale(frame):
            self.grey_samples += 1
        if getattr(water, "confident", False) and getattr(water, "mask", None) is not None:
            self.water_frac = max(self.water_frac, float(water.mask.mean()))

    def _maybe_keyframe(self, frame: np.ndarray, t: float) -> None:
        if self.classifier is None or self._pending.is_set():
            return
        if self.next_keyframe_t is not None and t < self.next_keyframe_t:
            return
        first = self.first_t if self.first_t is not None else t
        self.next_keyframe_t = t + (KEYFRAME_EVERY_S if t - first < EVIDENCE_S else RECHECK_S)
        self._pending.set()
        if not self.classifier.submit(frame, self._on_verdict):
            self._pending.clear()  # model unavailable / queue full: try at the next keyframe

    def _on_verdict(self, verdict: Any) -> None:  # the classifier's thread
        self.verdicts.append(verdict)
        self._pending.clear()

    def observed_s(self) -> float:
        """Stream seconds from the first frame to the latest colour sample."""
        if self.first_t is None or self.last_sample_t is None:
            return 0.0
        return max(0.0, self.last_sample_t - self.first_t)

    def suggestion(self) -> tuple[str | None, str]:
        """(suggested scene type or None, reason)."""
        if self.configured:
            return POOL, "water zones / pool calibration are drawn for this camera"
        verdicts = list(self.verdicts)
        pool_max = max((v.pool_prob for v in verdicts), default=0.0)
        if pool_max >= MODEL_POOL_PROB:
            return POOL, f"scene model: pool (probability {pool_max:.2f})"
        if self.water_frac >= POOL_MIN_WATER_FRAC and (len(verdicts) < MODEL_MIN_KEYFRAMES or self.classifier is None):
            return POOL, f"looks like pool water on {self.water_frac:.0%} of the frame"
        grey = self.samples > 0 and self.grey_samples / self.samples > GREY_SHARE_MAX
        if len(verdicts) >= MODEL_MIN_KEYFRAMES and pool_max < MODEL_CCTV_MAX:
            top = max({v.top_class for v in verdicts}, key=[v.top_class for v in verdicts].count)
            if self.water_frac >= COLOUR_VETO_FRAC:
                return POOL, f"pool-coloured water on {self.water_frac:.0%} of the frame (scene model saw {top})"
            if grey:
                return None, f"scene model sees {top}, but the image is greyscale / infrared: decide yourself"
            return STANDARD_CCTV, f"scene model: {top}, no pool in {len(verdicts)} keyframes (max {pool_max:.2f})"
        if len(verdicts) >= MODEL_MIN_KEYFRAMES:
            return None, f"scene model unsure (pool probability up to {pool_max:.2f}): decide yourself"
        if grey:
            return None, "greyscale / infrared image: water colour cannot be judged"
        if self.observed_s() < EVIDENCE_S or (self.classifier is not None and self._model_usable()):
            return None, f"observing ({len(verdicts)} of {MODEL_MIN_KEYFRAMES} keyframes)"
        return None, (
            f"no pool water recognised in {self.observed_s():.0f} s - a colour check cannot rule a pool out;"
            " set standard CCTV yourself if this camera sees no water"
        )

    def _model_usable(self) -> bool:
        return getattr(self.classifier, "error", None) is None

    def as_dict(self) -> dict[str, Any]:
        kind, reason = self.suggestion()
        verdicts = list(self.verdicts)
        return {
            "suggested": kind,
            "reason": reason,
            "water_frac": round(self.water_frac, 3),
            "observed_s": round(self.observed_s(), 1),
            "keyframes": len(verdicts),
            "pool_prob_max": round(max((v.pool_prob for v in verdicts), default=0.0), 3) if verdicts else None,
        }


def has_configured_water(geometry: Any) -> bool:
    """True when an admin drew where the water is: a WATER zone, or a pool calibration."""
    if geometry is None:
        return False
    if getattr(geometry, "calibration", None) is not None:
        return True
    return any(getattr(z, "zone_type", None) == "WATER" for z in getattr(geometry, "zones", ()))


def is_greyscale(frame: np.ndarray) -> bool:
    """A (near-)colourless frame: infrared night mode, or a monochrome camera."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        return True
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (SAMPLE_WIDTH, max(1, round(h * SAMPLE_WIDTH / max(w, 1)))))
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]
    return float(np.median(sat)) < GREY_SAT_MAX
