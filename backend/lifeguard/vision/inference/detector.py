# mypy: disable-error-code="assignment, index, misc, var-annotated"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""detection/detector.py — Multi-person, multi-camera person detector + tracker.

OPEN-SOURCE STACK (RTMDet + RTMPose + ByteTrack)
────────────────────────────────────────────────
Every person in the frame is returned, each with a persistent `track_id`:

  1. RTMDet (OpenMMLab, served as ONNX through rtmlib + onnxruntime) finds
     every person box with its confidence;
  2. ByteTrack (in-repo, detection/bytetrack) assigns persistent ids to those boxes;
  3. RTMPose (top-down, ONNX) extracts 17 COCO keypoints for each tracked box;
  4. output is the same fully typed, JSON-serialisable `FrameDetections`:
        camera_id, timestamp, frame_index, frame_shape,
        tracks: [ {track_id, bbox[x1,y1,x2,y2], confidence, keypoints,
                   foot_point, tentative} ]

One `MultiPersonDetector` instance is created PER CAMERA, and each owns its own
ByteTrack instance, so track ids never mix between cameras. The ONNX sessions
themselves are shared process-wide (see detection/rtm_backend.py), so an extra
camera costs a tracker, not another copy of the weights.

[v3] See MultiPersonDetector: v2's stages verbatim; no HOG fallback, provider guard. Pose (step 3) runs
when POSE_ENABLED and a pose model is configured: top-down RTMPose-m COCO on every tracked box
(inference/pose.py), skipped on alternate frames only for stable tracks (POSE_STRIDE, see stage_pose).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from lifeguard.vision.constants import (
    DET_ENHANCE,
    DET_INPUT_SIZE,
    DET_LOW_SCORE_THRESHOLD,
    DET_MOSAIC,
    DET_MOSAIC_FAR_FRAC,
    DET_MOSAIC_GAIN,
    DET_MOSAIC_MAX_CALLS,
    DET_MOSAIC_NEAR_FRAC,
    DET_ROI_CROP,
    DET_TILING,
    DET_ZOOM,
    DET_ZOOM_CONTEXT,
    DET_ZOOM_MATCH_IOU,
    DET_ZOOM_MAX_BOX_FRAC,
    DET_ZOOM_MAX_CROP,
    DET_ZOOM_MAX_REGIONS,
    DET_ZOOM_MIN_CROP,
    ID_LINK_NESTED,
    KEYPOINT_CONF_THRESHOLD,
    NONPERSON_DECK_STATIC_SECONDS,
    POSE_ENABLED,
    POSE_INPUT_RGB,
    POSE_MAX_PER_FRAME,
    POSE_SMALL_BOX_PX,
    POSE_STABLE_MIN_FRAMES,
    POSE_STRIDE,
    SHADOW_IOA,
    SUBMERGED_FOLLOW,
    SUBMERGED_FOLLOW_BG_ALPHA,
    SUBMERGED_FOLLOW_CONFIRM_SECONDS,
    SUBMERGED_FOLLOW_DARK_DELTA,
    SUBMERGED_FOLLOW_ISOLATION,
    SUBMERGED_FOLLOW_MAX_SECONDS,
    SUBMERGED_FOLLOW_MIN_TRACK_SECONDS,
    SUBMERGED_FOLLOW_MISS_SECONDS,
    SUBMERGED_FOLLOW_SEARCH,
    TENTATIVE_DISPLAY_MIN_SCORE,
    TENTATIVE_HIDE_OUTSIDE_WATER,
    TRACK_COAST_SECONDS,
    TRACK_FAINT_BIRTHS,
    TRACK_REFIND_LOW_IOU,
    ZONE_GATE,
    ZONE_GATE_AFTER_TRACK,
    ZONE_GATE_GRACE_SECONDS,
)
from lifeguard.vision.frame_rules import frames_at
from lifeguard.vision.inference import decode as rtm_backend
from lifeguard.vision.inference import enhance, models
from lifeguard.vision.inference import pose as pose_stage
from lifeguard.vision.tracking.liveness import Liveness
from lifeguard.vision.tracking.motion import SceneMotion
from lifeguard.vision.tracking.non_person import NonPersonFilter

logger = logging.getLogger(__name__)


# [rtmpose clone] Standard CCTV cameras also run RTMPose, for the skeleton overlay only. The original skips pose there
# (detection + tracking, nothing scored). A CCTV camera is still never scored, and nothing that decides anything
# reads these keypoints: stage_pose's output reaches only the displayed people (see _detect_rtm). Set
# LIFEGUARD_CCTV_SKELETON=0 to get the original behaviour (no pose on standard CCTV, no GPU cost there).
CCTV_SKELETON: bool = os.environ.get("LIFEGUARD_CCTV_SKELETON", "1").strip().lower() not in ("0", "false", "no", "off")

BBox = tuple[int, int, int, int]
Point = tuple[float, float]
Keypoints = list[list[float]]  # 17 × [x, y, conf] in frame pixels
# COCO keypoint indices
KP_NOSE, KP_L_ANKLE, KP_R_ANKLE = 0, 15, 16
# A loaded backend that fails this many frames in a row is treated as dead:
# the environment broke under us (blocked DLL, GPU fell off the bus, …) and
# continuing would produce a run full of empty frames that reads as "safe".
MAX_CONSECUTIVE_FRAME_FAILURES = 30


# ═════════════════════════════════════════════════════════════════════════════
# Structured output
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class TrackedPerson:
    """A single person in a single frame with a tracker-assigned identity."""

    track_id: int  # -1 while ByteTrack has not confirmed the track
    bbox: BBox  # x1, y1, x2, y2 in ORIGINAL frame pixels
    confidence: float
    keypoints: Keypoints | None  # None when the backend has no pose head
    foot_point: Point  # ankle midpoint if available else bbox bottom-centre
    class_name: str = "person"
    # Detector's in-water vote in [0,1] (zone-aware v2 detector: P(in water |
    # person)); None for detectors without the in/out-of-water classes.
    water_prob: float | None = None
    # Tracking v3: this person's track was started by evidence (a persistent,
    # moving, body-coloured chain of FAINT boxes — tracking.FaintBirths), not
    # by a confident detection. Internal; not part of the wire format.
    faint_birth: bool = False
    # Pose was deliberately not run on this frame (stable track, POSE_STRIDE): keypoints None means "not
    # looked", never "no body found". Internal; not part of the wire format.
    pose_skipped: bool = False

    @property
    def tentative(self) -> bool:
        return self.track_id < 0

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def aspect_ratio(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(y2 - y1, 1) / max(x2 - x1, 1)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 4),
            "keypoints": self.keypoints,
            "foot_point": [round(self.foot_point[0], 1), round(self.foot_point[1], 1)],
            "tentative": self.tentative,
            "class": self.class_name,
        }
        if self.water_prob is not None:  # additive field; absent for pre-v2 detectors
            d["water_prob"] = round(self.water_prob, 3)
        return d

    def to_legacy_detection(self, frame_idx: int) -> dict[str, Any]:
        """Shape consumed by MotionTracker.analyze() (bbox / frame_idx / confidence)."""
        return {
            "bbox": list(self.bbox),
            "frame_idx": frame_idx,
            "confidence": self.confidence,
            "class": self.class_name,
            "track_id": self.track_id,
            "keypoints": self.keypoints,
        }


@dataclass
class FrameDetections:
    """All tracked people found in one frame of one camera."""

    camera_id: str
    timestamp: float  # UNIX seconds when the frame was processed
    frame_index: int
    frame_shape: tuple[int, int, int]  # (height, width, channels)
    tracks: list[TrackedPerson] = field(default_factory=list)
    inference_ms: float = 0.0
    backend: str = "rtm"  # "rtm" | "unavailable"
    # [2.35.0] TRACK_COAST_SECONDS: people whose track found no box this frame - {track_id, bbox (last observed),
    # unseen_s}. Display only (drawn dashed): never in `tracks`, never scored, never alert evidence.
    coasting: list[dict[str, Any]] = field(default_factory=list)

    @property
    def confirmed_tracks(self) -> list[TrackedPerson]:
        return [t for t in self.tracks if not t.tentative]

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "frame_shape": list(self.frame_shape),
            "tracks": [t.to_dict() for t in self.tracks],
            "inference_ms": round(self.inference_ms, 2),
            "backend": self.backend,
            "person_count": len(self.tracks),
            **({"coasting": self.coasting} if self.coasting else {}),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
def _cap_torso_hist(frame: np.ndarray, b: Any) -> np.ndarray | None:
    """[2.35.0] Hue-saturation histogram (16 x 8, normalised) of the top 40 % of a box: cap, head and torso. CPU, only
    called for a pair ID_LINK_NESTED is about to link."""
    h, w = frame.shape[:2]
    x1, y1 = int(max(0, b[0])), int(max(0, b[1]))
    x2 = int(min(w, b[2]))
    y2 = int(min(h, b[1] + 0.4 * (b[3] - b[1])))
    if x2 - x1 < 2 or y2 - y1 < 2 or frame.ndim != 3:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    s = float(hist.sum())
    return hist / s if s > 0 else None


def _clip_bbox(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> BBox:
    x1i = int(max(0, min(w - 1, round(x1))))
    y1i = int(max(0, min(h - 1, round(y1))))
    x2i = int(max(x1i + 1, min(w, round(x2))))
    y2i = int(max(y1i + 1, min(h, round(y2))))
    return (x1i, y1i, x2i, y2i)


def foot_point_from(bbox: BBox, keypoints: Keypoints | None) -> Point:
    """
    The point used for zone classification: mean of both ankles when they are
    confidently detected, else the bottom-centre of the bounding box.
    """
    if keypoints and len(keypoints) > KP_R_ANKLE:
        ankles = [keypoints[i] for i in (KP_L_ANKLE, KP_R_ANKLE) if keypoints[i][2] >= KEYPOINT_CONF_THRESHOLD]
        if ankles:
            return (
                float(sum(a[0] for a in ankles) / len(ankles)),
                float(sum(a[1] for a in ankles) / len(ankles)),
            )
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def _iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def fuse_zoom(
    boxes: np.ndarray, scores: np.ndarray, water: np.ndarray, zb: np.ndarray, zs: np.ndarray, zw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge zoom-pass boxes into the full-frame detections. A zoom box on a
    person the full frame already has (IoU >= DET_ZOOM_MATCH_IOU) only RAISES
    that box's score (and takes the finer zoomed geometry when it is the more
    confident one). A zoom box nested with a full-frame box (>= 70 % of the
    smaller inside the other) without matching it is dropped — a part of that
    person or a crop-sized artefact, never a second person. Anything else is a
    person the full frame missed and is appended. The 4th array is each box's FULL-FRAME score (0 for an
    appended zoom box): a CLAHE-enlarged crop finds faint "people" in deck
    clutter and on a ball, so zoom boxes may extend / re-find a track but never
    seed a faint birth.
    """
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4).copy()
    scores = np.asarray(scores, np.float32).reshape(-1).copy()
    water = np.asarray(water, np.float32).reshape(-1).copy()
    if len(water) != len(boxes):
        water = np.full(len(boxes), np.nan, np.float32)
    zb = np.asarray(zb, np.float32).reshape(-1, 4)
    zs = np.asarray(zs, np.float32).reshape(-1)
    base = scores.copy()  # full-frame scores (faint births use only these)
    add_b: list[np.ndarray] = []
    add_s: list[float] = []
    add_w: list[float] = []
    for k in np.argsort(-zs, kind="stable"):
        b = zb[k]
        j, best, nested = -1, -1.0, False
        for i in range(len(boxes)):
            a = boxes[i]
            ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = ix * iy
            if inter <= 0:
                continue
            aa, ab = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
            iou = inter / max(aa + ab - inter, 1e-9)
            if iou >= DET_ZOOM_MATCH_IOU and iou > best:
                j, best = i, iou
            elif inter / max(min(aa, ab), 1e-9) >= 0.7:
                nested = True
        if j >= 0:
            if zs[k] > scores[j]:
                scores[j], boxes[j] = zs[k], b
            continue
        if nested:
            continue  # overlaps a full-frame person without matching it
        if any(_iou(tuple(b), tuple(x)) >= DET_ZOOM_MATCH_IOU for x in add_b):
            continue  # two zoom crops saw the same person
        add_b.append(b.astype(np.float32))
        add_s.append(float(zs[k]))
        add_w.append(float(zw[k]) if len(zw) == len(zb) else np.nan)
    if add_b:
        boxes = np.concatenate([boxes, np.asarray(add_b, np.float32)])
        scores = np.concatenate([scores, np.asarray(add_s, np.float32)])
        water = np.concatenate([water, np.asarray(add_w, np.float32)])
        base = np.concatenate([base, np.zeros(len(add_b), np.float32)])
    return boxes, scores, water, base


_model_load_lock = threading.Lock()

BACKEND_UNLOADED = "unloaded"
BACKEND_RTM = "rtm"  # RTMDet on ONNX Runtime + ByteTrack (v2 wire name "yolo" dropped)
BACKEND_UNAVAILABLE = "unavailable"


class DetectorUnavailableError(RuntimeError):
    """No working detector: the camera is NOT MONITORED.

    Deliberately loud and REPEATED (v2 raised it once per instance, then returned empty frames that
    read as "nobody in the pool"): every call on a dead detector raises.
    """

    def __init__(self, camera_id: str, reason: str) -> None:
        self.camera_id = camera_id
        self.load_error = reason
        super().__init__(f"no working detector on '{camera_id}': {reason}")


class MultiPersonDetector:
    """
    Detects and tracks every person in a frame - one instance per camera (its own ByteTrack), ONNX
    sessions shared process-wide (inference/models.py).

    [v3] From the v2 baseline detection/detector.py. Kept verbatim: the detect / zoom / track /
    assemble stages. Removed: the HOG fallback (plan: no degraded mode - a missing or broken detector
    raises DetectorUnavailableError and the pipeline marks the camera NOT_MONITORED + FAULT), the rtmlib
    pose objects, the YOLO_*
    aliases and the "yolo" wire name. Every switch comes from the versioned thresholds file.

    Pose: with POSE_ENABLED and a `pose_model_path`, the pose model is loaded with the detector (a missing or
    broken pose model makes the detector unavailable - never a silent pose-less run) and every frame's tracked
    boxes get keypoints. `pose_model_path=None` is an explicit no-pose detector, shown in `status()`.
    """

    def __init__(
        self,
        camera_id: str,
        det_model_path: str,
        provider: str = "auto",
        conf: float = DET_LOW_SCORE_THRESHOLD,
        pose_model_path: str | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.pose_model_path = pose_model_path
        self.pose_enabled = bool(POSE_ENABLED) and bool(pose_model_path)
        self._pose_model: Any = None  # PoseModel (shared, inference/models.py)
        self.pose_infer: Any = None  # live/pose_batcher.py: crops of every camera in one call (None = direct)
        self.pose_stable: set[int] = set()  # track ids the behaviour stage calls stable (may skip pose)
        self._pose_frame = 0
        self._track_age: dict[int, int] = {}
        self._last_pose: dict[int, int] = {}
        self.pose_runs = 0
        self.pose_skips = 0
        self.det_model_path = det_model_path
        self.provider = provider
        self.conf = conf
        self.imgsz = DET_INPUT_SIZE
        self._det_model: Any = None  # DetModel (shared, inference/models.py)
        self._tracker: Any = None  # tracking.tracker.PersonTracker (per camera)
        self._det_names: list[str] | None = None
        # In-water votes aligned with the last stage_detect / stage_track output
        # (NaN = the detector has no in/out-of-water classes).
        self.last_det_water: np.ndarray = np.zeros(0, np.float32)
        self.last_track_water: np.ndarray = np.zeros(0, np.float32)
        self.fps: float = 25.0
        self.roi: tuple[int, int, int, int] | None = None
        self.enhance_modes = enhance.parse_modes(DET_ENHANCE)
        self.tiling = str(DET_TILING or "off").lower()
        self.roi_crop = bool(DET_ROI_CROP)
        self.motion = SceneMotion()
        self.zoom = bool(DET_ZOOM)
        self.faint_births = bool(TRACK_FAINT_BIRTHS)
        # [v4] standard CCTV camera (vision/scene/scene_type.py): people are detected and tracked, never posed
        self.surveillance = False
        self.water_test: Any | None = None
        self.follower: Any | None = None  # [2.40.0] SUBMERGED_FOLLOW (vision/tracking/submerged.py), built on first use
        self.outside_water_test: Any | None = None  # [2.33.0] set_outside_water_test
        self.zone_gate: Any | None = None  # [2.43.0] set_zone_gate (ZONE_GATE)
        self._gate_recent: dict[int, tuple[float, BBox]] = {}  # public id -> (last time inside the gate, last box)
        self.gate_dropped = 0  # boxes the zone gate removed (diagnostics)
        self.last_track_faint: np.ndarray = np.zeros(0, bool)
        self.last_birth_scores: np.ndarray = np.zeros(0, np.float32)
        self.zoom_calls = 0
        self.mosaic_calls = 0  # [v7 F4] extra detector calls of the resolution mosaic
        self.liveness = Liveness(camera_id)  # [v7 F3] static person-shaped objects outside the water
        self.non_person = NonPersonFilter(camera_id)  # [2.18.0] lens occluders and fixtures, anywhere
        self.nested_ioa = 0.0  # [2.18.0] > 0: drop boxes nested in another (finite clips: DET_NESTED_IOA_CLIP)
        self.birth_guard_iou: float | None = None  # [2.18.0] finite clips: TRACK_BIRTH_GUARD_IOU_CLIP
        self._backend = BACKEND_UNLOADED
        self._load_error: str | None = None
        self.load_exception: BaseException | None = None
        self._consecutive_failures = 0
        self._lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────────────
    @property
    def backend(self) -> str:
        return self._backend

    @property
    def healthy(self) -> bool:
        return self._backend == BACKEND_RTM

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def status(self) -> dict[str, Any]:
        return {
            "backend": self._backend,
            "healthy": self.healthy,
            "model": self.det_model_path,
            "provider": self.provider,
            "active_provider": models.active_provider(self._det_model) if self._det_model else None,
            "error": self._load_error,
            "pose": {
                "enabled": self.pose_enabled,
                "model": self.pose_model_path,
                "stride": POSE_STRIDE,
                "runs": self.pose_runs,
                "skips": self.pose_skips,
                "active_provider": models.active_provider(self._pose_model) if self._pose_model else None,
                "reason": None
                if self.pose_enabled
                else ("POSE_ENABLED=false" if not POSE_ENABLED else "no pose model"),
            },
        }

    def load(self) -> str:
        """Load the detector session + a fresh tracker (idempotent). Never raises: a failure leaves
        the backend "unavailable" with the reason in `load_error` / `load_exception`."""
        if self._backend != BACKEND_UNLOADED:
            return self._backend
        with _model_load_lock:
            if self._backend != BACKEND_UNLOADED:
                return self._backend
            try:
                self._det_model = models.load_detector_model(self.det_model_path, self.provider, self.imgsz)
                models.warmup(self._det_model)
                self._det_names = models.class_names_for(self._det_model, self.det_model_path)
                if self.pose_enabled:
                    self._pose_model = models.load_pose_model(str(self.pose_model_path), self.provider, POSE_INPUT_RGB)
                    models.warmup(self._pose_model)
                self._tracker = self._new_tracker()
                self._backend, self._load_error = BACKEND_RTM, None
                logger.info(
                    "[detector:%s] loaded %s on %s",
                    self.camera_id,
                    self.det_model_path,
                    models.active_provider(self._det_model),
                )
            except Exception as exc:  # ProviderError, missing / corrupt model, CUDA error ...
                self._backend, self.load_exception = BACKEND_UNAVAILABLE, exc
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.exception("[detector:%s] NO WORKING DETECTOR: %s", self.camera_id, self._load_error)
        return self._backend

    def _new_tracker(self) -> Any:
        """ByteTrack at the REAL analysed frame rate with a lost buffer in seconds (tracking/tracker.py)."""
        from lifeguard.vision.tracking.tracker import PersonTracker

        tracker = PersonTracker(self.fps)
        tracker.set_birth_guard(self._effective_birth_guard())
        return tracker

    def _effective_birth_guard(self) -> float | None:
        """[2.19.0] The birth guard only for DETR detectors (D-FINE): with RTMDet it lost 2 of 7 held-out dev events."""
        return self.birth_guard_iou if getattr(self._det_model, "detr", False) else None

    def set_birth_guard(self, iou: float | None) -> None:
        """[2.18.0] Finite clips: the tracker's birth guard (TRACK_BIRTH_GUARD_IOU_CLIP); live cameras keep None."""
        self.birth_guard_iou = iou
        if self._tracker is not None and hasattr(self._tracker, "set_birth_guard"):
            self._tracker.set_birth_guard(self._effective_birth_guard())

    def reset(self) -> None:
        """Forget tracker state (call after a stream reconnect)."""
        with self._lock:
            self.motion.reset()
            if self._tracker is not None:
                self._tracker.reset()
            self._track_age.clear()
            self._last_pose.clear()
            self.pose_stable = set()
            self.liveness.reset()
            self.non_person.reset()
            if self.follower is not None:
                self.follower.reset()
            self._gate_recent.clear()

    def detect(self, frame: np.ndarray, frame_index: int = 0, timestamp: float | None = None) -> FrameDetections:
        """
        Detection + tracking on a BGR frame. One bad frame is tolerated (logged with its stack,
        empty result); MAX_CONSECUTIVE_FRAME_FAILURES in a row, or no detector at all, raise
        DetectorUnavailableError - on every call, so a dead detector can never read as "safe".
        """
        ts = timestamp if timestamp is not None else time.time()
        if frame is None or frame.size == 0:
            return FrameDetections(self.camera_id, ts, frame_index, (0, 0, 0), [], 0.0, self._backend)
        if self._backend == BACKEND_UNLOADED:
            self.load()
        if self._backend == BACKEND_UNAVAILABLE:
            raise DetectorUnavailableError(self.camera_id, self._load_error or "detector unavailable")
        shape = (int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2]) if frame.ndim == 3 else 1)
        t0 = time.perf_counter()
        with self._lock:
            try:
                tracks = self._detect_rtm(frame, shape, ts)
                self._consecutive_failures = 0
            except Exception as exc:
                tracks = []
                self._consecutive_failures += 1
                n = self._consecutive_failures
                if n == 1 or n % 10 == 0:
                    logger.warning(
                        "[detector:%s] frame %d failed (%d in a row)", self.camera_id, frame_index, n, exc_info=exc
                    )
                if n >= MAX_CONSECUTIVE_FRAME_FAILURES:
                    self._backend = BACKEND_UNAVAILABLE
                    self._load_error = f"{n} consecutive frame failures: {type(exc).__name__}: {exc}"
                    logger.error("[detector:%s] detector marked UNAVAILABLE - %s", self.camera_id, self._load_error)
            coasting = self._coasting(ts, tracks) if TRACK_COAST_SECONDS > 0 else []
            if ZONE_GATE and coasting and self.zone_gate is not None:
                coasting = self._gate_coasting(ts, coasting)
            if SUBMERGED_FOLLOW:
                coasting = self._follow_submerged(frame, ts, tracks, coasting)
        if self._backend == BACKEND_UNAVAILABLE:
            raise DetectorUnavailableError(self.camera_id, self._load_error or "detector unavailable")
        ms = (time.perf_counter() - t0) * 1000.0
        return FrameDetections(self.camera_id, ts, frame_index, shape, tracks, ms, self._backend, coasting)

    def _coasting(self, ts: float, tracks: list[TrackedPerson]) -> list[dict[str, Any]]:
        """[2.35.0] Ids the tracker still carries but that found no box this frame (last observed box). Left out: ids
        shown this frame, ids the non-person filter hides, and a last box lying on a box shown this frame (IoA-small >=
        SHADOW_IOA): that body is visible under another id (RTMDet's duplicate ids alternate on a splashing body)."""
        from lifeguard.vision.tracking.tracker import ioa_small

        stitcher = getattr(self._tracker, "stitcher", None)
        if stitcher is None:
            return []
        shown = {t.track_id for t in tracks}
        boxes = [t.bbox for t in tracks]
        hidden = set(getattr(self.non_person, "hidden_ids", set()) or set())
        return [
            {"track_id": int(pid), "bbox": [round(float(v), 1) for v in box], "unseen_s": round(float(gap), 2)}
            for pid, box, gap in stitcher.coasting(ts, TRACK_COAST_SECONDS)
            if pid not in shown and pid not in hidden and not any(ioa_small(box, b) >= SHADOW_IOA for b in boxes)
        ]

    def _follow_submerged(
        self, frame: np.ndarray, ts: float, tracks: list[TrackedPerson], coasting: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """[2.40.0] SUBMERGED_FOLLOW (vision/tracking/submerged.py): a swimmer lost under water is followed on the
        image; the stitcher's lost record moves with them and the followed box is listed with the coasting boxes
        ("followed": true - dashed, never scored). Replaces that id's plain coasting box."""
        stitcher = getattr(self._tracker, "stitcher", None)
        if stitcher is None or not hasattr(stitcher, "follow"):
            return coasting
        if self.follower is None:
            from lifeguard.vision.tracking.submerged import FollowConfig, SubmergedFollower

            self.follower = SubmergedFollower(
                FollowConfig(
                    dark_delta=SUBMERGED_FOLLOW_DARK_DELTA,
                    max_seconds=SUBMERGED_FOLLOW_MAX_SECONDS,
                    min_track_seconds=SUBMERGED_FOLLOW_MIN_TRACK_SECONDS,
                    miss_seconds=SUBMERGED_FOLLOW_MISS_SECONDS,
                    search=SUBMERGED_FOLLOW_SEARCH,
                    bg_alpha=SUBMERGED_FOLLOW_BG_ALPHA,
                    confirm_seconds=SUBMERGED_FOLLOW_CONFIRM_SECONDS,
                    isolation=SUBMERGED_FOLLOW_ISOLATION,
                )
            )
        shown = {
            int(p.track_id): tuple(float(v) for v in p.bbox)
            for p in tracks
            if p.track_id > 0 and not getattr(p, "tentative", False)
        }
        hidden = set(getattr(self.non_person, "hidden_ids", set()) or set())
        followed = self.follower.step(frame, ts, shown, stitcher.lost_ids(), self.water_test, hidden)  # type: ignore[arg-type]
        if not followed:
            return coasting
        ids = set()
        out = []
        for pid, box, secs in followed:
            stitcher.follow(pid, box, ts)
            ids.add(pid)
            out.append(
                {
                    "track_id": int(pid),
                    "bbox": [round(float(v), 1) for v in box],
                    "unseen_s": round(float(secs), 2),
                    "followed": True,
                }
            )
        return [c for c in coasting if c["track_id"] not in ids] + out

    def take_linked_ids(self) -> set[int]:
        """[2.35.0] Public ids ID_LINK_NESTED folded into another id since the last call (the scene forgets them)."""
        stitcher = getattr(self._tracker, "stitcher", None)
        return stitcher.take_retired() if stitcher is not None else set()

    def set_link_protected(self, track_ids: set[int]) -> None:
        """[2.35.0] Ids whose risk is up (CAUTION / DROWNING): ID_LINK_NESTED never retires them."""
        stitcher = getattr(self._tracker, "stitcher", None)
        if stitcher is not None:
            stitcher.protected = set(track_ids)

    def prefetch(self, frame: np.ndarray) -> bool:
        """[v4] Start `frame`'s detector call now (live/scheduler.BatchedDetModel.prefetch): the camera calls this for
        its NEXT queued frame before finishing the current one. Only when one detector call on the whole frame is all
        stage_detect makes (no enhancement, ROI crop or tiling - the defaults); zoom re-detection uses its own canvas
        and is never prefetched. Returns False when not applicable. Results are identical to not prefetching."""
        dm = self._det_model
        if (
            self._backend != BACKEND_RTM
            or self.enhance_modes
            or self.roi_crop
            or self.tiling != "off"
            or not hasattr(dm, "prefetch")
            or frame is None
            or frame.size == 0
        ):
            return False
        dm.prefetch(frame)
        return True

    # ── stages (verbatim from v2) ─────────────────────────────────────────────
    def set_fps(self, fps: float) -> None:
        """Tell the tracker the analysed frame rate (= source fps: every frame is analysed)."""
        if fps and fps > 0:
            self.fps = float(fps)
            if self._tracker is not None and hasattr(self._tracker, "set_fps"):
                self._tracker.set_fps(fps)

    def retime(self, fps: float) -> None:
        """[dynamic fps] The analysed rate stepped on a running stream: tracks and motion model are kept."""
        if fps and fps > 0:
            self.fps = float(fps)
            if self._tracker is not None and hasattr(self._tracker, "retime"):
                self._tracker.retime(fps)

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        """Pool region (px) used by DET_ROI_CROP / DET_TILING; None = whole frame."""
        self.roi = tuple(int(v) for v in roi) if roi else None

    def set_water_test(self, fn: Any | None) -> None:
        """point (px) → True / False / None (unknown): is it in the water? Gates faint births."""
        self.water_test = fn

    def set_zone_gate(self, fn: Any | None) -> None:
        """[2.43.0] boxes (N x 4 px) -> bool mask (True = inside the drawn water gate) or None (no gate: whole frame).
        Read only when ZONE_GATE is on."""
        self.zone_gate = fn

    def _apply_zone_gate(self, boxes: np.ndarray, scores: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """[2.43.0] Drop detector boxes outside the drawn water gate before tracking (deck / outside the pool: no box,
        track, pose, score or alert). Kept outside the gate: a box overlapping (IoU >= 0.3) the last box of a public
        track that was inside the gate within ZONE_GATE_GRACE_SECONDS (a swimmer climbing out keeps their id)."""
        inside = self.zone_gate(boxes) if self.zone_gate is not None else None
        if inside is None:
            self._gate_recent.clear()
            return boxes, scores
        keep = np.asarray(inside, bool).copy()
        if not keep.all() and self._gate_recent:
            recent = [b for (t_in, b) in self._gate_recent.values() if t - t_in <= ZONE_GATE_GRACE_SECONDS]
            for i in np.flatnonzero(~keep).tolist():
                bi = tuple(float(v) for v in boxes[i][:4])
                if any(_iou(bi, rb) >= 0.3 for rb in recent):  # type: ignore[arg-type]
                    keep[i] = True
        if keep.all():
            return boxes, scores
        self.gate_dropped += int((~keep).sum())
        n = len(boxes)
        if len(self.last_birth_scores) == n:
            self.last_birth_scores = self.last_birth_scores[keep]
        if len(self.last_det_water) == n:
            self.last_det_water = self.last_det_water[keep]
        return boxes[keep], scores[keep]

    def _gate_tracked(self, boxes: np.ndarray, ids: np.ndarray, t: float) -> np.ndarray | None:
        """[2.43.0] ZONE_GATE_AFTER_TRACK: the tracker saw every box (ids exactly as without the gate); keep a reported
        box only inside the gate, or a public id that was inside within ZONE_GATE_GRACE_SECONDS. None = no gate."""
        inside = self.zone_gate(boxes) if self.zone_gate is not None else None
        if inside is None:
            self._gate_recent.clear()
            return None
        keep = np.asarray(inside, bool).copy()
        for i in np.flatnonzero(~keep).tolist():
            rec = self._gate_recent.get(int(ids[i]))
            if int(ids[i]) >= 0 and rec is not None and t - rec[0] <= ZONE_GATE_GRACE_SECONDS:
                keep[i] = True
        self.gate_dropped += int((~keep).sum())
        return keep

    def _note_gate(self, t: float, people: list[TrackedPerson]) -> None:
        """[2.43.0] Remember each public track's last box and the last time it was inside the gate (grace)."""
        ids = [p for p in people if p.track_id >= 0]
        if ids and self.zone_gate is not None:
            inside = self.zone_gate(np.asarray([p.bbox for p in ids], np.float32))
            if inside is not None:
                for p, ok in zip(ids, inside, strict=True):
                    prev = self._gate_recent.get(p.track_id)
                    t_in = t if ok else (prev[0] if prev is not None else None)
                    if t_in is not None:
                        self._gate_recent[p.track_id] = (t_in, p.bbox)
        stale = [k for k, (t_in, _) in self._gate_recent.items() if t - t_in > ZONE_GATE_GRACE_SECONDS + 5.0]
        for k in stale:
            del self._gate_recent[k]

    def _gate_coasting(self, t: float, coasting: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """[2.43.0] Coasting boxes (display only) outside the gate are shown only while their id is in its grace."""
        inside = self.zone_gate(np.asarray([c["bbox"] for c in coasting], np.float32)) if self.zone_gate else None
        if inside is None:
            return coasting
        out = []
        for c, ok in zip(coasting, inside, strict=True):
            rec = self._gate_recent.get(int(c["track_id"]))
            if ok or (rec is not None and t - rec[0] <= ZONE_GATE_GRACE_SECONDS):
                out.append(c)
        return out

    def set_outside_water_test(self, fn: Any | None) -> None:
        """[2.33.0] box (px) → True only when it is entirely outside the camera's drawn water polygon (no zone: False).
        Read only by NONPERSON_DECK_STATIC_SECONDS and TENTATIVE_HIDE_OUTSIDE_WATER."""
        self.outside_water_test = fn

    # ── RTMDet → ByteTrack → pose, in stages ──────────────────────────────────
    # detect() runs the stages back to back; the live runtime batches the
    # detector call of every camera (live/scheduler.py).
    def stage_detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Person boxes (frame px) + scores, everything >= DET_LOW_SCORE_THRESHOLD."""
        h, w = frame.shape[:2]
        dm = self._det_model
        in_h, in_w = dm.model_input_size
        roi = self.roi if (self.roi_crop or self.tiling != "off") else None
        x1, y1, x2, y2 = roi if (self.roi_crop and roi) else (0, 0, w, h)
        img = frame[y1:y2, x1:x2]
        scale = 1.0
        if self.enhance_modes:
            # condition the image at detector resolution (cheap), then detect on it as-is
            scale = min(in_h / img.shape[0], in_w / img.shape[1])
            if scale < 1.0:
                img = cv2.resize(
                    img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA
                )
            else:
                scale = 1.0
            img = enhance.apply(img, self.enhance_modes)
        # person classes only (COCO/single-class: label 0; zone-aware v2 detector:
        # person_in_water + person_out_of_water, one box per person + water vote)
        boxes, scores, water = rtm_backend.run_detector_people(dm, img, score_thr=self.conf, names=self._det_names)
        if self.nested_ioa > 0 and len(boxes) > 1 and not getattr(dm, "detr", False):
            # [2.18.0] clips: a box mostly inside another (a splashing arm, the head of a struggling swimmer) is a
            # part of that person, not a second one - it otherwise becomes a shadow released as a new id after 1.5 s
            keep = rtm_backend._drop_nested(boxes, scores, self.nested_ioa)
            boxes, scores, water = boxes[keep], scores[keep], water[keep] if len(water) == len(boxes) else water
        boxes = boxes / scale
        if len(boxes):
            boxes = boxes + np.array([x1, y1, x1, y1], np.float32)
        if self.faint_births:
            self.motion.update(frame, self.fps)
        # [v3] Phase 8: DET_TILING="wbf" tiles the water ROI (or the whole frame when no ROI is known yet) at
        # native resolution, fuses full-frame + tiles by weighted box fusion, then full-frame NMS (plan 7.2).
        if self.tiling == "wbf" and roi is None:
            roi = (0, 0, w, h)
        if self.tiling != "off" and roi is not None:
            tb, ts, tw = [boxes.reshape(-1, 4)], [scores.reshape(-1)], [water.reshape(-1)]
            keep_tile = getattr(self, "tile_keep", None)  # [v3 Phase 11] DET_TILING_REGION (live/responsibility.py)
            for tx1, ty1, tx2, ty2 in enhance.tile_windows(roi, min(in_h, in_w), 0.2, w, h):
                if keep_tile is not None and not keep_tile((tx1, ty1, tx2, ty2)):
                    continue
                timg = frame[ty1:ty2, tx1:tx2]
                if self.enhance_modes:
                    timg = enhance.apply(timg, self.enhance_modes)
                b, sc, wv = rtm_backend.run_detector_people(dm, timg, score_thr=self.conf, names=self._det_names)
                keep = ~enhance.cut_by_tile(b, (tx1, ty1, tx2, ty2), w, h)
                if keep.any():
                    tb.append(b[keep] + np.array([tx1, ty1, tx1, ty1], np.float32))
                    ts.append(sc[keep])
                    tw.append(wv[keep])
            if self.tiling == "wbf":
                fb, fs, fw = enhance.weighted_box_fusion(np.concatenate(tb), np.concatenate(ts), np.concatenate(tw))
                boxes, scores, k = enhance.merge_boxes(fb, fs, iou=0.45, return_index=True)
                water = fw[k]
            else:
                boxes, scores, k = enhance.merge_boxes(np.concatenate(tb), np.concatenate(ts), return_index=True)
                water = np.concatenate(tw)[k]
        # [v7 F4] resolution mosaic: re-scan the water rows where people are small at DET_MOSAIC_GAIN x the
        # full-frame resolution (the far end, DET_MOSAIC_FAR_FRAC, is the owner's accepted exception)
        if DET_MOSAIC and self.roi is not None:
            boxes, scores, water = self._mosaic_detect(frame, boxes, scores, water)
        birth = scores.reshape(-1).astype(np.float32)
        if self.zoom and self._tracker is not None and hasattr(self._tracker, "focus_regions"):
            regions = list(self._tracker.focus_regions())
            if regions:
                zb, zs, zw = self._zoom_detect(frame, regions[:DET_ZOOM_MAX_REGIONS])
                boxes, scores, water, birth = fuse_zoom(
                    boxes.reshape(-1, 4), scores.reshape(-1), water.reshape(-1), zb, zs, zw
                )
        self.last_birth_scores = birth
        self.last_det_water = water.astype(np.float32).reshape(-1)
        return boxes.astype(np.float32).reshape(-1, 4), scores.astype(np.float32).reshape(-1)

    def _mosaic_detect(
        self, frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray, water: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full-frame boxes + the mosaic's boxes (enhance.mosaic_*), fused like the tiling path (weighted box fusion,
        then containment / NMS merge). At most DET_MOSAIC_MAX_CALLS extra detector calls; nothing when the region
        would not be seen at a higher resolution than the full-frame pass."""
        if self.roi is None:
            return boxes, scores, water
        h, w = frame.shape[:2]
        size = min(self._det_model.model_input_size)
        plan = enhance.mosaic_region(self.roi, w, h, size, DET_MOSAIC_GAIN, DET_MOSAIC_FAR_FRAC, DET_MOSAIC_NEAR_FRAC)
        if plan is None:
            return boxes, scores, water
        region, scale = plan
        mb, ms, mw = [boxes.reshape(-1, 4)], [scores.reshape(-1)], [water.reshape(-1)]
        for placements in enhance.mosaic_plan(region, scale, size, 0.2, int(DET_MOSAIC_MAX_CALLS)):
            canvas = enhance.mosaic_canvas(frame, placements, size)
            b, sc, wv = rtm_backend.run_detector_people(
                self._det_model, canvas, score_thr=self.conf, names=self._det_names
            )
            self.mosaic_calls += 1
            fb, k = enhance.mosaic_unmap(b, placements, region, w, h)
            if len(k):
                mb.append(fb)
                ms.append(sc[k])
                mw.append(wv[k] if len(wv) == len(b) else np.full(len(k), np.nan, np.float32))
        if len(mb) == 1:
            return boxes, scores, water
        fb, fs, fw = enhance.weighted_box_fusion(np.concatenate(mb), np.concatenate(ms), np.concatenate(mw))
        fb, fs, k = enhance.merge_boxes(fb, fs, iou=0.45, return_index=True)
        return fb, fs, fw[k]

    def _zoom_detect(
        self, frame: np.ndarray, regions: Sequence[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Multi-scale re-detection: each region (the last-seen box of a swimmer
        just lost in the water) becomes a square crop of DET_ZOOM_CONTEXT × its
        size, enlarged into one quadrant of a 2×2 mosaic the size of the
        detector input (one region: the whole input), CLAHE'd, and the mosaic
        is detected in ONE call. Per crop, only the best box overlapping the
        region by TRACK_REFIND_LOW_IOU is kept and mapped back to frame px —
        it can re-find that person's track; it can never start a track.
        """
        dm = self._det_model
        in_h, in_w = dm.model_input_size
        h, w = frame.shape[:2]
        n = len(regions)
        cells = (
            [(0, 0, in_w, in_h)]
            if n == 1
            else [
                (cx * (in_w // 2), cy * (in_h // 2), (cx + 1) * (in_w // 2), (cy + 1) * (in_h // 2))
                for cy in range(2)
                for cx in range(2)
            ]
        )
        canvas = np.zeros((in_h, in_w, 3), np.uint8)
        maps = []  # (cell, crop x0, y0, zoom, placed size)
        for r, cell in zip(regions, cells):
            cw, chh = cell[2] - cell[0], cell[3] - cell[1]
            rx1, ry1, rx2, ry2 = (float(v) for v in r)
            side = DET_ZOOM_CONTEXT * max(rx2 - rx1, ry2 - ry1, 1.0)
            side = int(min(max(side, DET_ZOOM_MIN_CROP), DET_ZOOM_MAX_CROP, w, h))
            cx, cy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
            x0 = int(min(max(cx - side / 2.0, 0), w - side))
            y0 = int(min(max(cy - side / 2.0, 0), h - side))
            crop = frame[y0 : y0 + side, x0 : x0 + side]
            if crop.size == 0:
                continue
            zoom = min(cw, chh) / float(side)
            sz = max(1, min(cw, chh, int(round(side * zoom))))
            img = cv2.resize(crop, (sz, sz), interpolation=cv2.INTER_LINEAR if zoom > 1 else cv2.INTER_AREA)
            img = enhance.apply(img, ("clahe",))
            canvas[cell[1] : cell[1] + sz, cell[0] : cell[0] + sz] = img
            maps.append((cell, x0, y0, sz / float(side), sz, np.asarray(r, np.float32)))
        if not maps:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)
        self.zoom_calls += 1
        b, s, wv = rtm_backend.run_detector_people(dm, canvas, score_thr=self.conf, names=self._det_names)
        best: dict[int, tuple[float, list[float], float]] = {}  # crop → (score, box, water vote)
        for k in range(len(b)):
            bx1, by1, bx2, by2 = (float(v) for v in b[k])
            mx, my = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            for ci, (cell, x0, y0, zoom, sz, anchor) in enumerate(maps):
                if not (cell[0] <= mx < cell[0] + sz and cell[1] <= my < cell[1] + sz):
                    continue
                m = 2.0
                if bx1 < cell[0] - m or by1 < cell[1] - m or bx2 > cell[0] + sz + m or by2 > cell[1] + sz + m:
                    break  # crosses into another crop
                if (bx2 - bx1) * (by2 - by1) > DET_ZOOM_MAX_BOX_FRAC * sz * sz:
                    break  # fills the crop: CLAHE texture, not a person
                fb = [
                    x0 + (bx1 - cell[0]) / zoom,
                    y0 + (by1 - cell[1]) / zoom,
                    x0 + (bx2 - cell[0]) / zoom,
                    y0 + (by2 - cell[1]) / zoom,
                ]
                # one box per crop: the person it was aimed at, where they were last seen
                if _iou(tuple(fb), tuple(anchor)) >= TRACK_REFIND_LOW_IOU and float(s[k]) > best.get(ci, (-1.0,))[0]:
                    best[ci] = (float(s[k]), fb, float(wv[k]) if len(wv) == len(b) else np.nan)
                break
        out = list(best.values())
        return (
            np.asarray([o[1] for o in out], np.float32).reshape(-1, 4),
            np.asarray([o[0] for o in out], np.float32),
            np.asarray([o[2] for o in out], np.float32),
        )

    def stage_track(self, boxes: np.ndarray, scores: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Feed every box to ByteTrack. Returns (boxes, scores, ids) to report:
        tracked people (public id), plus untracked boxes scoring at least
        TENTATIVE_DISPLAY_MIN_SCORE as tentative (id -1) so nothing a person
        could be is hidden from the operator.
        """
        if hasattr(self._tracker, "surveillance"):
            self._tracker.surveillance = self.surveillance
        if hasattr(self._tracker, "coast_water"):
            self._tracker.coast_water = self.water_test  # [2.24.0] read only when TRACK_LOST_SECONDS_WATER is on
        if self.faint_births:
            birth = self.last_birth_scores if len(self.last_birth_scores) == len(boxes) else scores
            idx, ids = self._tracker.update(
                boxes, scores, t, evidence=self.motion.evidence, in_water=self.water_test, birth_scores=birth
            )
        else:
            idx, ids = self._tracker.update(boxes, scores, t)
        if len(boxes) == 0:
            self.last_track_water = np.zeros(0, np.float32)
            self.last_track_faint = np.zeros(0, bool)
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        tracked = set(idx.tolist())  # includes shadows (id -1): a second box on a
        shadow = ids < 0  # person already reported — dropped, not shown
        idx, ids = idx[~shadow], ids[~shadow]
        extra = [i for i in range(len(boxes)) if i not in tracked and scores[i] >= TENTATIVE_DISPLAY_MIN_SCORE]
        sel = np.concatenate([idx, np.asarray(extra, int)]).astype(int)
        all_ids = np.concatenate([ids, np.full(len(extra), -1, int)]).astype(int)
        water = self.last_det_water
        self.last_track_water = water[sel] if len(water) == len(boxes) else np.full(len(sel), np.nan, np.float32)
        faint_ids = getattr(self._tracker, "faint_ids", set())
        self.last_track_faint = np.asarray([int(i) in faint_ids for i in all_ids], bool)
        return boxes[sel], scores[sel], all_ids

    @staticmethod
    def assemble(
        shape: tuple[int, int, int],
        boxes: np.ndarray,
        scores: np.ndarray,
        ids: np.ndarray,
        pose_idx: np.ndarray,
        kpts: np.ndarray,
        kp_conf: np.ndarray,
        water: np.ndarray | None = None,
        faint: np.ndarray | None = None,
        pose_skipped: np.ndarray | None = None,
    ) -> list[TrackedPerson]:
        h, w = shape[0], shape[1]
        slot = {int(i): j for j, i in enumerate(pose_idx.tolist())}
        use_water = water is not None and len(water) == len(boxes)
        use_faint = faint is not None and len(faint) == len(boxes)
        use_skip = pose_skipped is not None and len(pose_skipped) == len(boxes)
        out: list[TrackedPerson] = []
        for i in range(len(boxes)):
            bbox = _clip_bbox(*boxes[i].tolist(), w=w, h=h)
            keypoints: Keypoints | None = None
            j = slot.get(i)
            if j is not None and j < len(kpts):
                keypoints = [
                    [float(kpts[j][k][0]), float(kpts[j][k][1]), float(kp_conf[j][k])] for k in range(kpts.shape[1])
                ]
            wp = float(water[i]) if use_water and np.isfinite(water[i]) else None
            out.append(
                TrackedPerson(
                    track_id=int(ids[i]),
                    bbox=bbox,
                    confidence=float(scores[i]),
                    keypoints=keypoints,
                    foot_point=foot_point_from(bbox, keypoints),
                    water_prob=wp,
                    faint_birth=bool(faint[i]) if use_faint else False,
                    pose_skipped=bool(pose_skipped[i]) if use_skip else False,
                )
            )
        return out

    def set_pose_stable(self, track_ids: set[int]) -> None:
        """Tracks the behaviour stage judges stable (not drowning, low risk, pose-reliable): only these may skip."""
        self.pose_stable = set(track_ids)

    def pose_plan(self, boxes: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(indices that get RTMPose this frame, mask of boxes whose pose is deliberately skipped).

        Owner rule (thresholds 2.5.0, unchanged by 2.5.1): a track may skip pose on a frame only if ALL hold - confirmed
        id, seen for at least POSE_STABLE_MIN_FRAMES frames (rescaled to the fps by frames_at, [v7] switch), box at
        least POSE_SMALL_BOX_PX tall (far-end swimmers always get pose), judged stable by the behaviour stage, and posed
        within the last POSE_STRIDE - 1 frames. Anything new, re-found, small, at risk or unreliable is posed on every
        frame."""
        self._pose_frame += 1
        n = len(boxes)
        stride = max(1, int(POSE_STRIDE))
        stable_frames = frames_at(POSE_STABLE_MIN_FRAMES, self.fps)  # [v7] same seconds at any fps (switch)
        skipped = np.zeros(n, bool)
        run: list[int] = []
        seen = set()
        for i in range(n):
            tid = int(ids[i])
            if tid >= 0:
                seen.add(tid)
                self._track_age[tid] = self._track_age.get(tid, 0) + 1
            stable = (
                stride > 1
                and tid >= 0
                and self._track_age[tid] >= stable_frames
                and float(boxes[i][3] - boxes[i][1]) >= POSE_SMALL_BOX_PX
                and tid in self.pose_stable
                and self._pose_frame - self._last_pose.get(tid, -(10**9)) < stride
            )
            if stable:
                skipped[i] = True
            else:
                run.append(i)
        for tid in list(self._track_age):
            if tid not in seen:  # gone (or re-found under a new id): starts over as new
                self._track_age.pop(tid, None)
                self._last_pose.pop(tid, None)
        run_idx = np.asarray(run, int)
        if len(run_idx):
            run_idx = run_idx[pose_stage.pose_order(boxes[run_idx], POSE_MAX_PER_FRAME)]
        for i in run_idx.tolist():
            if int(ids[i]) >= 0:
                self._last_pose[int(ids[i])] = self._pose_frame
        self.pose_runs += len(run_idx)
        self.pose_skips += int(skipped.sum())
        return run_idx, skipped

    def stage_pose(
        self, frame: np.ndarray, boxes: np.ndarray, ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(pose_idx, keypoints, keypoint scores, skipped mask) for the tracked boxes; top-down RTMPose crops."""
        if self._pose_model is None or len(boxes) == 0 or (self.surveillance and not CCTV_SKELETON):
            return np.empty((0,), int), np.zeros((0, 17, 2)), np.zeros((0, 17)), np.zeros(len(boxes), bool)
        idx, skipped = self.pose_plan(boxes, ids)
        if len(idx) == 0:
            return idx, np.zeros((0, 17, 2)), np.zeros((0, 17)), skipped
        kpts, conf = pose_stage.run_pose(self._pose_model, frame, boxes[idx], self.pose_infer)
        return idx, kpts, conf, skipped

    def _detect_rtm(self, frame: np.ndarray, shape: tuple[int, int, int], t: float = 0.0) -> list[TrackedPerson]:
        """RTMDet -> ByteTrack -> RTMPose (top-down, on the tracked boxes) -> assemble (v2's stages)."""
        boxes, scores = self.stage_detect(frame)
        if ZONE_GATE and not ZONE_GATE_AFTER_TRACK and len(boxes):
            boxes, scores = self._apply_zone_gate(boxes, scores, t)
        stitcher = getattr(self._tracker, "stitcher", None) if ID_LINK_NESTED else None
        if stitcher is not None:
            stitcher.colour_fn = lambda b: _cap_torso_hist(frame, b)  # [2.35.0] colour veto (this frame only)
        boxes, scores, ids = self.stage_track(boxes, scores, t)
        if stitcher is not None:
            stitcher.colour_fn = None
        if len(boxes) == 0:
            return []
        if ZONE_GATE and ZONE_GATE_AFTER_TRACK:
            gk = self._gate_tracked(boxes, ids, t)
            if gk is not None and not gk.all():
                boxes, scores, ids = boxes[gk], scores[gk], ids[gk]
                if len(self.last_track_water) == len(gk):
                    self.last_track_water = self.last_track_water[gk]
                if len(self.last_track_faint) == len(gk):
                    self.last_track_faint = self.last_track_faint[gk]
                if len(boxes) == 0:
                    return []
        # [v7 F3] person-shaped objects outside the water (photo, poster, slide, monitor) are not reported or posed
        static = self.liveness.static_mask(frame, boxes, ids, t, self.surveillance, self.water_test)
        if static.any():
            keep = ~static
            boxes, scores, ids = boxes[keep], scores[keep], ids[keep]
            if len(self.last_track_water) == len(keep):
                self.last_track_water = self.last_track_water[keep]
            if len(self.last_track_faint) == len(keep):
                self.last_track_faint = self.last_track_faint[keep]
            if len(boxes) == 0:
                return []
        pose_idx, kpts, kp_conf, skipped = self.stage_pose(frame, boxes, ids)
        if stitcher is not None and not self.surveillance:  # [2.35.0] skeletons for ID_LINK_NESTED's two-skeleton veto
            for j, i in enumerate(pose_idx.tolist()):
                if j < len(kpts) and int(ids[i]) > 0:
                    stitcher.note_pose(int(ids[i]), kpts[j], kp_conf[j], t)
        people = self.assemble(
            shape,
            boxes,
            scores,
            ids,
            pose_idx,
            kpts,
            kp_conf,
            water=self.last_track_water,
            faint=self.last_track_faint,
            pose_skipped=skipped,
        )
        # [2.18.0] lens occluders (a finger) and fixtures (lane-rope anchor posts) are not people: not reported
        slot = {int(i): j for j, i in enumerate(pose_idx.tolist())}
        # [rtmpose clone] standard CCTV: the non-person (fixture / occluder) rules see no joints, as without pose,
        # so the skeleton overlay cannot change which boxes are shown
        joints = [
            int((kp_conf[slot[i]] >= KEYPOINT_CONF_THRESHOLD).sum())
            if not self.surveillance and i in slot and slot[i] < len(kp_conf)
            else None
            for i in range(len(boxes))
        ]
        test = self.outside_water_test
        deck = NONPERSON_DECK_STATIC_SECONDS > 0 or TENTATIVE_HIDE_OUTSIDE_WATER
        outside = [bool(test(b)) for b in boxes] if deck and test is not None else None
        hide = self.non_person.mask(frame, boxes, ids, joints, t, outside)
        if TENTATIVE_HIDE_OUTSIDE_WATER and outside is not None:
            # [2.33.0] an untracked box entirely outside the drawn water (a pump, a shirt on a rack) is not shown;
            # it is never scored, and a person there is shown as soon as the tracker confirms them
            hide = hide | (np.asarray(outside, bool) & (np.asarray(ids) < 0))
        if hide.any():
            people = [p for p, h in zip(people, hide, strict=True) if not h]
        if ZONE_GATE:
            self._note_gate(t, people)
        return people
