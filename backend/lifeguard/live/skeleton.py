"""[rtmpose clone] Skeleton overlay for the live view - display only, never scored.

The pipeline already runs RTMPose (top-down, on the detector's tracked person crops, GPU, batched by
live/pose_batcher.PoseBatcher) and leaves 17 COCO keypoints on each TrackedPerson. This module only DRAWS them:
no inference, nothing on the capture or analysis path. Drawing happens in the viewer's encode thread
(live/view.render) or in the browser (overlay=client: the keypoints ride along in boxes_json under "k").

Pose is deliberately skipped on some frames for stable tracks (POSE_STRIDE: keypoints None, pose_skipped True).
So the skeleton does not flicker, the last pose of that track is reused, moved and scaled with the track's
current box (PoseMemory). A remembered pose expires after POSE_MEMORY_S seconds.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np

# COCO-17 limbs (RTMPose body order): face, arms, torso, legs.
LIMBS: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)  # fmt: skip
SKELETON = (255, 200, 0)  # BGR cyan-blue: distinct from the green / grey / red boxes
JOINT = (0, 220, 255)  # BGR amber
POSE_MEMORY_S = 1.0  # a skipped-pose frame reuses the track's last pose for at most this long
_MIN_CONF = 0.3  # display cut-off for a keypoint (KEYPOINT_CONF_THRESHOLD is the analysis cut-off)


class PoseMemory:
    """Last real pose per (camera, track id), so a pose-skipped frame still shows the skeleton. Bounded: entries
    older than POSE_MEMORY_S are dropped on every update. Thread-safe (one encode thread per viewer)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last: dict[tuple[str, int], tuple[float, tuple[int, int, int, int], np.ndarray]] = {}

    def poses(self, camera: str, detections: Any | None) -> list[tuple[int, np.ndarray]]:
        """[(track index in detections.tracks, 17x3 float32 [x, y, conf] in frame pixels)] for every track that has
        a real pose this frame, or a recent remembered one moved onto its current box."""
        out: list[tuple[int, np.ndarray]] = []
        now = time.monotonic()
        with self._lock:
            for i, t in enumerate(getattr(detections, "tracks", None) or []):
                key = (camera, int(t.track_id))
                kp = getattr(t, "keypoints", None)
                if kp is not None and len(kp) >= 17:
                    arr = np.asarray(kp, dtype=np.float32)[:17]
                    if t.track_id >= 0:
                        self._last[key] = (now, tuple(t.bbox), arr)
                    out.append((i, arr))
                elif t.track_id >= 0 and key in self._last:
                    seen, (ox1, oy1, ox2, oy2), arr = self._last[key]
                    if now - seen <= POSE_MEMORY_S:
                        x1, y1, x2, y2 = t.bbox
                        sx = (x2 - x1) / max(ox2 - ox1, 1)
                        sy = (y2 - y1) / max(oy2 - oy1, 1)
                        moved = arr.copy()
                        moved[:, 0] = x1 + (arr[:, 0] - ox1) * sx
                        moved[:, 1] = y1 + (arr[:, 1] - oy1) * sy
                        out.append((i, moved))
            for k in [k for k, v in self._last.items() if now - v[0] > POSE_MEMORY_S]:
                del self._last[k]
        return out


MEMORY = PoseMemory()


_LIMB_A = np.array([a for a, _ in LIMBS])
_LIMB_B = np.array([b for _, b in LIMBS])


def draw(out: np.ndarray, poses: list[tuple[int, np.ndarray]], scale: float, thick: int) -> None:
    """Draws the skeletons onto `out` (already scaled by `scale`) in place: limbs picked with numpy, one polylines
    call for every limb of every person, then the joints. Non-antialiased (LINE_8): measured ~3x cheaper than
    LINE_AA (1.2 vs 3.3 ms for 10 people on a 1920 px frame)."""
    if not poses:
        return
    segs: list[np.ndarray] = []
    joints: list[np.ndarray] = []
    for _, kp in poses:
        xy = (kp[:, :2] * scale).astype(np.int32)
        ok = kp[:, 2] >= _MIN_CONF
        keep = ok[_LIMB_A] & ok[_LIMB_B]
        if keep.any():
            segs.extend(np.stack((xy[_LIMB_A[keep]], xy[_LIMB_B[keep]]), axis=1))
        joints.append(xy[ok])
    if segs:
        cv2.polylines(out, segs, False, SKELETON, thick, cv2.LINE_8)
    r = max(2, thick + 1)
    for x, y in np.concatenate(joints).tolist():
        cv2.circle(out, (x, y), r, JOINT, -1, cv2.LINE_8)


def to_json(poses: list[tuple[int, np.ndarray]], track_ids: list[int], scale: float) -> list[list[int]]:
    """[[track id, x0, y0, c0, ..., x16, y16, c16], ...] in the scaled frame's pixels, confidence as 0..100 - about
    150 bytes per person in the part header."""
    rows: list[list[int]] = []
    for i, kp in poses:
        k = np.nan_to_num(kp, nan=0.0)
        flat = np.column_stack((k[:, 0] * scale, k[:, 1] * scale, np.clip(k[:, 2], 0.0, 1.0) * 100 + 0.5))
        rows.append([int(track_ids[i]), *flat.astype(np.int32).ravel().tolist()])
    return rows
