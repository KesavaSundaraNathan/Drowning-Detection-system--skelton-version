"""Alert snapshots and recordings (plan Phase 5; v2 `save_recording` rebuilt).

Snapshot: when a live alert fires, the frame it fired on is written as
`alert_<id>.jpg` (person box drawn) under snapshots_dir; the alert row stores the NAME only.

Recording: every live camera keeps a small ring of recent frames (sampled at recording_fps,
downscaled to recording_width, about recording_pre_s seconds). An AUDIBLE alert starts a clip:
the ring (before the alarm) + recording_post_s seconds after it, written as `alert_<id>.webm` under
recordings_dir by a background thread, then attached to the alert. Recording never delays delivery.

Serving: `contained()` resolves a stored NAME to a regular file strictly INSIDE its directory; any
other value (absolute path, `..`, another volume, missing file) is refused - same rule as
alerts/snapshots.py. Recordings are WebM/VP8 (browser <video>); see RECORDING_FOURCC.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from lifeguard.alerts.snapshots import snapshot_paths, write_jpeg

logger = logging.getLogger(__name__)

SNAPSHOT_COLOUR = (0, 0, 255)  # BGR red
# Phase 6 decision: WebM / VP8 (libvpx, BSD; royalty-free) - plays in every current browser's <video>
# and is written by the FFmpeg already bundled in opencv-python-headless. MPEG-4 Part 2 (Phase 5,
# not browser-playable) files are still served, as downloads.
RECORDING_EXT, RECORDING_FOURCC = ".webm", "VP80"
MEDIA_TYPES = {".webm": "video/webm", ".mp4": "video/mp4", ".jpg": "image/jpeg"}


def contained(base: Path, name: str | None) -> Path | None:
    """`base/name` only if `name` is a plain file name of an existing regular file inside `base`."""
    if not name or "/" in name or "\\" in name or ":" in name or name in (".", ".."):
        return None
    try:
        root = base.resolve()
        p = (root / name).resolve()
        p.relative_to(root)
    except (OSError, ValueError):
        return None
    return p if p.is_file() else None


def save_snapshot(
    snapshots_dir: Path, alert_id: int, frame: np.ndarray, bbox: tuple[int, int, int, int] | None
) -> str | None:
    """Write the annotated alert frame; returns its NAME (or None on failure)."""
    img = frame.copy()
    if bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), SNAPSHOT_COLOUR, max(2, img.shape[1] // 400))
    path = snapshot_paths(snapshots_dir, alert_id)["annotated"]
    return path.name if write_jpeg(path, img) else None


def _shrink(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return frame.copy()
    return cv2.resize(frame, (width, max(2, int(h * width / w)) // 2 * 2), interpolation=cv2.INTER_AREA)


@dataclass
class _Clip:
    alert_id: int
    frames: list[np.ndarray]
    until: float
    on_done: Callable[[str | None], None]
    done: threading.Event = field(default_factory=threading.Event)


class Recorder:
    """One per live camera. `push()` is called from the camera's analysis thread for every frame."""

    def __init__(
        self,
        camera_id: str,
        recordings_dir: Path,
        *,
        pre_s: float = 5.0,
        post_s: float = 5.0,
        fps: float = 10.0,
        width: int = 640,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.camera_id = camera_id
        self.recordings_dir = recordings_dir
        self.pre_s, self.post_s, self.fps, self.width = pre_s, post_s, fps, width
        self.clock = clock
        self._ring: deque[np.ndarray] = deque(maxlen=max(1, round(pre_s * fps)))
        self._next_sample = 0.0
        self._clips: list[_Clip] = []
        self._lock = threading.Lock()
        self.written = 0
        self.failed = 0

    def push(self, frame: np.ndarray, overlay: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        """`overlay` ([Stage 4]): draws the live view's boxes on a kept frame (returns the frame at `width`)."""
        now = self.clock()
        if now < self._next_sample:
            return
        self._next_sample = now + 1.0 / self.fps
        small = overlay(frame) if overlay is not None else _shrink(frame, self.width)
        finished: list[_Clip] = []
        with self._lock:
            self._ring.append(small)
            for clip in self._clips:
                clip.frames.append(small)
                if now >= clip.until:
                    finished.append(clip)
            self._clips = [c for c in self._clips if c not in finished]
        for clip in finished:
            threading.Thread(target=self._write, args=(clip,), name=f"rec-{clip.alert_id}", daemon=True).start()

    def start_clip(self, alert_id: int, on_done: Callable[[str | None], None]) -> threading.Event:
        """Begin a clip for `alert_id`: the pre-alarm ring now, post-alarm frames as they arrive."""
        with self._lock:
            clip = _Clip(alert_id, list(self._ring), self.clock() + self.post_s, on_done)
            self._clips.append(clip)
        return clip.done

    def flush(self) -> None:
        """Write unfinished clips now (camera stopped)."""
        with self._lock:
            clips, self._clips = self._clips, []
        for clip in clips:
            self._write(clip)

    def _write(self, clip: _Clip) -> None:
        name: str | None = None
        try:
            if clip.frames:
                self.recordings_dir.mkdir(parents=True, exist_ok=True)
                name = f"alert_{clip.alert_id}{RECORDING_EXT}"
                h, w = clip.frames[-1].shape[:2]
                vw = cv2.VideoWriter(
                    str(self.recordings_dir / name), cv2.VideoWriter.fourcc(*RECORDING_FOURCC), self.fps, (w, h)
                )
                if not vw.isOpened():
                    raise OSError("VideoWriter did not open")
                for f in clip.frames:
                    vw.write(f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)))
                vw.release()
                self.written += 1
        except (OSError, cv2.error):
            logger.exception("[rec:%s] recording for alert %d failed", self.camera_id, clip.alert_id)
            self.failed += 1
            name = None
        try:
            clip.on_done(name)
        except Exception:
            logger.exception("[rec:%s] attaching recording %s failed", self.camera_id, name)
        clip.done.set()
