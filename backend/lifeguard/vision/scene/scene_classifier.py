"""[v4] Learned pool-vs-other scene check: zero-shot CLIP ViT-B/32 (tools/export_scene_clip.py, MODELS.json
"clip_vit_b32_scene") on KEYFRAMES, never per frame.

The ONNX graph maps an RGB 224x224 image to probabilities over scene classes (pool_indoor, pool_outdoor,
pool_underwater, office, meeting_room, corridor, home_room, shop_warehouse, outdoor_dry; names and the pool subset
come from the model's metadata). A CCTV frame is wide and the pool may fill only one side of it, so every keyframe
is scored as VIEWS: the whole frame squashed to a square, plus left / centre / right squares. ``pool_prob`` of a
keyframe = the HIGHEST pool probability over its views (a pool anywhere in the shot counts).

One background thread owns the session (``SceneClassifierService``): a camera's analysis thread only hands over a
downscaled copy of the frame and never waits; the model loads lazily on that thread (CUDA, else CPU - TensorRT is
not used: a ~350 MB engine build is not worth it for a few frames per camera). Any failure leaves the scene
suggestion on its colour-only evidence; it never touches monitoring.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SIZE = 224
SUBMIT_WIDTH = 640  # px: what a camera thread copies and hands over (the views are cut from this)


@dataclass(frozen=True)
class SceneVerdict:
    pool_prob: float  # highest pool probability over the views
    top_class: str  # most likely class: of the most pool-like view if that one is a pool, else of the full frame
    view_pool: tuple[float, ...]  # per view: full, left, centre, right


def views(frame_bgr: np.ndarray) -> np.ndarray:
    """(4, 3, 224, 224) float32 RGB in [0, 1]: whole frame squashed, then left / centre / right squares."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    s = min(h, w)
    crops = [rgb]
    for x0 in (0, (w - s) // 2, w - s) if w >= h else (0, (h - s) // 2, h - s):
        crops.append(rgb[:, x0 : x0 + s] if w >= h else rgb[x0 : x0 + s, :])
    out = [cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for c in crops]
    return np.ascontiguousarray(np.stack(out).transpose(0, 3, 1, 2), dtype=np.float32) / 255.0


class SceneClassifier:
    def __init__(self, path: str, session: Any = None) -> None:
        self.path = path
        if session is None:
            session = _make_session(path)
        self.session = session
        meta = session.get_modelmeta().custom_metadata_map
        self.classes: list[str] = json.loads(meta["classes"])
        pool = set(json.loads(meta["pool_classes"]))
        self.pool_idx = np.array([i for i, c in enumerate(self.classes) if c in pool], int)
        self.input_name = session.get_inputs()[0].name

    def classify(self, frame_bgr: np.ndarray) -> SceneVerdict:
        probs = np.asarray(self.session.run(None, {self.input_name: views(frame_bgr)})[0], np.float64)
        pool = probs[:, self.pool_idx].sum(axis=1)
        best = int(np.argmax(pool))
        top = self.classes[int(np.argmax(probs[best if pool[best] >= 0.5 else 0]))]
        return SceneVerdict(float(pool[best]), top, tuple(round(float(p), 4) for p in pool))


def _make_session(path: str) -> Any:
    from lifeguard.vision.inference.session import ProviderError, make_session

    try:
        sess, _ = make_session(path, "cuda")
    except ProviderError:
        logger.warning("scene model: CUDA not available - running it on the CPU (keyframes only)")
        sess, _ = make_session(path, "cpu")
    return sess


Callback = Callable[[SceneVerdict], None]


class SceneClassifierService:
    """One thread, one session, a small bounded queue. submit() never blocks; a full queue drops the keyframe
    (the camera asks again at its next keyframe)."""

    def __init__(self, path: str, factory: Callable[[str], SceneClassifier] = SceneClassifier) -> None:
        self.path = path
        self._factory = factory
        self._q: queue.Queue[tuple[np.ndarray, Callback] | None] = queue.Queue(maxsize=8)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.error: str | None = None
        self.classified = 0
        self.dropped = 0

    def submit(self, frame_bgr: np.ndarray, callback: Callback) -> bool:
        if self.error is not None:
            return False
        h, w = frame_bgr.shape[:2]
        small = (
            cv2.resize(frame_bgr, (SUBMIT_WIDTH, max(1, round(h * SUBMIT_WIDTH / w))), interpolation=cv2.INTER_AREA)
            if w > SUBMIT_WIDTH
            else frame_bgr.copy()
        )
        self._ensure_thread()
        try:
            self._q.put_nowait((small, callback))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="scene-classifier", daemon=True)
                self._thread.start()

    def _loop(self) -> None:
        try:
            clf = self._factory(self.path)
            logger.info("scene model loaded: %s (%d classes)", self.path, len(clf.classes))
        except Exception as exc:  # noqa: BLE001 - the suggestion falls back to colour; logged once
            self.error = f"{type(exc).__name__}: {exc}"
            logger.warning("scene model unavailable (%s) - pool/CCTV suggestion uses colour only", self.error)
            return
        while True:
            item = self._q.get()
            if item is None:
                return
            frame, callback = item
            try:
                callback(clf.classify(frame))
                self.classified += 1
            except Exception:
                logger.exception("scene classification failed for one keyframe")

    def stop(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._q.put(None)
            self._thread.join(5.0)

    def status(self) -> dict[str, Any]:
        return {"model": self.path, "error": self.error, "classified": self.classified, "dropped": self.dropped}


_service_lock = threading.Lock()
_service: SceneClassifierService | None = None


def service() -> SceneClassifierService | None:
    """The process-wide service for the configured scene model, or None (no model: colour-only suggestion)."""
    global _service
    from lifeguard.vision import config as vision_config

    path = vision_config.current().scene_model_path
    with _service_lock:
        if path is None:
            return None
        if _service is None or _service.path != path:
            _service = SceneClassifierService(path)
        return _service
