"""Cross-camera dynamic batching of RTMPose crops: ONE pose call per round for every camera.

Each camera thread runs the unchanged detector stages; at the pose stage it crops and normalises its tracked
boxes on its own thread (inference/pose.prepare) and hands the crops to ``PoseBatcher.infer``. Each round the
batcher thread:
  1. waits for the first request;
  2. gathers more until it holds ``max_crops`` crops, or every registered camera has a request pending, or
     ``gather_ms`` has passed;
  3. concatenates every camera's crops and runs inference/pose.infer once (chunks of the largest batch
     bucket, each padded to a bucket - the TensorRT profile points);
  4. hands every camera its own slice of simcc_x / simcc_y back.

Failure isolation and stall handling are the detector scheduler's (live/scheduler.py): a failed batched run is
retried per camera, so one bad request cannot blank the others; a camera waiting longer than ``stall_s`` gets
SchedulerStalledError (a failed frame for its detector, never a silent pose-less frame).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lifeguard.live.scheduler import SchedulerStalledError, SchedulerStoppedError
from lifeguard.vision.inference import pose
from lifeguard.vision.inference.models import PoseModel

logger = logging.getLogger(__name__)


@dataclass
class _PoseRequest:
    x: np.ndarray  # (n, 3, H, W) normalised crops
    client: str
    submitted: float
    done: threading.Event = field(default_factory=threading.Event)
    sx: np.ndarray | None = None
    sy: np.ndarray | None = None
    error: BaseException | None = None
    started: bool = False


class PoseBatcher:
    def __init__(
        self, model: PoseModel, gather_ms: float = 5.0, stall_s: float = 10.0, name: str = "pose-batcher"
    ) -> None:
        self.model = model
        self.max_crops = model.max_batch
        self.gather_s = gather_ms / 1000.0
        self._buffers: dict[int, np.ndarray] = {}  # [v7] bucket-sized input buffers, this thread only
        self.stall_s = stall_s
        self.name = name
        self._pending: deque[_PoseRequest] = deque()
        self._inflight: list[_PoseRequest] = []
        self._cv = threading.Condition()
        self._clients: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.rounds = 0
        self.crops = 0
        self.requests = 0
        self.isolated_retries = 0
        self.failed_requests = 0
        self.crops_hist: Counter[int] = Counter()
        self.last_round_ms = 0.0
        self.busy_ms = 0.0  # [dynamic fps] total pose session time (GPU busy fraction, live/rate.py)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=self.name, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
        with self._cv:
            for req in self._pending:
                req.error = SchedulerStoppedError("pose batcher stopped")
                req.done.set()
            self._pending.clear()

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def register(self, client: str) -> None:
        with self._cv:
            self._clients.add(client)

    def unregister(self, client: str) -> None:
        with self._cv:
            self._clients.discard(client)
            self._cv.notify_all()

    # ── camera side ───────────────────────────────────────────────────────────
    def infer(self, x: np.ndarray, client: str, timeout: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """SimCC outputs for this camera's crops, from the next shared round; blocks the camera thread."""
        if len(x) == 0:
            return pose.infer(self.model, x)
        if self._stop.is_set() or not self.alive:
            raise SchedulerStoppedError("pose batcher is not running")
        req = _PoseRequest(np.ascontiguousarray(x, dtype=np.float32), client, time.monotonic())
        with self._cv:
            self._pending.append(req)
            self._cv.notify_all()
        if not req.done.wait(self.stall_s if timeout is None else timeout):
            with self._cv:
                if not req.started and req in self._pending:
                    self._pending.remove(req)
            raise SchedulerStalledError(f"no pose result for {client} within {self.stall_s:.1f} s")
        if req.error is not None:
            raise req.error
        if req.sx is None or req.sy is None:
            raise RuntimeError("pose batcher answered without outputs")
        return req.sx, req.sy

    def infer_fn(self, client: str) -> Any:
        """The detector's pose_infer hook for one camera (inference/pose.InferFn)."""
        return lambda _model, x: self.infer(x, client)

    # ── batcher side ──────────────────────────────────────────────────────────
    def oldest_pending_s(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        with self._cv:
            reqs = list(self._pending) + list(self._inflight)
        return max((now - r.submitted for r in reqs), default=0.0)

    def stalled(self, now: float | None = None) -> bool:
        return (self._thread is not None and not self.alive and not self._stop.is_set()) or (
            self.oldest_pending_s(now) > self.stall_s
        )

    def _gather(self) -> list[_PoseRequest]:
        with self._cv:
            while not self._pending and not self._stop.is_set():
                self._cv.wait(0.5)
            if self._stop.is_set():
                return []
            deadline = time.monotonic() + self.gather_s
            while not self._stop.is_set():
                if sum(len(r.x) for r in self._pending) >= self.max_crops:
                    break
                if self._clients and self._clients <= {r.client for r in self._pending}:
                    break  # every camera's crops are in this round
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._cv.wait(left)
            batch: list[_PoseRequest] = []
            total = 0
            while self._pending and (not batch or total + len(self._pending[0].x) <= self.max_crops):
                r = self._pending.popleft()
                r.started = True
                batch.append(r)
                total += len(r.x)
            self._inflight = batch
            return batch

    def _loop(self) -> None:
        while not self._stop.is_set():
            batch = self._gather()
            if not batch:
                continue
            try:
                self._process(batch)
            except BaseException as exc:  # never leave a camera thread waiting forever
                logger.exception("pose round failed")
                for r in batch:
                    if not r.done.is_set():
                        r.error = exc
                        r.done.set()
            finally:
                with self._cv:
                    self._inflight = []

    def _process(self, batch: list[_PoseRequest]) -> None:
        t0 = time.perf_counter()
        try:
            sx, sy = pose.infer_parts(self.model, [r.x for r in batch], self._buffers)  # [v7] one copy
        except Exception:
            if len(batch) == 1:
                raise
            logger.warning("batched pose run of %d requests failed; retrying each alone", len(batch), exc_info=True)
            self.isolated_retries += 1
            for r in batch:
                try:
                    r.sx, r.sy = pose.infer(self.model, r.x)
                except Exception as exc:  # noqa: BLE001 - raised on that camera's thread only
                    self.failed_requests += 1
                    r.error = exc
                r.done.set()
            return
        n = sum(len(r.x) for r in batch)
        self.last_round_ms = (time.perf_counter() - t0) * 1000.0
        self.busy_ms += self.last_round_ms
        self.rounds += 1
        self.crops += n
        self.requests += len(batch)
        self.crops_hist[n] += 1
        pos = 0
        for r in batch:
            k = len(r.x)
            r.sx, r.sy = sx[pos : pos + k], sy[pos : pos + k]
            pos += k
            r.done.set()

    def stats(self) -> dict[str, Any]:
        return {
            "alive": self.alive,
            "max_crops": self.max_crops,
            "buckets": list(self.model.buckets),
            "rounds": self.rounds,
            "crops": self.crops,
            "mean_crops_per_round": round(self.crops / self.rounds, 2) if self.rounds else 0.0,
            "mean_cameras_per_round": round(self.requests / self.rounds, 2) if self.rounds else 0.0,
            "crops_hist": dict(sorted(self.crops_hist.items())),
            "last_round_ms": round(self.last_round_ms, 2),
            "oldest_pending_s": round(self.oldest_pending_s(), 3),
            "isolated_retries": self.isolated_retries,
            "failed_requests": self.failed_requests,
            "clients": len(self._clients),
        }


def attach_batched_pose(detector: Any, batcher: PoseBatcher) -> None:
    """Route a loaded detector's RTMPose call through the shared batcher (no-op for a pose-less detector)."""
    if getattr(detector, "_pose_model", None) is not None:
        detector.pose_infer = batcher.infer_fn(str(detector.camera_id))
