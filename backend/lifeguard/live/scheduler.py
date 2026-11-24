"""Batched GPU scheduler (plan Phase 4, fixes audit #6: one camera at a time on one GPU thread).

ONE scheduler thread owns the detector session. Each camera's analysis thread runs the unchanged
vision pipeline; when the detector stage reaches the model it calls ``BatchedDetModel.run``, which
letterboxes the frame on the camera's own thread and hands the tensor to ``BatchScheduler.infer``.

Each round the scheduler:
  1. waits for the first pending frame;
  2. gathers more until it has ``batch_max`` frames, or every registered camera has one pending,
     or ``batch_gather_ms`` has passed;
  3. makes ONE onnxruntime run over the stacked batch, through IOBinding (inputs bound from host
     memory, outputs to preallocated host buffers);
  4. hands every camera its own slice of the outputs.

Failure isolation: a malformed tensor fails only its own request. If a batched run raises, every
request of the round is retried on its own, so one bad frame cannot blank the other cameras.
A model with a static batch of 1 still works (one run per frame, same thread), so the original
single-batch export stays usable.

Stall detection: ``oldest_pending_s()`` is the age of the oldest unanswered request; the runtime
raises a GPU_SCHEDULER_STALLED FAULT when it passes ``scheduler_stall_s``, and a camera waiting
longer than that gets ``SchedulerStalledError`` (counted as a failed frame by the detector).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lifeguard.vision.inference.models import DetModel

logger = logging.getLogger(__name__)


PREFETCH_KEEP = 2  # [v4] prefetched detections a camera may have outstanding


class SchedulerStalledError(RuntimeError):
    pass


class SchedulerStoppedError(RuntimeError):
    pass


@dataclass
class _Request:
    x: np.ndarray
    client: str
    submitted: float
    done: threading.Event = field(default_factory=threading.Event)
    outputs: list[np.ndarray] | None = None
    error: BaseException | None = None
    started: bool = False


class BatchScheduler:
    def __init__(
        self,
        session: Any,
        batch_max: int = 8,
        gather_ms: float = 10.0,
        stall_s: float = 10.0,
        use_iobinding: bool = True,
        name: str = "gpu-scheduler",
    ) -> None:
        self.session = session
        inp = session.get_inputs()[0]
        self.input_name: str = inp.name
        self.output_names: list[str] = [o.name for o in session.get_outputs()]
        lead = inp.shape[0]
        self.dynamic_batch = not (isinstance(lead, int) and lead > 0)
        self.input_chw: tuple[Any, ...] = tuple(inp.shape[1:])
        self.batch_max = batch_max if self.dynamic_batch else 1
        if not self.dynamic_batch and batch_max > 1:
            logger.warning("detector has a static batch of %s: the scheduler runs one frame per GPU call", lead)
        self.gather_s = gather_ms / 1000.0
        self.stall_s = stall_s
        self.use_iobinding = use_iobinding and hasattr(session, "io_binding")
        self.name = name
        self._pending: deque[_Request] = deque()
        self._inflight: list[_Request] = []
        self._cv = threading.Condition()
        self._clients: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # statistics
        self.rounds = 0
        self.frames = 0
        self.isolated_retries = 0
        self.failed_requests = 0
        self.batch_hist: Counter[int] = Counter()
        self.last_round_at: float | None = None
        self.last_round_ms = 0.0
        self.busy_ms = 0.0  # [dynamic fps] total session time: GPU busy fraction = d(busy_ms) / d(wall ms)
        self.ms_ewma: dict[int, float] = {}

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
                req.error = SchedulerStoppedError("scheduler stopped")
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
    def infer(self, x: np.ndarray, client: str, timeout: float | None = None) -> list[np.ndarray]:
        """Run one (1, C, H, W) tensor through the next round; blocks the calling camera thread."""
        return self.wait(self.submit(x, client), client, timeout)

    def submit(self, x: np.ndarray, client: str) -> _Request:
        """[v4] Queue one (1, C, H, W) tensor (copied) for the next round WITHOUT waiting; `wait()` collects it.
        A camera uses this to start the next frame's detection while it finishes the current one."""
        if self._stop.is_set() or not self.alive:
            raise SchedulerStoppedError("scheduler is not running")
        req = _Request(np.array(x, dtype=np.float32, copy=True), client, time.monotonic())
        with self._cv:
            self._pending.append(req)
            self._cv.notify_all()
        return req

    def wait(self, req: _Request, client: str, timeout: float | None = None) -> list[np.ndarray]:
        """The outputs of a submitted request; the stall limit counts from its submission."""
        left = (self.stall_s if timeout is None else timeout) - (time.monotonic() - req.submitted)
        if not req.done.wait(max(0.0, left)):
            with self._cv:
                if not req.started and req in self._pending:
                    self._pending.remove(req)
            raise SchedulerStalledError(f"no GPU result for {client} within {self.stall_s:.1f} s")
        if req.error is not None:
            raise req.error
        if req.outputs is None:
            raise RuntimeError("scheduler answered without outputs")
        return req.outputs

    # ── scheduler side ────────────────────────────────────────────────────────
    def oldest_pending_s(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        with self._cv:
            reqs = list(self._pending) + list(self._inflight)
        return max((now - r.submitted for r in reqs), default=0.0)

    def stalled(self, now: float | None = None) -> bool:
        return (self._thread is not None and not self.alive and not self._stop.is_set()) or (
            self.oldest_pending_s(now) > self.stall_s
        )

    def _gather(self) -> list[_Request]:
        with self._cv:
            while not self._pending and not self._stop.is_set():
                self._cv.wait(0.5)
            if self._stop.is_set():
                return []
            deadline = time.monotonic() + self.gather_s
            while len(self._pending) < self.batch_max and not self._stop.is_set():
                waiting = {r.client for r in self._pending}
                if self._clients and self._clients <= waiting:
                    break  # every registered camera has a frame in this round
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._cv.wait(left)
            batch = [self._pending.popleft() for _ in range(min(self.batch_max, len(self._pending)))]
            for r in batch:
                r.started = True
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
                logger.exception("scheduler round failed")
                for r in batch:
                    if not r.done.is_set():
                        r.error = exc
                        r.done.set()
            finally:
                with self._cv:
                    self._inflight = []

    def _valid(self, req: _Request) -> bool:
        shape = req.x.shape
        ok = (
            len(shape) == 4
            and shape[0] == 1
            and all(not isinstance(d, int) or d <= 0 or d == s for d, s in zip(self.input_chw, shape[1:], strict=True))
        )
        if not ok:
            req.error = ValueError(f"tensor shape {shape} does not fit model input (1, {self.input_chw})")
            self.failed_requests += 1
            req.done.set()
        return ok

    def _process(self, batch: list[_Request]) -> None:
        batch = [r for r in batch if self._valid(r)]
        if not batch:
            return
        if not self.dynamic_batch:
            for r in batch:
                self._run_alone(r)
            return
        t0 = time.perf_counter()
        try:
            outs = self._run(np.concatenate([r.x for r in batch], axis=0))
        except Exception as exc:
            if len(batch) == 1:
                self._fail(batch[0], exc)
                return
            logger.warning("batched run of %d frames failed; retrying each frame alone", len(batch), exc_info=True)
            self.isolated_retries += 1
            for r in batch:
                self._run_alone(r)
            return
        self._account(len(batch), t0)
        for i, r in enumerate(batch):
            r.outputs = [np.ascontiguousarray(o[i : i + 1]) for o in outs]
            r.done.set()

    def _run_alone(self, r: _Request) -> None:
        t0 = time.perf_counter()
        try:
            r.outputs = list(self._run(r.x))
            self._account(1, t0)
        except Exception as exc:  # noqa: BLE001 - forwarded to (and raised on) that camera's thread only
            self._fail(r, exc)
            return
        r.done.set()

    def _fail(self, r: _Request, exc: BaseException) -> None:
        self.failed_requests += 1
        r.error = exc
        r.done.set()

    def _account(self, n: int, t0: float) -> None:
        ms = (time.perf_counter() - t0) * 1000.0
        self.rounds += 1
        self.frames += n
        self.batch_hist[n] += 1
        self.last_round_ms = ms
        self.busy_ms += ms
        self.last_round_at = time.monotonic()
        prev = self.ms_ewma.get(n)
        self.ms_ewma[n] = ms if prev is None else 0.9 * prev + 0.1 * ms

    def _run(self, x: np.ndarray) -> list[np.ndarray]:
        if self.use_iobinding:
            io = self.session.io_binding()
            io.bind_cpu_input(self.input_name, x)
            for name in self.output_names:
                io.bind_output(name)
            self.session.run_with_iobinding(io)
            return list(io.copy_outputs_to_cpu())
        return list(self.session.run(self.output_names, {self.input_name: x}))

    def warm(self, sizes: list[int]) -> None:
        """[Phase 11 WP5] One untimed run per batch size (the algorithm search for that shape) on the caller's
        thread, before start(): a capacity-cache hit warms the sizes the starting cameras will use."""
        h, w = (int(d) if isinstance(d, int) and d > 0 else 640 for d in self.input_chw[1:])
        c = int(self.input_chw[0]) if isinstance(self.input_chw[0], int) else 3
        for b in sizes:
            if 1 <= b <= self.batch_max:
                self._run(np.zeros((b, c, h, w), np.float32))

    def measure(self, sizes: list[int], repeats: int = 3) -> dict[int, float]:
        """Median latency (ms) of one run per batch size, on the caller's thread (before start())."""
        h, w = (int(d) if isinstance(d, int) and d > 0 else 640 for d in self.input_chw[1:])
        c = int(self.input_chw[0]) if isinstance(self.input_chw[0], int) else 3
        out: dict[int, float] = {}
        rng = np.random.default_rng(0)
        for b in sizes:
            if b > self.batch_max:
                continue
            x = rng.normal(0.0, 1.0, (b, c, h, w)).astype(np.float32)
            self._run(x)  # warm: algorithm search for this shape
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                self._run(x)
                times.append((time.perf_counter() - t0) * 1000.0)
            out[b] = float(np.median(times))
        return out

    def stats(self) -> dict[str, Any]:
        mean_batch = self.frames / self.rounds if self.rounds else 0.0
        age = None if self.last_round_at is None else round(time.monotonic() - self.last_round_at, 3)
        return {
            "alive": self.alive,
            "dynamic_batch": self.dynamic_batch,
            "batch_max": self.batch_max,
            "rounds": self.rounds,
            "frames": self.frames,
            "mean_batch": round(mean_batch, 3),
            "batch_hist": dict(sorted(self.batch_hist.items())),
            "last_round_ms": round(self.last_round_ms, 2),
            "last_round_age_s": age,
            "oldest_pending_s": round(self.oldest_pending_s(), 3),
            "isolated_retries": self.isolated_retries,
            "failed_requests": self.failed_requests,
            "clients": len(self._clients),
            "iobinding": self.use_iobinding,
        }


class BatchedDetModel(DetModel):
    """A camera's view of the shared detector: same preprocessing and metadata, inference via the scheduler.

    The vision code (decode, run_detector_people) sees a DetModel and is unchanged - this keeps the
    Phase 3 parity of everything around the model call.
    """

    def __init__(self, base: DetModel, scheduler: BatchScheduler, client: str) -> None:
        # Deliberately NOT calling DetModel.__init__: share the loaded session + metadata.
        self.__dict__.update(base.__dict__)
        self._tls = threading.local()  # own letterbox buffers per camera thread
        self.base = base
        self.scheduler = scheduler
        self.client = client
        self.calls = 0
        # [v4] prefetched detections: frame key -> (request, ratio, frame), oldest first (the camera's own thread only).
        # The entry holds the frame itself: while it exists its buffer cannot be freed and its address reused by
        # another frame, so a key can never match different pixels.
        self._prefetched: dict[tuple[int, tuple[int, ...], tuple[int, ...]], tuple[Any, float, np.ndarray]] = {}
        self.prefetch_hits = 0
        self.prefetch_misses = 0  # prefetched, never asked for (the frame was dropped / its detection not needed)

    @staticmethod
    def _key(frame: np.ndarray) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        """Identity of the pixels: a full-frame view (frame[0:h, 0:w]) has the same key as the frame."""
        return int(frame.__array_interface__["data"][0]), tuple(frame.shape), tuple(frame.strides)

    def prefetch(self, frame: np.ndarray) -> None:
        """[v4] Start this frame's detection now; run() on the same pixels later collects it instead of waiting for a
        new round. The camera calls it for the NEXT queued frame before finishing the current one, so the GPU round
        of frame n+1 overlaps the pose / tracking / behaviour of frame n (the per-camera chain was latency-bound:
        ~33 ms detector wait + ~15 ms pose wait back to back). Outputs are the ones a later run() would get."""
        key = self._key(frame)
        if key in self._prefetched:
            return
        x, ratio = self.preprocess(frame)
        self._prefetched[key] = (self.scheduler.submit(x, self.client), ratio, frame)
        while len(self._prefetched) > PREFETCH_KEEP:  # never collected: forget the oldest
            self._prefetched.pop(next(iter(self._prefetched)))
            self.prefetch_misses += 1

    def run(self, frame: np.ndarray) -> tuple[list[np.ndarray], float]:
        self.calls += 1
        hit = self._prefetched.pop(self._key(frame), None) if self._prefetched else None
        if hit is not None:
            req, ratio, _ = hit
            self.prefetch_hits += 1
            return self.scheduler.wait(req, self.client), ratio
        x, ratio = self.preprocess(frame)
        return self.scheduler.infer(x, self.client), ratio


def attach_batched(detector: Any, scheduler: BatchScheduler) -> bool:
    """Load `detector` (a MultiPersonDetector) and route its model calls through `scheduler`.

    Returns False (detector left unavailable) when it cannot load - the camera then goes
    NOT_MONITORED with a FAULT on its first frame, exactly like the single-camera path.
    """
    detector.load()
    if not detector.healthy:
        return False
    base = detector._det_model
    if isinstance(base, BatchedDetModel):
        return True
    detector._det_model = BatchedDetModel(base, scheduler, str(detector.camera_id))
    return True
