"""Capacity and admission control (plan Phase 4).

At startup the runtime measures the detector's latency for every batch size up to ``batch_max``
on the real GPU, plus the per-frame CPU cost around it (letterbox, decode, tracking, behaviour).
From that profile it computes, for N cameras, the frame rate at which EVERY camera is analysed:

    gpu_cycle_ms(N) = (N // B) * lat(B) + lat(N % B)      one round per B cameras
    cycle_ms(N)     = gpu_cycle_ms(N) + N * overhead_ms    CPU work assumed NOT to overlap (conservative)
    analysed_fps(N) = 1000 / cycle_ms(N)

[v4] With a GPU provider (``overlap=True``) every camera's CPU work runs on its own thread WHILE the GPU serves
the others, so the serial sum above over-states the cycle about 2x (measured on the dev box, RTX 4060 laptop,
TensorRT FP16, 4 site cameras with admission bypassed: 79 analysed fps in total, i.e. a 50.6 ms round of 4, where
the serial model predicted 82 ms and admitted only 2 cameras). The overlap model is

    gpu_ms(N)   = gpu_cycle_ms(N) + pose_gpu_ms(N)          detector + ONE batched pose call for every camera
    chain_ms(N) = overhead_ms + pose_ms + lat(min(N, B))    one camera's own critical path
    cycle_ms(N) = max(gpu_ms(N), chain_ms(N)) + gather_ms   + the scheduler's gather window

pose_gpu_ms(N) is the measured session time of one pose call over N * POSE_CAPACITY_PEOPLE crops (live/pose_batcher.py
batches every camera's crops into one call, padded to a bucket). [v4 fix] It used to be N * pose_ms, pose_ms being ONE
camera's whole pose latency (crop preparation + its own call): pose then counted ~4x (startup 2026-10-15: 4 x 10.1 ms
charged where the batched call of 24 crops takes ~7-10 ms), and a fourth camera that measured clean was refused.
A CPU provider keeps the serial model: there the "GPU" work competes with the camera threads for the same cores.

This is a PRIOR: pose cost grows with the swimmers in view, which no startup probe can know. Measured on the same box
(2026-10-15, pool mode, every camera 15 fps; with the next-frame detection prefetch of live/scheduler.py):
  4 cameras, 0-2 people each ..... 15.0 fps, 0 dropped          3 cameras, 8-11 swimmers each (1080p) ... 15.0 fps
  4 cameras, 6-9 swimmers each ... 12-13 fps, frames dropped    4 standard-CCTV cameras (no pose) ....... 15.0 fps
So the runtime also decides from MEASUREMENT (live/runtime.py): a camera LAGGING for LAG_FAULT_S raises
CAMERA_LAGGING, and while any camera lags no further camera is admitted (CapacityExceededError(behind=...)).

Camera N+1 is refused (API: HTTP 409 with the computed limit) when
``analysed_fps(N+1) * HEADROOM < required fps`` - cameras never silently lag. The required fps is
TARGET_ANALYSIS_FPS, or the real source fps of the fastest running camera if that is higher. HEADROOM keeps
a 10 % margin for decode jitter so the admitted set stays at ``live_analysed_fraction >= 0.99``.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

HEADROOM = 0.9
# Tracking + behaviour cost per frame that the synthetic startup probe cannot exercise (no real people to
# track). Phase 4 smoke on the dev box (1 camera, crowded 1080p clip web_lane_pool_1080p): the worker's
# whole frame took 22.4 ms of which 14.6 ms GPU -> 7.8 ms CPU, while the probe alone measures ~3 ms.
# This allowance closes that gap with a margin.
TRACK_BEHAVIOUR_MS = 5.0
# [docs/proposals/001,002] People per camera frame assumed by the pose cost for the UNIFORM-AVERAGE
# admission model (top-down pose: cost scales with people). Raised from 6 - low even on average for a
# pool camera - to 10; the SEPARATE, stricter skewed-load check below is what actually budgets for one
# camera carrying far more than the average (a localised crowd), rather than pushing this one number up
# to the brief's full people-count target and over-provisioning every camera, all the time, to guard
# against a load the uniform model cannot represent correctly anyway (see cycle_ms_skewed).
POSE_CAPACITY_PEOPLE = 10
# [docs/proposals/002] Worst-case people in ONE camera's view during a localised spike - the brief's "up
# to 20 people... across all 4 cameras combined" could in principle all cluster in front of one camera.
SPIKE_CAPACITY_PEOPLE = 20


class CapacityExceededError(RuntimeError):
    def __init__(
        self, requested: int, limit: int, analysed_fps: float, target_fps: float, *, behind: str | None = None
    ) -> None:
        if behind is None:
            msg = (
                f"camera {requested} would be analysed at {analysed_fps:.1f} fps (< target {target_fps:g} fps "
                f"with {round((1 - HEADROOM) * 100)} % headroom); this server can monitor at most {limit} camera(s)"
            )
        else:  # [v4] measured, not modelled: a running camera already cannot keep up
            msg = (
                f"camera {behind!r} is already falling behind ({analysed_fps:.1f} of {target_fps:g} fps analysed): "
                f"no camera is added while one cannot keep up"
            )
        super().__init__(msg)
        self.requested = requested
        self.limit = limit
        self.analysed_fps = analysed_fps
        self.target_fps = target_fps
        self.behind = behind

    def detail(self) -> dict[str, Any]:
        return {
            "code": "capacity_exceeded",
            "limit": self.limit,
            "requested": self.requested,
            "analysed_fps": round(self.analysed_fps, 2),
            "target_fps": self.target_fps,
            "behind": self.behind,
        }


@dataclass(frozen=True)
class CapacityProfile:
    batch_ms: dict[int, float]  # measured detector latency per batch size
    overhead_ms: float  # per-frame CPU cost outside the GPU call
    batch_max: int
    measured_at: float = field(default_factory=time.time)
    # [Phase 8] detector calls per analysed frame: 1, or 1 + tiles with DET_TILING (full frame + every tile).
    # Without it admission would count one call per frame and over-admit cameras when tiling is on.
    calls_per_frame: int = 1
    # [v4] overlap model (module docstring): pose cost per frame on the GPU, the scheduler's gather window, and
    # whether CPU work overlaps the GPU (GPU providers only). pose_ms is NOT part of overhead_ms in this model.
    pose_ms: float = 0.0
    # [v4] pose session time per batch size (bucketed: 1, 4, 8, 16, 32): the GPU share of pose in the overlap model
    pose_batch_ms: dict[int, float] = field(default_factory=dict)
    pose_people: int = POSE_CAPACITY_PEOPLE  # crops per camera frame assumed by admission
    gather_ms: float = 0.0
    overlap: bool = False

    def __post_init__(self) -> None:
        if not self.batch_ms or 1 not in self.batch_ms:
            raise ValueError("capacity profile needs at least the batch-1 latency")
        # [Phase 6] a bad pose table fails closed (refused at load) instead of under-estimating the pose cost
        for b, ms in self.pose_batch_ms.items():
            if int(b) <= 0 or not (float(ms) > 0.0):
                raise ValueError(f"invalid pose batch entry {b}: {ms} ms")

    def latency_ms(self, b: int) -> float:
        """Latency of one batch of b frames (linear interpolation between measured sizes)."""
        if b <= 0:
            return 0.0
        if b in self.batch_ms:
            return self.batch_ms[b]
        sizes = sorted(self.batch_ms)
        lo = max((s for s in sizes if s < b), default=sizes[0])
        hi = min((s for s in sizes if s > b), default=None)
        if hi is None:  # beyond the largest measured size: scale per frame
            return self.batch_ms[lo] * b / lo
        frac = (b - lo) / (hi - lo)
        return self.batch_ms[lo] + frac * (self.batch_ms[hi] - self.batch_ms[lo])

    def cycle_ms(self, n: int) -> float:
        if n <= 0:
            return 0.0
        b = max(1, self.batch_max)
        calls = n * max(1, self.calls_per_frame)
        det = (calls // b) * self.latency_ms(b) + self.latency_ms(calls % b)
        if not self.overlap:
            return det + n * (self.overhead_ms + self.pose_ms)
        gpu = det + self.pose_gpu_ms(n)
        chain = self.overhead_ms + self.pose_ms + self.latency_ms(min(calls, b))
        return max(gpu, chain) + self.gather_ms

    def pose_gpu_ms(self, n: int) -> float:
        """GPU time of the batched pose call(s) for n cameras' crops; n * pose_ms when no batch table was measured."""
        if self.pose_people <= 0:
            return n * self.pose_ms
        return self._pose_gpu_ms_for_crops(n * self.pose_people)

    def _pose_gpu_ms_for_crops(self, crops: int) -> float:
        """[Phase 6] With no measured table (pose disabled, or not measured), the UNBATCHED cost: one camera's
        pose_ms per pose_people crops, rounded up - the uniform path stays n * pose_ms exactly and a spike is never
        cheaper than linear. (Was an IndexError on an empty table, from cycle_ms_skewed / as_dict.)"""
        if crops <= 0:
            return 0.0
        if not self.pose_batch_ms:
            return math.ceil(crops / max(1, self.pose_people)) * self.pose_ms
        sizes = sorted(self.pose_batch_ms)
        biggest = sizes[-1]
        full, rest = divmod(crops, biggest)
        ms = full * self.pose_batch_ms[biggest]
        if rest:
            ms += self.pose_batch_ms[min(s for s in sizes if s >= rest)]
        return ms

    def analysed_fps(self, n: int) -> float:
        cyc = self.cycle_ms(n)
        return float("inf") if cyc <= 0 else 1000.0 / cyc

    def cycle_ms_skewed(self, n: int, spike_people: int) -> float:
        """[docs/proposals/002] Worst case where ONE of the n admitted cameras has `spike_people` (a
        localised crowd - up to SPIKE_CAPACITY_PEOPLE, see below) and the rest have `self.pose_people`.
        cycle_ms(n) alone only models every camera at the uniform average; a real spike concentrated on
        one camera is a different, and worse, load than that average - this checks it explicitly rather
        than only relying on the live CAMERA_LAGGING fault to notice after the fact."""
        if n <= 0:
            return 0.0
        b = max(1, self.batch_max)
        calls = n * max(1, self.calls_per_frame)
        det = (calls // b) * self.latency_ms(b) + self.latency_ms(calls % b)
        # the spiked camera's own pose cost scales with its crop count (the rest stay at the baseline)
        spike_pose_ms = self.pose_ms * (spike_people / self.pose_people) if self.pose_people else self.pose_ms
        if not self.overlap:
            return det + max(0, n - 1) * (self.overhead_ms + self.pose_ms) + (self.overhead_ms + spike_pose_ms)
        crops = max(0, n - 1) * self.pose_people + spike_people
        gpu = det + self._pose_gpu_ms_for_crops(crops)
        chain = self.overhead_ms + spike_pose_ms + self.latency_ms(min(calls, b))
        return max(gpu, chain) + self.gather_ms

    def analysed_fps_skewed(self, n: int, spike_people: int) -> float:
        cyc = self.cycle_ms_skewed(n, spike_people)
        return float("inf") if cyc <= 0 else 1000.0 / cyc

    def limit(self, target_fps: float, hard_max: int = 256) -> int:
        """Largest N whose every camera is analysed at >= target_fps (with HEADROOM)."""
        n = 0
        while n < hard_max and self.analysed_fps(n + 1) * HEADROOM >= target_fps:
            n += 1
        return n

    def limit_skewed(self, target_fps: float, spike_people: int, hard_max: int = 256) -> int:
        """Largest N that still clears target_fps even with one camera spiked to `spike_people`."""
        n = 0
        while n < hard_max and self.analysed_fps_skewed(n + 1, spike_people) * HEADROOM >= target_fps:
            n += 1
        return n

    def admit(self, requested: int, target_fps: float) -> None:
        """Raise CapacityExceededError if `requested` concurrent cameras would fall below target."""
        if requested <= 0:
            return
        fps = self.analysed_fps(requested)
        if fps * HEADROOM < target_fps:
            raise CapacityExceededError(requested, self.limit(target_fps), fps, target_fps)

    def admit_skewed(self, requested: int, spike_people: int, target_fps: float) -> None:
        """Raise CapacityExceededError if `requested` cameras could not sustain target_fps with one of
        them spiked to `spike_people` - admit() alone only checks the uniform-average case."""
        if requested <= 0:
            return
        fps = self.analysed_fps_skewed(requested, spike_people)
        if fps * HEADROOM < target_fps:
            raise CapacityExceededError(requested, self.limit_skewed(target_fps, spike_people), fps, target_fps)

    def as_dict(self, target_fps: float | None = None) -> dict[str, Any]:
        d = asdict(self)
        d["batch_ms"] = {str(k): round(v, 2) for k, v in sorted(self.batch_ms.items())}
        d["overhead_ms"] = round(self.overhead_ms, 2)
        d["pose_ms"] = round(self.pose_ms, 2)
        d["pose_batch_ms"] = {str(k): round(v, 2) for k, v in sorted(self.pose_batch_ms.items())}
        if target_fps is not None:
            d["target_fps"] = target_fps
            d["limit"] = self.limit(target_fps)
            d["limit_skewed"] = self.limit_skewed(target_fps, SPIKE_CAPACITY_PEOPLE)
            d["headroom"] = HEADROOM
        return d


def measure_overhead_ms(model: Any, frame_hw: tuple[int, int] = (1080, 1920), repeats: int = 5) -> float:
    """CPU cost per frame: letterbox + normalise a `frame_hw` frame and decode a detector output, plus
    TRACK_BEHAVIOUR_MS. `model` is a DetModel (its preprocess is what each camera thread runs). The runtime passes
    settings.capacity_frame_size [Phase 11 WP4]."""
    from lifeguard.vision.inference.decode import decode_detections

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (*frame_hw, 3), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    in_h, in_w = model.model_input_size
    ncls = len(getattr(model, "_det_names", None) or []) or 80
    fake = rng.random((1, (in_h // 8) * (in_w // 8) * 21 // 16, 4 + ncls), dtype=np.float32) * 0.05
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _x, ratio = model.preprocess(frame)
        decode_detections([fake], ratio, (in_h, in_w), 0.1, 0.45, "xyxy_scores")
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)) + TRACK_BEHAVIOUR_MS


def measure_pose_batch_ms(model: Any, repeats: int = 5) -> dict[int, float]:
    """[v4] Session time of one pose call per bucket size (zeros: the GPU cost does not depend on the pixels)."""
    w, h = model.model_input_size
    out: dict[int, float] = {}
    for b in model.buckets:
        x = np.zeros((b, 3, h, w), np.float32)
        model.session.run(model.output_names, {model.input_name: x})
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model.session.run(model.output_names, {model.input_name: x})
            times.append((time.perf_counter() - t0) * 1000.0)
        out[int(b)] = float(np.median(times))
    return out


def measure_pose_ms(model: Any, frame_hw: tuple[int, int] = (1080, 1920), repeats: int = 3) -> float:
    """Per-frame cost of top-down RTMPose (inference/pose.py) on a `frame_hw` frame with POSE_CAPACITY_PEOPLE
    people, every one posed (no POSE_STRIDE credit taken: admission must hold when every track is at risk). It runs
    outside the batched detector call, so it is added to the per-frame overhead (no overlap assumed)."""
    from lifeguard.vision.inference import pose

    h, w = frame_hw
    rng = np.random.default_rng(1)
    frame = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (5, 5), 0)
    bw, bh = max(8, w // 16), max(16, h // 6)
    k = POSE_CAPACITY_PEOPLE
    xs = np.array([(i + 1) * w / (k + 1) for i in range(k)], np.float32)
    boxes = np.stack([xs - bw / 2, np.full(k, h / 2 - bh / 2), xs + bw / 2, np.full(k, h / 2 + bh / 2)], 1)
    pose.run_pose(model, frame, boxes.astype(np.float32))  # warm this thread's buffers
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        pose.run_pose(model, frame, boxes.astype(np.float32))
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))
