"""The live runtime: capture -> per-camera analysis -> ONE batched GPU scheduler (plan Phase 4).

    rt = LiveRuntime(settings, det_model_path, provider, fault_sink=..., fire_sink=...)
    rt.start()                      # load the model, measure capacity, start the scheduler + monitor
    rt.start_camera(spec)           # admission-controlled: CapacityExceededError -> HTTP 409
    rt.status() / rt.heartbeat()    # API status, cross-process heartbeat payload
    rt.stop()

The monitor thread (every MONITOR_PERIOD_S) raises FAULTs - edge-triggered, once per episode:
  * CAMERA_STALLED   last frame older than STALL_S -> FAULT + reconnect request;
  * GPU_SCHEDULER_STALLED  a frame waited longer than scheduler_stall_s for the GPU;
and tracks LAGGING cameras (oldest queued frame older than lag_threshold_s), reported in the API
heartbeat for the supervisor.

[Phase 5] Cameras are independent (owner directive 2026-11-13): no cross-camera identity, no alert deferred to
another camera, no detection region set by another camera. Network cameras still stamp frames with one runtime-wide
clock (``stream_clock``).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from lifeguard.alerts.media import Recorder
from lifeguard.core.config import Settings
from lifeguard.core.startup import TIMELINE, StartupTimeline
from lifeguard.live import capacity_cache
from lifeguard.live.capacity import (
    POSE_CAPACITY_PEOPLE,
    SPIKE_CAPACITY_PEOPLE,
    CapacityExceededError,
    CapacityProfile,
    measure_overhead_ms,
    measure_pose_batch_ms,
    measure_pose_ms,
)
from lifeguard.live.capture import CaptureWorker, Opener, force_rtsp_tcp
from lifeguard.live.frames import FrameQueue
from lifeguard.live.policy import audible, zone_states
from lifeguard.live.pose_batcher import PoseBatcher, attach_batched_pose
from lifeguard.live.rate import RateController
from lifeguard.live.scheduler import BatchScheduler, attach_batched
from lifeguard.live.worker import CameraWorker
from lifeguard.vision.faults import NOT_MONITORED, Fault
from lifeguard.vision.inference.session import GPU_PROVIDERS, short_name
from lifeguard.vision.scene import scene_classifier
from lifeguard.vision.scene.hysteresis import AlertFire

logger = logging.getLogger(__name__)

CAMERA_STALLED = "CAMERA_STALLED"
# [v4] a camera analysed slower than it streams for LAG_FAULT_S: its oldest frames are being dropped. No admission
# model can know how many swimmers a camera will see, so the runtime says so from MEASUREMENT (and refuses more cameras)
CAMERA_LAGGING = "CAMERA_LAGGING"
LAG_FAULT_S = 10.0
LAG_CLEAR_S = 10.0
GPU_SCHEDULER_STALLED = "GPU_SCHEDULER_STALLED"
MONITOR_PERIOD_S = 0.5
# [2026-11-18] capacity spot checks run while the cameras are pre-opening; one noisy take (5.02 vs 4.44 ms, overhead
# 15.6 vs 8.5 ms) re-measured everything under that load and refused all 4 cameras. Retry the spot check first.
SPOT_RETRIES = 2
SPOT_RETRY_PAUSE_S = 0.5
QUEUE_FPS_GUESS = 30.0  # queue length = live_queue_s * this (before the source fps is known)

LiveFaultSink = Callable[[Fault, str | None], None]  # (fault, camera_id or None for system)


class LiveFireSink(Protocol):
    """(camera_id, fire, audible, frame image[, muted_reason]) - muted_reason only when an alert does not sound for
    another reason than its zone schedule ([Phase 11 WP2] "deferred to <camera>")."""

    def __call__(
        self, camera_id: str, fire: AlertFire, audible: bool, image: Any, muted_reason: str | None = None
    ) -> None: ...


LiveClearSink = Callable[[str, str], None]  # (fault code, source id)


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    name: str
    source: str  # decrypted URL or absolute file path - never logged or returned
    source_type: str  # "rtsp" | "file"
    mode: str = "Medium"
    geometry_loader: Callable[[int, int], Any] | None = None  # (w, h) -> ZoneGeometry | None
    replay: Any = None  # [Phase 11 WP6] live/replay.ReplayFeed (evaluation only; source_type "replay")
    scene_type: str = "pool"  # [v4] "pool" | "standard_cctv" (db Camera.scene_type)


class _Camera:
    def __init__(self, spec: CameraSpec, queue: FrameQueue, capture: CaptureWorker, worker: CameraWorker) -> None:
        self.spec = spec
        self.queue = queue
        self.capture = capture
        self.worker = worker
        self.stalled = False
        self.healthy_seen = False  # first healthy check clears a stall FAULT a previous process left open
        self.lagging = False
        self.stall_faults = 0
        self.geometry: Any = None
        # [v4] sustained lag (CAMERA_LAGGING) + measured analysis rate (monitor: analysed frames per second)
        self.lag_since: float | None = None
        self.ok_since: float | None = None
        self.lag_fault = False
        self.analysed_fps: float | None = None
        self._rate_ref: tuple[float, int] | None = None


class LiveRuntime:
    def __init__(
        self,
        settings: Settings,
        det_model_path: str,
        provider: str,
        fault_sink: LiveFaultSink,
        fire_sink: LiveFireSink,
        *,
        opener: Opener | None = None,
        realtime_files: bool = True,
        clear_sink: LiveClearSink | None = None,
    ) -> None:
        self.clear_sink = clear_sink
        self.recorders: dict[str, Recorder] = {}
        self.settings = settings
        self.det_model_path = det_model_path
        self.provider = provider
        self.fault_sink = fault_sink
        self.fire_sink = fire_sink
        self.opener = opener
        self.realtime_files = realtime_files
        self.scheduler: BatchScheduler | None = None
        self.pose_batcher: PoseBatcher | None = None  # RTMPose crops of every camera, one call per round
        self._pose_model: Any = None
        self.profile: CapacityProfile | None = None
        self.started_at: float | None = None
        self._cams: dict[str, _Camera] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._monitor: threading.Thread | None = None
        self.scheduler_stalled = False
        # [dynamic fps] live analysis rate between target_analysis_fps (floor) and max_analysis_fps (live/rate.py)
        self.rate = RateController(settings.target_analysis_fps, settings.max_analysis_fps)
        self.gpu_busy: float | None = None  # detector + pose session ms per wall ms (EWMA over the monitor passes)
        self._gpu_ref: tuple[float, float] | None = None
        self.active_provider: str | None = None
        self.start_error: str | None = None
        self.rtsp: dict[str, Any] | None = None  # [Phase 11 WP4] force_rtsp_tcp() result
        # [Phase 11 WP5] startup timeline (process-wide) + where the capacity table came from (cache / measured)
        self.timeline: StartupTimeline = TIMELINE
        self.capacity_info: dict[str, Any] = {}
        self._cache_key: dict[str, Any] | None = None  # [Phase 6] set by _batch_latencies when the cache is usable
        self._cache_cpu: dict[str, Any] | None = None  # the cached CPU costs after a GPU cache hit
        # [Phase 11] one stream clock for every network camera
        self.clock_origin = time.monotonic()
        self._scene_types: dict[str, str] = {}  # [v4] camera id -> current scene type (admin may change it live)

    def stream_clock(self, now: float) -> float:
        """Runtime-wide stream time of a frame decoded at monotonic `now` (same origin for every camera)."""
        return now - self.clock_origin

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self, expected_cameras: int = 1) -> None:
        """Load the model, determine capacity, start the scheduler + monitor. `expected_cameras` = the enabled
        cameras about to be started: on a capacity-cache hit batch sizes 2..that are warmed before admission."""
        from lifeguard.vision.constants import DET_INPUT_SIZE
        from lifeguard.vision.inference import models

        bm = self.settings.batch_max
        base = models.load_detector_model(
            self.det_model_path, self.provider, DET_INPUT_SIZE, trt_batch=(1, min(4, bm), bm)
        )
        models.warmup(base)
        s = self.settings
        self.rtsp = force_rtsp_tcp()  # [Phase 11 WP4] before any camera opens
        sched = BatchScheduler(base.session, s.batch_max, s.batch_gather_ms, s.scheduler_stall_s)
        calls = detector_calls_per_frame(base, s.capacity_frame_size)
        t0 = time.perf_counter()
        batch_ms = self._batch_latencies(base, sched, calls, expected_cameras)
        self.capacity_info["capacity_phase_s"] = round(time.perf_counter() - t0, 3)
        # [Phase 11 WP4] the per-frame CPU cost is measured on a frame of the cameras' size (was full HD)
        frame_hw = (s.capacity_frame_size[1], s.capacity_frame_size[0])
        overhead, pose_ms, pose_batch_ms = self._cpu_costs(base, frame_hw, batch_ms)
        if pose_ms is not None:
            self.capacity_info["pose_ms"] = round(pose_ms, 2)
        # [v4] CPU work overlaps the GPU only on a GPU provider (capacity.py docstring)
        active = short_name(str(base.session.get_providers()[0]))
        self.active_provider = active  # [v4] what really runs ("auto" resolves to tensorrt / cuda / ... here)
        self.profile = CapacityProfile(
            batch_ms,
            overhead,
            sched.batch_max,
            calls_per_frame=calls,
            pose_ms=pose_ms or 0.0,
            pose_batch_ms=pose_batch_ms,
            pose_people=POSE_CAPACITY_PEOPLE,
            gather_ms=s.batch_gather_ms,
            overlap=active in GPU_PROVIDERS,
        )
        self.timeline.mark("capacity", **self.capacity_info)
        logger.info(
            "live capacity (%s): %s -> at most %d camera(s) at %g fps",
            self.capacity_info.get("source"),
            self.profile.as_dict(),
            self.profile.limit(s.target_analysis_fps),
            s.target_analysis_fps,
        )
        sched.start()
        self.scheduler = sched
        if self._pose_model is not None:
            pb = PoseBatcher(self._pose_model, gather_ms=s.batch_gather_ms / 2, stall_s=s.scheduler_stall_s)
            pb.start()
            self.pose_batcher = pb
        self._clear(GPU_SCHEDULER_STALLED, "system")  # left open by a previous process?
        self.started_at = time.monotonic()
        self._stop.clear()
        self._monitor = threading.Thread(target=self._monitor_loop, name="live-monitor", daemon=True)
        self._monitor.start()

        self.timeline.mark("runtime_started")

    @staticmethod
    def _pose_model_path() -> str | None:
        """The pose model every camera will run, or None when pose is off."""
        from lifeguard.vision import config as vision_config
        from lifeguard.vision.constants import POSE_ENABLED

        path = vision_config.current().pose_model_path
        return path if (POSE_ENABLED and path) else None

    def _load_pose_model(self) -> Any:
        from lifeguard.vision.constants import POSE_INPUT_RGB
        from lifeguard.vision.inference import models

        self._pose_model = None
        path = self._pose_model_path()
        if path is None:
            return None
        model = models.load_pose_model(path, self.provider, POSE_INPUT_RGB)
        models.warmup(model)
        self._pose_model = model
        return model

    def _cpu_costs(
        self, base: Any, frame_hw: tuple[int, int], batch_ms: dict[int, float]
    ) -> tuple[float, float | None, dict[int, float]]:
        """(overhead_ms, pose_ms or None, pose_batch_ms) for admission. [Phase 6] On a capacity-cache hit the overhead
        is measured as a spot check against the cached value (capacity_cache tolerances); within them every cached CPU
        cost is reused, so a restart admits exactly what the measured start did. Otherwise they are measured and
        written to the cache with the batch table."""
        s = self.settings
        model = self._load_pose_model()
        fixed = s.capacity_overhead_ms is not None
        overhead = s.capacity_overhead_ms if s.capacity_overhead_ms is not None else measure_overhead_ms(base, frame_hw)
        cached = self._cache_cpu
        if cached is not None and not fixed:
            # [dynamic fps / 2026-11-18] a spot check taken while the cameras are still opening (pre-open runs in
            # parallel) can read 2x slow and refuse every camera: retry before giving up on the cache
            for _ in range(SPOT_RETRIES):
                if capacity_cache.spot_ok(overhead, cached["overhead_ms"]):
                    break
                time.sleep(SPOT_RETRY_PAUSE_S)
                overhead = measure_overhead_ms(base, frame_hw)
        if cached is not None:
            if fixed or capacity_cache.spot_ok(overhead, cached["overhead_ms"]):
                self.capacity_info["cpu"] = "cache"
                if not fixed:
                    self.capacity_info["overhead_spot_ms"] = round(overhead, 2)
                    overhead = cached["overhead_ms"]
                return overhead, cached["pose_ms"], cached["pose_batch_ms"]
            self.capacity_info = {
                **self.capacity_info,
                "source": "measured",
                "reason": f"spot_check overhead {overhead:.2f} ms vs cached {cached['overhead_ms']:.2f} ms",
                "batch_ms": "cache",  # the GPU spot check passed
            }
        elif self.capacity_info.get("source") == "cache":  # a GPU hit on a file without CPU costs
            self.capacity_info = {**self.capacity_info, "source": "measured", "reason": "cpu_missing"}
        pose_ms = measure_pose_ms(model, frame_hw) if model is not None else None
        if not fixed:  # [2026-11-18] a fresh measurement: the quieter of two takes (start-up opens add CPU noise)
            time.sleep(SPOT_RETRY_PAUSE_S)
            overhead = min(overhead, measure_overhead_ms(base, frame_hw))
            if model is not None and pose_ms is not None:
                pose_ms = min(pose_ms, measure_pose_ms(model, frame_hw))
        pose_batch_ms = measure_pose_batch_ms(model) if model is not None else {}
        if self._cache_key is not None:
            cpu = {"overhead_ms": overhead, "pose_ms": pose_ms, "pose_batch_ms": pose_batch_ms}
            path = s.data_dir / capacity_cache.FILE_NAME
            self.capacity_info["cache_written"] = capacity_cache.save(path, self._cache_key, batch_ms, cpu=cpu)
        return overhead, pose_ms, pose_batch_ms

    def _batch_latencies(self, base: Any, sched: BatchScheduler, calls: int, expected_cameras: int) -> dict[int, float]:
        """[Phase 11 WP5] The GPU batch-latency table: from the capacity cache after a passing batch-1 spot check
        (then batch sizes 2..expected cameras are warmed), else the full 1..batch_max sweep, written back to the
        cache. Rules pre-registered in eval/site/levers_phase11.json "gates_wp5". [Phase 6] The cache is written by
        _cpu_costs, once the CPU costs are known."""
        s = self.settings
        sizes = list(range(1, sched.batch_max + 1))
        path = s.data_dir / capacity_cache.FILE_NAME
        key: dict[str, Any] | None = None
        reason = "disabled"
        if s.capacity_cache:
            try:
                session = base.session
                active = str(session.get_providers()[0])
                device = capacity_cache.device_identity(short_name(active))
                if device is None:
                    reason = "no_device_identity"
                else:
                    key = capacity_cache.cache_key(
                        model_path=self.det_model_path,
                        input_hw=tuple(base.model_input_size),
                        provider=self.provider,
                        active_provider=active,
                        provider_options=dict(session.get_provider_options().get(active, {})),
                        iobinding=sched.use_iobinding,
                        dynamic_batch=sched.dynamic_batch,
                        batch_max=sched.batch_max,
                        calls_per_frame=calls,
                        frame_size=s.capacity_frame_size,
                        device=device,
                        pose_model_path=self._pose_model_path(),
                    )
            except Exception:  # the cache must never block startup: measure instead
                logger.warning("capacity cache key failed - measuring", exc_info=True)
                key, reason = None, "key_error"
        self._cache_key, self._cache_cpu = key, None
        if key is not None:
            cached, reason = capacity_cache.load(path, key, s.capacity_cache_max_age_h)
            if cached is not None:
                spot = sched.measure([1], repeats=capacity_cache.SPOT_REPEATS)[1]
                for _ in range(SPOT_RETRIES):  # [2026-11-18] start-up noise: retry before a full re-measure
                    if capacity_cache.spot_ok(spot, cached[1]):
                        break
                    time.sleep(SPOT_RETRY_PAUSE_S)
                    spot = sched.measure([1], repeats=capacity_cache.SPOT_REPEATS)[1]
                if capacity_cache.spot_ok(spot, cached[1]):
                    warm = list(range(2, min(max(1, expected_cameras), sched.batch_max) + 1))
                    sched.warm(warm)
                    self.capacity_info = {
                        "source": "cache",
                        "spot_ms": round(spot, 2),
                        "cached_ms": round(cached[1], 2),
                        "warmed": [1, *warm],
                    }
                    self._cache_cpu = capacity_cache.load_cpu(path, key)
                    return cached
                reason = f"spot_check {spot:.2f} ms vs cached {cached[1]:.2f} ms"
        logger.info("capacity cache not used (%s): measuring batch sizes %s", reason, sizes)
        measured = sched.measure(sizes)
        self.capacity_info = {"source": "measured", "reason": reason, "cache_written": False}  # _cpu_costs writes it
        return measured

    def stop(self) -> None:
        self._stop.set()
        for cid in list(self._cams):
            self.stop_camera(cid)
        if self._monitor is not None:
            self._monitor.join(timeout=5)
        if self.scheduler is not None:
            self.scheduler.stop()
        if self.pose_batcher is not None:
            self.pose_batcher.stop()

    @property
    def running(self) -> bool:
        return self.scheduler is not None and self.scheduler.alive and not self._stop.is_set()

    @property
    def capacity_limit(self) -> int:
        """[Phase 6] What admission honours: the uniform limit AND the one-camera crowd spike (admit_skewed)."""
        if self.profile is None:
            return 0
        fps = self.required_fps()
        return min(self.profile.limit(fps), self.profile.limit_skewed(fps, SPIKE_CAPACITY_PEOPLE))

    def required_fps(self) -> float:
        """The analysis rate every camera needs: TARGET_ANALYSIS_FPS, or the fastest running camera's
        real source fps if that is higher (a 30 fps camera analysed at 25 fps would drop frames)."""
        with self._lock:
            # [dynamic fps] a camera above the floor can always step back: count what it NEEDS, not its current rate
            observed = [
                getattr(c.capture, "required_fps", c.capture.fps)
                for c in self._cams.values()
                if c.capture.frame_size is not None
            ]
        return max([self.settings.target_analysis_fps, *observed])

    # ── cameras ───────────────────────────────────────────────────────────────
    def camera_ids(self) -> list[str]:
        with self._lock:
            return list(self._cams)

    def start_camera(
        self,
        spec: CameraSpec,
        *,
        enforce_capacity: bool = True,
        adopt: Callable[[str, str, str], CaptureWorker | None] | None = None,
    ) -> None:
        """Start monitoring `spec`. Raises CapacityExceededError (nothing started) above the limit.

        `enforce_capacity=False` is for the evaluation harness only (A/B with the same camera count
        as the v2 baseline); the API never passes it."""
        if self.scheduler is None or self.profile is None:
            raise RuntimeError("live runtime is not started")
        with self._lock:
            if spec.camera_id in self._cams:
                return
            if enforce_capacity:
                self.profile.admit(len(self._cams) + 1, self.required_fps())
                # [docs/proposals/002] admit() alone only checks the uniform-average case; this also
                # refuses a count that could not sustain target_fps if one camera alone saw a localised
                # crowd of SPIKE_CAPACITY_PEOPLE, rather than only finding out from CAMERA_LAGGING later.
                self.profile.admit_skewed(len(self._cams) + 1, SPIKE_CAPACITY_PEOPLE, self.required_fps())
                self._admit_measured(len(self._cams) + 1)
            s = self.settings
            # [v4] a network camera is queued at <= target_analysis_fps (capture max_fps): size the queue for THAT
            # rate, so live_queue_s really is the most lag a camera can build up (30 fps guess: 4 s at 15 fps)
            queue_fps = s.target_analysis_fps if spec.source_type == "rtsp" else QUEUE_FPS_GUESS
            queue = FrameQueue(max(2, math.ceil(s.live_queue_s * queue_fps)))
            # [Phase 11 WP7] admitted: take over the capture opened in parallel with the guard + capacity phase (if any)
            capture = adopt(spec.camera_id, spec.source, spec.source_type) if adopt is not None else None
            if capture is not None:
                capture.queue, capture.clock = queue, self.stream_clock
            else:
                capture = CaptureWorker(
                    spec.camera_id,
                    spec.source,
                    spec.source_type,
                    queue,
                    s.stall_s,
                    opener=self.opener,
                    realtime=self.realtime_files,
                    clock=self.stream_clock,
                    first_frame_timeout_s=s.first_frame_timeout_s,
                    fps_fallback=s.camera_fps_fallback,
                    replay=spec.replay,
                    max_fps=s.target_analysis_fps,
                )
            cam_ref: dict[str, Any] = {}  # "cam" -> _Camera, "det" -> preloaded detector
            replay_hooks = self._replay_hooks(spec, queue)
            worker_ref = replay_hooks.pop("_worker_ref", None)
            worker = CameraWorker(
                spec.camera_id,
                queue,
                self._pipeline_factory(spec, cam_ref),
                self._camera_fault_sink(spec.camera_id),
                self._on_fire,
                fps=lambda: capture.fps,
                frame_size=lambda: capture.frame_size,
                clear_sink=self._clear,
                frame_hook=self._recorder_hook(spec.camera_id),
                geometry_loader=self._geometry_reloader(spec, cam_ref, capture),
                **replay_hooks,
            )
            if worker_ref is not None:
                worker_ref["w"] = worker
            self._scene_types[spec.camera_id] = spec.scene_type
            cam = _Camera(spec, queue, capture, worker)
            cam_ref["cam"] = cam
            self._cams[spec.camera_id] = cam
            self.scheduler.register(spec.camera_id)
            if self.pose_batcher is not None:
                self.pose_batcher.register(spec.camera_id)
            # [dynamic fps] admission was computed at the floor: every camera goes back to it, then may climb again
            self._apply_rate(self.rate.reset(time.monotonic()), list(self._cams.values()))
            # [Phase 8] load + attach this camera's detector BEFORE frames flow: built lazily on the first frame,
            # its ~1 s load overflowed the bounded queue at every camera start (staged acceptance: 127 frames)
            try:
                cam_ref["det"] = self._new_detector(spec)
            except Exception:
                capture.stop()  # an adopted capture is already running behind its gate: never leak it
                self._cams.pop(spec.camera_id, None)
                self._scene_types.pop(spec.camera_id, None)
                self.scheduler.unregister(spec.camera_id)
                if self.pose_batcher is not None:
                    self.pose_batcher.unregister(spec.camera_id)
                raise
            if capture.gated:
                capture.open_gate()  # adopted: frames flow from here on
            else:
                capture.start()
            worker.start()
            logger.info("[live:%s] monitoring started (%d camera(s) running)", spec.camera_id, len(self._cams))
            self.timeline.camera(spec.camera_id, "admitted")

    @staticmethod
    def _replay_hooks(spec: CameraSpec, queue: FrameQueue) -> dict[str, Any]:
        """[Phase 11 WP6] A replayed camera reports every processed frame to the lockstep coordinator (+ the trace)."""
        feed = spec.replay
        if feed is None:
            return {}
        feed.coordinator.register(spec.camera_id, queue.maxlen)
        cid = spec.camera_id
        worker_ref: dict[str, Any] = {}

        def done(frame: Any) -> None:
            try:
                if feed.on_frame is not None:
                    feed.on_frame(cid, frame, worker_ref.get("w"))
            finally:
                feed.coordinator.done(cid, frame.t)

        def start(frame: Any) -> None:
            if feed.on_start is not None:
                feed.on_start(cid, frame)

        hooks: dict[str, Any] = {"on_frame_done": done, "on_frame_start": start}
        hooks["_worker_ref"] = worker_ref
        return hooks

    def stop_camera(self, camera_id: str) -> bool:
        with self._lock:
            cam = self._cams.pop(camera_id, None)
        if cam is None:
            return False
        cam.capture.stop()
        cam.worker.stop()
        self._scene_types.pop(camera_id, None)
        rec = self.recorders.pop(camera_id, None)
        if rec is not None:
            rec.flush()
        if self.scheduler is not None:
            self.scheduler.unregister(camera_id)
        if self.pose_batcher is not None:
            self.pose_batcher.unregister(camera_id)
        logger.info("[live:%s] monitoring stopped", camera_id)
        return True

    def _new_detector(self, spec: CameraSpec) -> Any:
        from lifeguard.vision import config as vision_config
        from lifeguard.vision.inference.detector import MultiPersonDetector

        if self.scheduler is None:
            raise RuntimeError("live runtime is not started")
        pose = vision_config.current().pose_model_path
        det = MultiPersonDetector(spec.camera_id, self.det_model_path, self.provider, pose_model_path=pose)
        if attach_batched(det, self.scheduler) and self.pose_batcher is not None:
            attach_batched_pose(det, self.pose_batcher)
        return det

    def _pipeline_factory(
        self, spec: CameraSpec, cam_ref: dict[str, Any]
    ) -> Callable[[float, tuple[int, int] | None], Any]:
        from lifeguard.vision.pipeline import SourcePipeline

        def make(fps: float, frame_size: tuple[int, int] | None) -> Any:
            if self.scheduler is None:
                raise RuntimeError("live runtime is not started")
            det = cam_ref.pop("det", None) or self._new_detector(spec)  # preloaded by start_camera
            geometry = None
            if spec.geometry_loader is not None and frame_size is not None:
                try:
                    geometry = spec.geometry_loader(*frame_size)
                except Exception:
                    logger.exception("[live:%s] zone geometry failed to load - whole frame monitored", spec.camera_id)
            if "cam" in cam_ref:
                cam_ref["cam"].geometry = geometry
            pipe = SourcePipeline(
                spec.camera_id,
                spec.mode,
                geometry,
                fps,
                frame_size,
                det,
                fault_sink=lambda f: self._fault(f, spec.camera_id),
                scene_type=self._scene_types.get(spec.camera_id, spec.scene_type),
                scene_classifier=scene_classifier.service(),
            )
            return pipe

        return make

    # ── sinks ─────────────────────────────────────────────────────────────────
    def _fault(self, fault: Fault, camera_id: str | None) -> None:
        try:
            self.fault_sink(fault, camera_id)
        except Exception:
            logger.exception("FAULT sink failed for %s on %s - logged only", fault.code, fault.source_id)
            logger.error("FAULT %s on %s: %s", fault.code, fault.source_id, fault.message)

    def _camera_fault_sink(self, camera_id: str) -> Callable[[Fault], None]:
        def sink(fault: Fault) -> None:
            self._fault(fault, camera_id)

        return sink

    def _on_fire(self, camera_id: str, fire: AlertFire, image: Any) -> None:
        if self._scene_types.get(camera_id) == "standard_cctv":  # [v4] the pipeline never scores; belt and braces
            logger.warning("[live:%s] behaviour alert from a standard CCTV camera dropped", camera_id)
            return
        self.fire_sink(camera_id, fire, audible(fire), image)

    def _clear(self, code: str, source_id: str) -> None:
        if self.clear_sink is None:
            return
        try:
            self.clear_sink(code, source_id)
        except Exception:
            logger.exception("clearing FAULT %s on %s failed", code, source_id)

    def _recorder_hook(self, camera_id: str) -> Callable[[Any, Any, frozenset[int]], None] | None:
        s = self.settings
        if not s.recording_enabled:
            return None
        rec = Recorder(
            camera_id,
            s.recordings_dir,
            pre_s=s.recording_pre_s,
            post_s=s.recording_post_s,
            fps=s.recording_fps,
            width=s.recording_width,
        )
        self.recorders[camera_id] = rec

        def hook(image: Any, detections: Any = None, alerting: frozenset[int] = frozenset()) -> None:
            # [Stage 4] the recording shows what the live view shows: boxes, red while the alert is active, coasting
            # dashed; drawn only on the frames the recorder keeps
            from lifeguard.live import view

            overlay = None if detections is None else (lambda f: view.annotate(f, detections, rec.width, alerting))
            rec.push(image, overlay=overlay)

        return hook

    # ── monitor ───────────────────────────────────────────────────────────────
    def _monitor_loop(self) -> None:
        while not self._stop.wait(MONITOR_PERIOD_S):
            try:
                self.check()
            except Exception:
                logger.exception("live monitor check failed")

    def check(self, now: float | None = None) -> None:
        """One monitor pass (public for tests)."""
        now = time.monotonic() if now is None else now
        s = self.settings
        with self._lock:
            cams = list(self._cams.values())
        for cam in cams:
            # [Phase 11 WP4] grace is read BEFORE the age: the capture stores the first frame's time before it
            # leaves the grace, so a camera is never seen out of grace with a stale age
            grace = cam.capture.in_first_frame_grace(now)
            age = cam.capture.last_frame_age_s(now)
            # [Phase 11 WP6] a replayed recording waits for its lockstep turn: waiting is not a stall
            replay = getattr(cam.spec, "source_type", None) == "replay"
            if age > s.stall_s and not cam.stalled and not grace and not replay:
                cam.stalled = True
                cam.stall_faults += 1
                cause = getattr(
                    cam.capture, "last_error", None
                )  # [v4] e.g. the NVR refusing the stream: say so, not just "stalled"
                self._fault(
                    Fault(
                        CAMERA_STALLED,
                        cam.spec.camera_id,
                        f"no frame for {age:.1f} s (STALL_S={s.stall_s:g}); reconnecting"
                        + (f" - last error: {cause}" if cause else ""),
                        {"last_frame_age_s": round(age, 2), "last_error": cause},
                    ),
                    cam.spec.camera_id,
                )
                cam.capture.request_reconnect(f"stalled {age:.1f} s")
            elif age <= s.stall_s and cam.capture.frames_read > 0 and (cam.stalled or not cam.healthy_seen):
                if cam.stalled:
                    logger.warning("[live:%s] frames flowing again after a stall", cam.spec.camera_id)
                cam.stalled = False
                cam.healthy_seen = True
                self._clear(CAMERA_STALLED, cam.spec.camera_id)
            cam.lagging = cam.queue.lag_s(now) > s.lag_threshold_s
            self._check_lag(cam, now)
            # [Phase 11 WP5] startup timeline: first frame + first analysed frame (MONITORED), once per camera
            first_frame = getattr(cam.capture, "first_frame_at", None)
            if first_frame is not None:
                self.timeline.camera(cam.spec.camera_id, "first_frame", first_frame)
            first_analysed = getattr(cam.worker, "first_analysed_at", None)
            if first_analysed is not None:
                self.timeline.camera(cam.spec.camera_id, "monitored", first_analysed)
        sched = self.scheduler
        if sched is not None:
            pb = self.pose_batcher
            stalled = sched.stalled(now) or (pb is not None and pb.stalled(now))
            if stalled and not self.scheduler_stalled:
                self._fault(
                    Fault(
                        GPU_SCHEDULER_STALLED,
                        "system",
                        f"a frame has waited "
                        f"{max(sched.oldest_pending_s(now), pb.oldest_pending_s(now) if pb else 0.0):.1f} s "
                        f"for the GPU (limit {s.scheduler_stall_s:g} s) or a scheduler thread died",
                        {"alive": sched.alive, "pose_alive": pb.alive if pb is not None else None},
                    ),
                    None,
                )
            elif self.scheduler_stalled and not stalled:
                logger.warning("GPU scheduler recovered")
                self._clear(GPU_SCHEDULER_STALLED, "system")
            self.scheduler_stalled = stalled
        self._update_rate(now, cams)
        self.timeline.maybe_log_summary()  # [Phase 11 WP5] once, when every startup camera is MONITORED

    def _update_rate(self, now: float, cams: list[_Camera]) -> None:
        """[dynamic fps] GPU busy fraction since the last pass, then one RateController step (live/rate.py)."""
        sched, pb = self.scheduler, self.pose_batcher
        if sched is not None:
            busy = float(getattr(sched, "busy_ms", 0.0)) + (float(getattr(pb, "busy_ms", 0.0)) if pb else 0.0)
            ref = self._gpu_ref
            if ref is not None and now > ref[0]:
                frac = (busy - ref[1]) / ((now - ref[0]) * 1000.0)
                self.gpu_busy = frac if self.gpu_busy is None else 0.5 * self.gpu_busy + 0.5 * frac
            self._gpu_ref = (now, busy)
        rtsp = [c for c in cams if getattr(c.capture, "floor_fps", None)]
        if not rtsp or not self.rate.enabled:
            return
        worst = max(c.queue.lag_s(now) for c in rtsp)
        new = self.rate.update(now, worst, self.gpu_busy)
        if new is not None:
            logger.info(
                "[live] analysis rate %g fps (worst queue lag %.2f s, GPU busy %s)",
                new,
                worst,
                "?" if self.gpu_busy is None else f"{self.gpu_busy:.2f}",
            )
            self._apply_rate(new, rtsp)

    @staticmethod
    def _apply_rate(fps: float, cams: list[_Camera]) -> None:
        for c in cams:
            set_rate = getattr(c.capture, "set_rate", None)
            if set_rate is not None:
                set_rate(fps)

    def _check_lag(self, cam: _Camera, now: float) -> None:
        """[v4] Measured analysis rate + CAMERA_LAGGING: raised once when the camera has been LAGGING for
        LAG_FAULT_S, cleared after LAG_CLEAR_S without lag."""
        analysed = int(getattr(cam.worker, "analysed", 0) or 0)
        ref = cam._rate_ref
        if ref is None or now - ref[0] >= 5.0:
            if ref is not None and now > ref[0]:
                cam.analysed_fps = (analysed - ref[1]) / (now - ref[0])
            cam._rate_ref = (now, analysed)
        if cam.lagging:
            cam.ok_since = None
            cam.lag_since = cam.lag_since if cam.lag_since is not None else now
            if not cam.lag_fault and now - cam.lag_since >= LAG_FAULT_S:
                cam.lag_fault = True
                fps = f"{cam.analysed_fps:.1f}" if cam.analysed_fps is not None else "?"
                self._fault(
                    Fault(
                        CAMERA_LAGGING,
                        cam.spec.camera_id,
                        f"{cam.spec.name}: analysed {fps} fps of {cam.capture.fps:g}, "
                        f"{cam.queue.lag_s(now):.1f} s behind live for {now - cam.lag_since:.0f} s - frames are "
                        "being dropped. The server is overloaded: monitor fewer cameras, or set cameras that see no "
                        "pool to standard CCTV",
                        {"analysed_fps": cam.analysed_fps, "dropped": cam.queue.dropped},
                    ),
                    cam.spec.camera_id,
                )
            return
        cam.lag_since = None
        if cam.lag_fault:
            cam.ok_since = cam.ok_since if cam.ok_since is not None else now
            if now - cam.ok_since >= LAG_CLEAR_S:
                cam.lag_fault = False
                cam.ok_since = None
                logger.warning("[live:%s] analysis caught up again", cam.spec.camera_id)
                self._clear(CAMERA_LAGGING, cam.spec.camera_id)

    def _admit_measured(self, requested: int) -> None:
        """[v4] The model (CapacityProfile.admit) is a prior; a running camera that is MEASURED falling behind
        overrides it: no camera is added while one cannot keep up."""
        behind = [c for c in self._cams.values() if c.lag_fault or c.lagging]
        if not behind:
            return
        worst = min(behind, key=lambda c: c.analysed_fps if c.analysed_fps is not None else 0.0)
        raise CapacityExceededError(
            requested,
            len(self._cams),
            worst.analysed_fps if worst.analysed_fps is not None else 0.0,
            self.required_fps(),
            behind=worst.spec.name,
        )

    def _geometry_reloader(
        self, spec: CameraSpec, cam_ref: dict[str, _Camera], capture: CaptureWorker
    ) -> Callable[[], Any]:
        def load() -> Any:
            geometry = None
            if spec.geometry_loader is not None and capture.frame_size is not None:
                geometry = spec.geometry_loader(*capture.frame_size)
            if "cam" in cam_ref:
                cam_ref["cam"].geometry = geometry
            return geometry

        return load

    def set_scene_type(self, camera_id: str, scene_type: str) -> bool:
        """[v4] An admin switched the camera pool / standard CCTV: applied before its next analysed frame."""
        with self._lock:
            cam = self._cams.get(camera_id)
            if cam is None:
                return False
            self._scene_types[camera_id] = scene_type
        cam.worker.request_scene_type(scene_type)
        logger.warning("[live:%s] scene type -> %s", camera_id, scene_type)
        return True

    def reload_geometry(self, camera_id: str | None = None) -> int:
        """Zones / schedules changed (for `camera_id`, or for every camera): reload without a restart."""
        with self._lock:
            cams = [c for c in self._cams.values() if camera_id is None or c.spec.camera_id == camera_id]
        for c in cams:
            c.worker.request_geometry_reload()
        return len(cams)

    def latest_view(self, camera_id: str) -> tuple[Any, Any, frozenset[int]] | None:
        """(newest frame, newest detections) of a running camera, or None (operator view)."""
        with self._lock:
            cam = self._cams.get(camera_id)
        if cam is None or cam.worker.last_image is None:
            return None
        return cam.worker.last_image, cam.worker.last_detections, getattr(cam.worker, "alert_track_ids", frozenset())

    # ── status ────────────────────────────────────────────────────────────────
    def camera_status(self, cam: _Camera) -> dict[str, Any]:
        cap = cam.capture.status()
        wk = cam.worker.status()
        decoded = cap["frames_read"]
        fraction = round(wk["analysed"] / decoded, 4) if decoded else None
        monitoring = NOT_MONITORED if (cam.stalled or wk["monitoring"] == NOT_MONITORED) else wk["monitoring"]
        return {
            "camera_id": cam.spec.camera_id,
            "name": cam.spec.name,
            "monitoring": monitoring,
            "stalled": cam.stalled,
            "lagging": cam.lagging,
            "lag_fault": cam.lag_fault,
            "analysed_fps": None if cam.analysed_fps is None else round(cam.analysed_fps, 2),
            "lag_s": round(cam.queue.lag_s(), 3),
            "queued": len(cam.queue),
            "dropped": cam.queue.dropped,
            "analysed_fraction": fraction,
            "capture": cap,
            "analysis": wk,
            "zones": zone_states(cam.geometry),
            "scene": self._scene_status(cam),
        }

    def _scene_status(self, cam: _Camera) -> dict[str, Any]:
        """[v4] Scene type in force, the automatic suggestion, and the people the last analysed frame saw."""
        pipe = cam.worker.pipeline
        est = getattr(pipe, "scene_type_estimate", None)
        dets = cam.worker.last_detections
        return {
            "type": self._scene_types.get(cam.spec.camera_id, cam.spec.scene_type),
            "suggestion": est.as_dict() if est is not None else None,
            "people": len(getattr(dets, "confirmed_tracks", None) or []) if dets is not None else 0,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            cams = [self.camera_status(c) for c in self._cams.values()]
        s = self.settings
        return {
            "running": self.running,
            "provider": self.provider,
            "active_provider": self.active_provider,
            "model": self.det_model_path.replace("\\", "/").rsplit("/", 1)[-1],
            "scheduler": None
            if self.scheduler is None
            else {**self.scheduler.stats(), "stalled": self.scheduler_stalled},
            "pose_batcher": None if self.pose_batcher is None else self.pose_batcher.stats(),
            "capacity": None if self.profile is None else self.profile.as_dict(self.required_fps()),
            "capacity_limit": self.capacity_limit,  # [Phase 6] what admission enforces: min(limit, limit_skewed)
            "analysis_rate": {
                **self.rate.as_dict(),
                "gpu_busy": None if self.gpu_busy is None else round(self.gpu_busy, 3),
            },
            "cameras": cams,
            "any_lagging": any(c["lagging"] for c in cams),
            "stall_s": s.stall_s,
            "first_frame_timeout_s": s.first_frame_timeout_s,
            "rtsp": self.rtsp,
            "error": self.start_error,
            "scene_model": None if (svc := scene_classifier.service()) is None else svc.status(),  # [v4]
            "capacity_source": dict(self.capacity_info),
            "startup": self.timeline.as_dict(),
        }

    def heartbeat(self) -> dict[str, Any]:
        """Compact payload for the API heartbeat file (the supervisor reads it)."""
        with self._lock:
            cams = list(self._cams.values())
        sched = self.scheduler
        return {
            "live": {
                "running": self.running,
                "scheduler_alive": bool(sched and sched.alive),
                "scheduler_stalled": self.scheduler_stalled,
                "scheduler_last_round_age_s": None if sched is None else sched.stats()["last_round_age_s"],
                "capacity_limit": self.capacity_limit,
                "any_lagging": any(c.lagging for c in cams),
                "cameras": {
                    c.spec.camera_id: {
                        "last_frame_age_s": round(c.capture.last_frame_age_s(), 3),
                        "stalled": c.stalled,
                        "lagging": c.lagging,
                        "monitoring": NOT_MONITORED if c.stalled else c.worker.monitoring,
                        "frames_read": c.capture.frames_read,
                        "analysed": c.worker.analysed,
                    }
                    for c in cams
                },
            }
        }


def detector_calls_per_frame(model: Any, frame_wh: tuple[int, int] = (1920, 1080)) -> int:
    """[Phase 8] Detector calls one analysed frame costs: 1, or with DET_TILING 1 + the tiles of a worst-case
    frame with no water ROI yet (vision/inference/detector.py stage_detect). [Phase 11] Counted at the cameras'
    real frame size (settings.capacity_frame_size, 2560x1440 at the site) over the WHOLE frame: the responsible
    regions that restrict tiling at run time (DET_TILING_REGION) are not known before the cameras open, so
    admission keeps the worst case."""
    from lifeguard.vision.constants import DET_MOSAIC, DET_MOSAIC_MAX_CALLS, DET_TILING
    from lifeguard.vision.inference import enhance

    mosaic = int(DET_MOSAIC_MAX_CALLS) if DET_MOSAIC else 0  # [v7 F4] worst case: every mosaic canvas used
    if str(DET_TILING or "off").lower() == "off":
        return 1 + mosaic
    in_h, in_w = model.model_input_size
    w, h = frame_wh
    return 1 + mosaic + len(enhance.tile_windows((0, 0, w, h), min(in_h, in_w), 0.2, w, h))
