# mypy: disable-error-code="arg-type, assignment, attr-defined"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""
Drive a pipeline over one clip and record what it did.

The runner decodes EVERY source frame itself (ground truth is per source frame) and hands each one
to an adapter, which returns the people / alerts the operator would see on it.

TREES. The harness can measure two source layouts with the same code (plan Phase 3: "the harness
runs the new tree against v2-baseline ... with --backend-dir"):

    Tree("v3")                       this package: lifeguard.vision.*
    Tree("v2", backend_dir=...)      a v2 checkout: detection.*, config.*, services.* on sys.path

Only public entry points of the tree under test are used (pipeline, scene tracker, model view,
behaviour suite), so a v2 tree is driven exactly as its own harness drove it.

TRACE. `run_clip(..., trace=fn)` calls fn(record) for every frame with every reported person (id,
box, confidence, state, zone state) and every alert. Two traces (v2 tree vs v3 tree) are compared
frame by frame by lifeguard.eval.trace - the box / id / alert parity check.

[v3] From the v2 baseline evaluation/runner.py: StepResult, ClipRun, _people_from, _alerts_from,
_RenderSink, PipelineAdapter, Teacher and run_clip verbatim apart from the tree lookups and the
trace hook. Dropped: LegacyUploadAdapter (it reproduced the pre-Phase-3.1 frame-skipping upload
path; no tree that needs it exists - both frozen baselines were produced by PipelineAdapter).
"""

from __future__ import annotations

import importlib
import logging
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np

from lifeguard.eval.metrics import PredAlert, PredPerson

logger = logging.getLogger(__name__)

TraceFn = Callable[[dict[str, Any]], None]


# ═════════════════════════════════════════════════════════════════════════════
# The tree under test
# ═════════════════════════════════════════════════════════════════════════════
_MODULES = {
    #  role             v3 module                                v2 module
    "pipeline": ("lifeguard.vision.pipeline", "detection.pipeline"),
    "constants": ("lifeguard.vision.constants", "config.constants"),
    "geometry": ("lifeguard.vision.scene.geometry", "services.zone_service"),
    "model_view": ("lifeguard.vision.model_view", "detection.model_view"),
    "suite": ("lifeguard.eval.behaviour_suite", "detection.behavior_calibration"),
    "simulation": ("lifeguard.eval.simulation", "detection.simulation"),
    "decode": ("lifeguard.vision.inference.decode", "detection.rtm_backend"),
    "models": ("lifeguard.vision.inference.models", "detection.rtm_backend"),
}


class Tree:
    """Module lookup for the tree under test (v3 = this package; v2 = a checkout on sys.path)."""

    def __init__(self, layout: str = "v3", backend_dir: Path | None = None) -> None:
        if layout not in ("v3", "v2"):
            raise ValueError(f"unknown layout {layout!r}")
        self.layout = layout
        self.backend_dir = backend_dir
        if layout == "v2":
            if backend_dir is None or not (backend_dir / "detection" / "pipeline.py").is_file():
                raise ValueError(f"{backend_dir} is not a v2 backend (no detection/pipeline.py)")
            if str(backend_dir) not in sys.path:
                sys.path.insert(0, str(backend_dir))

    @staticmethod
    def detect(backend_dir: Path | None) -> Tree:
        if backend_dir is None:
            return Tree("v3")
        if (backend_dir / "detection" / "pipeline.py").is_file():
            return Tree("v2", backend_dir)
        raise ValueError(f"{backend_dir}: not a v2 backend (detection/pipeline.py); omit --backend-dir for this tree")

    def module(self, role: str) -> ModuleType:
        v3, v2 = _MODULES[role]
        return importlib.import_module(v3 if self.layout == "v3" else v2)

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"layout": self.layout, "backend_dir": str(self.backend_dir or "")}
        if self.layout == "v3":
            from lifeguard.vision import config as vision_config
            from lifeguard.vision.constants import ACTIVE_THRESHOLDS

            cfg = vision_config.current()
            out.update(
                thresholds=ACTIVE_THRESHOLDS.describe(),
                det_model=cfg.det_model_path,
                provider=cfg.provider,
                pose_model=cfg.pose_model_path,
            )
        else:
            settings = importlib.import_module("config.settings").settings
            out.update(
                det_model=getattr(settings, "RTM_DET_MODEL_PATH", ""),
                provider=getattr(settings, "RTM_PROVIDER", ""),
                device=getattr(settings, "RTM_DEVICE", ""),
            )
        return out

    def active_providers(self) -> dict[str, str]:
        try:
            if self.layout == "v3":
                m = self.module("models")
                return {str(k[:2]): m.active_provider(v) for k, v in m.cached_models().items()}
            rb = self.module("models")
            return {str(k[:2]): rb.active_provider(v) for k, v in list(rb._model_cache.items())}
        except (AttributeError, ImportError) as exc:
            logger.warning("active providers unavailable: %s", exc)
            return {}


_TREE: Tree = Tree("v3")


def set_tree(tree: Tree) -> None:
    global _TREE
    _TREE = tree


def tree() -> Tree:
    return _TREE


@dataclass
class StepResult:
    analysed: bool
    people: list[PredPerson]
    alerts: list[PredAlert]
    ms: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)


@dataclass
class ClipRun:
    clip_id: str
    adapter: str
    fps: float
    frames: int = 0
    analysed: int = 0
    wall_s: float = 0.0
    decode_ms: list[float] = field(default_factory=list)
    step_ms: list[float] = field(default_factory=list)  # analysed frames only
    stage_ms: dict[str, list[float]] = field(default_factory=dict)
    people: dict[int, list[PredPerson]] = field(default_factory=dict)  # frame -> people (stored frames only)
    alerts: list[PredAlert] = field(default_factory=list)
    track_spans: dict[int, list[float]] = field(default_factory=dict)  # track -> [first_t, last_t, n]
    people_per_frame: list[int] = field(default_factory=list)
    teacher: dict[str, Any] = field(default_factory=dict)
    first_result_s: float | None = None
    notes: list[str] = field(default_factory=list)


def _geometry_from_polygons(camera_id: str, polys: Sequence[Sequence[tuple[float, float]]], w: int, h: int) -> Any:
    zs = tree().module("geometry")
    zones = [
        zs.ZonePolygon(
            zone_id=i + 1,
            name=f"Water {i + 1}",
            zone_type="WATER",
            contour=np.array([[int(round(x)), int(round(y))] for x, y in p], np.int32).reshape(-1, 1, 2),
            sensitivity_multiplier=1.0,
            schedule_id=None,
        )
        for i, p in enumerate(polys)
    ]
    return zs.ZoneGeometry(camera_id, w, h, zones)


def _people_from(det: Any, scene: Any) -> list[PredPerson]:
    out: list[PredPerson] = []
    tracks = getattr(scene, "tracks", {}) or {}
    for p in det.tracks:
        ts = tracks.get(p.track_id) if p.track_id >= 0 else None
        in_water = None if ts is None else (ts.category == "SWIMMER")
        zone = None
        if ts is not None:
            zone = (
                "out_of_water"
                if ts.category != "SWIMMER"
                else {"EDGE": "edge"}.get(getattr(ts, "zone_state", ""), "in_water")
            )
        out.append(
            PredPerson(
                track_id=int(p.track_id),
                bbox=tuple(float(v) for v in p.bbox),
                conf=float(p.confidence),
                in_water=in_water,
                state=getattr(ts, "state", "") if ts is not None else "TENTATIVE",
                zone_state=zone,
            )
        )
    return out


def _alerts_from(fires: Sequence[Any], t: float, frame_idx: int) -> list[PredAlert]:
    return [
        PredAlert(
            t=t,
            frame=frame_idx,
            track_id=int(f.track_id),
            scenario=f.result.scenario,
            bbox=tuple(float(v) for v in f.bbox),
            ghost=bool(f.ghost),
            first=bool(f.first),
            gate=getattr(f, "gate", None),
        )
        for f in fires
        if getattr(f, "notify", True)  # [2.19.0] ONGOING updates are the same alert, not a new one
    ]


class _RenderSink:
    """Optional Model View render + VP8 encode - what the upload path does per analysed frame."""

    def __init__(self, enabled: bool, fps: float, max_fps: float | None = None) -> None:
        self.enabled = enabled
        self.fps = min(fps, max_fps) if max_fps else fps
        self._dt = 1.0 / self.fps if self.fps > 0 else 0.0
        self._last_t = -math.inf
        self._w: Any = None
        self._path: Path | None = None

    def write(self, frame: np.ndarray, items: list[dict[str, Any]], header: str, t: float | None = None) -> None:
        if not self.enabled:
            return
        if t is not None:
            if t - self._last_t < self._dt - 1e-6:
                return
            self._last_t = t
        render_model_view = tree().module("model_view").render_model_view
        img = render_model_view(frame, items, zones=None, header=header)
        h, w = img.shape[:2]
        tw = min(640, w) // 2 * 2
        th = max(2, int(round(h * tw / w)) // 2 * 2)
        if self._w is None:
            import tempfile

            self._path = Path(tempfile.gettempdir()) / f"lifeguard_eval_{id(self)}.webm"
            self._w = cv2.VideoWriter(str(self._path), cv2.VideoWriter_fourcc(*"VP80"), self.fps, (tw, th))
        self._w.write(cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA))

    def close(self) -> None:
        if self._w is not None:
            self._w.release()
            if self._path is not None:
                self._path.unlink(missing_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Adapter
# ═════════════════════════════════════════════════════════════════════════════
class PipelineAdapter:
    """The tree's SourcePipeline - analyses every frame."""

    name = "pipeline"

    def __init__(self, mode: str = "Medium", render: bool = False) -> None:
        self._P = tree().module("pipeline")
        self.mode = mode
        self.render = render

    def warmup(self) -> None:
        st = self._P.warmup_models()
        if isinstance(st, dict) and st.get("healthy") is False:
            raise RuntimeError(f"detector not available: {st.get('error')}")

    def start_clip(self, clip_id: str, fps: float, w: int, h: int, geometry: Any) -> None:
        self.sp = self._P.SourcePipeline(
            f"eval-{clip_id}", mode=self.mode, geometry=geometry, fps=fps, frame_size=(w, h)
        )
        try:  # [2.18.0] the harness measures the finite-clip path
            from lifeguard.vision.clip_clock import apply_clip_profile

            apply_clip_profile(getattr(self.sp, "detector", None))
        except ImportError:
            pass
        consts = tree().module("constants")  # the tree's render cap (named UPLOAD_RENDER_MAX_FPS before 2.21.0)
        cap = getattr(consts, "CLIP_RENDER_MAX_FPS", getattr(consts, "UPLOAD_RENDER_MAX_FPS", None))
        self.sink = _RenderSink(self.render, fps, cap)

    def step(self, frame_idx: int, t: float, frame: np.ndarray) -> StepResult:
        t0 = time.perf_counter()
        res = self.sp.process(frame, frame_idx, t)
        t1 = time.perf_counter()
        if self.render:
            self.sink.write(frame, self.sp.scene.render_items(), f"eval {frame_idx}", t)
        t2 = time.perf_counter()
        stages = dict(getattr(res, "stages_ms", {}) or {})
        stages["render"] = (t2 - t1) * 1000.0
        return StepResult(
            True,
            _people_from(res.detections, self.sp.scene),
            _alerts_from(res.fires, t, frame_idx),
            (t2 - t0) * 1000.0,
            stages,
        )

    def finish(self, frame_idx: int, t: float) -> list[Any]:
        """[2.18.0] Alerts the end-of-clip flush raises after the last frame (trees without it: none)."""
        finish = getattr(self.sp, "finish", None)
        return _alerts_from(finish(t), t, frame_idx) if callable(finish) else []

    def end_clip(self) -> None:
        self.sink.close()
        close = getattr(self.sp, "close", None)
        if callable(close):
            close()


def make_adapter(kind: str = "auto", mode: str = "Medium", render: bool = False) -> PipelineAdapter:
    if kind not in ("auto", "pipeline"):
        raise ValueError("only the pipeline adapter exists (see module docstring)")
    return PipelineAdapter(mode, render)


# ═════════════════════════════════════════════════════════════════════════════
# Teacher (unlabelled clips): high-resolution tiled inference of the stock person detector. NOT
# ground truth - it shares the model's blind spots - but a person it finds with high confidence that
# the live configuration misses is a real miss worth counting.
# ═════════════════════════════════════════════════════════════════════════════
class Teacher:
    def __init__(self, conf: float = 0.5, tile: int = 640, overlap: float = 0.25, min_h: float = 10.0) -> None:
        t = tree()
        if t.layout == "v3":
            from lifeguard.vision import config as vision_config

            cfg = vision_config.current()
            self.rb = t.module("decode")
            self.model = t.module("models").load_detector_model(cfg.det_model_path, cfg.provider, tile)
        else:
            settings = importlib.import_module("config.settings").settings
            self.rb = t.module("decode")
            dev = self.rb.resolve_device(settings.RTM_DEVICE, "onnxruntime")
            self.model = self.rb.load_detector_model(
                settings.RTM_DET_MODEL_PATH, tile, dev, "onnxruntime", score_thr=0.05
            )
        self.conf, self.tile, self.overlap, self.min_h = conf, tile, overlap, min_h

    def boxes(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        t = self.tile
        step = int(t * (1 - self.overlap))
        xs = list(range(0, max(w - t, 0) + 1, step)) or [0]
        ys = list(range(0, max(h - t, 0) + 1, step)) or [0]
        if xs[-1] + t < w:
            xs.append(w - t)
        if ys[-1] + t < h:
            ys.append(h - t)
        all_b, all_s = [], []
        for y in ys:
            for x in xs:
                crop = frame[max(y, 0) : y + t, max(x, 0) : x + t]
                b, s, lab = self.rb.run_detector(self.model, crop, score_thr=self.conf)
                keep = lab == 0
                b, s = b[keep], s[keep]
                if len(b):
                    b = b + np.array([max(x, 0), max(y, 0), max(x, 0), max(y, 0)], np.float32)
                    all_b.append(b)
                    all_s.append(s)
        b, s, lab = self.rb.run_detector(self.model, frame, score_thr=self.conf)
        if len(b):
            all_b.append(b[lab == 0])
            all_s.append(s[lab == 0])
        if not all_b:
            return np.zeros((0, 4), np.float32)
        bb, ss = np.concatenate(all_b), np.concatenate(all_s)
        idx = cv2.dnn.NMSBoxes(
            [[float(a), float(b_), float(c - a), float(d - b_)] for a, b_, c, d in bb], ss.tolist(), self.conf, 0.5
        )
        idx = np.asarray(idx).reshape(-1)
        bb = bb[idx]
        return bb[(bb[:, 3] - bb[:, 1]) >= self.min_h]


# ═════════════════════════════════════════════════════════════════════════════
# Clip loop
# ═════════════════════════════════════════════════════════════════════════════
def _trace_record(idx: int, t: float, r: StepResult) -> dict[str, Any]:
    return {
        "f": idx,
        "t": round(t, 6),
        "people": [
            [p.track_id, *(float(v) for v in p.bbox), float(p.conf), p.state, p.zone_state, p.in_water]
            for p in r.people
        ],
        "alerts": [
            [a.track_id, a.scenario, bool(a.ghost), bool(a.first), *(float(v) for v in a.bbox)] for a in r.alerts
        ],
    }


def _clip_clock(fps: float) -> Any:
    """The tree's ClipClock (vision/clip_clock.py, 2.18.0+), or None for trees analysed on every frame."""
    try:
        mod = importlib.import_module("lifeguard.vision.clip_clock")
    except ImportError:
        return None
    return mod.ClipClock(fps)


def run_clip(
    adapter: Any,
    clip: Any,
    *,
    max_seconds: float | None = None,
    use_zones: bool = False,
    drawn_zones: Sequence[Sequence[float]] | None = None,
    teacher: Teacher | None = None,
    teacher_every: int = 15,
    progress: Callable[[int, int], None] | None = None,
    trace: TraceFn | None = None,
) -> ClipRun:
    cap = cv2.VideoCapture(str(clip.video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {clip.video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or clip.fps or 25.0)
    if not math.isfinite(fps) or fps <= 0 or fps > 240:
        fps = clip.fps or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or clip.frame_count or 0)
    if max_seconds:
        total = min(total, int(max_seconds * fps))
    geometry = (
        _geometry_from_polygons(clip.camera_id, clip.water_polygons, w, h)
        if (use_zones and clip.water_polygons)
        else None
    )
    if drawn_zones:  # a drawn WATER zone, exactly as the live startup builds it (drawn_water precedence)
        geometry = _geometry_from_polygons(clip.camera_id, [[(x * w, y * h) for x, y in drawn_zones]], w, h)
        geometry.drawn_water = True
    run = ClipRun(clip.clip_id, adapter.name, fps)
    src = "drawn water zone" if drawn_zones else "annotated water polygons"
    run.notes.append(f"zones={src if geometry is not None else 'none (whole frame = water)'}")
    clock = _clip_clock(fps)  # [2.18.0] the upload worker's time base and analysis rate (None: older trees)
    if clock is not None:
        run.notes.append(f"analysed at {clock.analysis_fps:.1f} fps on container timestamps")
    adapter.start_clip(clip.clip_id, clock.analysis_fps if clock is not None else fps, w, h, geometry)
    t_hit = t_teacher = 0
    t_start = time.perf_counter()
    idx = 0
    last_t = 0.0
    held: list[Any] = []  # people of the last analysed frame: what the operator sees on a skipped frame
    # [Phase 3 Step 0] keep predictions on fully annotated frames and on every GT keyframe (events-only clips label
    # just the victim at sparse keyframes): metrics.victim_continuity reads them. Storage only, no behaviour change.
    keep: set[int] = set(clip.annotated) | {int(f) for tr in clip.tracks.values() for f in tr.get("_frames", ())}
    try:
        while total <= 0 or idx < total:
            td = time.perf_counter()
            if clock is None:
                ok, frame = cap.read()
                t = idx / fps
            else:
                ok = cap.grab()
                frame = None
                t = clock.stamp(cap, idx) if ok else last_t
                if ok and clock.keep(t):
                    ok, frame = cap.retrieve()
                elif ok:
                    if clip.labelled and idx in keep:
                        run.people[idx] = list(held)
                    run.people_per_frame.append(len(held))
                    run.frames += 1
                    idx += 1
                    continue
            if not ok or frame is None:
                break
            last_t = t
            run.decode_ms.append((time.perf_counter() - td) * 1000.0)
            r = adapter.step(idx, t, frame)
            held = list(r.people)
            if trace is not None:
                trace({"clip": clip.clip_id, **_trace_record(idx, t, r)})
            run.frames += 1
            if r.analysed:
                run.analysed += 1
                run.step_ms.append(r.ms)
                for k, v in r.stages.items():
                    run.stage_ms.setdefault(k, []).append(v)
                if run.first_result_s is None:
                    run.first_result_s = time.perf_counter() - t_start
            if clip.labelled and idx in keep:
                run.people[idx] = list(r.people)
            run.people_per_frame.append(len(r.people))
            for p in r.people:
                if p.track_id >= 0:
                    sp = run.track_spans.setdefault(p.track_id, [t, t, 0])
                    sp[1] = t
                    sp[2] += 1
            run.alerts.extend(r.alerts)
            if teacher is not None and idx % teacher_every == 0:
                tb = teacher.boxes(frame)
                if len(tb):
                    from lifeguard.eval.metrics import _assign, _similarity

                    pb = np.array([p.bbox for p in r.people], np.float32).reshape(-1, 4)
                    t_hit += len(_assign(_similarity(tb.astype(np.float32), pb, 0.3), 0.3))
                    t_teacher += len(tb)
            if progress and idx % 100 == 0:
                progress(idx, total)
            idx += 1
        if idx > 0 and hasattr(adapter, "finish"):
            last = idx - 1
            tail = adapter.finish(last, last_t)
            run.alerts.extend(tail)
            if trace is not None and tail:
                trace(
                    {
                        "clip": clip.clip_id,
                        "f": last,
                        "t": round(last_t, 6),
                        "people": [],
                        "alerts": [
                            [a.track_id, a.scenario, bool(a.ghost), bool(a.first), *(float(v) for v in a.bbox)]
                            for a in tail
                        ],
                    }
                )
    finally:
        cap.release()
        adapter.end_clip()
    run.wall_s = time.perf_counter() - t_start
    if teacher is not None:
        run.teacher = {
            "teacher_people": t_teacher,
            "found_by_pipeline": t_hit,
            "teacher_agreement": round(t_hit / t_teacher, 4) if t_teacher else None,
            "every_n_frames": teacher_every,
            "conf": teacher.conf,
            "tile": teacher.tile,
        }
    return run
