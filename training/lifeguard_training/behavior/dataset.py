"""
Behaviour training windows from labelled clips.

The windows are produced by the REAL SceneBehaviorTracker (same pooling to
10 Hz behaviour samples, same in-water / edge-distance bookkeeping as the live
cameras) and the same feature code the live classifier runs
(detection/behavior_model/features.py). Three frame sources:

  gt        ground-truth boxes from an aquaguard-eval/1 annotation,
            interpolated to BEHAVIOR_SAMPLE_HZ (no detector needed; isolates
            behaviour from detector noise)
  detector  the live detector + tracker on the clip's video; each detector
            track is labelled by the GT box it overlaps (live-faithful —
            includes the detector's real misses / jitter / id switches)
  sim       detection/simulation.py synthetic scenes (tests / pipeline smoke;
            NEVER a substitute for client footage)

Labels (one per window, at its END time):
  activity distress → DISTRESS, distress_active → ACTIVE_DROWNING,
  distress_passive → PASSIVE_DROWNING, any other in-water activity → NORMAL;
  an event interval on the track overrides the activity. Out-of-water /
  unlabelled / submerged windows are skipped (submersion stays rule-based).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lifeguard.vision.behaviour.classifier import SWIMMER, BehaviorClassifier
from lifeguard.vision.behaviour.engine import RiskEngine
from lifeguard.vision.behaviour.learned.features import CLASSES, window_features
from lifeguard.vision.constants import BEHAVIOR_MODEL_MIN_SECONDS, BEHAVIOR_SAMPLE_HZ
from lifeguard.vision.inference.detector import FrameDetections, TrackedPerson, foot_point_from
from lifeguard.vision.scene.scene_tracker import SceneBehaviorTracker

ACTIVITY_CLASS = {"distress": "DISTRESS", "distress_active": "ACTIVE_DROWNING", "distress_passive": "PASSIVE_DROWNING"}
SKIP_ACTIVITIES = ("submerged", "standing_deck", "walking", "sitting", "lying")
EVENT_CLASS = {"DISTRESS": "DISTRESS", "ACTIVE_DROWNING": "ACTIVE_DROWNING", "PASSIVE_DROWNING": "PASSIVE_DROWNING"}
SIM_CLASS = {
    "NORMAL_SWIMMING": "NORMAL",
    "RESTING": "NORMAL",
    "ACTIVE_DROWNING": "ACTIVE_DROWNING",
    "PASSIVE_DROWNING": "PASSIVE_DROWNING",
}

# label_fn(track_id, t, bbox) → (class name, gt track key, event id or "") | None
LabelFn = Callable[[int, float, tuple[float, float, float, float]], tuple[str, str, str] | None]


@dataclass
class Windows:
    seq: list[np.ndarray] = field(default_factory=list)
    summary: list[np.ndarray] = field(default_factory=list)
    y: list[int] = field(default_factory=list)
    clip: list[str] = field(default_factory=list)
    track: list[str] = field(default_factory=list)
    event: list[str] = field(default_factory=list)
    t: list[float] = field(default_factory=list)
    split: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)

    def extend(self, other: Windows) -> None:
        for k in self.__dataclass_fields__:
            getattr(self, k).extend(getattr(other, k))

    def save(self, path: Path, provenance: dict) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            seq=np.stack(self.seq).astype(np.float32),
            summary=np.stack(self.summary).astype(np.float32),
            y=np.asarray(self.y, np.int64),
            clip=np.asarray(self.clip),
            track=np.asarray(self.track),
            event=np.asarray(self.event),
            t=np.asarray(self.t, np.float64),
            split=np.asarray(self.split),
            classes=np.asarray(CLASSES),
            provenance=np.asarray(json.dumps(provenance)),
        )
        return path


def load_windows(paths: Sequence[Path]) -> dict[str, np.ndarray]:
    parts = [np.load(p, allow_pickle=False) for p in paths]
    for p in parts:
        if tuple(p["classes"].tolist()) != CLASSES:
            raise ValueError("window file class list differs from detection/behavior_model/features.py")
    keys = ("seq", "summary", "y", "clip", "track", "event", "t", "split")
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def windows_from_frames(
    frames: Iterable[FrameDetections], label_fn: LabelFn, clip_id: str, split: str, geometry=None, stride_s: float = 0.5
) -> Windows:
    """Run the live scene tracker over `frames`; emit one window per in-water track every `stride_s`."""
    sc = SceneBehaviorTracker(clip_id, "Medium", classifier=BehaviorClassifier(behavior_model=None))
    sc.set_geometry(geometry)
    engine = RiskEngine()
    out = Windows()
    next_t: float | None = None
    for det in frames:
        t = float(det.timestamp)
        sc.ingest(det)
        sc.expire(t)
        if next_t is None:
            next_t = t + stride_s
        if t + 1e-6 < next_t:
            continue
        next_t += stride_s
        for ts in list(sc.tracks.values()):
            if ts.last_seen < t - 1e-6 or ts.category != SWIMMER or ts.buffer.duration < BEHAVIOR_MODEL_MIN_SECONDS:
                continue
            lab = label_fn(ts.track_id, t, ts.last_bbox)
            if lab is None:
                continue
            cls, gt_key, ev = lab
            f = engine.extract(ts.buffer)
            seq, summ = window_features(ts.buffer, f)
            out.seq.append(seq)
            out.summary.append(summ)
            out.y.append(CLASSES.index(cls))
            out.clip.append(clip_id)
            out.track.append(f"{clip_id}:{gt_key}")
            out.event.append(f"{clip_id}:{ev}" if ev else "")
            out.t.append(t)
            out.split.append(split)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Annotated clips (aquaguard-eval/1)
# ═════════════════════════════════════════════════════════════════════════════
def _gt_label(clip, gt_id: int, frame: int) -> tuple[str, str, str] | None:
    for ev in clip.events:
        if ev.track_id == gt_id and ev.start_frame <= frame <= ev.end_frame:
            cls = EVENT_CLASS.get(ev.type)
            return (cls, str(gt_id), ev.id) if cls else None
    tr = clip.tracks.get(gt_id) or {}
    if tr.get("ignore"):
        return None
    from lifeguard.eval.annotations import _interval_lookup

    state = _interval_lookup(tr.get("state", []), frame)
    act = _interval_lookup(tr.get("activity", []), frame)
    if state != "in_water" and act not in ACTIVITY_CLASS:
        return None  # deck / edge / unlabelled: not a behaviour sample
    if act in ACTIVITY_CLASS:
        start = next((int(iv[0]) for iv in tr.get("activity", []) if int(iv[0]) <= frame <= int(iv[1])), frame)
        return ACTIVITY_CLASS[act], str(gt_id), f"act{gt_id}_{start}"  # one episode per labelled interval
    if act in SKIP_ACTIVITIES or act is None:
        return None
    return "NORMAL", str(gt_id), ""


def _gt_boxes_at(clip, frame: float) -> list[tuple[int, tuple[float, float, float, float], str]]:
    """GT boxes linearly interpolated at a (fractional) frame, visible tracks only."""
    out = []
    for tid, tr in clip.tracks.items():
        frames = tr["_frames"]
        if not frames or frame < frames[0] or frame > frames[-1]:
            continue
        k = int(np.searchsorted(frames, frame, side="right")) - 1
        f0 = frames[k]
        b0 = next(b for b in clip.boxes_by_frame[f0] if b.track_id == tid)
        if not b0.visible:
            continue
        if f0 == frame or k + 1 >= len(frames):
            out.append((tid, b0.bbox, b0.vis))
            continue
        f1 = frames[k + 1]
        b1 = next(b for b in clip.boxes_by_frame[f1] if b.track_id == tid)
        if not b1.visible or f1 - f0 > 3 * max(clip.fps, 1.0):  # no interpolation across long gaps
            out.append((tid, b0.bbox, b0.vis))
            continue
        a = (frame - f0) / (f1 - f0)
        out.append((tid, tuple(float(p + a * (q - p)) for p, q in zip(b0.bbox, b1.bbox)), b0.vis))
    return out


def _clip_geometry(clip):
    if not clip.water_polygons:
        return None
    from lifeguard.vision.scene.geometry import ZoneGeometry, ZonePolygon

    zones = [
        ZonePolygon(i, f"water{i}", "WATER", np.asarray(p, np.int32).reshape(-1, 1, 2), 1.0, None)
        for i, p in enumerate(clip.water_polygons)
    ]
    return ZoneGeometry(clip.camera_id, clip.width, clip.height, zones)


def windows_from_gt(clip, stride_s: float = 0.5, conf: float = 0.85) -> Windows:
    """Windows from a clip's GT boxes (interpolated to BEHAVIOR_SAMPLE_HZ)."""
    dt = 1.0 / BEHAVIOR_SAMPLE_HZ

    def frames() -> Iterator[FrameDetections]:
        n = int(clip.frame_count / clip.fps / dt)
        for k in range(n):
            t = k * dt
            fr = t * clip.fps
            people = [
                TrackedPerson(
                    tid,
                    tuple(int(round(v)) for v in bb),
                    conf,
                    None,
                    foot_point_from(tuple(int(round(v)) for v in bb), None),
                )
                for tid, bb, _vis in _gt_boxes_at(clip, fr)
            ]
            yield FrameDetections(clip.clip_id, 1000.0 + t, int(fr), (clip.height, clip.width, 3), people, 0.0, "gt")

    def label(track_id: int, t: float, _bbox) -> tuple[str, str, str] | None:
        return _gt_label(clip, track_id, int(round((t - 1000.0) * clip.fps)))

    return windows_from_frames(frames(), label, clip.clip_id, clip.split, _clip_geometry(clip), stride_s)


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _clip_frames(video_path: Path, detector) -> tuple[float, Iterator[FrameDetections]]:
    """(source fps, detector frames) at the upload worker's time base and analysis rate (vision/clip_clock.py), so
    training windows see the frame rate the served model will see. Timestamps are 1000 + clip seconds."""
    import cv2

    from lifeguard.vision.clip_clock import ClipClock, apply_clip_profile

    apply_clip_profile(detector)
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    fps = fps if 1.0 <= fps <= 240.0 else 25.0
    clock = ClipClock(fps)
    detector.reset()
    detector.set_fps(clock.analysis_fps)

    def frames() -> Iterator[FrameDetections]:
        k = 0
        try:
            while cap.grab():
                t = clock.stamp(cap, k)
                if clock.keep(t):
                    ok, img = cap.retrieve()
                    if ok and img is not None:
                        yield detector.detect(img, frame_index=k, timestamp=1000.0 + t)
                k += 1
        finally:
            cap.release()

    return fps, frames()


def windows_from_detector(clip, detector, stride_s: float = 0.5, min_iou: float = 0.3) -> Windows:
    """Windows from the live detector + tracker on the clip's video, labelled via GT box overlap."""
    fps, frames_iter = _clip_frames(clip.video_path, detector)

    def frames() -> Iterator[FrameDetections]:
        yield from frames_iter

    def label(track_id: int, t: float, bbox) -> tuple[str, str, str] | None:
        fr = int(round((t - 1000.0) * fps))
        best, best_iou = None, min_iou
        for tid, gb, _vis in _gt_boxes_at(clip, fr):
            v = _iou(bbox, gb)
            if v >= best_iou:
                best, best_iou = tid, v
        return None if best is None else _gt_label(clip, best, fr)

    return windows_from_frames(frames(), label, clip.clip_id, clip.split, _clip_geometry(clip), stride_s)


# ═════════════════════════════════════════════════════════════════════════════
# Sprint clips (eval/sprint_v1/clips.json: clip-level labels + event windows)
# ═════════════════════════════════════════════════════════════════════════════
EVENT_LEAD_S = 1.0  # a window is an event window once at least this much of it is past the onset


def label_sprint_windows(
    w: Windows, entry: dict, boxes: dict[tuple[str, float], tuple[float, float, float, float]]
) -> Windows:
    """Turn provisional NORMAL windows of one sprint clip into labelled ones (pure; unit-tested).

    Normal clip: every in-water window is NORMAL. Drowning clip with an event window: the VICTIM is the track seen
    most inside [onset, end], plus any track whose median box centre sits within 2 victim box heights of the
    victim's (the same person under another id); victim windows ending in [onset + EVENT_LEAD_S, end] get the event
    class, victim windows in [onset, onset + EVENT_LEAD_S) are dropped (mostly pre-onset motion), everything else -
    before the onset, after the end (several victims swim away), other people - stays NORMAL. A drowning clip
    without a window is not labelled at all (returns no windows)."""
    if entry["label"] != "drowning":
        return w
    if "onset" not in entry or "end" not in entry:
        return Windows()
    lo, hi = float(entry["onset"]), float(entry["end"])
    cls = CLASSES.index(entry.get("event_type", "ACTIVE_DROWNING"))
    rel = [t - 1000.0 for t in w.t]
    in_ev = [i for i in range(len(w)) if lo <= rel[i] <= hi]
    if not in_ev:
        return w
    counts: dict[str, int] = {}
    for i in in_ev:
        counts[w.track[i]] = counts.get(w.track[i], 0) + 1
    victim = max(counts, key=lambda k: counts[k])

    def centre(track: str) -> tuple[float, float, float] | None:
        bs = [boxes[(track, w.t[i])] for i in in_ev if w.track[i] == track and (track, w.t[i]) in boxes]
        if not bs:
            return None
        a = np.asarray(bs, np.float64)
        return (
            float(np.median((a[:, 0] + a[:, 2]) / 2)),
            float(np.median((a[:, 1] + a[:, 3]) / 2)),
            float(np.median(a[:, 3] - a[:, 1])),
        )

    vc = centre(victim)
    victims = {victim}
    if vc is not None:
        for track in counts:
            c = centre(track)
            if c is not None and math.hypot(c[0] - vc[0], c[1] - vc[1]) <= 2.0 * vc[2]:
                victims.add(track)
    out = Windows()
    for i in range(len(w)):
        if w.track[i] in victims and lo <= rel[i] < lo + EVENT_LEAD_S:
            continue
        keep = {k: getattr(w, k)[i] for k in w.__dataclass_fields__}
        if w.track[i] in victims and lo + EVENT_LEAD_S <= rel[i] <= hi:
            keep["y"], keep["event"] = cls, f"{entry['id']}:event"
        for k, v in keep.items():
            getattr(out, k).append(v)
    return out


def windows_from_sprint(entry: dict, root: Path, detector, stride_s: float = 0.5) -> Windows:
    """Windows of one eval/sprint_v1 clip from the live detector + tracker, labelled by `label_sprint_windows`."""
    if entry["label"] == "drowning" and "onset" not in entry:
        return Windows()  # no event window: nothing trustworthy to learn from
    _fps, frames_iter = _clip_frames(root / entry["video"], detector)
    boxes: dict[tuple[str, float], tuple[float, float, float, float]] = {}

    def label(track_id: int, t: float, bbox) -> tuple[str, str, str]:
        boxes[(f"{entry['id']}:{track_id}", t)] = tuple(float(v) for v in bbox)
        return "NORMAL", str(track_id), ""

    w = windows_from_frames(frames_iter, label, entry["id"], entry["split"], None, stride_s)
    return label_sprint_windows(w, entry, boxes)


# ═════════════════════════════════════════════════════════════════════════════
# Synthetic scenes (tests / smoke runs only)
# ═════════════════════════════════════════════════════════════════════════════
def windows_from_sim(
    scenarios: Sequence[str], seed: int, duration: float = 30.0, split: str = "dev", stride_s: float = 0.5
) -> Windows:
    from lifeguard.eval.simulation import simulate_scene

    scene = simulate_scene(scenarios, seed=seed, duration=duration, rate_hz=BEHAVIOR_SAMPLE_HZ)
    clip_id = f"sim{seed}"

    def label(track_id: int, t: float, _bbox) -> tuple[str, str, str] | None:
        truth = scene.truth.get(track_id)
        if truth is None:
            return None
        _scenario, lab, onset = truth
        cls = SIM_CLASS.get(lab)
        if cls is None:
            return None
        actor = scene.actor_of.get(track_id, track_id)
        if cls != "NORMAL" and onset is not None and (t - 1000.0) < onset:
            cls = "NORMAL"  # before the drowning starts
        return cls, str(actor), (f"a{actor}" if cls != "NORMAL" else "")

    return windows_from_frames((d for d, _f in scene.frames), label, clip_id, split, scene.geometry, stride_s)


def provenance(sources: Sequence[str]) -> dict:
    h = hashlib.sha256("\n".join(sorted(sources)).encode()).hexdigest()[:16]
    return {"sources": list(sources), "sources_sha": h, "classes": list(CLASSES)}
