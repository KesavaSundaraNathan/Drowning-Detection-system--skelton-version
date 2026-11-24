"""
[v3] Ported verbatim from the v2 baseline evaluation/annotations.py (imports only).

evaluation/annotations.py — the eval-set annotation format ("aquaguard-eval/1").

One JSON file per clip (eval/annotations/<clip_id>.json). The full spec, with
labelling rules, is in eval/README.md; this module is the executable version of
it: `load_clip()` validates a file and indexes it for the metrics.

Coordinates are PIXELS of the source video. Frame numbers are 0-based decode
order (frame N is the (N+1)-th frame cv2.VideoCapture returns).

Minimal shape:

    {
      "format": "aquaguard-eval/1",
      "clip_id": "poolA_cam1_2026-10-02_1400",
      "video": "clips/poolA_cam1_2026-10-02_1400.mp4",   # relative to eval/
      "camera_id": "poolA-cam1",
      "split": "test",                                  # "dev" | "test"
      "fps": 25.0, "width": 1920, "height": 1080, "frame_count": 15000,
      "annotated_frames": {"every": 5},                 # or "all", or [f0, f1, ...]
      "conditions": ["glare", "crowded"],               # clip-wide tags
      "frame_conditions": [[1200, 1800, ["splash"]]],   # [start, end, tags] inclusive
      "water_polygons": [[[x, y], ...]],                # optional (px) — pool outline
      "tracks": [
        {"id": 1, "age": "adult",
         "boxes": [[frame, x1, y1, x2, y2, "full"], ...],   # vis: full|partial|head|underwater|invisible
         "state": [[0, 900, "in_water"], [901, 960, "edge"], [961, 1500, "out_of_water"]],
         "activity": [[0, 600, "swimming"], [601, 900, "exiting"], [901, 1500, "walking"]]}
      ],
      "events": [
        {"id": "e1", "track_id": 3, "type": "ACTIVE_DROWNING",
         "start_frame": 4200, "end_frame": 4700, "staged": true}
      ]
    }
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT = "aquaguard-eval/1"

VISIBILITY = ("full", "partial", "head", "underwater", "invisible")
# Visible to the camera = counts toward recall. "invisible" (fully under, fully
# occluded) is still annotated so track continuity through it can be measured,
# but a detector cannot be blamed for missing it.
VISIBLE = frozenset(VISIBILITY[:-1])

# "edge" (ML core v2): on the pool edge — sitting on the coping with legs in
# the water, holding the wall, climbing out. Monitored, at reduced sensitivity.
STATES = ("in_water", "edge", "out_of_water")
MONITORED_STATES = ("in_water", "edge")

ACTIVITIES = (
    # in water — normal
    "swimming",
    "treading",
    "floating",
    "standing_water",
    "playing",
    "diving",
    "entering",
    "exiting",
    "resting_edge",
    # out of water
    "standing_deck",
    "walking",
    "sitting",
    "lying",
    # drowning-like (staged) — also listed as events
    # distress          = struggling but still able to call / wave for help (DISTRESS)
    # distress_active   = instinctive drowning response (ACTIVE_DROWNING)
    # distress_passive  = motionless face-down / sinking (PASSIVE_DROWNING)
    "distress",
    "distress_active",
    "distress_passive",
    "submerged",
)
# How a false alert is attributed (report buckets). Everything in-water that is
# not one of the explicit buckets lands in "other_in_water".
ACTIVITY_BUCKET = {
    "swimming": "normal_swimming",
    "treading": "normal_swimming",
    "floating": "normal_swimming",
    "resting_edge": "normal_swimming",
    "standing_water": "standing_in_water",
    "standing_deck": "out_of_water",
    "walking": "out_of_water",
    "sitting": "out_of_water",
    "lying": "out_of_water",
}
FALSE_ALERT_BUCKETS = (
    "normal_swimming",
    "standing_in_water",
    "out_of_water",
    "other_in_water",
    "no_person",
    "unattributed",
)

EVENT_TYPES = ("ACTIVE_DROWNING", "PASSIVE_DROWNING", "SUBMERSION", "FLOATABLE_FAILURE", "DISTRESS")
CONDITIONS = (
    "glare",
    "night",
    "low_res",
    "crowded",
    "far_end",
    "partial_submersion",
    "rain",
    "splash",
    "reflections",
    "compression",
    "underwater_lights",
)


class AnnotationError(ValueError):
    pass


@dataclass
class GTBox:
    track_id: int
    frame: int
    bbox: tuple[float, float, float, float]
    vis: str
    in_water: bool | None  # None = state not labelled at this frame (edge counts as in water)
    activity: str | None
    ignore: bool = False
    zone_state: str | None = None  # in_water | edge | out_of_water (None = not labelled)

    @property
    def visible(self) -> bool:
        return self.vis in VISIBLE and not self.ignore

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class GTEvent:
    id: str
    track_id: int
    type: str
    start_frame: int
    end_frame: int
    staged: bool = True


@dataclass
class ClipAnnotation:
    clip_id: str
    video_path: Path
    camera_id: str
    split: str
    fps: float
    width: int
    height: int
    frame_count: int
    labelled: bool
    conditions: set[str]
    annotated: set[int]
    boxes_by_frame: dict[int, list[GTBox]]
    tracks: dict[int, dict[str, Any]]
    events: list[GTEvent]
    frame_conditions: list[tuple[int, int, set[str]]] = field(default_factory=list)
    water_polygons: list[list[tuple[float, float]]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def conditions_at(self, frame: int) -> set[str]:
        tags = set(self.conditions)
        for s, e, t in self.frame_conditions:
            if s <= frame <= e:
                tags |= t
        return tags

    def track_box_at(self, track_id: int, frame: int) -> GTBox | None:
        """GT box of a track at `frame`, else the latest visible box before it."""
        best: GTBox | None = None
        for f in self.tracks.get(track_id, {}).get("_frames", ()):
            if f > frame:
                break
            b = next((x for x in self.boxes_by_frame.get(f, ()) if x.track_id == track_id), None)
            if b is not None and b.vis in VISIBLE:
                best = b
        return best


def _interval_lookup(intervals: Sequence[Sequence[Any]], frame: int) -> str | None:
    for iv in intervals:
        if int(iv[0]) <= frame <= int(iv[1]):
            return str(iv[2])
    return None


def load_clip(path: Path, eval_dir: Path | None = None) -> ClipAnnotation:
    """Load + validate one annotation file. Raises AnnotationError with the reason."""
    path = Path(path)
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AnnotationError(f"{path}: unreadable JSON ({exc})") from exc
    if d.get("format") != FORMAT:
        raise AnnotationError(f"{path}: format must be {FORMAT!r}, got {d.get('format')!r}")
    base = Path(eval_dir) if eval_dir else path.parent.parent
    video = Path(d.get("video", ""))
    if not video.is_absolute():
        video = (base / video).resolve()
    for key in ("clip_id", "fps", "width", "height", "frame_count"):
        if key not in d:
            raise AnnotationError(f"{path}: missing '{key}'")
    fps, fc = float(d["fps"]), int(d["frame_count"])
    split = d.get("split", "dev")
    if split not in ("dev", "test"):
        raise AnnotationError(f"{path}: split must be dev|test")
    tracks_raw = d.get("tracks") or []
    labelled = bool(d.get("labelled", bool(tracks_raw)))

    # Tier-B ("events only") clips have no person boxes: no frame is box-annotated.
    af = d.get("annotated_frames", "all" if tracks_raw else [])
    if af == "all":
        annotated = set(range(fc))
    elif isinstance(af, dict) and "every" in af:
        step = max(1, int(af["every"]))
        annotated = set(range(int(af.get("start", 0)), fc, step))
    elif isinstance(af, list):
        annotated = {int(x) for x in af}
    else:
        raise AnnotationError(f"{path}: annotated_frames must be 'all', {{'every': N}} or a list")
    if not labelled:
        annotated = set()

    unknown_cond = set(d.get("conditions", [])) - set(CONDITIONS)
    if unknown_cond:
        raise AnnotationError(f"{path}: unknown conditions {sorted(unknown_cond)} (allowed: {CONDITIONS})")
    frame_conditions = []
    for iv in d.get("frame_conditions", []) or []:
        tags = set(iv[2])
        if tags - set(CONDITIONS):
            raise AnnotationError(f"{path}: unknown frame condition {sorted(tags - set(CONDITIONS))}")
        frame_conditions.append((int(iv[0]), int(iv[1]), tags))

    boxes: dict[int, list[GTBox]] = {}
    tracks: dict[int, dict[str, Any]] = {}
    for tr in tracks_raw:
        tid = int(tr["id"])
        if tid in tracks:
            raise AnnotationError(f"{path}: duplicate track id {tid}")
        state_iv = tr.get("state", []) or []
        act_iv = tr.get("activity", []) or []
        for iv in state_iv:
            if iv[2] not in STATES:
                raise AnnotationError(f"{path}: track {tid} state {iv[2]!r} not in {STATES}")
        for iv in act_iv:
            if iv[2] not in ACTIVITIES:
                raise AnnotationError(f"{path}: track {tid} activity {iv[2]!r} not in {ACTIVITIES}")
        frames: list[int] = []
        for b in tr.get("boxes", []) or []:
            f = int(b[0])
            x1, y1, x2, y2 = (float(v) for v in b[1:5])
            vis = str(b[5]) if len(b) > 5 else "full"
            if vis not in VISIBILITY:
                raise AnnotationError(f"{path}: track {tid} frame {f} vis {vis!r} not in {VISIBILITY}")
            if x2 <= x1 or y2 <= y1:
                raise AnnotationError(f"{path}: track {tid} frame {f} degenerate box")
            st = _interval_lookup(state_iv, f)
            boxes.setdefault(f, []).append(
                GTBox(
                    track_id=tid,
                    frame=f,
                    bbox=(x1, y1, x2, y2),
                    vis=vis,
                    in_water=None if st is None else st in MONITORED_STATES,
                    activity=_interval_lookup(act_iv, f),
                    ignore=bool(tr.get("ignore", False)),
                    zone_state=st,
                )
            )
            frames.append(f)
        tracks[tid] = {
            "id": tid,
            "age": tr.get("age", "unknown"),
            "state": state_iv,
            "activity": act_iv,
            "ignore": bool(tr.get("ignore", False)),
            "_frames": sorted(frames),
        }

    events: list[GTEvent] = []
    for ev in d.get("events", []) or []:
        if ev.get("type") not in EVENT_TYPES:
            raise AnnotationError(f"{path}: event {ev.get('id')} type must be one of {EVENT_TYPES}")
        if int(ev["track_id"]) not in tracks:
            raise AnnotationError(f"{path}: event {ev.get('id')} references unknown track {ev['track_id']}")
        events.append(
            GTEvent(
                str(ev.get("id", len(events))),
                int(ev["track_id"]),
                ev["type"],
                int(ev["start_frame"]),
                int(ev["end_frame"]),
                bool(ev.get("staged", True)),
            )
        )

    polys = [[(float(x), float(y)) for x, y in poly] for poly in d.get("water_polygons", []) or []]
    return ClipAnnotation(
        clip_id=str(d["clip_id"]),
        video_path=video,
        camera_id=str(d.get("camera_id", d["clip_id"])),
        split=split,
        fps=fps,
        width=int(d["width"]),
        height=int(d["height"]),
        frame_count=fc,
        labelled=labelled,
        conditions=set(d.get("conditions", [])),
        annotated=annotated,
        boxes_by_frame=boxes,
        tracks=tracks,
        events=events,
        frame_conditions=frame_conditions,
        water_polygons=polys,
        raw=d,
    )


def load_manifest(eval_dir: Path) -> list[ClipAnnotation]:
    """Every clip listed in eval/manifest.json (or every annotations/*.json if none)."""
    eval_dir = Path(eval_dir)
    manifest = eval_dir / "manifest.json"
    paths: list[Path] = []
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        for entry in m.get("clips", []):
            p = Path(entry["annotation"])
            paths.append(p if p.is_absolute() else eval_dir / p)
    else:
        paths = sorted((eval_dir / "annotations").glob("*.json"))
    return [load_clip(p, eval_dir) for p in paths]
