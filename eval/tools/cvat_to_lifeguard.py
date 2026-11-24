"""
[v3] Ported from the v2 baseline eval/tools/cvat_to_lifeguard.py (Phase 8); output validated by
lifeguard.eval.annotations (backend/lifeguard/eval/annotations.py).

eval/tools/cvat_to_lifeguard.py — convert a CVAT "CVAT for video 1.1" export
into the Life Guard eval format (aquaguard-eval/1).

    python eval/tools/cvat_to_lifeguard.py annotations.xml \
        --video eval/clips/poolA_cam1_2026-10-02_1400.mp4 \
        --clip-id poolA_cam1_2026-10-02_1400 --camera-id poolA-cam1 --split test \
        --events eval/clips/poolA_cam1_2026-10-02_1400.events.csv \
        --every 5 --out eval/annotations/poolA_cam1_2026-10-02_1400.json

CVAT project setup (see eval/README.md §4):
  label "person" (rectangle tracks) with attributes
      visibility  select, mutable   full | partial | head | underwater | invisible
      state       select, mutable   in_water | edge | out_of_water
      activity    select, mutable   see ACTIVITIES in backend/evaluation/annotations.py
      age         select            adult | child
      ignore      checkbox          reflections, people behind glass, posters …
  tag labels (frame tags) named after conditions: glare, night, splash, …

events CSV columns: track_id,type,start_frame,end_frame[,staged]
  (track_id = the CVAT object id shown in the UI; type = ACTIVE_DROWNING |
  PASSIVE_DROWNING | SUBMERSION | FLOATABLE_FAILURE | DISTRESS)

Only stdlib + OpenCV (for fps / size of the video).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET  # nosec B405  # own CVAT export only, see convert()
from pathlib import Path

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


def _runs(values: list[tuple[int, str | None]]) -> list[list]:
    """[(frame, value)] sorted → [[start, end, value]] with end = frame before the next change."""
    out: list[list] = []
    for f, v in values:
        if v is None:
            continue
        if out and out[-1][2] == v:
            out[-1][1] = f
        else:
            if out:
                out[-1][1] = max(out[-1][1], f - 1)
            out.append([f, f, v])
    return out


def convert(
    xml_path: Path,
    video: Path,
    clip_id: str,
    camera_id: str,
    split: str,
    events_csv: Path | None,
    every: int | None,
    conditions: list[str],
) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # our own self-hosted CVAT export; CPython 3.12 expat: no external entities, expansion limits on
    root = ET.parse(xml_path).getroot()  # noqa: S314  # nosec B314
    tracks = []
    for tr in root.iter("track"):
        if tr.get("label") != "person":
            continue
        tid = int(tr.get("id"))
        boxes, states, acts = [], [], []
        age, ignore = "unknown", False
        for b in tr.iter("box"):
            f = int(b.get("frame"))
            if b.get("outside") == "1":
                states.append((f, None))
                continue
            attrs = {a.get("name"): (a.text or "").strip() for a in b.iter("attribute")}
            vis = attrs.get("visibility") or ("partial" if b.get("occluded") == "1" else "full")
            age = attrs.get("age", age) or age
            ignore = ignore or attrs.get("ignore", "false").lower() == "true"
            if every and f % every != 0:
                pass  # keep state/activity, skip box
            else:
                boxes.append(
                    [
                        f,
                        round(float(b.get("xtl")), 1),
                        round(float(b.get("ytl")), 1),
                        round(float(b.get("xbr")), 1),
                        round(float(b.get("ybr")), 1),
                        vis,
                    ]
                )
            states.append((f, attrs.get("state") or None))
            acts.append((f, attrs.get("activity") or None))
        states.sort(key=lambda x: x[0])
        acts.sort(key=lambda x: x[0])
        tracks.append(
            {
                "id": tid,
                "age": age,
                "ignore": ignore,
                "boxes": sorted(boxes),
                "state": _runs(states),
                "activity": _runs(acts),
            }
        )

    tag_frames: dict[str, list[int]] = {}
    for tg in root.iter("tag"):
        lab = tg.get("label")
        if lab in CONDITIONS:
            tag_frames.setdefault(lab, []).append(int(tg.get("frame")))
    frame_conditions = []
    for lab, frames in tag_frames.items():
        frames.sort()
        start = prev = frames[0]
        for f in [*frames[1:], None]:
            if f is None or f != prev + 1:
                frame_conditions.append([start, prev, [lab]])
                if f is not None:
                    start = f
            if f is not None:
                prev = f

    events = []
    if events_csv:
        with Path(events_csv).open(newline="", encoding="utf-8") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                events.append(
                    {
                        "id": f"e{i + 1}",
                        "track_id": int(row["track_id"]),
                        "type": row["type"].strip(),
                        "start_frame": int(row["start_frame"]),
                        "end_frame": int(row["end_frame"]),
                        "staged": str(row.get("staged", "true")).strip().lower() != "false",
                    }
                )

    return {
        "format": "aquaguard-eval/1",
        "clip_id": clip_id,
        "video": str(video).replace("\\", "/"),
        "camera_id": camera_id,
        "split": split,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "annotated_frames": {"every": every} if every else "all",
        "conditions": conditions,
        "frame_conditions": frame_conditions,
        "tracks": tracks,
        "events": events,
        "source": f"cvat:{xml_path.name}",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xml", type=Path)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--camera-id", required=True)
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--events", type=Path, default=None)
    ap.add_argument("--every", type=int, default=None, help="keep boxes only on every Nth frame")
    ap.add_argument("--conditions", default="", help="clip-wide condition tags, comma-separated")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    cond = [c for c in a.conditions.split(",") if c]
    bad = set(cond) - set(CONDITIONS)
    if bad:
        raise SystemExit(f"unknown conditions {sorted(bad)}; allowed {CONDITIONS}")
    doc = convert(a.xml, a.video, a.clip_id, a.camera_id, a.split, a.events, a.every, cond)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    # validate with the harness loader
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
    from lifeguard.eval.annotations import load_clip  # [v3] package path

    c = load_clip(a.out, a.out.parent.parent)
    sys.stdout.write(
        f"wrote {a.out}: {len(c.tracks)} tracks, {sum(len(v) for v in c.boxes_by_frame.values())} boxes, "
        f"{len(c.events)} events, {len(c.annotated)} annotated frames\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
