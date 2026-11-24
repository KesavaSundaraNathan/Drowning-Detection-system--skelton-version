"""Pre-annotations for the CVAT box-labelling pass (Phase 8; docs/DATA_PROTOCOL.md section 6).

    python tools/cvat_preannotate.py --trace build/phase8/levers/baseline/trace.jsonl.gz \
        --eval-dir eval/heldout_v1 --split test --out build/cvat/heldout_v1_test

For every clip of the split, from a harness trace (`lifeguard.eval.evaluate --trace`, i.e. the real pipeline:
detector + ByteTrack + ID stitching), it writes:

  <clip>.xml          CVAT for video 1.1: one `person` track per public track id, a keyframe box every
                      `--hz` (default 2 Hz, the plan's keyframe rate) while the track is visible, `outside`
                      when it ends; attributes visibility=full (the labeller corrects it), state from the
                      pipeline's zone state (in_water / edge / out_of_water), source="auto". Tentative
                      (unconfirmed, id -1) people are not pre-filled: the labeller adds anyone missed.
  <clip>.events.csv   the clip's event windows from the existing event-level ground truth, with
                      track_id = -1: the reviewer replaces it with the CVAT id of the victim's track.
  README.txt          what to import where (deploy/cvat/README.md has the full procedure).

These are SUGGESTIONS from a detector that caught 1 of 5 test events (Phase 7); only labels a person has
reviewed count (plan 7.1). After review, eval/tools/cvat_to_lifeguard.py converts the export back.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import xml.etree.ElementTree as ET  # nosec B405  # writing only
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATES = {"in_water", "edge", "out_of_water"}
LABEL_SPEC = ROOT / "deploy" / "cvat" / "labels.json"


def _labels_xml(parent: ET.Element) -> None:
    labels = ET.SubElement(parent, "labels")
    for lab in json.loads(LABEL_SPEC.read_text(encoding="utf-8")):
        el = ET.SubElement(labels, "label")
        ET.SubElement(el, "name").text = lab["name"]
        ET.SubElement(el, "type").text = lab.get("type", "any")
        attrs = ET.SubElement(el, "attributes")
        for at in lab.get("attributes", []):
            a = ET.SubElement(attrs, "attribute")
            ET.SubElement(a, "name").text = at["name"]
            ET.SubElement(a, "mutable").text = str(at.get("mutable", False)).lower()
            ET.SubElement(a, "input_type").text = at["input_type"]
            ET.SubElement(a, "default_value").text = str(at.get("default_value", ""))
            ET.SubElement(a, "values").text = "\n".join(str(v) for v in at.get("values", []))


def read_trace(path: Path) -> dict[str, dict[int, list[list[Any]]]]:
    out: dict[str, dict[int, list[list[Any]]]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out.setdefault(r["clip"], {})[int(r["f"])] = r["people"]
    return out


def clip_xml(clip: Any, frames: dict[int, list[list[Any]]], every: int) -> ET.Element:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    task = ET.SubElement(ET.SubElement(root, "meta"), "task")
    ET.SubElement(task, "name").text = clip.clip_id
    ET.SubElement(task, "size").text = str(clip.frame_count)
    ET.SubElement(task, "mode").text = "interpolation"
    ET.SubElement(task, "start_frame").text = "0"
    ET.SubElement(task, "stop_frame").text = str(clip.frame_count - 1)
    _labels_xml(task)
    size = ET.SubElement(task, "original_size")
    ET.SubElement(size, "width").text = str(clip.width)
    ET.SubElement(size, "height").text = str(clip.height)
    tracks: dict[int, list[tuple[int, list[Any]]]] = {}
    for f in sorted(frames):
        for p in frames[f]:
            if int(p[0]) > 0:
                tracks.setdefault(int(p[0]), []).append((f, p))
    for n, (tid, seen) in enumerate(sorted(tracks.items())):
        kept = [(f, p) for f, p in seen if f % every == 0]
        if not kept:
            continue
        tr = ET.SubElement(root, "track", id=str(n), label="person", source="auto")
        tr.set("lifeguard_track", str(tid))
        for f, p in kept:
            box = ET.SubElement(tr, "box", frame=str(f), keyframe="1", outside="0", occluded="0", z_order="0")
            box.set("xtl", f"{max(0.0, float(p[1])):.1f}")
            box.set("ytl", f"{max(0.0, float(p[2])):.1f}")
            box.set("xbr", f"{min(float(clip.width), float(p[3])):.1f}")
            box.set("ybr", f"{min(float(clip.height), float(p[4])):.1f}")
            ET.SubElement(box, "attribute", name="visibility").text = "full"
            ET.SubElement(box, "attribute", name="state").text = p[7] if p[7] in STATES else "in_water"
        last = seen[-1][0]
        if last + 1 < clip.frame_count:  # the track ends: an `outside` key so CVAT stops interpolating
            p = seen[-1][1]
            end = ET.SubElement(tr, "box", frame=str(last + 1), keyframe="1", outside="1", occluded="0", z_order="0")
            end.set("xtl", f"{max(0.0, float(p[1])):.1f}")
            end.set("ytl", f"{max(0.0, float(p[2])):.1f}")
            end.set("xbr", f"{min(float(clip.width), float(p[3])):.1f}")
            end.set("ybr", f"{min(float(clip.height), float(p[4])):.1f}")
    return root


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(ROOT / "backend"))
    from lifeguard.eval.annotations import load_manifest

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=("dev", "test"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hz", type=float, default=2.0)
    a = ap.parse_args(argv)
    trace = read_trace(a.trace)
    a.out.mkdir(parents=True, exist_ok=True)
    lines = []
    for clip in load_manifest(a.eval_dir):
        if clip.split != a.split:
            continue
        if clip.clip_id not in trace:
            raise SystemExit(f"{clip.clip_id}: not in the trace - run the harness on this split first")
        every = max(1, round(clip.fps / a.hz))
        root = clip_xml(clip, trace[clip.clip_id], every)
        ET.indent(root)
        ET.ElementTree(root).write(a.out / f"{clip.clip_id}.xml", encoding="utf-8", xml_declaration=True)
        with (a.out / f"{clip.clip_id}.events.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["track_id", "type", "start_frame", "end_frame", "staged"])
            for ev in clip.events:
                w.writerow([-1, ev.type, ev.start_frame, ev.end_frame, "false"])
        n = sum(1 for _ in root.iter("track"))
        lines.append(f"{clip.clip_id}: {n} tracks, keyframe every {every} frames, video {clip.video_path.name}")
    (a.out / "README.txt").write_text(
        "CVAT pre-annotations (suggestions - review every box). Import each <clip>.xml into its task with\n"
        "'Upload annotations' -> format 'CVAT 1.1'. Fill track_id in <clip>.events.csv after review.\n"
        "See deploy/cvat/README.md.\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
