"""Phase 3: build eval/underwater_v1, a victim-continuity set, from the lean underwater labels (owner, 2026-11-15).

    python eval/tools/lean_to_continuity.py --verify

Source: data/dataset/lean_underwater_v1 (extract_and_prelabel.py at 1 Hz over `Drowning Dataset/`, auto-label plus the
owner's MakeSense.ai fill-in, converted to COCO by training/tools/yolo_to_coco_lean.py). Image `<CLIP>/frame_<i>.jpg`
is video frame i * round(src_fps / 1.0) (extract_and_prelabel.sampled_frame_indices).

Per clip it writes annotations/<clip_id>.json (aquaguard-eval/1, split "dev"): one track (id 1, the subject) with a
box on every labelled 1 Hz frame, `annotated_frames` = those frames, no events, and `continuity_track: 1` - the key
that makes lifeguard.eval.metrics score the track's continuity over the whole clip. Frames without a box are
"subject not visible" and are not keyframes. Where a frame has several boxes, the one nearest the previous kept box
is the subject. The labels carry no visibility, so every box is "partial"; no state, no activity.

These clips were lean_v2's training data (train.json clips): `lean_split` in manifest.json records it, and a
continuity number for a model trained on a clip is not evidence. The footage has no written rights: local eval only,
never a release gate. Prints counts only; writes no media.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEAN = ROOT / "data" / "dataset" / "lean_underwater_v1"
VIDEOS = ROOT / "Drowning Dataset"
OUT = ROOT / "eval" / "underwater_v1"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _video_meta(p: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(p))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, fr = cap.read()
        if not ok:
            raise SystemExit(f"{p.name}: cannot decode")
        return fps, n, fr.shape[1], fr.shape[0]
    finally:
        cap.release()


def _seek_matches_sequential(p: Path, frames: list[int]) -> float:
    """Max mean-abs pixel difference between seeking to a frame (how the labels were extracted) and decoding up to it
    in order (how the harness reads it). ~0 means a label's frame number is the frame the harness analyses."""
    want = set(frames)
    seq: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(p))
    i = 0
    while want - seq.keys():
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            seq[i] = cv2.resize(fr, (320, 180)).astype(np.float32)
        i += 1
    cap.release()
    worst = 0.0
    cap = cv2.VideoCapture(str(p))
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok or f not in seq:
            return float("inf")
        worst = max(worst, float(np.abs(cv2.resize(fr, (320, 180)).astype(np.float32) - seq[f]).mean()))
    cap.release()
    return worst


def _pick(boxes: list[list[float]], prev: list[float] | None) -> list[float]:
    if len(boxes) == 1 or prev is None:
        return boxes[0]
    pc = np.array([(prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2])
    return min(boxes, key=lambda b: float(np.linalg.norm(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) - pc)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="check seek vs sequential decode on 3 labelled frames/clip")
    a = ap.parse_args()

    labels: dict[str, dict[int, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    lean_split: dict[str, str] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for split in ("train", "val"):
        d = json.loads((LEAN / "coco" / f"{split}.json").read_text(encoding="utf-8"))
        ims = {im["id"]: im for im in d["images"]}
        for im in d["images"]:
            clip = im["file_name"].replace("\\", "/").split("/")[0]
            lean_split[clip] = split
            sizes[clip] = (im["width"], im["height"])
        for an in d["annotations"]:
            im = ims[an["image_id"]]
            clip, name = im["file_name"].replace("\\", "/").split("/")[:2]
            i = int(Path(name).stem.split("_")[1])
            x, y, w, h = an["bbox"]
            labels[clip][i].append([x, y, x + w, y + h])

    (OUT / "annotations").mkdir(parents=True, exist_ok=True)
    entries, clips_meta = [], []
    for clip in sorted(lean_split):
        video = VIDEOS / f"{clip}.mov"
        if not video.is_file():
            raise SystemExit(f"{video} missing")
        fps, n, w, h = _video_meta(video)
        if (w, h) != sizes[clip]:
            raise SystemExit(f"{clip}: video decodes {w}x{h}, labels are {sizes[clip]} - orientation differs")
        step = max(1, round(fps / 1.0))
        boxes, prev, multi = [], None, 0
        for i in sorted(labels.get(clip, {})):
            cand = labels[clip][i]
            multi += len(cand) > 1
            b = _pick(cand, prev)
            prev = b
            boxes.append([i * step, *(round(v, 1) for v in b), "partial"])
        frames = [b[0] for b in boxes]
        diff = None
        if a.verify and frames:
            diff = _seek_matches_sequential(video, [frames[0], frames[len(frames) // 2], frames[-1]])
        cid = f"uw_{clip.lower()}"
        ann = {
            "format": "aquaguard-eval/1",
            "clip_id": cid,
            "video": f"../../Drowning Dataset/{clip}.mov",
            "camera_id": cid,
            "split": "dev",
            "labelled": bool(boxes),
            "fps": fps,
            "width": w,
            "height": h,
            "frame_count": n,
            "annotated_frames": frames,
            "conditions": [],
            "tracks": [{"id": 1, "age": "unknown", "boxes": boxes}] if boxes else [],
            "events": [],
            "continuity_track": 1,
        }
        (OUT / "annotations" / f"{cid}.json").write_text(json.dumps(ann, indent=1), encoding="utf-8")
        if boxes:  # a clip with no subject label has nothing to score: kept in clip_meta, not run
            entries.append({"clip_id": cid, "annotation": f"annotations/{cid}.json", "split": "dev", "labelled": True})
        clips_meta.append(
            {
                "clip_id": cid,
                "video": f"{clip}.mov",
                "video_sha256": _sha256(video),
                "lean_split": lean_split[clip],
                "keyframes": len(frames),
                "multi_box_frames": multi,
            }
        )
        print(  # noqa: T201
            f"{cid}: {fps:.2f} fps, {n} frames, step {step}, {len(frames)} keyframes, {multi} multi-box"
            + (f", seek-vs-sequential max diff {diff:.2f}" if diff is not None else "")
        )
    manifest = {
        "format": "aquaguard-eval-manifest/1",
        "name": "underwater_v1",
        "source": "data/dataset/lean_underwater_v1 (coco train+val), eval/tools/lean_to_continuity.py",
        "purpose": "victim track continuity at 1 Hz only (no events); lean_v2 trained on lean_split=train clips",
        "clips": entries,
        "clip_meta": clips_meta,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{OUT}: {len(entries)} clips, {sum(c['keyframes'] for c in clips_meta)} keyframes")  # noqa: T201


if __name__ == "__main__":
    main()
