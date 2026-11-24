"""Lean Data Strategy (Phase 3): convert extract_and_prelabel.py's YOLO .txt output + MakeSense.ai's
manual fill-in of the submerged misses into the COCO train/val json D-FINE's dataloader expects
(training/configs/dfine_l_v0.yml's ann_file convention - see training/tools/hardneg.py:145,224 for the
category_id convention this matches: 0 swimmer, 1 person).

Every box in data/dataset/lean_underwater_v1/labels is written as category_id 0 (swimmer) - owner
decision 2026-11-14: these 30 clips were selected specifically for underwater/submerged recall, so
mapping the dataset's single annotated class onto "person_out_of_water" (1) instead would be wrong more
often than right. Split is by CLIP, not by frame (adjacent 1 fps frames of the same clip are highly
correlated - a frame-level split would leak into val).

    training\\.venv\\Scripts\\python.exe training\\tools\\yolo_to_coco_lean.py
      [--dataset-dir data/dataset/lean_underwater_v1] [--val-frac 0.15]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

CATEGORIES = [{"id": 0, "name": "swimmer"}, {"id": 1, "name": "person"}]
MAPPED_CATEGORY_ID = 0  # every box in this dataset -> swimmer (owner decision 2026-11-14)


def build_split(dataset_dir: Path, clips: list[str]) -> dict:
    images, annotations = [], []
    image_id, ann_id = 0, 0
    for clip in clips:
        img_dir = dataset_dir / "images" / clip
        lbl_dir = dataset_dir / "labels" / clip
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue  # no label file at all (not even empty) - extraction didn't finish this frame
            w, h = Image.open(img_path).size
            images.append(
                {
                    "id": image_id,
                    "file_name": f"{clip}/{img_path.name}",
                    "width": w,
                    "height": h,
                }
            )
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                _cls, cx, cy, bw, bh = line.split()
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                x, y = cx - bw / 2.0, cy - bh / 2.0
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": MAPPED_CATEGORY_ID,
                        "bbox": [x, y, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
            image_id += 1
    return {"images": images, "annotations": annotations, "categories": CATEGORIES}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default="data/dataset/lean_underwater_v1")
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    clips = sorted(p.name for p in (dataset_dir / "images").iterdir() if p.is_dir())
    if not clips:
        raise SystemExit(f"no clips under {dataset_dir / 'images'} - run extract_and_prelabel.py first")

    n_val = max(1, round(len(clips) * args.val_frac)) if len(clips) > 1 else 0
    val_clips, train_clips = clips[-n_val:] if n_val else [], clips[: len(clips) - n_val]

    out_dir = dataset_dir / "coco"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_clips in (("train", train_clips), ("val", val_clips)):
        coco = build_split(dataset_dir, split_clips)
        (out_dir / f"{name}.json").write_text(json.dumps(coco), encoding="utf-8")
        print(  # noqa: T201
            f"{name}: {len(split_clips)} clips, {len(coco['images'])} images, "
            f"{len(coco['annotations'])} boxes (all category_id {MAPPED_CATEGORY_ID}) -> {out_dir / f'{name}.json'}"
        )


if __name__ == "__main__":
    main()
