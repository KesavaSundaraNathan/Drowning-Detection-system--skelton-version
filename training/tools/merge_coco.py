"""Phase 3 (retrain A): merge several COCO json files, each with its own image folder, into one json that D-FINE's
dataloader can read with a single img_folder. Every file_name becomes an absolute path (os.path.join with an
absolute second part ignores img_folder), image/annotation ids are renumbered, and the categories must match
across inputs (0 swimmer, 1 person - training/tools/yolo_to_coco_lean.py, hardneg.py). Prints counts only.

    training\\.venv\\Scripts\\python.exe training\\tools\\merge_coco.py --out data/dataset/merged_v2/train.json \\
        --part training/data/frames=training/data/coco/sprint_r1/train.json \\
        --part data/dataset/lean_underwater_v1/images=data/dataset/lean_underwater_v1/coco/train.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def merge(parts: list[tuple[Path, Path]], dedupe: bool = False) -> dict:
    images, annotations, categories = [], [], None
    # --dedupe: drop an image already merged (first input wins). Off by default: repeated file_names in these sets
    # are deliberate oversampling (seed_r1 = reviewed x2 + auto-accepted x1; sprint_r1 contains all of seed_r1).
    # Keys are os.path.normcase'd so the same file spelled in another case counts as a repeat on Windows.
    seen: set[str] = set()
    for img_folder, ann_file in parts:
        data = json.loads(ann_file.read_text(encoding="utf-8"))
        cats = sorted((c["id"], c["name"]) for c in data["categories"])
        if categories is None:
            categories = data["categories"]
        elif cats != sorted((c["id"], c["name"]) for c in categories):
            raise SystemExit(f"{ann_file}: categories {cats} differ from the first input's")
        boxes_per_image = Counter(a["image_id"] for a in data["annotations"])
        id_map: dict[int, int] = {}
        dup_ids: set[int] = set()
        for im in data["images"]:
            name = str((img_folder / im["file_name"]).resolve())
            key = os.path.normcase(name)
            if dedupe and key in seen:
                dup_ids.add(im["id"])
                if boxes_per_image[im["id"]]:
                    print(  # noqa: T201
                        f"WARNING: {ann_file}: dropped repeated image {name} (id {im['id']}) "
                        f"with {boxes_per_image[im['id']]} boxes; the first input's copy is kept"
                    )
                continue
            seen.add(key)
            id_map[im["id"]] = len(images)
            images.append({**im, "id": len(images), "file_name": name})
        kept = [a for a in data["annotations"] if a["image_id"] not in dup_ids]
        for a in kept:
            if a["image_id"] not in id_map:
                raise SystemExit(
                    f"{ann_file}: annotation {a.get('id')} refers to image_id {a['image_id']}, "
                    "which is not in this file's images"
                )
            annotations.append({**a, "id": len(annotations), "image_id": id_map[a["image_id"]]})
        per_cat = Counter(a["category_id"] for a in kept)
        print(  # noqa: T201
            f"{ann_file}: {len(data['images'])} images ({len(dup_ids)} already seen, dropped), "
            f"{len(kept)} boxes kept, per category {dict(per_cat)}"
        )
    return {"images": images, "annotations": annotations, "categories": categories or []}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--part", action="append", required=True, help="IMG_FOLDER=ANN_FILE (repeat)")
    ap.add_argument("--dedupe", action="store_true", help="drop repeated images (first input wins)")
    a = ap.parse_args()
    parts = [(Path(f), Path(j)) for f, j in (p.split("=", 1) for p in a.part)]
    merged = merge(parts, a.dedupe)
    missing = sum(not Path(im["file_name"]).is_file() for im in merged["images"])
    if missing:
        raise SystemExit(f"{missing} image files missing - not writing {a.out}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(merged), encoding="utf-8")
    print(f"{a.out}: {len(merged['images'])} images, {len(merged['annotations'])} boxes")  # noqa: T201


if __name__ == "__main__":
    main()
