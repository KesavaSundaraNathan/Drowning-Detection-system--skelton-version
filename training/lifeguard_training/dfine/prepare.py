"""
Our training set (scripts/export_training_set.py --classes zone: COCO json with
1-based category ids 1 = person_in_water, 2 = person_out_of_water) → the layout
D-FINE's custom-dataset loader reads with `remap_mscoco_category: False`, where
`category_id` IS the class index (0-based).
"""

from __future__ import annotations

import json
from pathlib import Path

from lifeguard_training.dfine import CLASSES


def prepare(dataset: Path, out: Path) -> dict[str, dict[str, int]]:
    dataset, out = Path(dataset), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        src = dataset / "annotations" / f"instances_{split}.json"
        if not src.exists():
            continue
        d = json.loads(src.read_text(encoding="utf-8"))
        names = [c["name"] for c in sorted(d["categories"], key=lambda c: c["id"])]
        if tuple(names) != CLASSES:
            raise ValueError(
                f"{src}: categories {names} != {list(CLASSES)} — export with "
                "scripts/export_training_set.py --classes zone"
            )
        old_to_new = {c["id"]: CLASSES.index(c["name"]) for c in d["categories"]}
        d["categories"] = [{"id": i, "name": n, "supercategory": "person"} for i, n in enumerate(CLASSES)]
        for a in d["annotations"]:
            a["category_id"] = old_to_new[a["category_id"]]
        (out / f"instances_{split}.json").write_text(json.dumps(d), encoding="utf-8")
        per = {n: sum(1 for a in d["annotations"] if a["category_id"] == i) for i, n in enumerate(CLASSES)}
        stats[split] = {"images": len(d["images"]), **per}
    if "train" not in stats or "val" not in stats:
        raise ValueError(f"{dataset}: needs annotations/instances_train.json and instances_val.json")
    return stats
