r"""Sprint retrain data (owner request 2026-11-12): auto-labelled frames of the NEW footage for a D-FINE-L fine-tune.

The pool-tuned detector (v0 / v0.1) was trained on ONE camera (Red Rescue). It finds 53 % of the people in a crowded
frame of that pool where the COCO RTMDet-m of the live path finds 2-8 %, but on the owner's phone clips it reports
dozens of junk tracks. This tool adds the other views:

  frames : DEV clips only (eval/sprint_v1 dev + eval/heldout_v1 dev; test clips are never read) -> JPEG frames,
           at most 1920 px wide, under training/data/frames/<clip id>/
  vote   : the three COCO voters of voters.py on every frame (raw boxes -> training/labels/raw_voters/sprint.jsonl)
  build  : consensus (>= 2 of 3 voters, consensus.py thresholds and clustering) -> boxes; class `swimmer` when the
           lower part of the box sits in water-coloured pixels, else `person`; lens occluders (a tall sliver on the
           frame edge, a box wider than 60 % of the frame) are dropped, so those frames teach "not a person";
           merged with the v0.1 training set -> training/data/coco/sprint_r1/train.json

Labels are NOT hand labels and nobody audited them: the model trained from them replaces the current one only if it
wins on the reviewed val frames and on the dev clips.

    training\.venv\Scripts\python.exe training\tools\sprint_labels.py frames|vote|build
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "training"
FR = TR / "data" / "frames"
RAW = TR / "labels" / "raw_voters" / "sprint.jsonl"
OUT = TR / "data" / "coco" / "sprint_r1"
BASE = TR / "data" / "coco" / "hardneg_r1" / "train.json"
MAX_W = 1920
FPS_PHONE, FPS_WEB, MAX_PER_CLIP = 1.5, 0.5, 70


def dev_clips() -> list[tuple[str, Path, float]]:
    out = []
    for c in json.loads((ROOT / "eval/sprint_v1/clips.json").read_text(encoding="utf-8"))["clips"]:
        if c["split"] == "dev" and c["session"] != "red_rescue":  # the pool camera is already in the base set
            out.append((f"sp_{c['id']}", ROOT / c["video"], FPS_PHONE if c["id"].startswith("ip_") else FPS_WEB))
    held = ROOT / "eval/heldout_v1"
    for c in json.loads((held / "manifest.json").read_text(encoding="utf-8"))["clips"]:
        if c["split"] == "dev":
            ann = json.loads((held / c["annotation"]).read_text(encoding="utf-8"))
            out.append((f"sp_{c['clip_id']}", (held / ann["video"]).resolve(), FPS_WEB))
    return out


def frames() -> None:
    for cid, video, rate in dev_clips():
        d = FR / cid
        if (d / "done").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(fps / rate), -(-n // MAX_PER_CLIP))
        k = saved = 0
        while cap.grab():
            if k % step == 0:
                ok, img = cap.retrieve()
                if ok:
                    if img.shape[1] > MAX_W:
                        s = MAX_W / img.shape[1]
                        img = cv2.resize(img, (MAX_W, round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(d / f"f{k:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    saved += 1
            k += 1
        (d / "done").write_text(str(saved))
        print(f"{cid}: {saved} frames (every {step})", flush=True)


def vote() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import voters as V

    keys = [f"{cid}/{p.name}" for cid, _v, _r in dev_clips() for p in sorted((FR / cid).glob("*.jpg"))]
    done = set()
    if RAW.exists():
        done = {json.loads(x)["key"] for x in RAW.read_text(encoding="utf-8").splitlines() if x.strip()}
    todo = [k for k in keys if k not in done]
    print(f"{len(keys)} frames, {len(todo)} to run", flush=True)
    vs = {"dfine": V.DFine(), "rtmdet": V.RTMDet(), "owlv2": V.OWLv2()}
    with RAW.open("a", encoding="utf-8") as f:
        for i, key in enumerate(todo):
            img = cv2.imread(str(FR / key))
            rec = {"key": key, "h": img.shape[0], "w": img.shape[1]}
            for name, v in vs.items():
                rec[name] = v(img)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(todo)}", flush=True)


def water_fraction(img: np.ndarray, box: np.ndarray) -> float:
    """Share of water-coloured pixels (the app's hue band) in a band around the lower 60 % of the box."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    ax1, ax2 = int(max(0, x1 - 0.4 * bw)), int(min(w, x2 + 0.4 * bw))
    ay1, ay2 = int(max(0, y1 + 0.4 * bh)), int(min(h, y2 + 0.3 * bh))
    if ax2 - ax1 < 2 or ay2 - ay1 < 2:
        return 0.0
    hsv = cv2.cvtColor(img[ay1:ay2, ax1:ax2], cv2.COLOR_BGR2HSV)
    m = (hsv[..., 0] >= 78) & (hsv[..., 0] <= 132) & (hsv[..., 1] >= 35) & (hsv[..., 2] >= 40)
    return float(m.mean())


def dedupe(kept: list[tuple]) -> tuple[list[tuple], int]:
    """One box per person: the consensus can leave 2-3 overlapping instances on one swimmer when the voters disagree
    on its extent (they then cannot pass MATCH_IOU with each other). Keep the best-supported box (most voters, then
    the highest summed score) and drop any other box with IoU >= 0.45 with it or lying >= 0.75 inside it / around it."""
    order = sorted(range(len(kept)), key=lambda i: (-kept[i][5], -kept[i][6]))
    out: list[tuple] = []
    for i in order:
        x1, y1, w, h = kept[i][:4]
        a = np.array([x1, y1, x1 + w, y1 + h])
        clash = False
        for o in out:
            b = np.array([o[0], o[1], o[0] + o[2], o[1] + o[3]])
            iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = iw * ih
            ua, ub = w * h, o[2] * o[3]
            if inter / (ua + ub - inter + 1e-9) >= 0.45 or inter / max(min(ua, ub), 1e-9) >= 0.75:
                clash = True
                break
        if not clash:
            out.append(kept[i])
    return out, len(kept) - len(out)


def build() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import consensus as C

    base = json.loads(BASE.read_text(encoding="utf-8"))
    images, anns = list(base["images"]), list(base["annotations"])
    iid, aid = max(i["id"] for i in images) + 1, max(a["id"] for a in anns) + 1
    stats = {"frames": 0, "boxes": 0, "swimmer": 0, "person": 0, "dropped_occluder": 0, "single_voter": 0, "empty": 0}
    for line in RAW.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        img = cv2.imread(str(FR / rec["key"]))
        h, w = rec["h"], rec["w"]
        kept = []
        for inst in C.cluster(rec, C.THR):
            if len(inst["voters"]) < 2:
                stats["single_voter"] += 1
                continue
            x1, y1, x2, y2 = (float(v) for v in inst["box"])
            bw, bh = x2 - x1, y2 - y1
            edge = x1 <= 0.01 * w or x2 >= 0.99 * w or y1 <= 0.01 * h or y2 >= 0.99 * h
            if bw >= 0.6 * w or bw * bh >= 0.3 * w * h or (edge and bh >= 0.25 * h and bh >= 4 * bw):
                stats["dropped_occluder"] += 1
                continue
            if bw < 4 or bh < 4:
                continue
            cls = 0 if water_fraction(img, inst["box"]) >= 0.3 else 1
            kept.append((x1, y1, bw, bh, cls, len(inst["voters"]), sum(inst["voters"].values())))
        kept, dup = dedupe(kept)
        stats["dropped_duplicate"] = stats.get("dropped_duplicate", 0) + dup
        images.append({"id": iid, "file_name": rec["key"], "width": w, "height": h})
        for x1, y1, bw, bh, cls, *_ in kept:
            anns.append({"id": aid, "image_id": iid, "category_id": cls, "bbox": [round(x1, 1), round(y1, 1), round(bw, 1),
                         round(bh, 1)], "area": round(bw * bh, 1), "iscrowd": 0})
            aid += 1
            stats["swimmer" if cls == 0 else "person"] += 1
        stats["frames"] += 1
        stats["boxes"] += len(kept)
        stats["empty"] += not kept
        iid += 1
    OUT.mkdir(parents=True, exist_ok=True)
    doc = dict(base, images=images, annotations=anns)
    (OUT / "train.json").write_text(json.dumps(doc), encoding="utf-8")
    stats["total_images"], stats["total_boxes"] = len(images), len(anns)
    (OUT / "summary.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    {"frames": frames, "vote": vote, "build": build}[sys.argv[1]]()
