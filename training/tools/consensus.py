"""Phase 4: consensus of the three voters (raw detections from voters.py) -> auto-accepted boxes + a review queue.

Rules (training/MASTER_PROMPT.md Phase 4, owner-approved guide training/docs/LABELING_GUIDE.md):
- per-voter score thresholds (PROVISIONAL until the first audit; tuned from it);
- AUTO-ACCEPT an instance when >= 2 of 3 voters match it (IoU >= 0.6) above their thresholds;
- refine accepted boxes with SAM 2.1 (box prompt); keep the refined box only if IoU(refined, consensus) >= 0.7;
- FLAG for review: single-voter boxes, voter disagreement (best cross-voter IoU 0.3-0.6), very small boxes
  (< 20 px tall), crowded overlaps (IoU between accepted people > 0.3), boxes touching the frame border or straddling
  the pool-water edge;
- CLASS: `swimmer` if any part of the box's lower 60 % lies in the pool-water mask, `person` if none does, flagged when
  only a sliver (< 10 %) is in the water (edge sitters, ambiguous).
A frame with any flag goes to the review queue (hardest first = most flags); a frame with none is auto-accepted.
Output: training/labels/auto/<set>_consensus.jsonl and <set>_summary.json. Keypoints: separate step (pose.py).

    training\\.venv\\Scripts\\python.exe training\\tools\\consensus.py [--set pool] [--no-sam] [--thr dfine=0.35,...]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "training"
FR = TR / "data" / "frames"
THR = {"dfine": 0.35, "rtmdet": 0.20, "owlv2": 0.15}  # PROVISIONAL; see the summary for the per-threshold yields
MATCH_IOU, DISAGREE_IOU, SAM_KEEP_IOU = 0.6, 0.3, 0.7
SMALL_PX, CROWD_IOU = 20, 0.3


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N,4) and b (M,4)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def cluster(rec: dict, thr: dict) -> list[dict]:
    """Greedy cross-voter clustering of boxes above threshold. Returns instances with their voters."""
    boxes = []
    for v in ("dfine", "rtmdet", "owlv2"):
        for b in rec.get(v, []):
            if b[4] >= thr[v]:
                boxes.append((v, np.array(b[:4], float), b[4]))
    boxes.sort(key=lambda x: -x[2])
    inst: list[dict] = []
    for v, b, s in boxes:
        best, best_iou = None, 0.0
        for c in inst:
            if v in c["voters"]:
                continue
            o = float(iou(b[None], c["box"][None])[0, 0])
            if o > best_iou:
                best, best_iou = c, o
        if best is not None and best_iou >= MATCH_IOU:
            best["voters"][v] = s
            best["members"].append(b)
            w = np.array(list(best["voters"].values()))
            best["box"] = (np.stack(best["members"]) * w[:, None]).sum(0) / w.sum()
        else:
            inst.append({"box": b.copy(), "voters": {v: s}, "members": [b], "best_cross_iou": best_iou})
    return inst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="pool")
    ap.add_argument("--no-sam", action="store_true")
    ap.add_argument("--thr", default="")
    a = ap.parse_args()
    thr = dict(THR)
    for kv in filter(None, a.thr.split(",")):
        k, v = kv.split("=")
        thr[k] = float(v)
    mask = cv2.imread(str(TR / "data" / "pool_mask" / "pool_water_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    splits = json.loads((TR / "data" / "splits.json").read_text())
    split_of = {c[:-4]: s for s in ("train", "val") for c in splits[s]}
    sam = None
    if not a.no_sam:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml",
                                            str(TR / "weights" / "sam2.1_hiera_base_plus.pt"), device="cuda"))
    raw = [json.loads(line) for line in (TR / "labels" / "raw_voters" / f"{a.set}.jsonl").open(encoding="utf-8")]
    out_dir = TR / "labels" / "auto"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats, flags_all = Counter(), Counter()
    records = []
    for rec in raw:
        h, w = rec["h"], rec["w"]
        inst = cluster(rec, thr)
        accepted = [c for c in inst if len(c["voters"]) >= 2]
        frame_flags = []
        for c in inst:
            if len(c["voters"]) == 1:
                c["flags"] = ["single_voter"] + (["disagreement"] if c["best_cross_iou"] >= DISAGREE_IOU else [])
        img = None
        if sam is not None and accepted and "@" not in rec["key"]:  # dense keys: refined in the temporal step
            img = cv2.cvtColor(cv2.imread(str(FR / rec["key"])), cv2.COLOR_BGR2RGB)
            import torch

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                sam.set_image(img)
                for c in accepted:
                    m, _, _ = sam.predict(box=c["box"], multimask_output=False)
                    ys, xs = np.nonzero(m[0] > 0)
                    if len(xs):
                        rb = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)
                        o = float(iou(rb[None], c["box"][None])[0, 0])
                        c["sam_iou"] = round(o, 3)
                        if o >= SAM_KEEP_IOU:
                            c["box"] = rb
                            stats["sam_refined"] += 1
                        else:
                            stats["sam_rejected"] += 1
        acc_boxes = np.array([c["box"] for c in accepted]) if accepted else np.zeros((0, 4))
        cross = iou(acc_boxes, acc_boxes)
        for i, c in enumerate(accepted):
            f = []
            x1, y1, x2, y2 = c["box"]
            if y2 - y1 < SMALL_PX:
                f.append("small")
            if len(accepted) > 1 and (np.delete(cross[i], i) > CROWD_IOU).any():
                f.append("crowded_overlap")
            if x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2:
                f.append("frame_border")
            xi1, yi1 = int(max(0, x1)), int(max(0, y1 + 0.4 * (y2 - y1)))
            xi2, yi2 = int(min(w, x2)), int(min(h, y2))
            sub = mask[yi1:yi2, xi1:xi2]
            frac = float(sub.mean()) if sub.size else 0.0
            c["water_frac"] = round(frac, 3)
            c["cls"] = "swimmer" if frac > 0 else "person"
            if 0 < frac < 0.10:  # only a sliver in the water: edge sitter or deck person next to the water
                f.append("pool_edge")
            c["flags"] = f
        instances = []
        for c in inst:
            fl = c.get("flags", [])
            frame_flags += fl
            instances.append({"box": [round(float(v), 1) for v in c["box"]], "voters": c["voters"],
                              "accepted": len(c["voters"]) >= 2, "cls": c.get("cls"), "flags": fl,
                              "water_frac": c.get("water_frac"), "sam_iou": c.get("sam_iou")})
        for fl in frame_flags:
            flags_all[fl] += 1
        status = "review" if frame_flags else "auto_accepted"
        stats[status] += 1
        stats["instances_accepted"] += len(accepted)
        stats["instances_flagged_single"] += sum(1 for c in inst if len(c["voters"]) == 1)
        records.append({"key": rec["key"], "split": split_of.get(rec["key"].split("/")[0].split("@")[0]),
                        "status": status, "n_flags": len(frame_flags), "instances": instances})
    with open(out_dir / f"{a.set}_consensus.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = {"thresholds": thr, "frames": len(records), **stats, "flags": dict(flags_all)}
    (out_dir / f"{a.set}_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
