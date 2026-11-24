"""Phase 6/10: evaluate ONNX detectors on a labelled COCO split through Life Guard's own decoder (SERVING venv).

Metrics (MASTER_PROMPT Phase 6), class-agnostic person detection at IoU 0.5 unless noted:
  AP50 (all-point), recall at 0.1 / 0.5 / 1.0 false positives per frame, the same for SMALL people (box height
  < 32 px), and the swimmer-vs-person class accuracy on matched boxes (in-water vote >= 0.5 = swimmer; models with
  no in/out-of-water class pair have no vote and get "—").
Curves need low scores, so the app's D-FINE score floor (DET_DETR_MIN_SCORE) is set to 0 here; the live system's
operating point is measured by the Life Guard eval harness, not by this script.
The test split is refused unless --final is passed (MASTER_PROMPT: test metrics only at the final evaluation).

    $env:PYTHONPATH="backend"; venv\\Scripts\\python.exe training\\tools\\eval_v0.py --ann training\\data\\coco\\seed_r1\\val.json \\
        --model backend\\models\\rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c_dynbatch.onnx --model training\\work_dirs\\onnx\\dfine_l_v0.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import lifeguard.vision.inference.decode as dec
from lifeguard.vision.inference.models import class_names_for, load_detector_model

ROOT = Path(__file__).resolve().parents[2]
FRAMES = ROOT / "training" / "data" / "frames"
SMALL_H = 32
FPPI = (0.1, 0.5, 1.0)
WORK_SCORE = 0.5


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def evaluate(model_path: str, images: list[dict], gts: dict[int, list[dict]], provider: str) -> dict:
    dm = load_detector_model(model_path, provider, 1024)
    names = class_names_for(dm, model_path)
    dets = []  # (score, is_tp, is_small_gt_matched, img, class_ok or None)
    n_gt = n_small = 0
    small_fp_dets = []
    t0 = time.time()
    for im in images:
        g = gts.get(im["id"], [])
        gb = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in g]).reshape(-1, 4)
        gsmall = np.array([a["bbox"][3] < SMALL_H for a in g], bool)
        gswim = np.array([a["category_id"] == 0 for a in g], bool)
        n_gt += len(g)
        n_small += int(gsmall.sum())
        b, s, w = dec.run_detector_people(dm, cv2.imread(str(FRAMES / im["file_name"])), score_thr=0.01, names=names)
        order = np.argsort(-s)
        b, s, w = b[order], s[order], w[order]
        ious = iou_matrix(b, gb)
        used = np.zeros(len(gb), bool)
        for i in range(len(b)):
            j = int(np.argmax(np.where(used, -1, ious[i]))) if len(gb) else -1
            if j >= 0 and not used[j] and ious[i, j] >= 0.5:
                used[j] = True
                cls_ok = None if np.isnan(w[i]) else bool((w[i] >= 0.5) == gswim[j])
                dets.append((float(s[i]), True, bool(gsmall[j]), cls_ok))
            else:
                dets.append((float(s[i]), False, False, None))
                small_fp_dets.append(float(s[i]))
    sec = (time.time() - t0) / max(len(images), 1)
    dets.sort(key=lambda d: -d[0])
    tp = np.cumsum([d[1] for d in dets])
    fp = np.cumsum([not d[1] for d in dets])
    tps = np.cumsum([d[2] for d in dets])
    rec = tp / max(n_gt, 1)
    prec = tp / np.maximum(tp + fp, 1)
    mprec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
    ap50 = float(np.sum(np.diff(np.concatenate([[0], rec])) * mprec)) if len(rec) else 0.0
    out = {"model": Path(model_path).name, "frames": len(images), "gt": n_gt, "gt_small": n_small, "AP50": ap50,
           "s_per_frame_cpu" if provider == "cpu" else "s_per_frame": round(sec, 3)}
    for f in FPPI:
        k = np.searchsorted(fp, f * len(images), side="right") - 1  # last detection index with fp <= budget
        out[f"recall@{f}fppi"] = float(rec[k]) if k >= 0 else 0.0
        out[f"small_recall@{f}fppi"] = float(tps[k] / n_small) if k >= 0 and n_small else None
    work = [d for d in dets if d[0] >= WORK_SCORE]  # the app's working score: false boxes and recall a user would see
    out[f"fp_per_frame@s{WORK_SCORE}"] = sum(not d[1] for d in work) / max(len(images), 1)
    out[f"recall@s{WORK_SCORE}"] = sum(d[1] for d in work) / max(n_gt, 1)
    cls = [d[3] for d in dets if d[1] and d[3] is not None]
    out["class_acc"] = float(np.mean(cls)) if cls else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--provider", default="cuda")
    ap.add_argument("--final", action="store_true", help="required for the test split")
    ap.add_argument("--out", help="write the results as JSON")
    a = ap.parse_args()
    if "test" in Path(a.ann).name.lower() and not a.final:
        print("refused: test-split metrics are only produced at the final evaluation (--final)")
        return 2
    coco = json.loads(Path(a.ann).read_text())
    seen, images = set(), []
    for im in coco["images"]:  # the train export repeats reviewed frames (2x weight); evaluate each frame once
        if im["file_name"] not in seen:
            seen.add(im["file_name"])
            images.append(im)
    gts: dict[int, list[dict]] = {}
    for an in coco["annotations"]:
        gts.setdefault(an["image_id"], []).append(an)
    dec.DET_DETR_MIN_SCORE = 0.0
    rows = [evaluate(m, images, gts, a.provider) for m in a.model]
    keys = ["AP50", *[f"recall@{f}fppi" for f in FPPI], *[f"small_recall@{f}fppi" for f in FPPI],
            f"recall@s{WORK_SCORE}", f"fp_per_frame@s{WORK_SCORE}", "class_acc"]
    print(f"{len(images)} frames, {rows[0]['gt']} people ({rows[0]['gt_small']} under {SMALL_H} px tall), IoU 0.5")
    print("| model | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for r in rows:
        print(f"| {r['model']} | " + " | ".join("—" if r[k] is None else f"{r[k]:.3f}" for k in keys) + " |")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
