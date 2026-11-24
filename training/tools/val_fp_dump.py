"""Diagnostic (SERVING venv): unmatched val detections at score >= 0.5 and unmatched val labels, per model.

Writes work_dirs/hardneg/val_fp_<model>.json: {"fp": [[file, box, score]], "fn": [[file, box]]} for a contact sheet.
    $env:PYTHONPATH="backend"; venv\\Scripts\\python.exe training\\tools\\val_fp_dump.py <model.onnx> ...
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

import lifeguard.vision.inference.decode as dec
from lifeguard.vision.inference.models import class_names_for, load_detector_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_v0 import FRAMES, iou_matrix  # noqa: E402

TR = Path(__file__).resolve().parents[1]
coco = json.loads((TR / "data/coco/seed_r1/val.json").read_text())
gts = {}
for a in coco["annotations"]:
    x, y, w, h = a["bbox"]
    gts.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
dec.DET_DETR_MIN_SCORE = 0.0
for m in sys.argv[1:]:
    dm = load_detector_model(m, "cuda", 1024)
    names = class_names_for(dm, m)
    fp, fn = [], []
    for im in coco["images"]:
        g = np.array(gts.get(im["id"], [])).reshape(-1, 4)
        b, s, _ = dec.run_detector_people(dm, cv2.imread(str(FRAMES / im["file_name"])), score_thr=0.5, names=names)
        ious = iou_matrix(b, g)
        used = np.zeros(len(g), bool)
        for i in np.argsort(-s):
            j = int(np.argmax(np.where(used, -1, ious[i]))) if len(g) else -1
            if j >= 0 and not used[j] and ious[i, j] >= 0.5:
                used[j] = True
            else:
                fp.append([im["file_name"], [float(t) for t in b[i]], float(s[i])])
        fn += [[im["file_name"], [float(t) for t in g[j]]] for j in np.where(~used)[0]]
    out = TR / "work_dirs/hardneg" / f"val_fp_{Path(m).stem}.json"
    out.write_text(json.dumps({"fp": fp, "fn": fn}))
    print(Path(m).name, "fp", len(fp), "fn (at score 0.5)", len(fn), "->", out)
