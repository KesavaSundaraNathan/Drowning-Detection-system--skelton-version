"""AUTO labels for the TEST A+B frames (owner decision 2026-10-30: drawing 300 frames by hand is too slow, "automate that step").

NOT hand labels. They are made only from detectors that are independent of v0 (COCO D-FINE-L, COCO RTMDet-L, OWLv2) plus the
distractor verifier, so the v0 score is not graded by v0's own opinion. Expect them to miss small / submerged swimmers and to keep
a few fixtures; the final test numbers must be reported as "auto-labelled test (not hand)".

Rule: a candidate box is kept when >= 2 voters agree (IoU >= 0.5, score >= 0.30 each), or one voter has score >= 0.55 and the
verifier says p(person) >= 0.60; every kept box also needs verifier p >= 0.40. Class: swimmer when any of the lower 60 % of the
box is over pool water, else person. Output: data/test_upload/test_labels_auto.json (same format as the HTML tool's result) and
data/coco/test_auto.json via annotator_pages.export_test.

    training\\.venv\\Scripts\\python.exe training\tools\test_autolabel.py
"""

import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import distractor_verifier as D
import voters as V

TR = D.TR
UP = TR / "data" / "test_upload"


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    i = max(w, 0) * max(h, 0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def main() -> int:
    ts = json.loads((TR / "data/test_sets.json").read_text())
    names = ts["A_keypoints"] + ts["B_boxes"]
    vs = {"dfine": V.DFine(), "rtmdet": V.RTMDet(), "owlv2": V.OWLv2()}
    cache = TR / "work_dirs" / "test_autolabel_cand.json"
    cand = []  # (name, box, {voter: score})
    if cache.exists():
        cand = [(n, b, vt) for n, b, vt in json.loads(cache.read_text())]
        names_todo = []
    else:
        names_todo = names
    for k, n in enumerate(names_todo):
        img = cv2.imread(str(UP / n))
        raw = {v: [b[:5] for b in f(img)] for v, f in vs.items()}
        pool = [(b[:4], v, b[4]) for v, L in raw.items() for b in L if b[4] >= 0.15]
        pool.sort(key=lambda x: -x[2])
        groups = []
        for box, v, s in pool:
            for g in groups:
                if iou(g["box"], box) >= 0.5:
                    g["votes"][v] = max(g["votes"].get(v, 0), s)
                    break
            else:
                groups.append({"box": box, "votes": {v: s}})
        cand += [(n, g["box"], g["votes"]) for g in groups]
        if (k + 1) % 50 == 0:
            print(k + 1, "/", len(names), flush=True)
    if names_todo:
        cache.write_text(json.dumps([(n, [float(t) for t in b], vt) for n, b, vt in cand]))
    cand = [(n, [min(max(float(t), 0.0), 959 if i % 2 == 0 else 1071) for i, t in enumerate(b)], vt) for n, b, vt in cand]
    cand = [c for c in cand if c[1][2] - c[1][0] >= 4 and c[1][3] - c[1][1] >= 4]
    # verifier
    v = joblib.load(TR / "work_dirs/verifier/verifier.joblib")
    items = [("../test_upload/" + n, b) for n, b, _ in cand]
    X = D.features(items)
    geo = np.array([[(b[2] - b[0]) / 100, (b[3] - b[1]) / 100,
                     float(D.water[int(b[1]):max(int(b[3]), int(b[1]) + 1), int(b[0]):max(int(b[2]), int(b[0]) + 1)].mean())] for _, b in items])
    Z = np.nan_to_num((np.hstack([X, geo]) - v["mu"]) / v["sd"])
    p = v["clf"].predict_proba(Z)[:, 1]
    out = {n: [] for n in names}
    for (n, b, votes), q in zip(cand, p):
        strong = sum(s >= 0.30 for s in votes.values())
        keep = (strong >= 2 or (max(votes.values()) >= 0.55 and q >= 0.60)) and q >= 0.40
        if not keep:
            continue
        x1, y1, x2, y2 = (int(round(t)) for t in b)
        y0 = int(y1 + 0.4 * (y2 - y1))
        cls = "swimmer" if D.water[max(y0, 0):max(y2, 1), max(x1, 0):max(x2, 1)].any() else "person"
        out[n].append({"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "cls": cls})
    rows = [{"id": n, "task": 0, "frame": k, "key": n, "boxes": out[n]} for k, n in enumerate(names)]
    path = UP / "test_labels_auto.json"
    path.write_text(json.dumps(rows, indent=1))
    nb = sum(len(r["boxes"]) for r in rows)
    print(f"{len(rows)} frames, {nb} auto boxes ({nb / len(rows):.1f} per frame) -> {path}")
    import annotator_pages as A
    A.export_test(str(path))
    (TR / "data/coco/test_hand.json").rename(TR / "data/coco/test_auto.json")  # never mistaken for hand labels
    return 0


if __name__ == "__main__":
    sys.exit(main())
