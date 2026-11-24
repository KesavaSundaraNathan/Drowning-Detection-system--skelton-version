"""Val clean-up (owner request 2026-10-30, after v0.1 failed): the 50 val frames hold non-person labels (floor T-markings,
deck bags, ladder, glare) and miss real swimmers, so val could not show whether a change removes distractors.

make  : questions for the owner -> data/audit/val_clean_r1.html (+ work_dirs/hardneg/val_clean_state.json)
        - every val label with verifier p < 0.3, or on a fixture spot with p < 0.5 ("is this label a person?");
        - every v0 / v0.1 detection at score >= 0.5 that matches no label (val_fp_dump.py), merged at IoU 0.5
          ("is there an unlabelled person here?").
apply : owner answers -> data/coco/seed_r1/val_clean.json (val.json is kept): labels called "not a person" are dropped,
        a changed class is applied, unlabelled people the owner confirms are added. People that neither model found stay
        unlabelled (a known limit).

    $env:PYTHONPATH="backend"; venv\\Scripts\\python.exe training\\tools\\val_fp_dump.py <v0.onnx> <v01.onnx>   # first
    training\\.venv\\Scripts\\python.exe training\\tools\\val_clean.py make|apply [--result file]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hardneg as H  # noqa: E402

TR = H.TR
VAL = TR / "data/coco/seed_r1/val.json"
OUT = TR / "data/coco/seed_r1/val_clean.json"
STATE = H.WD / "val_clean_state.json"
PAGE = TR / "data/audit/val_clean_r1.html"
CLS = {0: "swimmer", 1: "person"}


def make() -> int:
    import claude_redo as R
    import distractor_verifier as D

    fixtures = [f["box"] for f in json.loads((TR / "data/static_background.json").read_text())]
    coco = json.loads(VAL.read_text())
    img = {i["id"]: i["file_name"] for i in coco["images"]}
    rows = []
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        rows.append({"src": "label", "ann": a["id"], "key": img[a["image_id"]], "box": H.clip_box([x, y, x + w, y + h]),
                     "cls": CLS[a["category_id"]]})
    found = []
    for m in ("dfine_l_v0", "dfine_l_v01"):
        for f, b, s in json.loads((H.WD / f"val_fp_{m}.json").read_text())["fp"]:
            b = H.clip_box(b)
            if b[2] - b[0] >= 4 and b[3] - b[1] >= 4 and not any(k == f and H.iou(b, c) >= 0.5 for k, c, _ in found):
                found.append((f, b, s))
    rows += [{"src": "missing", "key": f, "box": b, "score": s} for f, b, s in found]
    p = H.verifier_p([(r["key"], r["box"]) for r in rows])
    ask = []
    for i, (r, q) in enumerate(zip(rows, p)):
        r["id"], r["p"] = i, round(float(q), 3)
        r["fixture"] = any(H.iou(r["box"], f) >= 0.3 for f in fixtures)
        if r["src"] == "label":
            if q < 0.3 or (r["fixture"] and q < 0.5):
                r["sug"], r["tag"] = ("not_a_person" if q < 0.3 else r["cls"]), f"val label ({r['cls']})"
                ask.append(r)
        else:
            r["sug"] = R.verdict_for_person(D.water, r["box"]) if q >= 0.5 else "not_a_person"
            r["tag"] = f"no label, model {r['score']:.0%}"
            ask.append(r)
    STATE.write_text(json.dumps(rows))
    H.write_sheet(ask, PAGE, "Val clean-up", "val_clean_r1_result.json")
    print(f"val labels {sum(r['src'] == 'label' for r in rows)}, asked {sum(r['src'] == 'label' for r in ask)}; "
          f"unlabelled detections {len(found)} (all asked); total {len(ask)} -> {PAGE}")
    print("suggestions", dict(Counter((r["src"], r["sug"]) for r in ask)))
    return 0


def apply(result: str) -> int:
    rows = {r["id"]: r for r in json.loads(STATE.read_text())}
    ans = {r["id"]: r["owner"] for r in json.loads(Path(result).read_text()) if r["owner"]}
    coco = json.loads(VAL.read_text())
    by_ann = {r["ann"]: ans.get(r["id"]) for r in rows.values() if r["src"] == "label"}
    fid = {i["file_name"]: i["id"] for i in coco["images"]}
    anns, dropped, changed = [], 0, 0
    for a in coco["annotations"]:
        v = by_ann.get(a["id"])
        if v == "not_a_person":
            dropped += 1
            continue
        if v in ("swimmer", "person") and v != CLS[a["category_id"]]:
            a = {**a, "category_id": 0 if v == "swimmer" else 1}
            changed += 1
        anns.append(a)
    aid, added = max(a["id"] for a in coco["annotations"]) + 1, 0
    for i, v in ans.items():
        r = rows[i]
        if r["src"] == "missing" and v in ("swimmer", "person"):
            x1, y1, x2, y2 = r["box"]
            anns.append({"id": aid, "image_id": fid[r["key"]], "category_id": 0 if v == "swimmer" else 1,
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            aid += 1
            added += 1
    OUT.write_text(json.dumps({**coco, "annotations": anns}))
    unanswered = sum(1 for r in rows.values() if "sug" in r and r["id"] not in ans)
    print(f"val {len(coco['annotations'])} -> {len(anns)} labels: dropped {dropped}, class changed {changed}, "
          f"added {added}; unanswered {unanswered} -> {OUT}")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "make":
        sys.exit(make())
    sys.exit(apply(sys.argv[sys.argv.index("--result") + 1] if "--result" in sys.argv
                   else str(Path.home() / "Downloads/val_clean_r1_result.json")))
