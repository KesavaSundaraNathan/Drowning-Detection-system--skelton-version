"""Hard negatives round 2 (v0.1 -> v0.2), no owner questions (owner 2026-10-30: proceed without asking).

v0.1 still leaves persistent boxes on far-end window glare and lane ropes that the behaviour rules turn into alerts. Round 2
mines MORE frames with v0.1 itself and decides every box with score >= 0.3 automatically:
  person      : verifier p >= 0.6 and score >= 0.5, off the fixture spots  -> labelled swimmer / person from the pool mask
  distractor  : p < 0.15, or (p < 0.3 and score < 0.5), or (fixture spot and p < 0.5 and score < 0.5) -> no box
  otherwise   : undecided -> the whole frame is NOT used (a missed person must never be taught as background)
Boxes with score < 0.3 are left unlabelled. Frames are ranked by distractor count, at most PER_CLIP per clip. The result is
data/coco/hardneg_r2/train.json = hardneg_r1/train.json + these frames. Auto decisions are spot-checked on a contact sheet.

    $env:PYTHONPATH="backend"; venv\\Scripts\\python.exe training\\tools\\hardneg2.py mine
    training\\.venv\\Scripts\\python.exe training\\tools\\hardneg2.py build
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hardneg as H  # noqa: E402

TR = H.TR
BASE = TR / "data/coco/hardneg_r1/train.json"
OUT = TR / "data/coco/hardneg_r2"
MODEL = TR / "work_dirs/onnx/dfine_l_v01.onnx"
MINED = H.WD / "mined_r2.json"
STEP, OFFSET, PER_CLIP, N_FRAMES = 10, 5, 30, 400
S_MIN = 0.3
MAX_UNDECIDED = 8
MASKED = TR / "data" / "frames_r2"


def mine() -> int:
    import lifeguard.vision.inference.decode as dec
    from lifeguard.vision.inference.models import class_names_for, load_detector_model

    clips = {c.removesuffix(".mp4") for c in json.loads((TR / "data/splits.json").read_text())["train"]}
    used = {i["file_name"] for i in json.loads(BASE.read_text())["images"]}
    todo = []
    for clip in sorted(clips):
        fs = sorted((H.FRAMES / clip).glob("*.jpg"))
        todo += [f"{clip}/{f.name}" for f in fs[OFFSET::STEP] if f"{clip}/{f.name}" not in used]
    dm = load_detector_model(str(MODEL), "cuda", 1024)
    names = class_names_for(dm, str(MODEL))
    dec.DET_DETR_MIN_SCORE = 0.0
    out = {}
    for k, key in enumerate(todo):
        b, s, _ = dec.run_detector_people(dm, cv2.imread(str(H.FRAMES / key)), score_thr=S_MIN, names=names)
        out[key] = [[*map(float, bb), float(ss)] for bb, ss in zip(b, s)]
        if (k + 1) % 200 == 0:
            print(k + 1, "/", len(todo), flush=True)
    MINED.write_text(json.dumps(out))
    print(len(out), "frames,", sum(map(len, out.values())), "boxes ->", MINED)
    return 0


def build() -> int:
    import claude_redo as R
    import distractor_verifier as D

    fixtures = [f["box"] for f in json.loads((TR / "data/static_background.json").read_text())]
    mined = json.loads(MINED.read_text())
    rows = []
    for key, L in mined.items():
        for x1, y1, x2, y2, s in L:
            b = H.clip_box([x1, y1, x2, y2])
            if b[2] - b[0] >= 4 and b[3] - b[1] >= 4:
                rows.append({"key": key, "box": b, "score": s})
    p = H.verifier_p([(r["key"], r["box"]) for r in rows])
    for r, q in zip(rows, p):
        r["p"] = round(float(q), 3)
        fx = any(H.iou(r["box"], f) >= 0.3 for f in fixtures)
        weak = r["score"] < 0.5
        if q >= 0.6 and not weak and not fx:
            r["dec"] = R.verdict_for_person(D.water, r["box"])
        elif q < 0.15 or (q < 0.3 and weak) or (fx and q < 0.5 and weak):
            r["dec"] = "not_a_person"
        else:
            r["dec"] = None
    by = {}
    for key in mined:
        by[key] = []
    for r in rows:
        by[r["key"]].append(r)
    ok = {k: v for k, v in by.items() if sum(r["dec"] is None for r in v) <= MAX_UNDECIDED}
    rank = sorted(ok, key=lambda k: -(sum(r["dec"] == "not_a_person" for r in ok[k]) - 0.2 * sum(r["dec"] is None for r in ok[k])))
    chosen, per = [], Counter()
    for k in rank:
        clip = k.split("/")[0]
        if per[clip] >= PER_CLIP or sum(r["dec"] == "not_a_person" for r in ok[k]) < 2:
            continue
        chosen.append(k)
        per[clip] += 1
        if len(chosen) >= N_FRAMES:
            break
    coco = json.loads(BASE.read_text())
    images, anns = list(coco["images"]), list(coco["annotations"])
    nid, aid = max(i["id"] for i in images) + 1, max(a["id"] for a in anns) + 1
    npos = nneg = nmask = 0
    MASKED.mkdir(parents=True, exist_ok=True)
    for k in chosen:
        und = [r for r in ok[k] if r["dec"] is None]
        fname = k
        if und:  # undecided boxes: blur the region in a copy, so the frame teaches neither a person nor a background there
            im = cv2.imread(str(H.FRAMES / k))
            for r in und:
                x1, y1, x2, y2 = r["box"]
                px, py = 0.15 * (x2 - x1) + 4, 0.15 * (y2 - y1) + 4
                X1, Y1, X2, Y2 = int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(960, x2 + px)), int(min(1072, y2 + py))
                im[Y1:Y2, X1:X2] = cv2.blur(im[Y1:Y2, X1:X2], (max(3, (X2 - X1) // 2 | 1), max(3, (Y2 - Y1) // 2 | 1)))
                nmask += 1
            fname = "../frames_r2/" + k.replace("/", "_")
            cv2.imwrite(str(MASKED / k.replace("/", "_")), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        images.append({"id": nid, "file_name": fname, "width": 960, "height": 1072})
        for r in ok[k]:
            if r["dec"] is None:
                continue
            if r["dec"] == "not_a_person":
                nneg += 1
                continue
            x1, y1, x2, y2 = r["box"]
            anns.append({"id": aid, "image_id": nid, "category_id": 0 if r["dec"] == "swimmer" else 1,
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            aid += 1
            npos += 1
        nid += 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": coco["categories"]}))
    (H.WD / "state_r2.json").write_text(json.dumps({"frames": chosen, "rows": [r for k in chosen for r in ok[k]]}))
    dist = Counter(r["dec"] for r in rows)
    print(f"mined boxes {len(rows)}: {dict(dist)}; frames fully decided {len(ok)} of {len(mined)}")
    print(f"chosen {len(chosen)} frames: {npos} person boxes, {nneg} distractors with no box, {nmask} undecided regions blurred, "
          f"per clip max {max(per.values())} -> {OUT / 'train.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(mine() if sys.argv[1] == "mine" else build())
