"""Hard negatives for the v0 fine-tune (owner request 2026-10-30: "detect the people and leave glare and water markers").

v0 fires on pennants, floor T-markings, lane-rope floats and glare. A detector learns "not a person" only from frames where
those things are in view and carry NO box, so this tool adds such frames, fully labelled, and cleans the old labels:

  mine  (SERVING venv)  : v0 over every 25th frame of the train-split clips (frames not already in train.json), boxes >= 0.15
                          -> work_dirs/hardneg/mined.json
  score (training venv) : distractor verifier p(person) for every mined box and every train.json box, static-fixture overlap
                          -> auto person / auto distractor / ask the owner; picks the frames with the most distractors
                          -> data/audit/hardneg_r1.html (answer sheet, suggestions pre-selected) + work_dirs/hardneg/state.json
  build (training venv) : owner answers + auto decisions -> data/coco/hardneg_r1/train.json (old train with the boxes the owner
                          called "not a person" removed, plus the new frames x1). Val is unchanged.

Safety: a person missed by v0 on a new frame would be taught as background, so a frame is used only when every box on it is
decided. Auto person: verifier p >= 0.6 and v0 score >= 0.5, off the fixture spots. Auto distractor: p < 0.15, or p < 0.3 with
a weak v0 score (< 0.5), or a weak box on a fixture spot with p < 0.5. Everything in between goes to the owner.

    $env:PYTHONPATH="backend"; venv\\Scripts\\python.exe training\\tools\\hardneg.py mine
    training\\.venv\\Scripts\\python.exe training\\tools\\hardneg.py score|build [--result file]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

TR = Path(__file__).resolve().parents[1]
FRAMES = TR / "data" / "frames"
WD = TR / "work_dirs" / "hardneg"
TRAIN = TR / "data" / "coco" / "seed_r1" / "train.json"
OUT = TR / "data" / "coco" / "hardneg_r1"
MODEL = TR.parent / "backend" / "models" / "lifeguard_dfine_l_pool_v0.onnx"
STEP = 25          # every 25th extracted frame
N_FRAMES = 300     # new frames to add
P_PERSON, S_PERSON, P_DISTRACTOR = 0.6, 0.5, 0.15
Q_BUDGET = 350     # owner questions on new frames (the old-label questions come on top)


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    i = max(w, 0) * max(h, 0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def mine() -> int:
    import lifeguard.vision.inference.decode as dec
    from lifeguard.vision.inference.models import class_names_for, load_detector_model

    train_clips = {c.removesuffix(".mp4") for c in json.loads((TR / "data/splits.json").read_text())["train"]}
    used = {i["file_name"] for i in json.loads(TRAIN.read_text())["images"]}
    todo = []
    for clip in sorted(train_clips):
        fs = sorted((FRAMES / clip).glob("*.jpg"))
        todo += [f"{clip}/{f.name}" for f in fs[::STEP] if f"{clip}/{f.name}" not in used]
    dm = load_detector_model(str(MODEL), "cuda", 1024)
    names = class_names_for(dm, str(MODEL))
    dec.DET_DETR_MIN_SCORE = 0.0
    out = {}
    for k, key in enumerate(todo):
        b, s, w = dec.run_detector_people(dm, cv2.imread(str(FRAMES / key)), score_thr=0.15, names=names)
        out[key] = [[*map(float, bb), float(ss), None if np.isnan(ww) else float(ww)] for bb, ss, ww in zip(b, s, w)]
        if (k + 1) % 200 == 0:
            print(k + 1, "/", len(todo), flush=True)
    WD.mkdir(parents=True, exist_ok=True)
    (WD / "mined.json").write_text(json.dumps(out))
    print(len(out), "frames,", sum(map(len, out.values())), "boxes ->", WD / "mined.json")
    return 0


def verifier_p(items):
    import joblib

    import distractor_verifier as D

    v = joblib.load(TR / "work_dirs/verifier/verifier.joblib")
    X = D.features(items)
    geo = np.array([[(b[2] - b[0]) / 100, (b[3] - b[1]) / 100,
                     float(D.water[int(b[1]):max(int(b[3]), int(b[1]) + 1), int(b[0]):max(int(b[2]), int(b[0]) + 1)].mean())]
                    for _, b in items])
    return v["clf"].predict_proba(np.nan_to_num((np.hstack([X, geo]) - v["mu"]) / v["sd"]))[:, 1]


def clip_box(b):
    x1, y1, x2, y2 = (min(max(float(t), 0.0), 959 if i % 2 == 0 else 1071) for i, t in enumerate(b))
    return [x1, y1, x2, y2]


def write_sheet(ask: list[dict], page: Path, title: str, download: str) -> None:
    """One-page answer sheet: a card per row (key, box, p, sug, tag, fixture, id); saves [{owner, id, sug}] as `download`."""
    import claude_audit as A

    cards = []
    for i, r in enumerate(ask):
        cards.append(f"""<div class="card" id="c{i}"><img src="data:image/jpeg;base64,{A.crop_b64(r['key'], r['box'])}">
<p>#{i+1} &middot; {r['tag']} &middot; suggestion: <b>{r['sug'].replace('_', ' ')}</b> ({r['p']:.0%} person){' &middot; fixture spot' if r['fixture'] else ''}</p>
<button onclick="set({i},'swimmer')">Person in water</button><button onclick="set({i},'person')">Person on deck</button><button onclick="set({i},'not_a_person')">NOT a person</button></div>""")
    meta = [{"id": r["id"], "sug": r["sug"]} for r in ask]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>
body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:16px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px}}
.card{{background:#222;padding:8px;border-radius:6px;border:3px solid #444}}.card img{{max-width:100%}}.card.swimmer{{border-color:#2a2}}.card.person{{border-color:#29d}}.card.not_a_person{{border-color:#d33}}
button{{font-size:15px;padding:6px 10px;margin-right:4px}}#top{{position:sticky;top:0;background:#111;padding:6px 0;z-index:2}}#top button{{font-size:18px;padding:10px 20px;margin:6px}}</style></head><body>
<h2>{title}: {len(ask)} boxes</h2><p>The yellow box is the question: is there a person inside it? Glare, flags, floor markings, lane-rope floats and
shadows are NOT a person. A loose box that holds a person counts as a person. Grey = not answered (keeps the suggestion if you press Accept).
Keys: hover a card and press 1 (in water), 2 (deck) or 3 (not a person).</p>
<div id="top"><button onclick="acc()">Accept all suggestions for unanswered</button><button onclick="save()">Save results</button> <span id="count"></span></div>
<div class="grid">{''.join(cards)}</div><script>const meta={json.dumps(meta)};const res={{}};let hov=null;
document.querySelectorAll('.card').forEach((c,i)=>{{c.onmouseenter=()=>hov=i;}});
document.onkeydown=e=>{{if(hov===null)return;const v={{'1':'swimmer','2':'person','3':'not_a_person'}}[e.key];if(v)set(hov,v);}};
function set(i,v){{res[i]=v;document.getElementById('c'+i).className='card '+v;cnt();}}
function acc(){{meta.forEach((m,i)=>{{if(!res[i])set(i,m.sug)}});}}
function cnt(){{document.getElementById('count').textContent=Object.keys(res).length+' / '+meta.length+' answered';}}cnt();
function save(){{const out=meta.map((m,i)=>Object.assign({{owner:res[i]||null}},m));const a=document.createElement('a');
a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{{type:'application/json'}}));a.download='{download}';a.click();}}</script></body></html>"""
    page.write_text(html, encoding="utf-8")


def score() -> int:
    import claude_redo as R
    import distractor_verifier as D

    fixtures = [f["box"] for f in json.loads((TR / "data/static_background.json").read_text())]
    mined = json.loads((WD / "mined.json").read_text())
    rows = []  # one per box: source new|old
    for key, L in mined.items():
        for x1, y1, x2, y2, s, w in L:
            b = clip_box([x1, y1, x2, y2])
            if b[2] - b[0] >= 4 and b[3] - b[1] >= 4:
                rows.append({"src": "new", "key": key, "box": b, "score": s})
    coco = json.loads(TRAIN.read_text())
    img = {i["id"]: i["file_name"] for i in coco["images"]}
    seen = set()
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        k = (img[a["image_id"]], round(x, 1), round(y, 1))
        if k in seen:  # reviewed frames appear twice in train.json
            continue
        seen.add(k)
        rows.append({"src": "old", "key": img[a["image_id"]], "box": clip_box([x, y, x + w, y + h]), "score": 1.0,
                     "cls": "swimmer" if a["category_id"] == 0 else "person"})
    p = verifier_p([(r["key"], r["box"]) for r in rows])
    for r, q in zip(rows, p):
        r["p"] = round(float(q), 3)
        r["fixture"] = any(iou(r["box"], f) >= 0.3 for f in fixtures)
        weak = r["score"] < S_PERSON
        if r["src"] == "old":  # old labels: ask only where the verifier or the fixture map strongly doubts the box
            r["auto"] = None if q < P_DISTRACTOR or (r["fixture"] and q < 0.3) else "keep"
        elif q >= P_PERSON and not weak and not r["fixture"]:
            r["auto"] = R.verdict_for_person(D.water, r["box"])
        elif q < P_DISTRACTOR or (q < 0.3 and weak) or (r["fixture"] and q < 0.5 and weak):
            r["auto"] = "not_a_person"
        else:
            r["auto"] = None
    # choose the new frames with the fewest questions (then the most distractors) until the question budget is used
    by = {}
    for r in rows:
        if r["src"] == "new":
            by.setdefault(r["key"], []).append(r)
    rank = sorted(by, key=lambda k: (sum(r["auto"] is None for r in by[k]), -sum(r["auto"] == "not_a_person" for r in by[k])))
    chosen, budget = [], 0
    for k in rank:
        nq = sum(r["auto"] is None for r in by[k])
        if budget + nq > Q_BUDGET or len(chosen) >= N_FRAMES:
            break
        chosen.append(k)
        budget += nq
    keep = set(chosen)
    rows = [r for r in rows if r["src"] == "old" or r["key"] in keep]
    for i, r in enumerate(rows):
        r["id"] = i
    ask = [r for r in rows if r["auto"] is None]
    for r in ask:
        r["sug"] = R.verdict_for_person(D.water, r["box"]) if r["p"] >= 0.5 else "not_a_person"
    WD.mkdir(parents=True, exist_ok=True)
    (WD / "state.json").write_text(json.dumps({"frames": chosen, "rows": rows}))
    for r in ask:
        r["tag"] = "old label" if r["src"] == "old" else f"v0 {r['score']:.0%}"
    page = TR / "data/audit/hardneg_r1.html"
    write_sheet(ask, page, "Hard negatives", "hardneg_r1_result.json")
    new = [r for r in rows if r["src"] == "new"]
    old = [r for r in rows if r["src"] == "old"]
    print(f"new frames {len(chosen)}: boxes {len(new)} {dict(Counter(r['auto'] for r in new))}")
    print(f"old labels {len(old)}: {dict(Counter(r['auto'] for r in old))}")
    print(f"owner questions {len(ask)} ({sum(r['src'] == 'old' for r in ask)} old, {sum(r['src'] == 'new' for r in ask)} new) -> {page}")
    return 0


def build(result: str) -> int:
    st = json.loads((WD / "state.json").read_text())
    ans = {r["id"]: r["owner"] for r in json.loads(Path(result).read_text()) if r["owner"]}
    rows = st["rows"]
    for r in rows:
        r["final"] = ans.get(r["id"], r["auto"])
    coco = json.loads(TRAIN.read_text())
    img = {i["id"]: i["file_name"] for i in coco["images"]}
    drop = {(r["key"], tuple(round(t, 1) for t in r["box"])) for r in rows if r["src"] == "old" and r["final"] == "not_a_person"}
    undecided_old = sum(r["src"] == "old" and r["final"] is None for r in rows)
    anns = []
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        if (img[a["image_id"]], tuple(round(t, 1) for t in clip_box([x, y, x + w, y + h]))) not in drop:
            anns.append(a)
    removed = len(coco["annotations"]) - len(anns)
    by = {}
    for r in rows:
        if r["src"] == "new":
            by.setdefault(r["key"], []).append(r)
    images, nid, aid = list(coco["images"]), max(i["id"] for i in coco["images"]) + 1, max(a["id"] for a in anns) + 1
    added = skipped = nboxes = 0
    for key in st["frames"]:
        L = by.get(key, [])
        if any(r["final"] is None for r in L):
            skipped += 1
            continue
        images.append({"id": nid, "file_name": key, "width": 960, "height": 1072})
        for r in L:
            if r["final"] in ("swimmer", "person"):
                x1, y1, x2, y2 = r["box"]
                anns.append({"id": aid, "image_id": nid, "category_id": 0 if r["final"] == "swimmer" else 1,
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
                aid += 1
                nboxes += 1
        nid += 1
        added += 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": coco["categories"]}))
    msg = (f"old labels removed as not-a-person: {removed} (undecided kept: {undecided_old}); new frames added {added} "
           f"({nboxes} person boxes, skipped {skipped} with an undecided box) -> {OUT / 'train.json'}")
    print(msg)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    cmd = sys.argv[1]
    if cmd == "mine":
        sys.exit(mine())
    if cmd == "score":
        sys.exit(score())
    sys.exit(build(sys.argv[sys.argv.index("--result") + 1] if "--result" in sys.argv else str(Path.home() / "Downloads/hardneg_r1_result.json")))
