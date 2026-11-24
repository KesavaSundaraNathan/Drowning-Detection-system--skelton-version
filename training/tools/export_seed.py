"""Phase 5/6: export the checked SEED (CVAT tasks "CHECK R1 train" / "CHECK R1 val") to COCO for v0 training, with QA.

Only jobs marked Completed are exported (the rest join the next retrain, MASTER_PROMPT: training may start at 80 %).
QA (Phase 5, written to training\\reports\\label_qa_R1.md); a job that fails a blocking check is left out and listed:
  blocking: a `proposal` whose verdict is still "?" (or an old `candidate` box), a zero-area box;
  a `proposal` becomes a swimmer / person box from its verdict; verdict not_a_person drops it;
  reported: boxes under 4 px, duplicates (IoU > 0.9), class vs pool mask disagreement (water fraction 0 / > 0.5).
Train set = reviewed train frames x2 (2x sampling weight) + auto-accepted consensus frames of the train split x1
(status "auto_accepted", none of them in the seed). Val = reviewed val frames only. Categories: 0 swimmer, 1 person.
Images stay in training\\data\\frames (file_name = "<clip>/<frame>.jpg").

    training\\.venv\\Scripts\\python.exe training\\tools\\export_seed.py [--all-jobs]   # --all-jobs: pipeline smoke only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import requests

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "training"
BASE = "http://localhost:8080"
OUT = TR / "data" / "coco" / "seed_r1"
CATS = [{"id": 0, "name": "swimmer"}, {"id": 1, "name": "person"}]
CID = {"swimmer": 0, "person": 1}


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def session() -> requests.Session:
    c = dict(line.split("=", 1) for line in (TR / "cvat/credentials.txt").read_text().split() if "=" in line)
    s = requests.Session()
    s.headers["Authorization"] = "Token " + s.post(f"{BASE}/api/auth/login", json={"username": c["username"],
                                                                                    "password": c["password"]}).json()["key"]
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-jobs", action="store_true", help="export every job, completed or not (pipeline smoke only)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    seed = json.loads((TR / "data" / "seed_r1.json").read_text())
    water = cv2.imread(str(TR / "data/pool_mask/pool_water_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    s = session()
    got = s.get(f"{BASE}/api/labels", params={"project_id": 1, "page_size": 100}).json()["results"]
    labels = {lb["id"]: lb["name"] for lb in got}
    verdict_id = next((a["id"] for lb in got if lb["name"] == "proposal" for a in lb["attributes"] if a["name"] == "verdict"), None)
    tasks = [t for t in s.get(f"{BASE}/api/tasks", params={"search": "CHECK R1", "page_size": 50}).json()["results"]
             if t["name"].startswith("CHECK R1")]

    frames: dict[str, list[dict]] = {"train": [], "val": []}  # {"key", "boxes": [(cls, xyxy)]}
    qa_rows, excluded, n_jobs, n_done = [], [], 0, 0
    for t in sorted(tasks, key=lambda t: t["id"]):
        split = "train" if " train " in t["name"] else "val"
        meta = s.get(f"{BASE}/api/tasks/{t['id']}/data/meta").json()["frames"]
        ann = s.get(f"{BASE}/api/tasks/{t['id']}/annotations").json()
        jobs = s.get(f"{BASE}/api/jobs", params={"task_id": t["id"], "page_size": 50}).json()["results"]
        by_frame: dict[int, list[dict]] = {}
        for sh in ann["shapes"]:
            if sh["type"] == "rectangle":
                by_frame.setdefault(sh["frame"], []).append(sh)
        for j in sorted(jobs, key=lambda j: j["start_frame"]):
            n_jobs += 1
            done = j["state"] == "completed"
            n_done += done
            if not (done or a.all_jobs):
                continue
            problems, notes = [], []
            job_frames = []
            for fi in range(j["start_frame"], j["stop_frame"] + 1):
                key = seed[split][int(meta[fi]["name"][:4])]["key"]
                boxes = []
                for sh in by_frame.get(fi, []):
                    name, b = labels[sh["label_id"]], [float(v) for v in sh["points"][:4]]
                    w, h = b[2] - b[0], b[3] - b[1]
                    if name == "proposal":
                        name = next((x["value"] for x in sh["attributes"] if x["spec_id"] == verdict_id), "?")
                        if name == "not_a_person":
                            continue
                    if name in ("candidate", "?"):
                        problems.append(f"{key}: box not decided (verdict ?)")
                        continue
                    if name not in CID:
                        continue
                    if w <= 0 or h <= 0:
                        problems.append(f"{key}: zero-area {name} box")
                        continue
                    if min(w, h) < 4:
                        notes.append(f"{key}: tiny {name} box {w:.0f}x{h:.0f} px")
                    x0, y0, x1, y1 = (int(round(v)) for v in b)
                    wf = float(water[max(y0, 0):max(y1, 1), max(x0, 0):max(x1, 1)].mean()) if x1 > x0 and y1 > y0 else 0.0
                    if name == "swimmer" and wf == 0:
                        notes.append(f"{key}: swimmer box with no pool-water pixels")
                    if name == "person" and wf > 0.5:
                        notes.append(f"{key}: person box {wf:.0%} over pool water")
                    boxes.append((name, b))
                for i in range(len(boxes)):
                    for k in range(i + 1, len(boxes)):
                        if iou(boxes[i][1], boxes[k][1]) > 0.9:
                            notes.append(f"{key}: duplicate boxes (IoU > 0.9)")
                job_frames.append({"key": key, "boxes": boxes})
            blocked = bool(problems) and done  # --all-jobs smoke keeps unreviewed jobs, candidates simply dropped
            qa_rows.append((t["name"], j["id"], j["state"], len(job_frames), len(problems), len(notes),
                            "excluded" if blocked else "exported",
                            [f"{m} (x{c})" for m, c in Counter(problems + notes).most_common(6)]))
            if blocked:
                excluded.append(j["id"])
            else:
                frames[split].extend(job_frames)

    seed_keys = {x["key"] for v in seed.values() for x in v}
    auto = [json.loads(l) for l in (TR / "labels/auto/pool_consensus.jsonl").open(encoding="utf-8")]
    auto = [r for r in auto if r["split"] == "train" and r["status"] == "auto_accepted" and r["key"] not in seed_keys]

    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val"):
        images, anns = [], []
        entries = [(f, 2 if split == "train" else 1) for f in frames[split]]
        entries += [({"key": r["key"], "boxes": [(i["cls"], i["box"]) for i in r["instances"] if i["accepted"]]}, 1)
                    for r in (auto if split == "train" else [])]
        for f, rep in entries:
            for _ in range(rep):  # repeated image entries = sampling weight
                iid = len(images)
                images.append({"id": iid, "file_name": f["key"], "width": int(water.shape[1]), "height": int(water.shape[0])})
                for cls, b in f["boxes"]:
                    w, h = b[2] - b[0], b[3] - b[1]
                    anns.append({"id": len(anns), "image_id": iid, "category_id": CID[cls], "bbox": [b[0], b[1], w, h],
                                 "area": w * h, "iscrowd": 0})
        (out / f"{split}.json").write_text(json.dumps({"info": {"description": "Life Guard seed R1"}, "licenses": [], "images": images, "annotations": anns, "categories": CATS}))
        counts[split] = (len(images), len({i["file_name"] for i in images}), len(anns))
        print(f"{split}: {counts[split][0]} image entries ({counts[split][1]} unique frames), {len(anns)} boxes")

    rep = [f"# Label QA, round 1 (seed export){' — SMOKE (--all-jobs)' if a.all_jobs else ''}", "",
           f"Jobs Completed: {n_done}/{n_jobs} ({n_done / max(n_jobs, 1):.0%}). Excluded by blocking checks: {excluded or 'none'}.",
           f"Export: train {counts['train'][0]} entries ({counts['train'][1]} unique frames, reviewed x2 + {len(auto)} "
           f"auto-accepted x1), val {counts['val'][0]} frames; boxes train {counts['train'][2]}, val {counts['val'][2]}.", "",
           "| Task | Job | State | Frames | Blocking | Notes | Result | First items |", "|---|---|---|---|---|---|---|---|"]
    for r in qa_rows:
        rep.append(f"| {r[0][:12]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {'; '.join(r[7]) or '—'} |")
    (TR / "reports" / ("label_qa_R1_smoke.md" if a.all_jobs else "label_qa_R1.md")).write_text("\n".join(rep) + "\n",
                                                                                                encoding="utf-8")
    if counts["val"][0] == 0 and not a.all_jobs:
        print("no reviewed val frames: v0 cannot be validated (ask the owner for a val job)")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
