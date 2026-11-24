"""Phase 4 option A (owner, 2026-10-28): a ~250-frame SEED for full human review, pre-labelled, in CVAT.

Selection (train 200 / val 50) from the consensus round 1 (`training\\labels\\auto\\pool_consensus.jsonl`):
  difficulty = confident lone-voter boxes (D-FINE >= 0.40, RTMDet >= 0.30, OWLv2 >= 0.30) + crowded overlaps +
               pool-edge + small flags;
  diversity  = DINOv2 k-means over the split's pool frames (k = number to pick); from each cluster the hardest frame.
Pre-labels: accepted instances as `swimmer` / `person` (pool mask rule); confident lone-voter boxes as `candidate`
(owner converts or deletes); a `review` tag per frame with the reason. Tasks: "SEED R1 train" (jobs of 30) and
"SEED R1 val" (jobs of 25). Writes training\\data\\seed_r1.json.

    training\\.venv\\Scripts\\python.exe training\\tools\\seed_review.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "training"
BASE = "http://localhost:8080"
LONE = {"dfine": 0.40, "rtmdet": 0.30, "owlv2": 0.30}
N = {"train": 200, "val": 50}


def difficulty(r: dict) -> tuple[int, list[str]]:
    reasons = []
    lone = sum(1 for i in r["instances"] if not i["accepted"] and next(iter(i["voters"].values())) >=
               LONE[next(iter(i["voters"]))])
    if lone:
        reasons.append(f"{lone} candidate(s): only one model confident")
    fl = [f for i in r["instances"] if i["accepted"] for f in i["flags"] if f in ("crowded_overlap", "pool_edge", "small")]
    for f in sorted(set(fl)):
        reasons.append(f"{fl.count(f)} x {f.replace('_', ' ')}")
    return lone * 2 + len(fl), reasons


def main() -> int:
    recs = {r["key"]: r for r in map(json.loads, (TR / "labels/auto/pool_consensus.jsonl").open(encoding="utf-8"))}
    picked: dict[str, list[dict]] = {}
    for split, n in N.items():
        keys = [k for k, r in recs.items() if r["split"] == split]
        emb = []
        for k in keys:
            clip, f = k.split("/")
            meta = json.loads((TR / "data/frames" / clip / "index.json").read_text())
            idx = next(i for i, m in enumerate(meta) if m["file"] == f)
            emb.append(np.load(TR / "data/frames" / clip / "dinov2.npy", mmap_mode="r")[idx].astype(np.float32))
        lab = KMeans(n_clusters=n, n_init=3, random_state=0).fit_predict(np.stack(emb))
        sel = []
        for c in range(n):
            members = [k for k, l in zip(keys, lab) if l == c]
            best = max(members, key=lambda k: difficulty(recs[k])[0])
            d, why = difficulty(recs[best])
            sel.append({"key": best, "difficulty": d, "reasons": why})
        picked[split] = sorted(sel, key=lambda x: -x["difficulty"])  # hardest first
        print(f"{split}: {len(sel)} frames, median difficulty {np.median([x['difficulty'] for x in sel])}")
    (TR / "data" / "seed_r1.json").write_text(json.dumps(picked, indent=1))

    c = dict(line.split("=", 1) for line in (TR / "cvat/credentials.txt").read_text().split() if "=" in line)
    s = requests.Session()
    s.headers["Authorization"] = "Token " + s.post(f"{BASE}/api/auth/login", json={"username": c["username"],
                                                                                    "password": c["password"]}).json()["key"]
    labels = {l["name"]: l for l in s.get(f"{BASE}/api/labels", params={"project_id": 1, "page_size": 100}).json()["results"]
              if not l.get("parent_id")}
    act = {n: next(a["id"] for a in labels[n]["attributes"] if a["name"] == "activity") for n in ("swimmer", "person")}
    rr = next(a["id"] for a in labels["review"]["attributes"] if a["name"] == "review_reason")
    for split, seg in (("train", 30), ("val", 25)):
        name = f"SEED R1 {split} ({len(picked[split])} frames, pre-labelled: correct everything)"
        if any(t["name"] == name for t in s.get(f"{BASE}/api/tasks", params={"search": "SEED R1"}).json()["results"]):
            print(f"exists: {name}")
            continue
        # upload order = hardest first; file names keep that order under natural sorting
        files = []
        for n_, x in enumerate(picked[split]):
            clip, f = x["key"].split("/")
            files.append((f"{n_:04d}_{clip[11:]}_{f}", (TR / "data/frames" / clip / f).read_bytes()))
        tid = s.post(f"{BASE}/api/tasks", json={"name": name, "project_id": 1, "segment_size": seg}).json()["id"]
        r = s.post(f"{BASE}/api/tasks/{tid}/data", data={"image_quality": 95, "sorting_method": "natural"},
                   files=[(f"client_files[{i}]", (fn, b, "image/jpeg")) for i, (fn, b) in enumerate(files)], timeout=1800)
        for _ in range(120):  # wait for CVAT to finish creating the task data
            if s.get(f"{BASE}/api/tasks/{tid}").json().get("size"):
                break
            time.sleep(2)
        shapes, tags = [], []
        for fi, x in enumerate(picked[split]):
            for inst in recs[x["key"]]["instances"]:
                if inst["accepted"]:
                    cls = inst["cls"]
                    shapes.append({"type": "rectangle", "frame": fi, "label_id": labels[cls]["id"], "points": inst["box"],
                                   "attributes": [{"spec_id": act[cls], "value": "unknown"}], "occluded": False,
                                   "z_order": 0, "group": 0, "source": "auto"})
                else:
                    (v, sc), = inst["voters"].items()
                    if sc >= LONE[v]:
                        shapes.append({"type": "rectangle", "frame": fi, "label_id": labels["candidate"]["id"],
                                       "points": inst["box"], "attributes": [], "occluded": False, "z_order": 0,
                                       "group": 0, "source": "auto"})
            tags.append({"frame": fi, "label_id": labels["review"]["id"], "group": 0, "source": "auto",
                         "attributes": [{"spec_id": rr, "value": "; ".join(x["reasons"]) or "diversity pick"}]})
        r2 = s.put(f"{BASE}/api/tasks/{tid}/annotations", json={"version": 0, "shapes": shapes, "tags": tags, "tracks": []},
                   timeout=300)
        jobs = sorted(j["id"] for j in s.get(f"{BASE}/api/jobs", params={"task_id": tid, "page_size": 50}).json()["results"])
        print(f"task {tid} '{name}': upload {r.status_code}, pre-labels {r2.status_code} "
              f"({len(shapes)} boxes, {len(tags)} tags), jobs {jobs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
