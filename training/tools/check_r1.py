"""Auto-label method (owner, 2026-10-28): the SEED as a one-key-per-box CHECK in CVAT's Attribute Annotation Mode.

Replaces the edit-every-frame seed review (tasks "SEED R1 ..." are renamed "OLD ... superseded", nothing deleted).
Same 250 frames (training\\data\\seed_r1.json). Every box proposal is a `proposal` rectangle with the attribute
`verdict` (? / swimmer / person / not_a_person):
  - boxes 2 of 3 detectors agreed on: verdict preset to the pool-mask class (swimmer / person); Tab to accept;
  - boxes only one detector saw (D-FINE >= 0.30, RTMDet >= 0.20, OWLv2 >= 0.20; de-cluttered: dropped when IoU >= 0.4
    with a kept box): verdict "?" - one key decides.
The owner then adds anyone still missing as a plain `swimmer` / `person` box. export_seed.py turns verdicts into
labels; a job with any "?" left is not exported. Tasks: "CHECK R1 train" (jobs of 30) and "CHECK R1 val" (jobs of 25).

    training\\.venv\\Scripts\\python.exe training\\tools\\check_r1.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "training"
BASE = "http://localhost:8080"
THR = {"dfine": 0.30, "rtmdet": 0.20, "owlv2": 0.20}
DECLUTTER_IOU = 0.4
VERDICTS = ["?", "swimmer", "person", "not_a_person"]


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def proposals(rec: dict) -> list[tuple[list[float], str]]:
    """[(box, preset verdict)] for one frame."""
    kept = [(i["box"], i["cls"]) for i in rec["instances"] if i["accepted"]]
    lone = [i for i in rec["instances"] if not i["accepted"] and any(s >= THR[v] for v, s in i["voters"].items())]
    for i in sorted(lone, key=lambda i: -max(i["voters"].values())):
        if all(iou(i["box"], k[0]) < DECLUTTER_IOU for k in kept):
            kept.append((i["box"], "?"))
    return kept


def main() -> int:
    seed = json.loads((TR / "data" / "seed_r1.json").read_text())
    recs = {r["key"]: r for r in map(json.loads, (TR / "labels/auto/pool_consensus.jsonl").open(encoding="utf-8"))}
    c = dict(line.split("=", 1) for line in (TR / "cvat/credentials.txt").read_text().split() if "=" in line)
    s = requests.Session()
    s.headers["Authorization"] = "Token " + s.post(f"{BASE}/api/auth/login", json={"username": c["username"],
                                                                                    "password": c["password"]}).json()["key"]
    get_labels = lambda: {lb["name"]: lb for lb in s.get(f"{BASE}/api/labels", params={"project_id": 1, "page_size": 100})  # noqa: E731
                          .json()["results"] if not lb.get("parent_id")}
    labels = get_labels()
    if "proposal" not in labels:
        r = s.patch(f"{BASE}/api/projects/1", json={"labels": [{
            "name": "proposal", "type": "rectangle", "color": "#ff00cc",
            "attributes": [{"name": "verdict", "mutable": True, "input_type": "radio", "default_value": "?",
                            "values": VERDICTS}]}]})
        r.raise_for_status()
        labels = get_labels()
    verdict_id = next(a["id"] for a in labels["proposal"]["attributes"] if a["name"] == "verdict")

    for t in s.get(f"{BASE}/api/tasks", params={"search": "SEED R1", "page_size": 50}).json()["results"]:
        if not t["name"].startswith("OLD"):
            s.patch(f"{BASE}/api/tasks/{t['id']}", json={"name": f"OLD {t['name'][:80]} - superseded by CHECK R1"})
            print(f"renamed task {t['id']}")

    for split, seg in (("train", 30), ("val", 25)):
        name = f"CHECK R1 {split} ({len(seed[split])} frames): one key per box, then add anyone missing"
        if any(t["name"] == name for t in s.get(f"{BASE}/api/tasks", params={"search": "CHECK R1"}).json()["results"]):
            print(f"exists: {name}")
            continue
        files = []
        for n, x in enumerate(seed[split]):  # same file naming as the seed tasks: index prefix = position in seed_r1.json
            clip, f = x["key"].split("/")
            files.append((f"{n:04d}_{clip[11:]}_{f}", (TR / "data/frames" / clip / f).read_bytes()))
        tid = s.post(f"{BASE}/api/tasks", json={"name": name, "project_id": 1, "segment_size": seg}).json()["id"]
        r = s.post(f"{BASE}/api/tasks/{tid}/data", data={"image_quality": 95, "sorting_method": "natural"},
                   files=[(f"client_files[{i}]", (fn, b, "image/jpeg")) for i, (fn, b) in enumerate(files)], timeout=1800)
        for _ in range(120):
            if s.get(f"{BASE}/api/tasks/{tid}").json().get("size"):
                break
            time.sleep(2)
        shapes, n_pre = [], 0
        for fi, x in enumerate(seed[split]):
            for box, v in proposals(recs[x["key"]]):
                n_pre += v != "?"
                shapes.append({"type": "rectangle", "frame": fi, "label_id": labels["proposal"]["id"], "points": box,
                               "attributes": [{"spec_id": verdict_id, "value": v}], "occluded": False, "z_order": 0,
                               "group": 0, "source": "auto"})
        r2 = s.put(f"{BASE}/api/tasks/{tid}/annotations", json={"version": 0, "shapes": shapes, "tags": [], "tracks": []},
                   timeout=300)
        jobs = sorted(j["id"] for j in s.get(f"{BASE}/api/jobs", params={"task_id": tid, "page_size": 50}).json()["results"])
        print(f"task {tid} '{name[:16]}': upload {r.status_code}, proposals {r2.status_code}: {len(shapes)} boxes "
              f"({n_pre} preset, {len(shapes) - n_pre} '?'), jobs {jobs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
