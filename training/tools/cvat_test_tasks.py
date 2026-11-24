"""Phase 9 early start: the hand-labelled TEST set as CVAT tasks in project "Life Guard swimmers v1" (no pre-labels).

  Test A: 100 of the 300 test frames — boxes + class + keypoints (jobs of 25)
  Test B: the other 200 — boxes + class (jobs of 25)
  Test C: the dense 10-s segment at 25 fps (all frames) — boxes + class + track ids (jobs of 50, overlap 5)
Images are uploaded to the LOCAL CVAT only. Idempotent: an existing task with the same name is skipped.

    training\\.venv\\Scripts\\python.exe training\\tools\\cvat_test_tasks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "training"
BASE = "http://localhost:8080"


def session() -> requests.Session:
    c = dict(line.split("=", 1) for line in (TR / "cvat" / "credentials.txt").read_text().split() if "=" in line)
    s = requests.Session()
    r = s.post(f"{BASE}/api/auth/login", json={"username": c["username"], "password": c["password"]}, timeout=30)
    r.raise_for_status()
    s.headers["Authorization"] = f"Token {r.json()['key']}"
    return s


def create(s: requests.Session, pid: int, name: str, files: list[Path], seg: int, overlap: int = 0) -> None:
    if any(t["name"] == name for t in s.get(f"{BASE}/api/tasks", params={"search": name[:30]}).json()["results"]):
        print(f"exists: {name}")
        return
    body = {"name": name, "project_id": pid, "segment_size": seg}
    if overlap:
        body["overlap"] = overlap
    tid = s.post(f"{BASE}/api/tasks", json=body, timeout=60).json()["id"]
    up = [(f"client_files[{i}]", (f.name, open(f, "rb"), "image/jpeg")) for i, f in enumerate(files)]
    r = s.post(f"{BASE}/api/tasks/{tid}/data", data={"image_quality": 95, "sorting_method": "natural"}, files=up,
               timeout=1800)
    print(f"task {tid} '{name}': {len(files)} frames, upload {r.status_code}")


def main() -> int:
    samp = json.loads((TR / "data" / "sampling.json").read_text())
    s = session()
    pid = next(p["id"] for p in s.get(f"{BASE}/api/projects").json()["results"] if p["name"] == "Life Guard swimmers v1")
    # stage copies with clip-prefixed names so frames from different clips never collide
    stage = TR / "data" / "test_upload"
    stage.mkdir(parents=True, exist_ok=True)
    files = []
    for t in sorted(samp["test_frames"], key=lambda x: (x["clip"], x["frame"])):
        src = TR / "data" / "frames" / t["clip"][:-4] / t["file"]
        dst = stage / f"{t['clip'][11:-4]}_{t['file']}"
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        files.append(dst)
    rng = np.random.default_rng(20261028)
    kp = set(rng.choice(len(files), 100, replace=False).tolist())
    a = [f for i, f in enumerate(files) if i in kp]
    b = [f for i, f in enumerate(files) if i not in kp]
    (TR / "data" / "test_sets.json").write_text(json.dumps({"A_keypoints": [f.name for f in a],
                                                             "B_boxes": [f.name for f in b]}, indent=1))
    create(s, pid, "TEST A - boxes + class + keypoints (100 frames, hand only)", a, 25)
    create(s, pid, "TEST B - boxes + class (200 frames, hand only)", b, 25)
    d = samp["test_dense"]
    dense_dir = TR / "data" / "test_dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(ROOT / "training_dataset" / d["clip"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, d["start_frame"])
    dense = []
    for i in range(d["start_frame"], d["end_frame"] + 1):
        ok, fr = cap.read()
        if not ok:
            break
        f = dense_dir / f"f{i:06d}.jpg"
        if not f.exists():
            cv2.imwrite(str(f), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        dense.append(f)
    create(s, pid, f"TEST C - dense 10 s {d['clip'][11:-4]} {d['start_s']}-{d['end_s']} s (boxes + track ids)", dense,
           50, 5)
    for t in s.get(f"{BASE}/api/tasks", params={"project_id": pid}).json()["results"]:
        jobs = s.get(f"{BASE}/api/jobs", params={"task_id": t["id"], "page_size": 50}).json()["results"]
        print(t["id"], t["name"][:60], "jobs:", [j["id"] for j in jobs])
    return 0


if __name__ == "__main__":
    sys.exit(main())
