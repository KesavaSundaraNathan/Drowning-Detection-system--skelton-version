"""Phase 1: a CVAT task for the owner to draw the pool-water mask once (one camera). Frames are extracted to
training\\data\\pool_mask\\ and uploaded to the LOCAL CVAT only; Claude does not view them.

Project "Life Guard pool mask": polygon label `pool_water` (the water surface, including lane ropes) and polygon label
`exclude` (optional: areas never to be labelled, e.g. a TV screen or poster).

    training\\.venv\\Scripts\\python.exe training\\tools\\cvat_pool_mask_task.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import requests

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "training"
BASE = "http://localhost:8080"
CLIPS = ["2026-10-22-18-14-59.mp4", "2026-10-22-18-46-32.mp4", "2026-10-22-19-13-23.mp4"]  # early, middle, late


def main() -> int:
    out = TR / "data" / "pool_mask"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for c in CLIPS:
        cap = cv2.VideoCapture(str(ROOT / "training_dataset" / c))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
        ok, fr = cap.read()
        if not ok:
            print(f"cannot read {c}")
            return 1
        f = out / f"{c[:-4]}_mid.png"
        cv2.imwrite(str(f), fr)
        files.append(f)
    cred = dict(line.split("=", 1) for line in (TR / "cvat" / "credentials.txt").read_text().split() if "=" in line)
    s = requests.Session()
    r = s.post(f"{BASE}/api/auth/login", json={"username": cred["username"], "password": cred["password"]}, timeout=30)
    r.raise_for_status()
    s.headers["Authorization"] = f"Token {r.json()['key']}"
    name = "Life Guard pool mask"
    found = [p for p in s.get(f"{BASE}/api/projects", params={"search": name}).json()["results"] if p["name"] == name]
    if found:
        pid = found[0]["id"]
    else:
        pid = s.post(f"{BASE}/api/projects", json={"name": name, "labels": [
            {"name": "pool_water", "type": "polygon", "color": "#3b82f6", "attributes": []},
            {"name": "exclude", "type": "polygon", "color": "#9ca3af", "attributes": []},
        ]}, timeout=60).json()["id"]
    task = s.post(f"{BASE}/api/tasks", json={"name": "Draw the pool water once (3 frames, same camera)",
                                             "project_id": pid}, timeout=60).json()
    tid = task["id"]
    with_files = [("client_files[%d]" % i, (f.name, open(f, "rb"), "image/png")) for i, f in enumerate(files)]
    r = s.post(f"{BASE}/api/tasks/{tid}/data", data={"image_quality": 95, "sorting_method": "natural"},
               files=with_files, timeout=300)
    print("upload", r.status_code)
    jobs = s.get(f"{BASE}/api/jobs", params={"task_id": tid}).json()["results"]
    print(f"project {pid}, task {tid}, job {jobs[0]['id'] if jobs else '?'}: "
          f"{BASE}/tasks/{tid}/jobs/{jobs[0]['id'] if jobs else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
