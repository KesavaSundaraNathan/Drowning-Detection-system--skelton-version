"""Watchdog hook (every 5 min): queue v0 (export, D-FINE-L, RTMDet-L, ONNX export + val eval) once the seed CHECK is ready.

Rule (MASTER_PROMPT: training may start when 80 % of a round's jobs are done): >= 80 % of the CHECK R1 jobs Completed
with no `proposal` left on "?"
and at least one val job Completed (v0 needs reviewed val frames). Queues once; the marker
training\\data\\seed_r1_trigger.json records when. Exits quietly when CVAT is not reachable.

    training\\.venv\\Scripts\\python.exe training\\tools\\seed_trigger.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "training"
BASE = "http://localhost:8080"
MARKER = TR / "data" / "seed_r1_trigger.json"
PY = str(TR / ".venv" / "Scripts" / "python.exe")
QUEUE = [r"C:\Program Files\Python312\python.exe", str(TR / "tools" / "queue" / "jobqueue.py"), "add"]
NOTIFY = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(TR / "tools" / "notify.ps1")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if MARKER.exists():
        return 0
    try:
        c = dict(line.split("=", 1) for line in (TR / "cvat/credentials.txt").read_text().split() if "=" in line)
        s = requests.Session()
        s.headers["Authorization"] = "Token " + s.post(f"{BASE}/api/auth/login", timeout=10, json={
            "username": c["username"], "password": c["password"]}).json()["key"]
        tasks = [t for t in s.get(f"{BASE}/api/tasks", params={"search": "CHECK R1", "page_size": 50}, timeout=10)
                 .json()["results"] if t["name"].startswith("CHECK R1")]
        jobs = [(t["name"], j) for t in tasks for j in
                s.get(f"{BASE}/api/jobs", params={"task_id": t["id"], "page_size": 50}, timeout=10).json()["results"]]
    except (requests.RequestException, KeyError, ValueError):
        return 0  # CVAT down (Docker not started yet): try again at the next watchdog call
    vid = next((a["id"] for lb in s.get(f"{BASE}/api/labels", params={"project_id": 1, "page_size": 100}, timeout=10)
                .json()["results"] if lb["name"] == "proposal" for a in lb["attributes"] if a["name"] == "verdict"), None)

    def checked(j: dict) -> bool:
        """Completed AND no proposal left on "?" (a finished job with open boxes does not count)."""
        if j["state"] != "completed":
            return False
        shapes = s.get(f"{BASE}/api/jobs/{j['id']}/annotations", timeout=60).json()["shapes"]
        return not any(x["spec_id"] == vid and x["value"] == "?" for sh in shapes for x in sh["attributes"])

    ok = {j["id"]: checked(j) for _, j in jobs}
    done = [j for _, j in jobs if ok[j["id"]]]
    val_done = [j for n, j in jobs if " val " in n and ok[j["id"]]]
    open_done = [j["id"] for _, j in jobs if j["state"] == "completed" and not ok[j["id"]]]
    if open_done:
        print(f"finished but with boxes still on '?': jobs {open_done} (not counted)")
    frac = len(done) / max(len(jobs), 1)
    print(f"CHECK R1: {len(done)}/{len(jobs)} jobs completed with every box answered ({frac:.0%}), val {len(val_done)}")
    if frac < 0.8 or not val_done or a.dry_run:
        return 0
    ids = []
    for name, cmd, resume in (
        ("export_seed_r1", [PY, "training/tools/export_seed.py"], ""),
        ("train_dfine_l_v0", [PY, "training/tools/train_dfine.py", "-c", "training/configs/dfine_l_v0.yml",
                              "-t", "training/weights/dfine_l_coco.pth"], "--resume"),
        ("train_rtmdet_l_v0", [PY, "training/tools/train_rtmdet.py", "training/configs/rtmdet_l_v0.py"], "--resume"),
        ("post_v0_export_eval", [PY, "training/tools/post_v0.py"], ""),
    ):
        ids.append(subprocess.run([*QUEUE, "--name", name, f"--resume-arg={resume}", "--", *cmd],
                                  capture_output=True, text=True, check=True).stdout.strip())
    MARKER.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "completed": [j["id"] for j in done],
                                  "of": len(jobs), "queued": ids}, indent=1))
    subprocess.run([*NOTIFY, "-Title", "Life Guard training",
                    "-Message", f"Seed {frac:.0%} reviewed: v0 training queued (D-FINE-L, RTMDet-L, then export + eval)"], check=False)
    print("queued", ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
