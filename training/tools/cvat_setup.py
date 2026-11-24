"""Create the Life Guard labelling project in the local CVAT (training/MASTER_PROMPT.md, Phase 0), idempotently.

Schema
- rectangle labels `swimmer` and `person`, each with the track attribute `activity` (mutable select);
- skeleton label `pose`: COCO-17 keypoints with the COCO edges (visibility per point: CVAT's occluded / outside flags);
- tag label `review` with the text attribute `review_reason` (CVAT has no frame-level attributes; a tag is per frame).

    training\\.venv\\Scripts\\python.exe training\\tools\\cvat_setup.py
Credentials: training\\cvat\\credentials.txt (written by the setup step, ignored by git).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

TR = Path(__file__).resolve().parents[1]
CRED = TR / "cvat" / "credentials.txt"
BASE = "http://localhost:8080"
PROJECT = "Life Guard swimmers v1"
ACTIVITY = ["freestyle", "breaststroke", "backstroke", "butterfly", "treading_vertical", "floating", "standing_wading",
            "entering_diving", "underwater", "unknown"]
KPTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
        "right_ankle"]
EDGES = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
         (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]  # COCO-17, 0-based
# a standing-figure template (x, y in a 100 x 100 box) so the skeleton is drawn sensibly when placed
POS = [(50, 10), (47, 8), (53, 8), (44, 10), (56, 10), (40, 25), (60, 25), (35, 40), (65, 40), (32, 55), (68, 55),
       (44, 55), (56, 55), (43, 72), (57, 72), (42, 90), (58, 90)]


def skeleton_svg() -> str:
    parts = [
        f'<line x1="{POS[a][0]}" y1="{POS[a][1]}" x2="{POS[b][0]}" y2="{POS[b][1]}" stroke="black" '
        f'data-type="edge" data-node-from="{a + 1}" data-node-to="{b + 1}"></line>'
        for a, b in EDGES
    ]
    parts += [
        f'<circle r="1.5" stroke="black" fill="#b3b3b3" cx="{x}" cy="{y}" stroke-width="0.1" data-type="element node" '
        f'data-element-id="{i + 1}" data-node-id="{i + 1}" data-label-name="{n}"></circle>'
        for i, (n, (x, y)) in enumerate(zip(KPTS, POS))
    ]
    return "".join(parts)


def labels() -> list[dict]:
    activity = {"name": "activity", "mutable": True, "input_type": "select", "default_value": "unknown",
                "values": ACTIVITY}
    return [
        {"name": "swimmer", "type": "rectangle", "color": "#33cc66", "attributes": [activity]},
        {"name": "person", "type": "rectangle", "color": "#3b82f6", "attributes": [activity]},
        {"name": "pose", "type": "skeleton", "color": "#ffdc00", "attributes": [],
         "sublabels": [{"name": n, "type": "points", "attributes": []} for n in KPTS], "svg": skeleton_svg()},
        {"name": "review", "type": "tag", "color": "#eb2828",
         "attributes": [{"name": "review_reason", "mutable": True, "input_type": "text", "default_value": "",
                         "values": [""]}]},
    ]


def main() -> int:
    cred = dict(line.split("=", 1) for line in CRED.read_text().split() if "=" in line)
    s = requests.Session()
    r = s.post(f"{BASE}/api/auth/login", json={"username": cred["username"], "password": cred["password"]}, timeout=30)
    r.raise_for_status()
    s.headers["Authorization"] = f"Token {r.json()['key']}"
    existing = s.get(f"{BASE}/api/projects", params={"search": PROJECT}, timeout=30).json().get("results", [])
    if any(p["name"] == PROJECT for p in existing):
        pid = next(p["id"] for p in existing if p["name"] == PROJECT)
        print(f"project exists: {pid}")
    else:
        r = s.post(f"{BASE}/api/projects", json={"name": PROJECT, "labels": labels()}, timeout=60)
        if r.status_code >= 400:
            print(r.status_code, r.text[:2000])
            return 1
        pid = r.json()["id"]
        print(f"project created: {pid}")
    got = s.get(f"{BASE}/api/labels", params={"project_id": pid, "page_size": 100}, timeout=30).json()["results"]
    summary = {lb["name"]: {"type": lb["type"], "attributes": [a["name"] for a in lb["attributes"]],
                            "sublabels": len(lb.get("sublabels") or [])} for lb in got if not lb.get("parent_id")}
    print(json.dumps(summary, indent=1))
    ok = summary.get("pose", {}).get("sublabels") == 17 and {"swimmer", "person", "pose", "review"} <= set(summary)
    print("schema OK" if ok else "schema INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
