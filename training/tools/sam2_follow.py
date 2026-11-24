"""Test set C (dense 10-s segment, CVAT task "TEST C"): SAM 2 follows the people the OWNER boxed (owner, 2026-10-28).

Allowed by the owner's change to the test-set rule: the only prompts are boxes the owner drew by hand; no detector or
automatic label is involved. For every owner-drawn `swimmer` / `person` rectangle (a plain shape; the frame it is on is
where that person starts), SAM 2.1 (b+) propagates the person's mask forward through the segment; each frame's mask
becomes a box. Each prompt shape is replaced by a CVAT track with that label: a keyframe on every frame, `outside`
where the mask is empty. The owner then scrubs through and fixes the frames where it slipped.
Run again after adding people who appear later: shapes that are already tracks are left alone.

    training\\.venv\\Scripts\\python.exe training\\tools\\sam2_follow.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests
import torch

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "training"
BASE = "http://localhost:8080"
TASK_PREFIX = "TEST C"
MIN_PIXELS = 20  # a mask smaller than this is "outside" (the person is gone or hidden)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report the prompts, write nothing")
    ap.add_argument("--task-prefix", default=TASK_PREFIX, help="tests use a scratch task")
    ap.add_argument("--frames-dir", default=str(TR / "data" / "test_dense"))
    a = ap.parse_args()
    c = dict(line.split("=", 1) for line in (TR / "cvat/credentials.txt").read_text().split() if "=" in line)
    s = requests.Session()
    s.headers["Authorization"] = "Token " + s.post(f"{BASE}/api/auth/login", json={"username": c["username"],
                                                                                    "password": c["password"]}).json()["key"]
    task = next(t for t in s.get(f"{BASE}/api/tasks", params={"search": a.task_prefix}).json()["results"]
                if t["name"].startswith(a.task_prefix))
    labels = {lb["id"]: lb["name"] for lb in s.get(f"{BASE}/api/labels", params={"task_id": task["id"], "page_size": 100})
              .json()["results"]}
    meta = s.get(f"{BASE}/api/tasks/{task['id']}/data/meta").json()["frames"]
    ann = s.get(f"{BASE}/api/tasks/{task['id']}/annotations").json()
    prompts = [sh for sh in ann["shapes"] if sh["type"] == "rectangle" and labels[sh["label_id"]] in ("swimmer", "person")]
    if any(sh.get("source") not in ("manual", None) for sh in prompts):
        print("refused: a prompt shape is not hand-drawn (source != manual); only the owner's boxes may prompt SAM 2")
        return 2
    print(f"task {task['id']}: {len(meta)} frames, {len(prompts)} hand-drawn prompt boxes, {len(ann['tracks'])} tracks so far")
    if a.dry_run or not prompts:
        return 0

    frames_dir = Path(a.frames_dir)
    names = [m["name"] for m in meta]
    missing = [n for n in names if not (frames_dir / n).exists()]
    if missing:
        print(f"missing local frames: {missing[:3]}")
        return 1
    # SAM 2 reads a folder of JPEGs named by integer index: link the task frames in CVAT order
    seq = frames_dir.parent / f"{frames_dir.name}_seq"
    seq.mkdir(exist_ok=True)
    for i, n in enumerate(names):
        p = seq / f"{i:05d}.jpg"
        if not p.exists():
            p.write_bytes((frames_dir / n).read_bytes())

    from sam2.build_sam import build_sam2_video_predictor

    pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_b+.yaml", str(TR / "weights/sam2.1_hiera_base_plus.pt"),
                                      device="cuda")
    tracks = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = pred.init_state(str(seq), offload_video_to_cpu=True)
        for start in sorted({sh["frame"] for sh in prompts}):  # one propagation per start frame, all its people together
            pred.reset_state(state)
            group = [sh for sh in prompts if sh["frame"] == start]
            for oid, sh in enumerate(group):
                pred.add_new_points_or_box(state, frame_idx=start, obj_id=oid, box=np.array(sh["points"][:4], np.float32))
            boxes: dict[int, dict[int, list[float] | None]] = {oid: {} for oid in range(len(group))}
            for fi, oids, logits in pred.propagate_in_video(state, start_frame_idx=start):
                for oid, lg in zip(oids, logits):
                    m = (lg[0] > 0).cpu().numpy()
                    ys, xs = np.nonzero(m)
                    boxes[oid][fi] = ([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
                                      if len(xs) >= MIN_PIXELS else None)
            for oid, sh in enumerate(group):
                shapes, last = [], sh["points"][:4]
                for fi in range(start, len(names)):
                    b = sh["points"][:4] if fi == start else boxes[oid].get(fi)
                    shapes.append({"type": "rectangle", "frame": fi, "points": b or last, "outside": b is None,
                                   "occluded": False, "keyframe": True, "attributes": []})
                    last = b or last
                tracks.append({"frame": start, "label_id": sh["label_id"], "group": 0, "source": "semi-auto",
                               "shapes": shapes, "attributes": []})  # activity: set by the owner later (unknown default)
            print(f"start frame {start}: {len(group)} people followed to frame {len(names) - 1}")
    r1 = s.patch(f"{BASE}/api/tasks/{task['id']}/annotations", params={"action": "create"},
                 json={"version": 0, "shapes": [], "tags": [], "tracks": tracks})
    if not r1.ok:
        print(r1.status_code, r1.text[:800])
        return 1
    r2 = s.patch(f"{BASE}/api/tasks/{task['id']}/annotations", params={"action": "delete"},
                 json={"version": 0, "shapes": prompts, "tags": [], "tracks": []})
    r2.raise_for_status()
    print(f"wrote {len(tracks)} tracks; the {len(prompts)} prompt boxes became their first frames")
    (TR / "logs" / "sam2_follow_last.json").write_text(json.dumps({"tracks": len(tracks), "prompts": len(prompts)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
