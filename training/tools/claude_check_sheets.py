"""Seed CHECK with Claude's vision (owner-approved 2026-10-29: crops of the 250 SEED frames only, never the test set,
may be sent to Anthropic; a 10 % owner audit follows).

make : every `proposal` still on "?" in the CHECK R1 tasks -> numbered crop sheets (6 x 5 tiles). Each tile: the box
       in yellow, the other proposals in thin cyan (duplicates are visible), 50 % context around it.
       Writes <out>\\sheet_NNN.png and <out>\\index.json (tile -> task, shape id, frame).
apply: reads <out>\\answers.json {"sheet_NNN": {"n": [tile numbers not a person], "u": [unsure]}} (every other tile on
       a judged sheet is a person) and writes the verdicts to CVAT: person -> swimmer / person by the pool-mask rule
       (any of the lower 60 % of the box in the water = swimmer, as consensus.py), "n" -> not_a_person, "u" -> stays "?".
       Records every answer in training\\labels\\auto\\claude_check_r1.jsonl.

    training\\.venv\\Scripts\\python.exe training\\tools\\claude_check_sheets.py make  --out <dir>
    training\\.venv\\Scripts\\python.exe training\\tools\\claude_check_sheets.py apply --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "training"
BASE = "http://localhost:8080"
COLS, ROWS, TW, TH = 5, 4, 260, 300


def session() -> requests.Session:
    c = dict(line.split("=", 1) for line in (TR / "cvat/credentials.txt").read_text().split() if "=" in line)
    s = requests.Session()
    s.headers["Authorization"] = "Token " + s.post(f"{BASE}/api/auth/login", json={"username": c["username"],
                                                                                    "password": c["password"]}).json()["key"]
    return s


def check_tasks(s: requests.Session) -> list[dict]:
    return [t for t in s.get(f"{BASE}/api/tasks", params={"search": "CHECK R1", "page_size": 50}).json()["results"]
            if t["name"].startswith("CHECK R1")]


def verdict_spec(s: requests.Session) -> tuple[int, int]:
    lb = next(lb for lb in s.get(f"{BASE}/api/labels", params={"project_id": 1, "page_size": 100}).json()["results"]
              if lb["name"] == "proposal")
    return lb["id"], next(a["id"] for a in lb["attributes"] if a["name"] == "verdict")


def tile(img: np.ndarray, box: list[float], others: list[list[float]], n: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = max(bw * 1.25, 60), max(bh * 1.25, 70)  # box + 75 % context each side, at least 120 x 140 px
    X1, Y1 = int(max(0, cx - half_w)), int(max(0, cy - half_h))
    X2, Y2 = int(min(w, cx + half_w)), int(min(h, cy + half_h))
    crop = img[Y1:Y2, X1:X2].copy()
    for o in others:
        cv2.rectangle(crop, (int(o[0] - X1), int(o[1] - Y1)), (int(o[2] - X1), int(o[3] - Y1)), (255, 255, 0), 1)
    cv2.rectangle(crop, (int(x1 - X1), int(y1 - Y1)), (int(x2 - X1), int(y2 - Y1)), (0, 255, 255), 2)
    r = min(TW / crop.shape[1], (TH - 22) / crop.shape[0])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * r)), max(1, int(crop.shape[0] * r))), interpolation=cv2.INTER_CUBIC)
    t = np.full((TH, TW, 3), 40, np.uint8)
    t[22:22 + crop.shape[0], :crop.shape[1]] = crop
    cv2.putText(t, str(n), (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(t, f"{int(bh)}px", (TW - 52, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return t


def make(out: Path, layout: str = "frames") -> int:
    s = session()
    seed = json.loads((TR / "data" / "seed_r1.json").read_text())
    pid, vid = verdict_spec(s)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for t in check_tasks(s):
        split = "train" if " train " in t["name"] else "val"
        meta = s.get(f"{BASE}/api/tasks/{t['id']}/data/meta").json()["frames"]
        shapes = s.get(f"{BASE}/api/tasks/{t['id']}/annotations").json()["shapes"]
        by_frame: dict[int, list[dict]] = {}
        for sh in shapes:
            if sh["type"] == "rectangle":
                by_frame.setdefault(sh["frame"], []).append(sh)
        for fi, shs in sorted(by_frame.items()):
            key = seed[split][int(meta[fi]["name"][:4])]["key"]
            for sh in shs:
                v = next((a["value"] for a in sh["attributes"] if a["spec_id"] == vid), None)
                if sh["label_id"] == pid and v == "?":
                    others = [o["points"] for o in shs if (layout == "frames" or o is not sh) and not any(
                        a["spec_id"] == vid and a["value"] == "not_a_person" for a in o["attributes"])]
                    items.append({"task": t["id"], "id": sh["id"], "frame": fi, "key": key, "box": sh["points"][:4],
                                  "others": others})
    if layout == "frames":  # one image per frame: its "?" boxes numbered in yellow, the other boxes thin cyan
        index = {}
        frames: dict[tuple[int, str], list[dict]] = {}
        for it in items:
            frames.setdefault((it["task"], it["key"]), []).append(it)
        for k, ((task, key), its) in enumerate(frames.items()):
            name = f"frame_{k:03d}"
            img = cv2.imread(str(TR / "data/frames" / key))
            z = 1.4
            img = cv2.resize(img, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
            for o in its[0]["others"]:
                if not any(o == it["box"] for it in its):
                    cv2.rectangle(img, (int(o[0] * z), int(o[1] * z)), (int(o[2] * z), int(o[3] * z)), (255, 255, 0), 1)
            for j, it in enumerate(its, 1):
                x1, y1, x2, y2 = (int(v * z) for v in it["box"])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                tx, ty = x1 + 2, y1 + 16
                cv2.rectangle(img, (x1, y1), (x1 + 13 * len(str(j)) + 4, y1 + 20), (0, 0, 0), -1)
                cv2.putText(img, str(j), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                index.setdefault(name, []).append({k2: it[k2] for k2 in ("task", "id", "frame", "key", "box")})
            cv2.imwrite(str(out / f"{name}.png"), img)
        (out / "index.json").write_text(json.dumps(index))
        print(f"{len(items)} '?' boxes -> {len(index)} frame images in {out}")
        return 0
    per = COLS * ROWS
    index = {}
    imgs: dict[str, np.ndarray] = {}
    for k in range(0, len(items), per):
        name = f"sheet_{k // per:03d}"
        sheet = np.full((ROWS * TH, COLS * TW, 3), 20, np.uint8)
        for j, it in enumerate(items[k:k + per]):
            if it["key"] not in imgs:
                imgs = {it["key"]: cv2.imread(str(TR / "data/frames" / it["key"]))}  # items are frame-ordered
            r, c = divmod(j, COLS)
            sheet[r * TH:(r + 1) * TH, c * TW:(c + 1) * TW] = tile(imgs[it["key"]], it["box"], it["others"], j + 1)
            index.setdefault(name, []).append({k2: it[k2] for k2 in ("task", "id", "frame", "key", "box")})
        cv2.imwrite(str(out / f"{name}.png"), sheet)
    (out / "index.json").write_text(json.dumps(index))
    print(f"{len(items)} '?' boxes -> {len(index)} sheets in {out}")
    return 0


def apply(out: Path) -> int:
    s = session()
    pid, vid = verdict_spec(s)
    index = json.loads((out / "index.json").read_text())
    answers = json.loads((out / "answers.json").read_text())
    water = cv2.imread(str(TR / "data/pool_mask/pool_water_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    per_task: dict[int, dict[int, str]] = {}
    log = []
    for name, tiles in index.items():
        if name not in answers:
            continue
        no, unsure = set(answers[name].get("n", [])), set(answers[name].get("u", []))
        for j, it in enumerate(tiles, 1):
            if j in unsure:
                v = "?"
            elif j in no:
                v = "not_a_person"
            else:
                x1, y1, x2, y2 = (int(round(x)) for x in it["box"])
                y0 = int(y1 + 0.4 * (y2 - y1))  # lower 60 % of the box
                v = "swimmer" if water[max(y0, 0):max(y2, 1), max(x1, 0):max(x2, 1)].any() else "person"
            per_task.setdefault(it["task"], {})[it["id"]] = v
            log.append({**it, "sheet": name, "tile": j, "verdict": v})
    n = 0
    for task, verdicts in per_task.items():
        shapes = s.get(f"{BASE}/api/tasks/{task}/annotations").json()["shapes"]
        upd = []
        for sh in shapes:
            if sh["id"] in verdicts:
                cur = next((a["value"] for a in sh["attributes"] if a["spec_id"] == vid), None)
                if cur != "?":  # the owner answered it meanwhile: keep the owner's answer
                    continue
                sh["attributes"] = [{"spec_id": vid, "value": verdicts[sh["id"]]}]
                upd.append(sh)
        if upd:
            r = s.patch(f"{BASE}/api/tasks/{task}/annotations", params={"action": "update"},
                        json={"version": 0, "shapes": upd, "tags": [], "tracks": []})
            r.raise_for_status()
            n += len(upd)
    with (TR / "labels/auto/claude_check_r1.jsonl").open("a", encoding="utf-8") as f:
        for row in log:
            f.write(json.dumps(row) + "\n")
    from collections import Counter
    print(f"wrote {n} verdicts to CVAT: {dict(Counter(r['verdict'] for r in log))}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "apply"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", choices=["frames", "tiles"], default="frames")
    a = ap.parse_args()
    return make(Path(a.out), a.layout) if a.cmd == "make" else apply(Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
