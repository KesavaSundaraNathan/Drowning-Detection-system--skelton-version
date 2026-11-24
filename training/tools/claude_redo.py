"""Round 2 of Claude's seed check, after the owner's audit failed round 1 (25.1 % wrong, 2026-10-29).
New rule: a box is a person when any visible part of a person (even a dark head or a partly submerged body, small or far)
is inside it; loose, duplicate and group boxes that hold a person count as a person.

make  --out DIR : contact sheets (24 numbered tiles each) of the re-check candidates (not_a_person / ? boxes mostly on
                  water, not already settled by the owner's audit) -> DIR/sheet_NNN.png + DIR/index.json
apply --out DIR : DIR/answers.json {"sheet_NNN": [tile numbers that ARE a person]} (every other tile -> not_a_person) and
                  --owner : the owner's audited verdicts (training/data/audit result) are written first, as ground truth.
Everything written is logged to labels/auto/claude_check_r2.jsonl; the owner's own CVAT answers are never overwritten.
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import claude_check_sheets as c

AUD = Path.home() / "Downloads" / "audit_r1_result.json"
CS = json.loads  # alias


def verdict_for_person(water, box):
    x1, y1, x2, y2 = (int(round(x)) for x in box)
    y0 = int(y1 + 0.4 * (y2 - y1))
    return "swimmer" if water[max(y0, 0):max(y2, 1), max(x1, 0):max(x2, 1)].any() else "person"


def audited() -> dict:
    return {r["id"]: r for r in json.loads(AUD.read_text())}


def make(out: Path) -> int:
    cand = json.loads((c.TR / "data/audit/recheck_cand.json").read_text())
    aud = audited()
    items = [x for x in cand if x[1] not in aud]
    out.mkdir(parents=True, exist_ok=True)
    per, T = 24, 240
    index = {}
    for s0 in range(0, len(items), per):
        name = f"sheet_{s0 // per:03d}"
        tiles = []
        for j, (k, bid, b, v, _f) in enumerate(items[s0:s0 + per], 1):
            task, key = k.split("|", 1)
            img = cv2.imread(str(c.TR / "data/frames" / key))
            x1, y1, x2, y2 = b
            bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hw, hh = max(bw * 1.5, 70), max(bh * 1.5, 70)
            X1, Y1, X2, Y2 = int(max(0, cx - hw)), int(max(0, cy - hh)), int(min(img.shape[1], cx + hw)), int(min(img.shape[0], cy + hh))
            cr = img[Y1:Y2, X1:X2].copy()
            cv2.rectangle(cr, (int(x1 - X1), int(y1 - Y1)), (int(x2 - X1), int(y2 - Y1)), (0, 255, 255), 1)
            r = (T - 4) / max(cr.shape[:2])
            cr = cv2.resize(cr, None, fx=r, fy=r, interpolation=cv2.INTER_CUBIC)
            t = np.zeros((T, T, 3), np.uint8)
            t[:cr.shape[0], :cr.shape[1]] = cr
            cv2.putText(t, str(j), (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles.append(t)
            index.setdefault(name, []).append({"task": int(task), "id": bid, "key": key, "box": b})
        while len(tiles) % 6:
            tiles.append(np.zeros((T, T, 3), np.uint8))
        rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
        cv2.imwrite(str(out / f"{name}.png"), np.vstack(rows))
    (out / "index.json").write_text(json.dumps(index))
    print(f"{len(items)} boxes -> {len(index)} sheets")
    return 0


def apply(out: Path, owner_only: bool) -> int:
    s = c.session()
    pid, vid = c.verdict_spec(s)
    water = cv2.imread(str(c.TR / "data/pool_mask/pool_water_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    want: dict[int, dict[int, str]] = {}
    log = []
    frames = json.loads((c.TR / "data/audit/frames_state.json").read_text())
    boxes = {i: (k, b) for k, L in frames.items() for i, b, v in L}
    for bid, r in audited().items():
        if r["owner"] == "wrong":  # flip Claude's round-1 answer
            v = verdict_for_person(water, boxes[bid][1]) if r["verdict"] == "not_a_person" else "not_a_person"
        else:
            v = r["verdict"]
        want.setdefault(r["task"], {})[bid] = v
        log.append({"id": bid, "task": r["task"], "verdict": v, "src": "owner_audit"})
    if not owner_only:
        index = json.loads((out / "index.json").read_text())
        answers = json.loads((out / "answers.json").read_text())
        for name, tiles in index.items():
            if name not in answers:
                continue
            yes = set(answers[name])
            for j, it in enumerate(tiles, 1):
                v = verdict_for_person(water, it["box"]) if j in yes else "not_a_person"
                want.setdefault(it["task"], {})[it["id"]] = v
                log.append({"id": it["id"], "task": it["task"], "verdict": v, "src": f"claude_r2:{name}:{j}"})
    n = 0
    for task, vd in want.items():
        shapes = s.get(f"{c.BASE}/api/tasks/{task}/annotations").json()["shapes"]
        upd = []
        for sh in shapes:
            if sh["id"] in vd:
                sh["attributes"] = [{"spec_id": vid, "value": vd[sh["id"]]}]
                upd.append(sh)
        if upd:
            s.patch(f"{c.BASE}/api/tasks/{task}/annotations", params={"action": "update"},
                    json={"version": 0, "shapes": upd, "tags": [], "tracks": []}).raise_for_status()
            n += len(upd)
    with (c.TR / "labels/auto/claude_check_r2.jsonl").open("a", encoding="utf-8") as f:
        for row in log:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {n} verdicts")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "apply"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--owner-only", action="store_true")
    a = ap.parse_args()
    return make(Path(a.out)) if a.cmd == "make" else apply(Path(a.out), a.owner_only)


if __name__ == "__main__":
    sys.exit(main())
