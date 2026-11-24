"""Phase 4 temporal check on the dense segments (25 fps, 10 s each): link consensus boxes frame to frame and flag.

Linking (greedy, per segment, frame t -> t+1, at most 2 missing frames bridged): IoU >= 0.3, or centre distance
< 0.5 x box height (a "jump" link). Boxes used: accepted consensus instances and confident lone-voter boxes
(D-FINE >= 0.40, RTMDet >= 0.30, OWLv2 >= 0.30, the same `candidate` rule as the seed).
Flags (MASTER_PROMPT Phase 4):
  singleton  - an accepted box with no link in either direction (appears in 1 frame);
  jump       - a link made only by centre distance (IoU < 0.3): implausible motion or a box switching person;
  size_jump  - box area changes by more than 40 % between linked frames;
  gap        - a track missing for 1-2 frames between two detections (interpolation candidates for review).
Also reported: lone-voter tracks lasting >= 1 s (25 frames): probably real people that only one model sees.
Output: training\\labels\\auto\\dense_temporal.jsonl (per-frame flags) and training\\reports\\temporal_check_dense.md.

    training\\.venv\\Scripts\\python.exe training\\tools\\temporal_check.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "training"
LONE = {"dfine": 0.40, "rtmdet": 0.30, "owlv2": 0.30}
LINK_IOU, JUMP_FRAC, SIZE_JUMP, MAX_GAP, LONG = 0.3, 0.5, 0.4, 2, 25


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def link_score(a: list[float], b: list[float]) -> tuple[float, bool]:
    """(score, is_jump); score 0 = no link."""
    o = iou(a, b)
    if o >= LINK_IOU:
        return 1 + o, False
    h = max(a[3] - a[1], b[3] - b[1], 1.0)
    d = np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
    return (0.5 * (1 - d / (JUMP_FRAC * h)), True) if d < JUMP_FRAC * h else (0.0, False)


def main() -> int:
    segs: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    for line in (TR / "labels/auto/dense_consensus.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        clip, f = r["key"].split("@")
        boxes = []
        for i, inst in enumerate(r["instances"]):
            if inst["accepted"]:
                boxes.append({"i": i, "box": inst["box"], "acc": True})
            else:
                (v, sc), = inst["voters"].items() if len(inst["voters"]) == 1 else ((None, 0),)
                if v and sc >= LONE[v]:
                    boxes.append({"i": i, "box": inst["box"], "acc": False})
        segs[clip][int(f)] = boxes

    flags: dict[str, list[dict]] = defaultdict(list)
    stats: Counter = Counter()
    tracks_all = []
    for clip, frames in segs.items():
        fs = sorted(frames)
        active: list[dict] = []  # {"boxes": [(f, box, acc)], "last": f}
        tracks = []
        for f in fs:
            cur = frames[f]
            cands = []
            for ti, t in enumerate(active):
                if f - t["last"] > MAX_GAP + 1:
                    continue
                lb = t["boxes"][-1][1]
                for bi, b in enumerate(cur):
                    s, jump = link_score(lb, b["box"])
                    if s > 0:
                        cands.append((s, ti, bi, jump))
            used_t, used_b = set(), set()
            for s, ti, bi, jump in sorted(cands, reverse=True):
                if ti in used_t or bi in used_b:
                    continue
                used_t.add(ti)
                used_b.add(bi)
                t, b = active[ti], cur[bi]
                pf, pb, _ = t["boxes"][-1]
                key = f"{clip}@{f}"
                if jump:
                    flags[key].append({"i": b["i"], "flag": "jump"})
                    stats["jump"] += b["acc"]
                a0 = (pb[2] - pb[0]) * (pb[3] - pb[1])
                a1 = (b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1])
                if a0 > 0 and abs(a1 - a0) / a0 > SIZE_JUMP:
                    flags[key].append({"i": b["i"], "flag": "size_jump"})
                    stats["size_jump"] += b["acc"]
                if f - pf > 1:
                    for g in range(pf + 1, f):
                        flags[f"{clip}@{g}"].append({"flag": "gap", "box": [round((x + y) / 2, 1) for x, y in zip(pb, b["box"])]})
                    stats["gap_frames"] += f - pf - 1
                t["boxes"].append((f, b["box"], b["acc"]))
                t["last"] = f
            for bi, b in enumerate(cur):
                if bi not in used_b:
                    active.append({"boxes": [(f, b["box"], b["acc"])], "last": f, "start": f})
            done = [t for t in active if f - t["last"] > MAX_GAP + 1]
            tracks += done
            active = [t for t in active if f - t["last"] <= MAX_GAP + 1]
        tracks += active
        for t in tracks:
            n, n_acc = len(t["boxes"]), sum(x[2] for x in t["boxes"])
            if n == 1 and n_acc == 1:
                f, b, _ = t["boxes"][0]
                flags[f"{clip}@{f}"].append({"box": b, "flag": "singleton"})
                stats["singleton"] += 1
            if n_acc == 0 and n >= LONG:
                stats["lone_tracks_ge_1s"] += 1
                stats["lone_track_boxes_ge_1s"] += n
            if n_acc:
                stats["accepted_tracks"] += 1
            tracks_all.append((n, n_acc))
        stats["frames"] += len(fs)
        stats["accepted_boxes"] += sum(b["acc"] for v in frames.values() for b in v)
        stats["lone_boxes"] += sum(not b["acc"] for v in frames.values() for b in v)

    with (TR / "labels/auto/dense_temporal.jsonl").open("w", encoding="utf-8") as fo:
        for k in sorted(flags):
            fo.write(json.dumps({"key": k, "flags": flags[k]}) + "\n")
    acc_len = [n for n, a in tracks_all if a]
    lone_len = [n for n, a in tracks_all if not a]
    flagged_frames = len({k for k, v in flags.items() if any(x["flag"] != "gap" for x in v)})
    rep = f"""# Temporal check, dense segments (Phase 4)

Input: `labels\\auto\\dense_consensus.jsonl` ({stats['frames']:,} frames, {len(segs)} clips, 25 fps; consensus without SAM).
Boxes linked: {stats['accepted_boxes']:,} accepted + {stats['lone_boxes']:,} confident lone-voter boxes.

| Check | Count (accepted boxes) |
|---|---|
| singleton (1 frame only) | {stats['singleton']:,} |
| jump (linked only by centre distance) | {stats['jump']:,} |
| size change > 40 % between linked frames | {stats['size_jump']:,} |
| gap frames bridged (1–2 missing frames inside a track; interpolation candidates) | {stats['gap_frames']:,} |
| frames with at least one non-gap flag | {flagged_frames:,} of {stats['frames']:,} |

Tracks with accepted boxes: {stats['accepted_tracks']:,}; median length {np.median(acc_len) if acc_len else 0:.0f} frames
(p90 {np.percentile(acc_len, 90) if acc_len else 0:.0f}).
Lone-voter tracks lasting >= 1 s: **{stats['lone_tracks_ge_1s']:,}** ({stats['lone_track_boxes_ge_1s']:,} boxes); median lone
track length {np.median(lone_len) if lone_len else 0:.0f} frames. A box that one model keeps for a second or more is more
likely a real person the other voters miss than noise (not checked by eye), so these are review candidates.
Short accepted tracks (median above) and the many bridged gaps show that the COCO voters' agreement flickers from frame
to frame: per-frame consensus alone under-labels the dense segments.

Use: these flags go into the dense review / self-training round (Phase 7) once v0 replaces the COCO voters; no
dense labels are used for v0 (option A trains on the reviewed seed).
"""
    (TR / "reports/temporal_check_dense.md").write_text(rep, encoding="utf-8")
    print(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
