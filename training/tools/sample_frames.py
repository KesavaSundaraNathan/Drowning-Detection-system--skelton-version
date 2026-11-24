"""Phase 2: frame sampling from the 2 fps extraction (training\\data\\frames). Numbers only; no frame is viewed.

Train / val (labelling pool for Option B):
  1. person count per frame with COCO D-FINE-M (score > 0.4) - for STRATIFICATION only, never used as a label;
  2. near-duplicates removed: pHash Hamming distance <= 6 to an already kept frame of the same clip;
  3. DINOv2 k-means per (split, count stratum 0 / 1-3 / 4-8 / 9+); frames picked round-robin over clusters so every
     look is represented; ~10 % hard negatives (count 0, or the most person-like empty frames);
  4. dense 10-s segments: 3 per train clip (2 for the 10-min-heavy ones if needed), 1 per val clip - chosen at
     different crowd levels, all 25 fps frames of the segment are listed (labelled later via propagation).
Test (hand labelling only - NO detector is run on test frames):
  300 frames spread over the 4 clips by DINOv2 k-means (+ pHash dedup) and one dense 10-s segment (middle of the
  most varied clip by embedding spread).

Output: training\\data\\sampling.json, training\\reports\\sampling.md

    training\\.venv\\Scripts\\python.exe training\\tools\\sample_frames.py [--pool 5000]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import imagehash
import numpy as np
import torch
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "training"
FR = TR / "data" / "frames"
RNG = np.random.default_rng(20261028)
STRATA = [(0, 0, "0"), (1, 3, "1-3"), (4, 8, "4-8"), (9, 10**6, "9+")]


def stratum(n: int) -> str:
    return next(s for lo, hi, s in STRATA if lo <= n <= hi)


def load_clip(clip: str) -> tuple[list[dict], np.ndarray]:
    d = FR / clip[:-4]
    return json.loads((d / "index.json").read_text()), np.load(d / "dinov2.npy").astype(np.float32)


def person_counts(clips: list[str]) -> dict[str, list[int]]:
    """COCO D-FINE-M person count per extracted frame (train / val only)."""
    cache = TR / "data" / "person_counts_trainval.json"
    if cache.exists():
        return json.loads(cache.read_text())
    sys.path.insert(0, str(TR / "third_party" / "dfine"))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(TR / "third_party/dfine/configs/dfine/dfine_hgnetv2_m_coco.yml"))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ck = torch.load(TR / "weights/dfine_m_coco.pth", map_location="cpu")
    cfg.model.load_state_dict(ck["ema"]["module"] if "ema" in ck else ck["model"])
    model, post = cfg.model.deploy().cuda().eval(), cfg.postprocessor.deploy()
    out = {}
    for clip in clips:
        meta, _ = load_clip(clip)
        counts = []
        for k in range(0, len(meta), 16):
            imgs = [cv2.imread(str(FR / clip[:-4] / m["file"])) for m in meta[k : k + 16]]
            h, w = imgs[0].shape[:2]
            x = torch.stack([torch.from_numpy(cv2.cvtColor(cv2.resize(i, (640, 640)), cv2.COLOR_BGR2RGB))
                             .permute(2, 0, 1) for i in imgs]).float().div(255).cuda()
            with torch.no_grad():
                lab, _, sc = post(model(x), torch.tensor([[w, h]] * len(imgs), device="cuda"))
            counts += [int(((lab[i] == 0) & (sc[i] > 0.4)).sum()) for i in range(len(imgs))]
        out[clip] = counts
        print(f"counts {clip}: median {int(np.median(counts))}", flush=True)
    cache.write_text(json.dumps(out))
    return out


def dedup(meta: list[dict], emb: np.ndarray, min_gap_s: float = 2.0, max_cos: float = 0.975, max_ph: int = 2) -> list[int]:
    """Near-duplicate removal for a FIXED camera (measured 2026-10-28: whole-frame pHash is ~2 apart at 0.5 s and ~4
    at 60 s, so a pHash threshold alone removes almost everything). A frame is dropped only when it is < min_gap_s
    after the last kept frame AND very similar to it (DINOv2 cosine >= max_cos and pHash <= max_ph)."""
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    kept: list[int] = []
    for i in range(len(meta)):
        if kept:
            j = kept[-1]
            close = meta[i]["t"] - meta[j]["t"] < min_gap_s
            same = float(e[i] @ e[j]) >= max_cos and (
                imagehash.hex_to_hash(meta[i]["phash"]) - imagehash.hex_to_hash(meta[j]["phash"]) <= max_ph
            )
            if close and same:
                continue
        kept.append(i)
    return kept


def pick_diverse(items: list[tuple[str, int]], embs: np.ndarray, n: int) -> list[tuple[str, int]]:
    """Round-robin over k-means clusters of the items' embeddings: n picks covering every cluster."""
    if len(items) <= n:
        return items
    k = max(2, min(n // 4, len(items) // 3, 64))
    lab = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(embs)
    buckets = defaultdict(list)
    for idx in RNG.permutation(len(items)):
        buckets[lab[idx]].append(items[idx])
    out = []
    while len(out) < n:
        for c in list(buckets):
            if buckets[c]:
                out.append(buckets[c].pop())
                if len(out) == n:
                    break
    return out


def dense_segments(meta: list[dict], counts: list[int] | None, n: int, fps: float = 25.0) -> list[dict]:
    """n non-overlapping 10-s segments at different crowd levels (by count quantiles when counts are known)."""
    total_t = meta[-1]["t"]
    starts = []
    if counts:
        c = np.array(counts)
        for q in np.linspace(0.2, 0.9, n):
            target = np.quantile(c, q)
            cand = [m["t"] for m, k in zip(meta, c) if k == round(target) and 5 < m["t"] < total_t - 15]
            cand = [t for t in cand if all(abs(t - s) > 20 for s in starts)]
            if cand:
                starts.append(float(RNG.choice(cand)))
    while len(starts) < n:
        t = float(RNG.uniform(5, total_t - 15))
        if all(abs(t - s) > 20 for s in starts):
            starts.append(t)
    return [{"start_s": round(s, 2), "end_s": round(s + 10, 2), "start_frame": int(s * fps),
             "end_frame": int((s + 10) * fps)} for s in sorted(starts)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=5000)
    a = ap.parse_args()
    splits = json.loads((TR / "data" / "splits.json").read_text())
    assert splits["status"].startswith("APPROVED"), "split not approved"
    tv = splits["train"] + splits["val"]
    counts = person_counts(tv)
    result = {"pool": [], "dense": {}, "test_frames": [], "test_dense": None, "params": vars(a)}
    stats = defaultdict(lambda: defaultdict(int))
    # ── train / val pool ──
    by_stratum = defaultdict(list)  # (split, stratum) -> [(clip, i)]
    embs_of = {}
    for split in ("train", "val"):
        for clip in splits[split]:
            meta, emb = load_clip(clip)
            embs_of[clip] = (meta, emb)
            keep = dedup(meta, emb)
            stats[split]["extracted"] += len(meta)
            stats[split]["after_dedup"] += len(keep)
            for i in keep:
                by_stratum[(split, stratum(counts[clip][i]))].append((clip, i))
            n_seg = 3 if split == "train" else 1
            result["dense"][clip] = dense_segments(meta, counts[clip], n_seg)
    n_val_target = int(a.pool * 0.15)
    targets = {"train": a.pool - n_val_target, "val": n_val_target}
    for split in ("train", "val"):
        groups = {s: by_stratum[(split, s)] for _, _, s in STRATA}
        avail = {s: len(v) for s, v in groups.items()}
        # ~10 % hard negatives from stratum "0"; the rest spread over 1-3 / 4-8 / 9+ (equal shares, capped by supply)
        want = {"0": int(targets[split] * 0.10)}
        rest = targets[split] - want["0"]
        share = {s: rest // 3 for s in ("1-3", "4-8", "9+")}
        for _ in range(3):  # move unfillable demand to strata that still have frames
            short = sum(max(0, share[s] - avail[s]) for s in share)
            share = {s: min(share[s], avail[s]) for s in share}
            open_s = [s for s in share if share[s] < avail[s]]
            for s in open_s:
                share[s] += short // max(1, len(open_s))
        want.update(share)
        for s, items in groups.items():
            n = min(want.get(s, 0), len(items))
            if n <= 0:
                continue
            e = np.stack([embs_of[c][1][i] for c, i in items])
            for c, i in pick_diverse(items, e, n):
                m = embs_of[c][0][i]
                result["pool"].append({"split": split, "clip": c, "frame": m["frame"], "t": m["t"], "file": m["file"],
                                       "stratum": s, "count_coco": counts[c][i]})
                stats[split][f"stratum_{s}"] += 1
    # ── test: hand labelling only, no detector ──
    items, embs = [], []
    for clip in splits["test"]:
        meta, emb = load_clip(clip)
        for i in dedup(meta, emb):
            items.append((clip, i))
            embs.append(emb[i])
        stats["test"]["extracted"] += len(meta)
    stats["test"]["after_dedup"] = len(items)
    for c, i in pick_diverse(items, np.stack(embs), 300):
        meta, _ = load_clip(c)
        result["test_frames"].append({"clip": c, "frame": meta[i]["frame"], "t": meta[i]["t"], "file": meta[i]["file"]})
    spread = {c: float(np.linalg.norm(load_clip(c)[1].std(0))) for c in splits["test"]}
    tclip = max(spread, key=spread.get)
    tmeta, _ = load_clip(tclip)
    seg = dense_segments(tmeta, None, 1)[0]
    result["test_dense"] = {"clip": tclip, **seg}
    stats["test"]["frames_selected"] = len(result["test_frames"])
    (TR / "data" / "sampling.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: dict(v) for k, v in stats.items()}, indent=1))
    print(json.dumps({"dense_train_val": sum(len(v) for v in result["dense"].values()), "test_dense": result["test_dense"]}))


if __name__ == "__main__":
    main()
