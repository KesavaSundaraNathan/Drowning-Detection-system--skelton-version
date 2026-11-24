"""
Train + evaluate the learned behaviour classifier.

Protocol (built for a SMALL dataset — ~10 videos per drowning category):
  1. Leave-one-video-out cross-validation on the dev split: every window is
     predicted by a model that never saw its video → honest out-of-fold (OOF)
     probabilities. Splitting by window would leak (neighbouring windows of
     the same video are near-duplicates).
  2. Per-class thresholds are tuned on the OOF probabilities: the lowest
     threshold whose OOF precision ≥ `target_precision` (falls back to the
     best-F1 threshold). Reported with the OOF metrics.
  3. The final model is trained on ALL dev videos with the same settings and
     exported (ONNX + JSON sidecar, detection/behavior_model/infer.py format).
  4. Test-split videos (if any) are evaluated ONCE by the final model and
     never used for any choice above.

Metrics: per-class precision / recall / F1 on windows, confusion matrix,
EVENT recall (a labelled drowning episode counts as detected if any of its
windows is called with the right class) with time-to-detect, and NORMAL
false-positive windows per hour of normal in-water footage. With this little
data every number carries a wide confidence interval — the report prints a
Wilson 95 % interval for event recall and says how many events it rests on.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lifeguard.vision.behaviour.learned.features import CLASSES, SEQ_CHANNELS, SEQ_LEN, SUMMARY_FIELDS
from lifeguard.vision.behaviour.learned.infer import FORMAT

DROWNING = [c for c in CLASSES if c != "NORMAL"]


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 128
    lr: float = 3e-3
    weight_decay: float = 1e-3
    width: int = 32
    dropout: float = 0.3
    label_smoothing: float = 0.05
    noise: float = 0.05  # gaussian noise on normalised inputs (augmentation)
    seed: int = 0
    target_precision: float = 0.8
    stride_s: float = 0.5  # window stride used when the dataset was built
    device: str = "cpu"


# ─────────────────────────────────────────────────────────────────────────────
def _norm_stats(seq: np.ndarray, summ: np.ndarray) -> dict[str, list[float]]:
    return {
        "seq_mean": seq.mean(axis=(0, 2)).tolist(),
        "seq_std": (seq.std(axis=(0, 2)) + 1e-6).tolist(),
        "sum_mean": summ.mean(axis=0).tolist(),
        "sum_std": (summ.std(axis=0) + 1e-6).tolist(),
    }


def _apply_norm(seq, summ, n):
    s = (seq - np.asarray(n["seq_mean"], np.float32)[None, :, None]) / np.asarray(n["seq_std"], np.float32)[
        None, :, None
    ]
    m = (summ - np.asarray(n["sum_mean"], np.float32)[None]) / np.asarray(n["sum_std"], np.float32)[None]
    return s.astype(np.float32), m.astype(np.float32)


def fit(seq: np.ndarray, summ: np.ndarray, y: np.ndarray, cfg: TrainConfig):
    """Train one BehaviorNet. Returns (model, norm stats)."""
    import torch
    from torch import nn

    from lifeguard_training.behavior.model import BehaviorNet

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    norm = _norm_stats(seq, summ)
    xs, xm = _apply_norm(seq, summ, norm)
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / (counts > 0).sum(), 0.0)
    model = BehaviorNet(xs.shape[1], xm.shape[1], len(CLASSES), cfg.width, cfg.dropout).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(w, dtype=torch.float32, device=cfg.device), label_smoothing=cfg.label_smoothing
    )
    ts, tm, ty = (torch.tensor(a, device=cfg.device) for a in (xs, xm, y.astype(np.int64)))
    n = len(y)
    for _ in range(cfg.epochs):
        model.train()
        perm = torch.tensor(rng.permutation(n), device=cfg.device)
        for i in range(0, n, cfg.batch_size):
            b = perm[i : i + cfg.batch_size]
            bs, bm = ts[b], tm[b]
            if cfg.noise > 0:
                bs = bs + cfg.noise * torch.randn_like(bs)
                bm = bm + cfg.noise * torch.randn_like(bm)
            opt.zero_grad()
            loss = loss_fn(model.logits(bs, bm), ty[b])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model, norm


def predict(model, norm, seq: np.ndarray, summ: np.ndarray, device: str = "cpu") -> np.ndarray:
    import torch

    xs, xm = _apply_norm(seq, summ, norm)
    with torch.no_grad():
        return model(torch.tensor(xs, device=device), torch.tensor(xm, device=device)).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Decisions + metrics
# ─────────────────────────────────────────────────────────────────────────────
def decide(probs: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """Class per window: the drowning class with the highest prob that clears its threshold, else NORMAL."""
    out = np.zeros(len(probs), np.int64)
    best = np.zeros(len(probs))
    for c in DROWNING:
        i = CLASSES.index(c)
        hit = (probs[:, i] >= thresholds.get(c, 0.5)) & (probs[:, i] > best)
        out[hit] = i
        best[hit] = probs[hit, i]
    return out


def tune_thresholds(probs: np.ndarray, y: np.ndarray, target_precision: float) -> dict[str, float]:
    thr: dict[str, float] = {}
    for c in DROWNING:
        i = CLASSES.index(c)
        pos = y == i
        if pos.sum() == 0:
            thr[c] = 0.6
            continue
        best_f1, best_t, chosen = -1.0, 0.6, None
        for t in np.linspace(0.3, 0.95, 27):
            pred = probs[:, i] >= t
            tp = float((pred & pos).sum())
            prec = tp / max(pred.sum(), 1)
            rec = tp / max(pos.sum(), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
            if chosen is None and prec >= target_precision and rec > 0:
                chosen = float(t)
        thr[c] = round(chosen if chosen is not None else best_t, 3)
    return thr


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def alert_replay(
    pred: np.ndarray, track: np.ndarray, t: np.ndarray, stride_s: float, k_pos: int, k_neg: int
) -> list[int]:
    """
    Rows where an alert RISES when window decisions go through the live alert
    hysteresis (scene_tracker.AlertHysteresis): k_pos consecutive positive
    windows raise it, k_neg consecutive negative windows clear it. Windows of
    one track more than 1.5 strides apart break the run (track gap).
    """
    rises: list[int] = []
    order = np.lexsort((t, track))
    prev_track, prev_t, pos, neg, active = None, None, 0, 0, False
    for r in order:
        if track[r] != prev_track or (prev_t is not None and t[r] - prev_t > 1.5 * stride_s):
            pos, neg, active = 0, 0, False
        prev_track, prev_t = track[r], t[r]
        if pred[r] != 0:
            pos, neg = pos + 1, 0
            if not active and pos >= k_pos:
                active = True
                rises.append(int(r))
        else:
            neg, pos = neg + 1, 0
            if active and neg >= k_neg:
                active = False
    return rises


def metrics(
    y: np.ndarray, pred: np.ndarray, event: np.ndarray, t: np.ndarray, stride_s: float, track: np.ndarray | None = None
) -> dict:
    K = len(CLASSES)
    conf = np.zeros((K, K), np.int64)
    for a, b in zip(y, pred):
        conf[a, b] += 1
    per = {}
    for i, c in enumerate(CLASSES):
        tp = int(conf[i, i])
        p = tp / max(conf[:, i].sum(), 1)
        r = tp / max(conf[i].sum(), 1)
        per[c] = {
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(2 * p * r / max(p + r, 1e-9), 3),
            "support": int(conf[i].sum()),
        }
    # event recall: an episode is detected when any of its windows gets its class
    ev_rows: dict[str, list[int]] = {}
    for k, e in enumerate(event):
        if e:
            ev_rows.setdefault(str(e), []).append(k)
    events = {}
    for c in DROWNING:
        ci = CLASSES.index(c)
        eps = [rows for rows in ev_rows.values() if y[rows[0]] == ci]
        hit, ttd = 0, []
        for rows in eps:
            rows = sorted(rows, key=lambda r: t[r])
            first = next((r for r in rows if pred[r] == ci), None)
            if first is not None:
                hit += 1
                ttd.append(float(t[first] - t[rows[0]]))
        lo, hi = wilson(hit, len(eps))
        events[c] = {
            "episodes": len(eps),
            "detected": hit,
            "recall": round(hit / max(len(eps), 1), 3),
            "recall_ci95": [round(lo, 3), round(hi, 3)],
            "median_time_to_detect_s": round(float(np.median(ttd)), 2) if ttd else None,
        }
    normal = y == 0
    normal_hours = normal.sum() * stride_s / 3600.0
    fp = int((normal & (pred != 0)).sum())
    out = {
        "windows": len(y),
        "confusion": conf.tolist(),
        "per_class": per,
        "events": events,
        "normal_hours": round(normal_hours, 3),
        "false_positive_windows": fp,
        "false_positive_windows_per_hour": round(fp / normal_hours, 1) if normal_hours > 0 else None,
    }
    if track is not None:
        from lifeguard.vision.constants import BEHAVIOR_MODE_PARAMS

        mp = BEHAVIOR_MODE_PARAMS["Medium"]
        rises = alert_replay(pred, track, t, stride_s, int(mp["positive_windows"]), int(mp["negative_windows"]))
        false_alerts = sum(1 for r in rises if y[r] == 0)
        ev_alerted = {str(event[r]) for r in rises if event[r] and pred[r] == y[r]}
        n_eps = sum(v["episodes"] for v in events.values())
        out["alerts"] = {
            "mode": "Medium",
            "raised": len(rises),
            "false_alerts": false_alerts,
            "false_alerts_per_hour": round(false_alerts / normal_hours, 2) if normal_hours > 0 else None,
            "episodes_alerted": len(ev_alerted),
            "episodes": n_eps,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
def cross_validate(data: dict[str, np.ndarray], cfg: TrainConfig, log=print) -> tuple[np.ndarray, dict]:
    dev = data["split"] != "test"
    clips = sorted(set(data["clip"][dev].tolist()))
    if len(clips) < 2:
        raise ValueError(f"leave-one-video-out needs ≥ 2 dev videos, got {len(clips)}")
    idx_dev = np.where(dev)[0]
    oof = np.zeros((len(idx_dev), len(CLASSES)), np.float32)
    for n_, clip in enumerate(clips):
        hold = data["clip"][idx_dev] == clip
        tr = idx_dev[~hold]
        if len(np.unique(data["y"][tr])) < 2:
            continue
        model, norm = fit(data["seq"][tr], data["summary"][tr], data["y"][tr], cfg)
        oof[hold] = predict(model, norm, data["seq"][idx_dev[hold]], data["summary"][idx_dev[hold]], cfg.device)
        log(f"  fold {n_ + 1}/{len(clips)}  held out {clip}  ({int(hold.sum())} windows)")
    y = data["y"][idx_dev]
    thr = tune_thresholds(oof, y, cfg.target_precision)
    m = metrics(y, decide(oof, thr), data["event"][idx_dev], data["t"][idx_dev], cfg.stride_s, data["track"][idx_dev])
    m.update({"folds": len(clips), "thresholds": thr, "target_precision": cfg.target_precision})
    return oof, m


def export(model, norm: dict, thresholds: dict[str, float], out: Path, info: dict) -> Path:
    import torch

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    seq = torch.zeros(1, len(SEQ_CHANNELS), SEQ_LEN)
    summ = torch.zeros(1, 2 * len(SUMMARY_FIELDS))
    torch.onnx.export(
        model,
        (seq, summ),
        str(out),
        input_names=["seq", "summary"],
        output_names=["probs"],
        dynamic_axes={"seq": {0: "n"}, "summary": {0: "n"}, "probs": {0: "n"}},
        opset_version=17,
    )
    meta = {
        "format": FORMAT,
        "name": out.stem,
        "classes": list(CLASSES),
        "seq_channels": list(SEQ_CHANNELS),
        "seq_len": SEQ_LEN,
        "summary_fields": list(SUMMARY_FIELDS),
        "norm": norm,
        "thresholds": thresholds,
        **info,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def run(data: dict[str, np.ndarray], out: Path, cfg: TrainConfig, provenance: dict | None = None, log=print) -> dict:
    t0 = time.time()
    dev = data["split"] != "test"
    counts = {c: int((data["y"][dev] == i).sum()) for i, c in enumerate(CLASSES)}
    log(f"dev windows per class: {counts}")
    _oof, cv = cross_validate(data, cfg, log)
    log("training the final model on every dev video")
    model, norm = fit(data["seq"][dev], data["summary"][dev], data["y"][dev], cfg)
    report: dict = {"cv": cv, "dev_class_windows": counts, "config": cfg.__dict__}
    test = ~dev
    if test.any():
        probs = predict(model, norm, data["seq"][test], data["summary"][test], cfg.device)
        report["test"] = metrics(
            data["y"][test],
            decide(probs, cv["thresholds"]),
            data["event"][test],
            data["t"][test],
            cfg.stride_s,
            data["track"][test],
        )
    info = {
        "cv": cv,
        "test": report.get("test"),
        "trained_on": {"dev_videos": sorted(set(data["clip"][dev].tolist())), "windows": counts, **(provenance or {})},
        "training": {
            "framework": "PyTorch (BSD-3) — trained from scratch, no pretrained weights",
            "seconds": round(time.time() - t0, 1),
        },
    }
    export(model, norm, cv["thresholds"], out, info)
    report["model"] = str(out)
    Path(out).with_suffix(".report.md").write_text(render_report(report), encoding="utf-8")
    return report


def render_report(r: dict) -> str:
    cv = r["cv"]
    lines = [
        "# Behaviour model report",
        "",
        f"Leave-one-video-out on {cv['folds']} dev videos, {cv['windows']} windows. "
        f"Thresholds tuned on out-of-fold predictions for precision ≥ {cv['target_precision']}.",
        "",
        "| class | precision | recall | F1 | windows |",
        "|---|---|---|---|---|",
    ]
    for c, v in cv["per_class"].items():
        lines.append(f"| {c} | {v['precision']} | {v['recall']} | {v['f1']} | {v['support']} |")
    lines += [
        "",
        "| episode type | detected / episodes | recall (95 % CI) | median time-to-detect |",
        "|---|---|---|---|",
    ]
    for c, v in cv["events"].items():
        lines.append(
            f"| {c} | {v['detected']}/{v['episodes']} | {v['recall']} {v['recall_ci95']} | "
            f"{v['median_time_to_detect_s']} s |"
        )
    lines += [
        "",
        f"Normal in-water footage: {cv['normal_hours']} h, false-positive windows/h: "
        f"{cv['false_positive_windows_per_hour']}",
    ]
    if cv.get("alerts"):
        a = cv["alerts"]
        lines.append(
            f"Through the live alert hysteresis ({a['mode']}): {a['false_alerts']} false alerts "
            f"({a['false_alerts_per_hour']}/h), {a['episodes_alerted']}/{a['episodes']} episodes alerted "
            f"with the right class."
        )
    lines += ["", f"Thresholds: `{json.dumps(cv['thresholds'])}`", ""]
    if r.get("test"):
        t = r["test"]
        lines += ["## Locked test split (evaluated once)", ""]
        for c, v in t["events"].items():
            lines.append(f"- {c}: {v['detected']}/{v['episodes']} episodes, recall CI95 {v['recall_ci95']}")
        lines.append(f"- false-positive windows/h: {t['false_positive_windows_per_hour']}")
        if t.get("alerts"):
            lines.append(f"- false alerts/h (Medium hysteresis): {t['alerts']['false_alerts_per_hour']}")
    lines += [
        "",
        "With ~10 videos per category these intervals are wide: treat the model as a second "
        "opinion next to the rule engine, not as validated life-safety accuracy.",
    ]
    return "\n".join(lines) + "\n"
