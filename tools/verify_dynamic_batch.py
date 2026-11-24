"""Check a dynamic-batch detector export with the SERVING onnxruntime (Phase 7, D-FINE).

    python tools/verify_dynamic_batch.py backend/models/<model>.onnx [--ref <model>.ref.npz] [--cuda] [--json out.json]

training/lifeguard_training/dfine/export.py runs in the training venv (torch, no onnxruntime) and saves
PyTorch outputs on real val frames next to the ONNX (<model>.ref.npz). This tool, in the serving venv,
proves on the runtime that will execute the model:

  * the graph's batch axis is symbolic (live/scheduler.py batches only such graphs);
  * CPU (deterministic reference path): a batch of N frames equals the N single-frame runs within --atol
    (batch independence), and every reference frame decodes to the same PEOPLE as PyTorch (export
    fidelity; the raw max difference is reported for information only);
  * every provider (CPU, and CUDA with --cuda; sessions from the production provider guard), single
    frames AND one batch, gives the same PEOPLE as CPU single frames on real --images, decoded exactly
    as the live path does (decode._person_view, class-agnostic NMS 0.7):
    each detection has a partner with IoU >= 0.9 and |score diff| <= 0.02, none left over. Raw tensors
    are not compared across providers: D-FINE's top-k query selection permutes near-tied queries on
    CUDA (Phase 7: raw diff 2.1, decoded detections identical), so only the detection SET is meaningful.

Exit 1 on any failure; the measured differences are printed (and written with --json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

CLASSES = ("person_in_water", "person_out_of_water")


def _dets(logits: np.ndarray, boxes: np.ndarray, thr: float) -> list[tuple[float, np.ndarray]]:
    """People of one frame exactly as the live decoder makes them (decode._person_view: one box per person,
    class-agnostic NMS at the DETR IoU 0.7): (score, cxcywh box) above thr."""
    from lifeguard.vision.inference.decode import _person_view, person_class_ids

    person, water, land = person_class_ids(CLASSES)
    cls = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1).astype(np.float32) * 1000.0
    b, sc, _w = _person_view(xyxy, cls, person, water, land, thr, 0.7)
    cxcywh = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
    return [(float(sc[k]), cxcywh[k]) for k in range(len(sc))]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


SCORE_TOL = 0.02


def same_detections(
    ref: list[tuple[float, np.ndarray]], got: list[tuple[float, np.ndarray]], thr: float
) -> dict[str, Any]:
    """Order-independent: every reference detection (score > thr) has a partner with IoU >= 0.9 and
    |score diff| <= SCORE_TOL among `got` (decoded down to thr - SCORE_TOL, so a detection sitting on the
    threshold is not counted as lost); a leftover counts only if clearly above it (>= thr + SCORE_TOL).
    DETR top-k query selection may permute near-tied queries between providers; the SET must agree."""
    free = list(range(len(got)))
    worst_iou, worst_score, unmatched = 1.0, 0.0, 0
    for sc, bx in sorted(ref, key=lambda d: -d[0]):
        cand = [(j, _iou(bx, got[j][1])) for j in free]
        j, iou = max(cand, key=lambda c: c[1], default=(None, 0.0))
        if j is None or iou < 0.9 or abs(got[j][0] - sc) > SCORE_TOL:
            unmatched += 1
            continue
        free.remove(j)
        worst_iou, worst_score = min(worst_iou, iou), max(worst_score, abs(got[j][0] - sc))
    return {
        "ref": len(ref),
        "got": len(got),
        "unmatched": unmatched + sum(got[j][0] >= thr + SCORE_TOL for j in free),
        "min_iou": worst_iou,
        "max_score_diff": worst_score,
    }


def check(
    onnx_path: Path, ref_path: Path, providers: list[str], atol: float, images: list[Path], thr: float = 0.3
) -> dict[str, Any]:
    from lifeguard.vision.inference.session import make_session  # the production provider guard

    ref = np.load(ref_path)
    xs = ref["images"]
    out: dict[str, Any] = {"model": onnx_path.name, "frames": int(xs.shape[0]), "atol": atol, "providers": {}}
    real = _load_images(images, xs.shape[-1]) if images else None
    real_ref: list[list[tuple[float, np.ndarray]]] | None = None
    for prov in providers:  # "cpu" first: it is the reference path
        sess, _ = make_session(str(onnx_path), prov)
        lead = sess.get_inputs()[0].shape[0]
        if isinstance(lead, int):
            raise SystemExit(f"{onnx_path.name}: batch axis is static ({lead}); expected a symbolic dimension")
        name = sess.get_inputs()[0].name
        singles = [sess.run(None, {name: xs[i : i + 1]}) for i in range(len(xs))]
        batched = sess.run(None, {name: xs})
        res: dict[str, Any] = {"batch_axis": str(lead)}
        if prov == "cpu":  # deterministic path: batch-independent bit for bit, same people as PyTorch
            res["max_abs_batch_vs_single"] = max(
                float(np.abs(batched[k][i] - singles[i][k][0]).max()) for i in range(len(xs)) for k in range(2)
            )
            # raw tensors are informational: near-tied queries may come out in another order
            res["max_abs_vs_torch_raw"] = max(
                float(np.abs(singles[i][k][0] - ref[key][i]).max())
                for i in range(len(xs))
                for k, key in enumerate(("logits", "boxes"))
            )
            low = thr - SCORE_TOL
            vs = [
                same_detections(
                    [d for d in _dets(ref["logits"][i], ref["boxes"][i], low) if d[0] > thr],
                    _dets(singles[i][0][0], singles[i][1][0], low),
                    thr,
                )
                for i in range(len(xs))
            ]
            res["people_vs_torch"] = {
                "people": sum(c["ref"] for c in vs),
                "unmatched": sum(c["unmatched"] for c in vs),
                "min_iou": min([c["min_iou"] for c in vs], default=1.0),
                "max_score_diff": max([c["max_score_diff"] for c in vs], default=0.0),
            }
            res["ok"] = res["max_abs_batch_vs_single"] <= atol and res["people_vs_torch"]["unmatched"] == 0
        else:
            res["ok"] = True
        if real is not None:  # decoded detections on real frames vs the CPU single-frame reference
            r_single = [sess.run(None, {name: real[i : i + 1]}) for i in range(len(real))]
            r_batch = sess.run(None, {name: real})
            low = thr - SCORE_TOL
            got_s = [_dets(o[0][0], o[1][0], low) for o in r_single]
            got_b = [_dets(r_batch[0][i], r_batch[1][i], low) for i in range(len(real))]
            if real_ref is None:
                real_ref = [[d for d in f if d[0] > thr] for f in got_s]
            cmp_s = [same_detections(a, b, thr) for a, b in zip(real_ref, got_s, strict=True)]
            cmp_b = [same_detections(a, b, thr) for a, b in zip(real_ref, got_b, strict=True)]
            res["real_frames"] = {
                "frames": len(real),
                "detections": sum(len(d) for d in real_ref),
                "single_unmatched": sum(c["unmatched"] for c in cmp_s),
                "batch_unmatched": sum(c["unmatched"] for c in cmp_b),
                "min_iou": min([c["min_iou"] for c in cmp_s + cmp_b], default=1.0),
                "max_score_diff": max([c["max_score_diff"] for c in cmp_s + cmp_b], default=0.0),
            }
            res["ok"] = res["ok"] and res["real_frames"]["single_unmatched"] == 0
            res["ok"] = res["ok"] and res["real_frames"]["batch_unmatched"] == 0
        out["providers"][prov] = res
    out["ok"] = all(p["ok"] for p in out["providers"].values())
    return out


def _load_images(paths: list[Path], size: int) -> np.ndarray:
    import cv2

    ims = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            raise SystemExit(f"cannot read {p}")
        ims.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (size, size)).astype(np.float32).transpose(2, 0, 1))
    return np.stack(ims) / np.float32(255.0)  # D-FINE preprocessing: RGB / 255, plain resize


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--ref", type=Path, default=None, help="default: <onnx>.ref.npz")
    ap.add_argument("--cuda", action="store_true", help="also check on CUDAExecutionProvider")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--images", nargs="*", type=Path, default=[], help="real frames for the decoded-detection check")
    a = ap.parse_args(argv)
    ref = a.ref or a.onnx.with_suffix(".ref.npz")
    providers = ["cpu"] + (["cuda"] if a.cuda else [])
    res = check(a.onnx, ref, providers, a.atol, a.images)
    text = json.dumps(res, indent=2)
    sys.stdout.write(text + "\n")
    if a.json is not None:
        a.json.write_text(text + "\n", encoding="utf-8", newline="\n")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
