"""Build behaviour training windows (.npz) for training/lifeguard_training/behavior/cli.py.

Runs the SERVING detector + tracker (the upload worker's time base and analysis rate, the non-person filter) over:
  * eval/heldout_v1 clips, labelled by their aquaguard-eval/1 annotations (windows_from_detector);
  * eval/sprint_v1 clips, labelled by clips.json (windows_from_sprint: normal clips all NORMAL, drowning clips by
    their event window; drowning clips without a window are skipped).

Run with the SERVING venv (onnxruntime-gpu; no torch needed), from the repo root:

    $env:PYTHONPATH = "backend;training"
    venv\\Scripts\\python training\\tools\\build_behavior_windows.py --split dev \\
        --det-model backend\\models\\lifeguard_dfine_m_devtrain_v1_dynbatch.onnx --out training\\data\\behavior\\dev.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=("dev", "test", "all"))
    ap.add_argument("--heldout", type=Path, default=REPO / "eval" / "heldout_v1")
    ap.add_argument("--sprint", type=Path, default=REPO / "eval" / "sprint_v1" / "clips.json")
    ap.add_argument("--det-model", type=Path, default=None)
    ap.add_argument("--provider", default="auto")
    ap.add_argument("--stride", type=float, default=0.5)
    ap.add_argument("--only", default="", help="comma-separated clip ids")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    from lifeguard.eval.annotations import load_manifest
    from lifeguard.vision import config as vision_config
    from lifeguard.vision.inference.detector import MultiPersonDetector
    from lifeguard_training.behavior.dataset import Windows, provenance, windows_from_detector, windows_from_sprint

    cfg = vision_config.from_env()
    det_model = str(a.det_model.resolve()) if a.det_model else cfg.det_model_path
    detector = MultiPersonDetector("behaviour-windows", det_model, a.provider, pose_model_path=cfg.pose_model_path)
    if detector.load() != "rtm":
        sys.stderr.write(f"detector not available: {detector.load_error}\n")
        return 2
    only = {c for c in a.only.split(",") if c}
    want = (lambda s: True) if a.split == "all" else (lambda s: s == a.split)
    all_w, sources = Windows(), []
    for clip in load_manifest(a.heldout):
        if not want(clip.split) or (only and clip.clip_id not in only):
            continue
        t0 = time.perf_counter()
        w = windows_from_detector(clip, detector, a.stride)
        all_w.extend(w)
        sources.append(f"heldout_v1:{clip.clip_id}")
        n_ev = len({e for e in w.event if e})
        _say(f"heldout {clip.clip_id:28s} windows={len(w):5d} events={n_ev} {time.perf_counter() - t0:.0f}s")
    for entry in json.loads(a.sprint.read_text(encoding="utf-8"))["clips"]:
        if not want(entry["split"]) or (only and entry["id"] not in only):
            continue
        t0 = time.perf_counter()
        w = windows_from_sprint(entry, REPO, detector, a.stride)
        if len(w):
            all_w.extend(w)
            sources.append(f"sprint_v1:{entry['id']}")
        n_ev = sum(1 for e in w.event if e)
        _say(f"sprint  {entry['id']:28s} windows={len(w):5d} event windows={n_ev:4d} {time.perf_counter() - t0:.0f}s")
    prov = provenance(sources)
    prov["detector"] = Path(det_model).name
    path = all_w.save(a.out, prov)
    _say(f"wrote {path} ({len(all_w)} windows from {len(sources)} clips)")
    return 0


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
