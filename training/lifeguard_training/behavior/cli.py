"""
scripts/train_behavior_model.py — train the learned behaviour classifier.

    cd backend
    python scripts/train_behavior_model.py ../eval/datasets/behavior_v1/gt.npz ../eval/datasets/behavior_v1/det.npz \
        --out models/behavior/behavior_v1.onnx
    # then, to use it live (after reading the report it writes next to the model):
    set BEHAVIOR_MODEL_PATH=models/behavior/behavior_v1.onnx

Leave-one-video-out cross-validation → thresholds → final model → ONNX +
JSON sidecar + <name>.report.md. PyTorch (BSD-3) only; trained from scratch
on the given windows, no pretrained weights involved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # [v3] repository root
sys.path.insert(0, str(REPO / "training"))
BACKEND = REPO / "backend"


def _register(out: Path, prov: dict, provenance_note: str | None = None, dev_only: bool = False) -> None:
    """List a model written under backend/models in models/MODELS.json (the licence gate reads it)."""
    import hashlib

    models = BACKEND / "models"
    try:
        out.resolve().relative_to(models.resolve())
    except ValueError:
        return  # scratch / test output: not a shippable model
    sources = [s for v in prov.values() if isinstance(v, dict) for s in v.get("sources", [])]
    synthetic = bool(sources) and all(str(s).startswith("sim:") for s in sources)
    if provenance_note:
        data_provenance = provenance_note  # caller knows the sources better than we can guess
    elif synthetic:
        data_provenance = "synthetic scenes (detection/simulation.py) — pipeline smoke test, never ship"
    else:
        data_provenance = f"trained from scratch on client footage: {sorted(set(sources))}"
    entry = {
        "file": out.name,
        "role": "behaviour",
        "key": out.stem,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "bytes": out.stat().st_size,
        "licence": "project code; PyTorch (BSD-3) training",
        "commercial_ok": not synthetic,
        "data_provenance": data_provenance,
        "sidecar": out.with_suffix(".json").name,
        "dev_only": bool(dev_only),
    }
    manifest_path = REPO / "backend" / "models" / "MODELS.json"  # [v3] aquaguard-models/3 manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["models"] = [m for m in manifest.get("models", []) if m.get("file") != entry["file"]] + [entry]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"registered {out.name} in models/MODELS.json (commercial_ok={not synthetic})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("windows", type=Path, nargs="+", help=".npz files from build_behavior_dataset.py")
    ap.add_argument("--out", type=Path, default=BACKEND / "models" / "behavior" / "behavior_v1.onnx")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--target-precision", type=float, default=0.8)
    ap.add_argument("--stride", type=float, default=0.5, help="window stride the dataset was built with")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--provenance-note",
        default=None,
        help="override the auto-generated MODELS.json data_provenance string (e.g. for an "
        "unreviewed/bootstrap dataset that is NOT the client's contracted footage)",
    )
    ap.add_argument(
        "--dev-only",
        action="store_true",
        help="mark the MODELS.json entry dev_only:true (never set as the shipped default)",
    )
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console (cp1252)

    import numpy as np

    from lifeguard_training.behavior.dataset import load_windows
    from lifeguard_training.behavior.train import TrainConfig, run

    data = load_windows(a.windows)
    prov = {"window_files": [str(p) for p in a.windows]}
    for p in a.windows:
        try:
            prov[p.name] = json.loads(str(np.load(p)["provenance"]))
        except (KeyError, ValueError) as exc:  # [v3] was a blind `except Exception: pass`
            print(f"note: {p.name} has no readable provenance ({exc})")
    cfg = TrainConfig(
        epochs=a.epochs, target_precision=a.target_precision, stride_s=a.stride, seed=a.seed, device=a.device
    )
    report = run(data, a.out, cfg, prov)
    _register(Path(a.out), prov, a.provenance_note, a.dev_only)
    print(Path(a.out).with_suffix(".report.md").read_text(encoding="utf-8"))
    print(f"model: {report['model']}  (+ .json sidecar, .report.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
