"""
scripts/train_detector_dfine.py — fine-tune the zone-aware person detector (D-FINE).

One-time setup (training machine; see documentation/ML_CORE_V2.md):
    git clone https://github.com/Peterande/D-FINE ../third_party/D-FINE      # Apache-2.0
    pip install -r requirements-training.txt
    # COCO-ONLY checkpoint from the D-FINE README "COCO" table, e.g. dfine_m_coco.pth
    # (never *_obj365* / *_obj2coco* — Objects365 is non-commercial; this script refuses them)

Then:
    python scripts/export_training_set.py --clips <clips.json> --annotations <reviewed> \
        --out datasets/client_v1 --classes zone
    python scripts/train_detector_dfine.py --dfine-repo ../third_party/D-FINE \
        --checkpoint ../third_party/weights/dfine_m_coco.pth --dataset datasets/client_v1 \
        --size m --imgsz 960 --epochs 80 --out models/lifeguard_dfine_m_v1.onnx
    python scripts/evaluate_pipeline.py --name dfine_v1 --split test \
        --set RTM_DET_MODEL_PATH=models/lifeguard_dfine_m_v1.onnx --compare <baseline report.json>

Steps: prepare (0-based categories) → YAML → `train.py -t <checkpoint>` → raw-logit
ONNX + metadata → models/MODELS.json (commercial_ok, provenance). `--dry-run`
stops after writing the YAML and prints the training command.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # [v3] repository root
sys.path.insert(0, str(REPO / "training"))
BACKEND = REPO / "backend"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dfine-repo", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True, help="COCO-only D-FINE checkpoint (.pth)")
    ap.add_argument("--dataset", type=Path, required=True, help="export_training_set.py --classes zone output")
    ap.add_argument("--size", default="m", choices=("n", "s", "m", "l", "x"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--python", default=sys.executable, help="interpreter with D-FINE's requirements")
    ap.add_argument(
        "--provenance",
        default=None,
        help="override the auto-generated MODELS.json data_provenance string — use this for "
        "any dataset that is NOT the client's reviewed, contracted footage (e.g. an "
        "internal/unreviewed bootstrap run); default wording claims '(client footage)'",
    )
    ap.add_argument(
        "--dev-only",
        action="store_true",
        help="mark the MODELS.json entry dev_only:true (never set as the shipped default)",
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from lifeguard_training.dfine import FORBIDDEN_CHECKPOINT_MARKERS
    from lifeguard_training.dfine.config import write_config
    from lifeguard_training.dfine.prepare import prepare

    ck = a.checkpoint.name.lower()
    if any(m in ck for m in FORBIDDEN_CHECKPOINT_MARKERS):
        raise SystemExit(
            f"refusing {a.checkpoint.name}: Objects365-pretrained weights are non-commercial. "
            "Use the COCO-only checkpoint (dfine_<size>_coco.pth)."
        )
    work = a.work_dir or (BACKEND / "models" / "dfine_runs" / f"{a.dataset.name}_{a.size}_{a.imgsz}")
    out = a.out or (BACKEND / "models" / f"lifeguard_dfine_{a.size}_{a.dataset.name}.onnx")
    ann = work / "annotations"
    stats = prepare(a.dataset, ann)
    print(f"dataset: {stats}")
    cfg = write_config(
        a.dfine_repo,
        a.size,
        a.dataset / "images",
        ann,
        work,
        a.imgsz,
        a.epochs,
        a.batch,
        a.workers,
        check_repo=not a.dry_run,
    )
    train_cmd = [
        a.python,
        "train.py",
        "-c",
        str(cfg.resolve()),
        "--use-amp",
        "--seed=0",
        "-t",
        str(a.checkpoint.resolve()),
    ]
    print("config:", cfg)
    print("train :", " ".join(train_cmd), f"  (cwd={a.dfine_repo})")
    if a.dry_run:
        return 0
    subprocess.run(train_cmd, cwd=str(a.dfine_repo), check=True)
    best = work / "run" / "best_stg2.pth"
    if not best.exists():
        best = next((p for p in (work / "run" / "best_stg1.pth", work / "run" / "last.pth") if p.exists()), None)
    if best is None:
        raise SystemExit(f"no checkpoint found in {work / 'run'}")
    provenance = a.provenance or (
        f"D-FINE {a.checkpoint.name} (COCO-only, Apache-2.0) fine-tuned on {a.dataset.name} (client footage)"
    )
    subprocess.run(
        [
            a.python,
            "-m",
            "lifeguard_training.dfine.export",
            "--repo",
            str(a.dfine_repo.resolve()),
            "--config",
            str(cfg.resolve()),
            "--checkpoint",
            str(best),
            "--imgsz",
            str(a.imgsz),
            "--out",
            str(out.resolve()),
            "--base-checkpoint",
            a.checkpoint.name,
            "--provenance",
            provenance,
        ]
        + (["--dev-only"] if a.dev_only else [])
        # PyTorch reference for tools/verify_dynamic_batch.py on real val frames, not noise: DETR's top-k
        # query order is unstable on noise, so decoded people on real frames are the fair comparison
        + ["--ref-images", *[str(p) for p in sorted((a.dataset / "images" / "val").glob("*.jpg"))[::10][:8]]],
        cwd=str(REPO / "training"),
        check=True,
    )
    print(f"done: {out} — evaluate it with `python -m lifeguard.eval.evaluate` before making it the default")
    return 0


if __name__ == "__main__":
    sys.exit(main())
