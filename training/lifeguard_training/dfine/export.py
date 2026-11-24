"""
Trained D-FINE checkpoint → ONNX the live path reads (rtm_backend FORMAT_DETR).

The export keeps the RAW decoder outputs (`pred_logits` (1,Q,C), `pred_boxes`
(1,Q,4) cxcywh in [0,1]) instead of D-FINE's own top-k post-processor, so the
live decoder can turn the two class scores into one box per person plus an
in-water vote (rtm_backend.run_detector_people). Preprocessing is RGB / 255 at a
plain resize, exactly what D-FINE trains with.

[v3] Phase 7: the batch axis is DYNAMIC (images (N, 3, imgsz, imgsz)), so the live
scheduler can batch cameras (live/scheduler.py reads a symbolic leading dim as
"dynamic"). The export is checked before it is registered: ONNX Runtime must match
PyTorch at batch 1, and every frame of a batch must equal its single-frame result
(tools/verify_dynamic_batch.py, serving venv, against the saved PyTorch reference).
The checkpoint is read with torch.load(weights_only=True) (CVE-2025-32434).

This file runs INSIDE the D-FINE repo's environment (it imports `src.core`),
launched by scripts/train_detector_dfine.py:

    python -m training.dfine.export --repo <D-FINE> --config <yml> --checkpoint <best.pth> \
        --imgsz 960 --out models/lifeguard_dfine_m_v1.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]  # [v3] training/ (holds the lifeguard_training package)
REPO = Path(__file__).resolve().parents[3]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from lifeguard_training.dfine import CLASSES  # noqa: E402

META = {
    "lifeguard.det_format": "detr_logits",
    "lifeguard.task": "detector",
    "lifeguard.classes": json.dumps(list(CLASSES)),
}


def add_metadata(onnx_path: Path, extra: dict) -> None:
    import onnx

    m = onnx.load(str(onnx_path))
    for k, v in {**META, **extra}.items():
        e = m.metadata_props.add()
        e.key, e.value = k, str(v)
    onnx.save(m, str(onnx_path))


def register(onnx_path: Path, provenance: str, source: dict, dev_only: bool = False) -> dict:
    """Add the exported model to backend/models/MODELS.json (checked by tools/check_licenses.py)."""
    h = hashlib.sha256()
    with open(onnx_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    entry = {
        "file": onnx_path.name,
        "role": "detector",
        "key": onnx_path.stem,
        "kind": "det",
        "source": source,
        "sha256": h.hexdigest(),
        "bytes": onnx_path.stat().st_size,
        "licence": "Apache-2.0 (D-FINE code)",
        # commercial_ok is about the CODE/WEIGHT lineage (Apache-2.0 D-FINE from a
        # COCO-only checkpoint) — that is true regardless of the training data, which
        # is why this stays True even for a dev_only entry; dev_only is the separate
        # "don't ship this" signal for unreviewed/internal training data.
        "commercial_ok": True,
        "data_provenance": provenance,
        "classes": list(CLASSES),
        "dev_only": bool(dev_only),
    }
    manifest_path = REPO / "backend" / "models" / "MODELS.json"  # [v3] aquaguard-models/3 manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["models"] = [m for m in manifest.get("models", []) if m.get("file") != entry["file"]] + [entry]
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    manifest_path.write_text(text, encoding="utf-8", newline="\n")  # LF on Windows too
    return entry


def export(
    repo: Path, config: Path, checkpoint: Path, imgsz: int, out: Path, ref_images: list[Path] | None = None
) -> Path:
    import torch
    from torch import nn

    sys.path.insert(0, str(repo))
    from src.core import YAMLConfig  # D-FINE

    cfg = YAMLConfig(str(config), resume=str(checkpoint))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    state = state["ema"]["module"] if "ema" in state else state["model"]
    cfg.model.load_state_dict(state)

    class Raw(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model.deploy()

        def forward(self, images: torch.Tensor):
            o = self.model(images)
            return o["pred_logits"], o["pred_boxes"]

    net = Raw(cfg.model).eval()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(2, 3, imgsz, imgsz, generator=gen)
    with torch.no_grad():
        logits, boxes = net(x)
    if logits.shape[-1] != len(CLASSES):
        raise ValueError(f"checkpoint predicts {logits.shape[-1]} classes, expected {len(CLASSES)}")
    batch = {0: "batch"}
    torch.onnx.export(
        net,
        x,
        str(out),
        input_names=["images"],
        output_names=["logits", "boxes"],
        dynamic_axes={"images": batch, "logits": batch, "boxes": batch},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # TorchScript exporter: the dynamic_axes contract above, as for the RTMDet graph
    )
    add_metadata(out, {"lifeguard.input_size": imgsz, "lifeguard.dynamic_batch": "true"})
    save_reference(net, out, imgsz, ref_images)
    return out


def save_reference(net, onnx_path: Path, imgsz: int, images: list[Path] | None = None, n: int = 3) -> Path:
    """PyTorch outputs next to the ONNX (<name>.ref.npz): on real frames when `images` are given (RGB / 255
    at a plain resize, as D-FINE trains), else on n fixed random inputs. The serving venv checks the export
    against them with its own ONNX Runtime: tools/verify_dynamic_batch.py (no ORT here)."""
    import numpy as np
    import torch

    if images:
        from PIL import Image  # pillow is in training.lock (torchvision); OpenCV is not

        xs = np.stack(
            [
                np.asarray(Image.open(p).convert("RGB").resize((imgsz, imgsz), Image.BILINEAR), np.float32).transpose(
                    2, 0, 1
                )
                / np.float32(255.0)
                for p in images
            ]
        )
    else:
        xs = np.random.default_rng(0).random((n, 3, imgsz, imgsz), dtype=np.float32)
    with torch.no_grad():
        outs = [net(torch.from_numpy(xs[i : i + 1])) for i in range(len(xs))]
    ref = onnx_path.with_suffix(".ref.npz")
    np.savez_compressed(
        ref,
        images=xs,
        logits=np.concatenate([o[0].numpy() for o in outs]),
        boxes=np.concatenate([o[1].numpy() for o in outs]),
    )
    return ref


def _rel(p: Path) -> str:
    """Repo-relative path in MODELS.json (no machine-specific prefix); absolute only outside the repo."""
    p = Path(p).resolve()
    return p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else str(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--provenance", default="")
    ap.add_argument("--base-checkpoint", default="")
    ap.add_argument("--dev-only", action="store_true")
    ap.add_argument("--ref-images", nargs="*", type=Path, default=[], help="real frames for the PyTorch reference")
    a = ap.parse_args(argv)
    out = export(a.repo, a.config, a.checkpoint, a.imgsz, a.out, a.ref_images)
    entry = register(
        out,
        a.provenance or "D-FINE COCO-only checkpoint fine-tuned on client footage",
        {
            "framework": "D-FINE (Apache-2.0)",
            "base_checkpoint": a.base_checkpoint,
            "config": _rel(a.config),
            "trained_checkpoint": _rel(a.checkpoint),
        },
        a.dev_only,
    )
    print(json.dumps({"onnx": str(out), **entry}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
