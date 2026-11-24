"""Phase 6/10: export a trained v0 detector to the ONNX layouts Life Guard's live path reads (training venv).

  dfine  -> FORMAT_DETR ("detr_logits"): raw `logits (N,Q,2)` + `boxes (N,Q,4)` cxcywh in [0,1]; input RGB / 255,
            plain resize (same contract as training\\lifeguard_training\\dfine\\export.py).
  rtmdet -> FORMAT_XYXY_SCORES ("xyxy_scores"): `(N,A,4+2)` xyxy in letterboxed input pixels + sigmoid class scores;
            input BGR, ImageNet mean/std, top-left letterbox padded with 114 (decode_detections layout (b)).
Batch axis dynamic, spatial size fixed (1024). Also writes <out>.ref.npz (a fixed input and the PyTorch outputs) for
training\\tools\\verify_v0_onnx.py, which adds the metadata and checks the file in the serving venv (onnx/ORT live
there; nothing new is installed in the training venv).

    training\\.venv\\Scripts\\python.exe training\\tools\\export_v0.py dfine  -c training\\configs\\dfine_l_v0.yml \\
        --checkpoint training\\work_dirs\\dfine_l_v0\\best_stg2.pth --out training\\work_dirs\\onnx\\dfine_l_v0.onnx
    training\\.venv\\Scripts\\python.exe training\\tools\\export_v0.py rtmdet -c training\\configs\\rtmdet_l_v0.py \\
        --checkpoint training\\work_dirs\\rtmdet_l_v0\\best_coco_bbox_mAP_epoch_N.pth \\
        --out training\\work_dirs\\onnx\\rtmdet_l_v0.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SIZE = 1024


def build_dfine(config: str, ckpt: str) -> nn.Module:
    sys.path.insert(0, str(ROOT / "training" / "third_party" / "dfine"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_dfine  # noqa: F401  (registers PoolPhotometric, referenced by the config)
    from src.core import YAMLConfig

    cfg = YAMLConfig(config)
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg.model.load_state_dict(state["ema"]["module"] if "ema" in state else state["model"])

    class Raw(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model.deploy()

        def forward(self, images: torch.Tensor):
            o = self.model(images)
            return o["pred_logits"], o["pred_boxes"]

    return Raw(cfg.model).eval()


def build_rtmdet(config: str, ckpt: str) -> nn.Module:
    from mmdet.apis import init_detector
    from mmdet.utils import register_all_modules

    register_all_modules()
    det = init_detector(config, ckpt, device="cpu")  # the checkpoint's state_dict holds the EMA weights
    head = det.bbox_head
    sizes = [(SIZE // sh, SIZE // sw) for sw, sh in head.prior_generator.strides]
    priors = torch.cat(head.prior_generator.grid_priors(sizes, device="cpu"), 0)[:, :2]  # (A, 2) x, y

    class Decoded(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.det = det
            self.register_buffer("priors", priors)

        def forward(self, images: torch.Tensor):
            cls, reg = self.det.bbox_head(self.det.extract_feat(images))
            b = images.shape[0]
            cls = torch.cat([c.permute(0, 2, 3, 1).reshape(b, -1, c.shape[1]) for c in cls], 1).sigmoid()
            dist = torch.cat([r.permute(0, 2, 3, 1).reshape(b, -1, 4) for r in reg], 1)  # l, t, r, b in pixels
            p = self.priors[None]
            xyxy = torch.cat([p - dist[..., :2], p + dist[..., 2:]], -1)
            return torch.cat([xyxy, cls], -1)

    return Decoded().eval()


FRAMES_DIR = ROOT / "training/data/frames"
REF_COUNT = 2  # ref_frames() takes sorted frames [100:102]
DFINE_CLASSES = 2  # swimmer, person


def ref_frames() -> list[Path]:
    """Two real training frames as the reference input (random noise makes DETR's top-k query choice unstable)."""
    frames = sorted(FRAMES_DIR.glob("*/f00*.jpg"))[100 : 100 + REF_COUNT]
    if len(frames) != REF_COUNT:
        raise SystemExit(
            f"need {REF_COUNT} reference frames (sorted {FRAMES_DIR}/*/f00*.jpg [100:102]), got {len(frames)}: "
            "the folder has fewer than 102 matching frames"
        )
    return frames


def preprocess(path: Path, kind: str) -> np.ndarray:
    """The same preprocessing as Life Guard's DetModel.preprocess for each format."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cv2.imread could not read reference frame {path}")
    if kind == "dfine":  # plain resize, RGB / 255
        return cv2.resize(img, (SIZE, SIZE))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    r = min(SIZE / img.shape[0], SIZE / img.shape[1])  # top-left letterbox, pad 114, BGR mean / std
    h, w = int(img.shape[0] * r), int(img.shape[1] * r)
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    canvas[:h, :w] = cv2.resize(img, (w, h))
    normalized = (canvas - np.array([103.53, 116.28, 123.675])) / np.array([57.375, 57.12, 58.395])
    return normalized.transpose(2, 0, 1).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["dfine", "rtmdet"])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    net = build_dfine(a.config, a.checkpoint) if a.kind == "dfine" else build_rtmdet(a.config, a.checkpoint)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    x = torch.from_numpy(np.stack([preprocess(f, a.kind) for f in ref_frames()]))
    with torch.no_grad():
        ref = net(x)
    ref = ref if isinstance(ref, tuple) else (ref,)
    if a.kind == "dfine" and ref[0].shape[-1] != DFINE_CLASSES:
        raise SystemExit(
            f"D-FINE logits have {ref[0].shape[-1]} classes (shape {tuple(ref[0].shape)}), expected {DFINE_CLASSES} "
            "(swimmer, person): check num_classes in the config and the checkpoint"
        )
    names = ["logits", "boxes"] if a.kind == "dfine" else ["dets"]
    torch.onnx.export(
        net,
        (x,),
        str(out),
        input_names=["images"],
        output_names=names,
        opset_version=17,
        dynamic_axes={"images": {0: "N"}, **{n: {0: "N"} for n in names}},
        dynamo=False,
    )
    np.savez(str(out) + ".ref.npz", x=x.numpy(), **{n: r.numpy() for n, r in zip(names, ref, strict=True)})
    shapes = ", ".join(f"{n} {tuple(r.shape)}" for n, r in zip(names, ref, strict=True))
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB); outputs {shapes}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
