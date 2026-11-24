"""Export the pool-vs-other scene classifier (OpenAI CLIP ViT-B/32, zero-shot) to ONNX - v4.

    build\\pose_export_venv\\Scripts\\python.exe tools\\export_scene_clip.py --register

Runs in the one-off export venv (torch + the `clip` package; never in the server's venv). It:

1. loads ViT-B/32 with `clip.load(..., download_root=build/clip_weights)`; the clip package downloads
   ViT-B-32.pt from openaipublic.azureedge.net and refuses a file whose sha256 differs from the one in its URL;
2. embeds the text prompts of every scene class (CLASSES; mean of the normalised prompt embeddings) ONCE, here;
3. exports image -> class probabilities: (N, 3, 224, 224) RGB in [0, 1] -> softmax(100 * cos(image, class))
   (N, C). Normalisation and the text side are inside the graph, so the server needs neither torch nor a
   tokenizer. Class names + which of them are pools are written into the ONNX metadata ("classes", "pool_classes");
4. checks ONNX Runtime against torch on real-looking inputs (max |dp| < 1e-3);
5. with --register, writes sha256 + size into backend/models/MODELS.json (entry "clip_vit_b32_scene").

The prompts are part of the model: changing them is a new export (new sha256) and a new measurement
(vision/scene/scene_type.py records the one this file was accepted with).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "backend" / "models"
OUT_NAME = "clip_vit_b32_scene.onnx"
MANIFEST_KEY = "clip_vit_b32_scene"

# class -> prompts. "pool_*" classes are pools; every other class is "not a pool".
CLASSES: dict[str, list[str]] = {
    "pool_indoor": [
        "a photo of an indoor swimming pool",
        "a security camera view of an indoor swimming pool",
        "people swimming in an indoor pool",
    ],
    "pool_outdoor": [
        "a photo of an outdoor swimming pool",
        "a security camera view of an outdoor swimming pool",
        "people swimming in an outdoor pool",
    ],
    "pool_underwater": ["an underwater view of a swimming pool", "a person underwater in a swimming pool"],
    "office": [
        "a security camera view of an office",
        "an office room with desks and computers",
        "a workshop or laboratory with workbenches",
    ],
    "meeting_room": ["a security camera view of a meeting room", "a conference room with a table and chairs"],
    "corridor": ["a security camera view of a corridor", "a hallway or lobby inside a building"],
    "home_room": ["a security camera view of a room in a house", "a living room or bedroom"],
    "shop_warehouse": ["a security camera view of a shop", "a warehouse or storage room"],
    "outdoor_dry": [
        "a security camera view of a street",
        "a parking lot",
        "the outside of a building with a yard or driveway",
    ],
}
POOL_CLASSES = [c for c in CLASSES if c.startswith("pool_")]
MEAN = (0.48145466, 0.4578275, 0.40821073)  # CLIP's own preprocessing
STD = (0.26862954, 0.26130258, 0.27577711)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(weights_dir: Path) -> object:
    import clip
    import torch

    model, _ = clip.load("ViT-B/32", device="cpu", jit=False, download_root=str(weights_dir))
    model = model.float().eval()
    with torch.no_grad():
        feats = []
        for prompts in CLASSES.values():
            t = model.encode_text(clip.tokenize(prompts))
            t = t / t.norm(dim=-1, keepdim=True)
            m = t.mean(0)
            feats.append(m / m.norm())
        text = torch.stack(feats)  # (C, 512)

    class SceneHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = model.visual
            self.register_buffer("text", text)
            self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            f = self.visual((x - self.mean) / self.std)
            f = f / f.norm(dim=-1, keepdim=True)
            return torch.softmax(100.0 * f @ self.text.t(), dim=-1)

    return SceneHead().eval()


def export(head: object, out: Path) -> None:
    import onnx
    import torch

    x = torch.rand(2, 3, 224, 224)
    torch.onnx.export(
        head,
        x,
        str(out),
        input_names=["pixels"],
        output_names=["probs"],
        dynamic_axes={"pixels": {0: "n"}, "probs": {0: "n"}},
        opset_version=17,
    )
    m = onnx.load(str(out))
    for k, v in {
        "classes": json.dumps(list(CLASSES)),
        "pool_classes": json.dumps(POOL_CLASSES),
        "prompts": json.dumps(CLASSES),
        "input": "RGB float32 [0,1], 224x224 (normalisation inside the graph)",
    }.items():
        e = m.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(m, str(out))


def check(head: object, out: Path) -> float:
    import numpy as np
    import onnxruntime as ort
    import torch

    rng = np.random.default_rng(0)
    x = rng.random((3, 3, 224, 224), dtype=np.float32)
    with torch.no_grad():
        ref = head(torch.from_numpy(x)).numpy()  # type: ignore[operator]
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"pixels": x})[0]
    return float(np.abs(ref - got).max())


def register(path: Path) -> None:
    manifest_path = MODELS_DIR / "MODELS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for m in manifest["models"]:
        if m.get("key") == MANIFEST_KEY:
            m["sha256"], m["bytes"] = sha256(path), path.stat().st_size
            break
    else:
        raise SystemExit(f"{MANIFEST_KEY} is not listed in MODELS.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-dir", type=Path, default=ROOT / "build" / "clip_weights")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / OUT_NAME)
    ap.add_argument("--register", action="store_true", help="write sha256 + size into MODELS.json")
    args = ap.parse_args()
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    head = build(args.weights_dir)
    export(head, args.out)
    worst = check(head, args.out)
    if worst > 1e-3:
        raise SystemExit(f"ONNX Runtime differs from torch by {worst:.2e} (> 1e-3): export rejected")
    print(f"exported {args.out} ({args.out.stat().st_size / 1e6:.1f} MB), ORT vs torch max |dp| {worst:.2e}")
    if args.register:
        register(args.out)
        print(f"registered {MANIFEST_KEY} in backend/models/MODELS.json (sha256 {sha256(args.out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
