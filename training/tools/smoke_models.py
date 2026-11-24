"""Phase 0 smoke test: run each labelling / training model once on frames from one training clip and print NUMBERS only
(output shapes, person detections, ms per frame on the GPU). No image is saved or viewed.

    training\\.venv\\Scripts\\python.exe training\\tools\\smoke_models.py [--only dfine,sam2,owlv2,dinov2,rtmdet]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "training"
W = TR / "weights"
os.environ.setdefault("HF_HOME", str(W / "hf"))
CLIP = ROOT / "training_dataset" / "2026-10-22-18-36-19.mp4"
DEV = "cuda"


def frames(n: int = 4) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(CLIP))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (k + 1) / (n + 1)))
        ok, fr = cap.read()
        if ok:
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    return out


def timed(fn, reps: int = 5) -> tuple[object, float]:
    out = fn()  # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000 / reps


def smoke_dfine(imgs, size: str) -> dict:
    sys.path.insert(0, str(TR / "third_party" / "dfine"))
    from src.core import YAMLConfig  # noqa: E402

    ck = torch.load(W / f"dfine_{size}_coco.pth", map_location="cpu")
    res = {}
    for inp in (640, 1024):
        # the decoder's anchors are built for eval_spatial_size: one model per input size
        cfg = YAMLConfig(
            str(TR / "third_party" / "dfine" / "configs" / "dfine" / f"dfine_hgnetv2_{size}_coco.yml"),
            eval_spatial_size=[inp, inp],
        )
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        sd = dict(ck["ema"]["module"] if "ema" in ck else ck["model"])
        if inp != 640:  # anchors / valid_mask are buffers computed from the input size, not learned weights
            sd = {k: v for k, v in sd.items() if not k.endswith(("decoder.anchors", "decoder.valid_mask"))}
        missing, unexpected = cfg.model.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.endswith(("anchors", "valid_mask")) for k in missing), (missing, unexpected)
        model = cfg.model.deploy().to(DEV).eval()
        post = cfg.postprocessor.deploy()
        x = torch.stack([torch.from_numpy(cv2.resize(i, (inp, inp))).permute(2, 0, 1).float() / 255 for i in imgs[:1]]).to(DEV)
        sizes = torch.tensor([[imgs[0].shape[1], imgs[0].shape[0]]], device=DEV)
        with torch.no_grad():
            (labels, boxes, scores), ms = timed(lambda post=post, model=model, x=x, sizes=sizes: post(model(x), sizes))
        person = int(((labels[0] == 0) & (scores[0] > 0.4)).sum())
        res[f"input_{inp}"] = {"ms": round(ms, 1), "persons_over_0.4": person}
    res["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    return res


def smoke_sam2(imgs) -> dict:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml", str(W / "sam2.1_hiera_base_plus.pt"), device=DEV)
    pred = SAM2ImagePredictor(model)
    h, w = imgs[0].shape[:2]
    box = np.array([w * 0.4, h * 0.4, w * 0.6, h * 0.6])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, ms_embed = timed(lambda: pred.set_image(imgs[0]), reps=3)
        (masks, sc, _), ms_box = timed(lambda: pred.predict(box=box, multimask_output=False), reps=5)
    return {"set_image_ms": round(ms_embed, 1), "box_prompt_ms": round(ms_box, 1), "mask_shape": list(masks.shape)}


def smoke_owlv2(imgs) -> dict:
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    n = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(n)
    model = Owlv2ForObjectDetection.from_pretrained(n).to(DEV).eval()
    texts = [["a person swimming", "a person's head in water", "a person standing"]]
    inputs = proc(text=texts, images=Image.fromarray(imgs[0]), return_tensors="pt").to(DEV)
    with torch.no_grad():
        out, ms = timed(lambda: model(**inputs), reps=3)
    h, w = imgs[0].shape[:2]
    r = proc.post_process_object_detection(out, threshold=0.2, target_sizes=torch.tensor([[max(h, w)] * 2], device=DEV))
    return {"ms": round(ms, 1), "boxes_over_0.2": int(len(r[0]["scores"]))}


def smoke_dinov2(imgs) -> dict:
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    n = "facebook/dinov2-base"
    proc = AutoImageProcessor.from_pretrained(n)
    model = AutoModel.from_pretrained(n).to(DEV).eval()
    inputs = proc(images=[Image.fromarray(i) for i in imgs], return_tensors="pt").to(DEV)
    with torch.no_grad():
        out, ms = timed(lambda: model(**inputs), reps=3)
    emb = out.pooler_output
    return {"ms_per_frame": round(ms / len(imgs), 1), "embedding": list(emb.shape)}


def smoke_rtmdet(imgs) -> dict:
    from mmdet.apis import inference_detector, init_detector

    cfg = TR / "third_party" / "mmdetection" / "configs" / "rtmdet" / "rtmdet_l_8xb32-300e_coco.py"
    model = init_detector(str(cfg), str(W / "rtmdet_l_coco_statedict.pth"), device=DEV)
    bgr = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR)
    res, ms = timed(lambda: inference_detector(model, bgr), reps=3)
    p = res.pred_instances
    return {"ms": round(ms, 1), "persons_over_0.4": int(((p.labels == 0) & (p.scores > 0.4)).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="dfine,sam2,owlv2,dinov2,rtmdet")
    a = ap.parse_args()
    imgs = frames()
    print(json.dumps({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "frame": list(imgs[0].shape)}))
    tests = {
        "dfine_l": lambda: smoke_dfine(imgs, "l"),
        "dfine_m": lambda: smoke_dfine(imgs, "m"),
        "sam2": lambda: smoke_sam2(imgs),
        "owlv2": lambda: smoke_owlv2(imgs),
        "dinov2": lambda: smoke_dinov2(imgs),
        "rtmdet": lambda: smoke_rtmdet(imgs),
    }
    want = a.only.split(",")
    for name, fn in tests.items():
        if not any(name.startswith(w) for w in want):
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            r = fn()
            r["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
            print(json.dumps({name: r}), flush=True)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({name: f"FAILED: {type(e).__name__}: {e}"[:400]}), flush=True)


if __name__ == "__main__":
    main()
