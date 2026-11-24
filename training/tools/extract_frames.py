"""Phase 2 prework (split-independent): every clip at 2 fps → JPEG frames + pHash + DINOv2 embedding.
No labels are produced (DINOv2 is a feature extractor, not a labeller), so test clips may be processed too.
Resumable: a clip with a finished `<clip>/index.json` is skipped; `--resume` is accepted for the job queue.

Output: training\\data\\frames\\<clip>\\f<frame>.jpg, <clip>\\index.json (frame, t, phash),
        <clip>\\dinov2.npy (N x 768, float16, same order as index.json)

    training\\.venv\\Scripts\\python.exe training\\tools\\extract_frames.py [--resume]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import imagehash
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training" / "data" / "frames"
os.environ.setdefault("HF_HOME", str(ROOT / "training" / "weights" / "hf"))
FPS_OUT = 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.parse_args()
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval().half()
    audit = json.loads((ROOT / "training" / "data" / "audit.json").read_text())
    clips = [r["clip"].replace(" (1)", "") for r in audit if "sha256" in r]
    t_all = time.time()
    for k, clip in enumerate(sorted(clips)):
        d = OUT / clip[:-4]
        if (d / "index.json").exists():
            print(f"[{k + 1}/{len(clips)}] {clip}: done earlier, skipped", flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(ROOT / "training_dataset" / clip))
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = fps / FPS_OUT
        embs, batch, meta = [], [], []
        i, nxt, t0 = 0, 0.0, time.time()
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i >= nxt:
                nxt += step
                f = d / f"f{i:06d}.jpg"
                cv2.imwrite(str(f), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rgb = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                meta.append({"frame": i, "t": round(i / fps, 3), "file": f.name, "phash": str(imagehash.phash(rgb))})
                batch.append(rgb)
                if len(batch) == 32:
                    embs.append(embed(proc, model, batch))
                    batch = []
            i += 1
        if batch:
            embs.append(embed(proc, model, batch))
        cap.release()
        np.save(d / "dinov2.npy", np.concatenate(embs).astype(np.float16))
        (d / "index.json").write_text(json.dumps(meta))  # written last: marks the clip complete
        el = time.time() - t0
        left = (time.time() - t_all) / (k + 1) * (len(clips) - k - 1)
        print(f"[{k + 1}/{len(clips)}] {clip}: {len(meta)} frames in {el:.0f} s; remaining ~ {left / 60:.0f} min",
              flush=True)
    print("all clips done", flush=True)


@torch.no_grad()
def embed(proc, model, imgs) -> np.ndarray:
    x = proc(images=imgs, return_tensors="pt")["pixel_values"].cuda().half()
    return model(pixel_values=x).pooler_output.float().cpu().numpy()


if __name__ == "__main__":
    main()
