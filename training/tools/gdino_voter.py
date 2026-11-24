"""Auto-label method (owner, 2026-10-28): Grounding DINO base as the 4th voter (Apache-2.0, labelling only, never shipped).

Text prompt "person. swimmer. head." (open vocabulary: swimmers and heads in the water that the COCO detectors miss),
full frame + 2x2 overlapping tiles (as D-FINE in voters.py), boxes >= MIN_SCORE merged by NMS. Output:
training\\labels\\raw_voters\\<set>_gdino.jsonl, one line per frame {"key", "gdino": [[x1,y1,x2,y2,score], ...]},
written incrementally and resumable. Never runs on test frames.

    training\\.venv\\Scripts\\python.exe training\\tools\\gdino_voter.py --set pool [--limit 50] [--resume]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voters import FR, TR, DenseReader, merge, tiles  # noqa: E402  (voters sets HF_HOME to training\weights\hf)

MODEL, REVISION = "IDEA-Research/grounding-dino-base", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
PROMPT = "person. swimmer. head."
MIN_SCORE = 0.15


class GDino:
    def __init__(self) -> None:
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        self.proc = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
        self.model = GroundingDinoForObjectDetection.from_pretrained(MODEL, revision=REVISION,
                                                                     use_safetensors=True).cuda().eval()

    @torch.no_grad()
    def run(self, img: np.ndarray, ox: int = 0, oy: int = 0) -> np.ndarray:
        from PIL import Image

        h, w = img.shape[:2]
        inp = self.proc(images=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), text=PROMPT,
                        return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out = self.model(**inp)
        r = self.proc.post_process_grounded_object_detection(out, inp["input_ids"], threshold=MIN_SCORE,
                                                            text_threshold=MIN_SCORE, target_sizes=[(h, w)])[0]
        b = r["boxes"].float().cpu().numpy().reshape(-1, 4)
        b[:, [0, 2]] += ox
        b[:, [1, 3]] += oy
        return np.concatenate([b, r["scores"].float().cpu().numpy()[:, None]], 1)

    def __call__(self, img: np.ndarray) -> list[list[float]]:
        out = [self.run(img)]
        for x1, y1, x2, y2 in tiles(*img.shape[:2]):
            out.append(self.run(img[y1:y2, x1:x2], x1, y1))
        return merge(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="pool", choices=["pool", "dense"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keys", help="file with one frame key per line (trial subsets)")
    a = ap.parse_args()
    src = TR / "labels" / "raw_voters" / f"{a.set}.jsonl"  # the same frames the three voters saw (train/val only)
    keys = [json.loads(line)["key"] for line in src.open(encoding="utf-8")]
    if a.keys:
        wanted = set(Path(a.keys).read_text().split())
        keys = [k for k in keys if k in wanted]
    test_clips = {c[:-4] for c in json.loads((TR / "data" / "splits.json").read_text())["test"]}
    assert not any(k.split("/")[0].split("@")[0] in test_clips for k in keys), "test frames must never reach a voter"
    out = TR / "labels" / "raw_voters" / f"{a.set}_gdino{'_trial' if a.keys else ''}.jsonl"
    done = {json.loads(line)["key"] for line in out.open(encoding="utf-8")} if out.exists() and a.resume else set()
    if out.exists() and not a.resume:
        out.unlink()
    todo = [k for k in keys if k not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(keys)} frames, {len(done)} done, {len(todo)} to run", flush=True)
    model, reader = GDino(), DenseReader()
    t0, nxt = time.time(), 0.0
    with out.open("a", encoding="utf-8") as fo:
        for i, k in enumerate(todo, 1):
            img = reader.read(k) if "@" in k else cv2.imread(str(FR / k))
            fo.write(json.dumps({"key": k, "gdino": model(img)}) + "\n")
            el = time.time() - t0
            if el >= nxt or i == len(todo):
                fo.flush()
                print(f"{i}/{len(todo)} ({i / len(todo):.0%}) elapsed {el / 60:.1f} min, remaining "
                      f"~{el / i * (len(todo) - i) / 60:.0f} min, {el / i:.2f} s/frame", flush=True)
                nxt = el + 600
    return 0


if __name__ == "__main__":
    sys.exit(main())
