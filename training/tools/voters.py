"""Phase 4 pre-run (owner-approved before GATE 2): raw detections of the three COCO voters on the train/val pool.
RAW ONLY: no label is accepted here; consensus, thresholds and review come in Phase 4 after GATE 2.

Per frame, person boxes in original pixels with scores down to 0.05:
  dfine  : D-FINE-L COCO, full frame at 1024 + 2x2 overlapping tiles at 640, merged by NMS (IoU 0.6)
  rtmdet : RTMDet-L COCO, full frame + the same tiles, merged by NMS
  owlv2  : OWLv2-base ensemble, full frame, prompts: "a person swimming", "a person's head in water", "a person"
Output: training\\labels\\raw_voters\\<set>.jsonl, one line per frame (appended; done frames are skipped on resume).

    training\\.venv\\Scripts\\python.exe training\\tools\\voters.py [--set pool] [--limit N] [--resume]
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
from torchvision.ops import nms

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "training"
FR = TR / "data" / "frames"
W = TR / "weights"
os.environ.setdefault("HF_HOME", str(W / "hf"))
MIN_SCORE = 0.05
PROMPTS = ["a person swimming", "a person's head in water", "a person"]


def tiles(h: int, w: int, overlap: float = 0.15) -> list[tuple[int, int, int, int]]:
    th, tw = int(h / (2 - overlap)), int(w / (2 - overlap))
    return [(x, y, x + tw, y + th) for y in (0, h - th) for x in (0, w - tw)]


def merge(boxes: list[np.ndarray]) -> list[list[float]]:
    b = np.concatenate([x for x in boxes if len(x)]) if any(len(x) for x in boxes) else np.zeros((0, 5))
    if not len(b):
        return []
    t = torch.from_numpy(b).float()
    keep = nms(t[:, :4], t[:, 4], 0.6).numpy()
    return [[round(float(v), 1) for v in r[:4]] + [round(float(r[4]), 4)] for r in b[keep]]


class DFine:
    def __init__(self) -> None:
        sys.path.insert(0, str(TR / "third_party" / "dfine"))
        from src.core import YAMLConfig

        ck = torch.load(W / "dfine_l_coco.pth", map_location="cpu")
        sd = ck["ema"]["module"] if "ema" in ck else ck["model"]
        self.models = {}
        for size in (640, 1024):
            cfg = YAMLConfig(str(TR / "third_party/dfine/configs/dfine/dfine_hgnetv2_l_coco.yml"),
                             eval_spatial_size=[size, size])
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
            s = sd if size == 640 else {k: v for k, v in sd.items() if not k.endswith(("anchors", "valid_mask"))}
            cfg.model.load_state_dict(s, strict=False)
            self.models[size] = (cfg.model.deploy().cuda().eval(), cfg.postprocessor.deploy())

    @torch.no_grad()
    def run(self, img: np.ndarray, size: int, ox: int = 0, oy: int = 0) -> np.ndarray:
        model, post = self.models[size]
        h, w = img.shape[:2]
        x = torch.from_numpy(cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        lab, box, sc = post(model(x.float().div(255)[None].cuda()), torch.tensor([[w, h]], device="cuda"))
        m = (lab[0] == 0) & (sc[0] >= MIN_SCORE)
        b = box[0][m].cpu().numpy()
        b[:, [0, 2]] += ox
        b[:, [1, 3]] += oy
        return np.concatenate([b, sc[0][m].cpu().numpy()[:, None]], 1)

    def __call__(self, img: np.ndarray) -> list[list[float]]:
        out = [self.run(img, 1024)]
        for x1, y1, x2, y2 in tiles(*img.shape[:2]):
            out.append(self.run(img[y1:y2, x1:x2], 640, x1, y1))
        return merge(out)


class RTMDet:
    def __init__(self) -> None:
        from mmdet.apis import init_detector

        cfg = TR / "third_party/mmdetection/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py"
        self.model = init_detector(str(cfg), str(W / "rtmdet_l_coco_statedict.pth"), device="cuda",
                                   cfg_options={"model.test_cfg.score_thr": MIN_SCORE})

    def run(self, img: np.ndarray, ox: int = 0, oy: int = 0) -> np.ndarray:
        from mmdet.apis import inference_detector

        p = inference_detector(self.model, img).pred_instances
        m = (p.labels == 0) & (p.scores >= MIN_SCORE)
        b = p.bboxes[m].cpu().numpy()
        b[:, [0, 2]] += ox
        b[:, [1, 3]] += oy
        return np.concatenate([b, p.scores[m].cpu().numpy()[:, None]], 1)

    def __call__(self, img: np.ndarray) -> list[list[float]]:
        out = [self.run(img)]
        for x1, y1, x2, y2 in tiles(*img.shape[:2]):
            out.append(self.run(img[y1:y2, x1:x2], x1, y1))
        return merge(out)


class OWLv2:
    def __init__(self) -> None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        n = "google/owlv2-base-patch16-ensemble"
        self.proc = Owlv2Processor.from_pretrained(n)
        self.model = Owlv2ForObjectDetection.from_pretrained(n).cuda().eval()

    @torch.no_grad()
    def __call__(self, img: np.ndarray) -> list[list[float]]:
        from PIL import Image

        h, w = img.shape[:2]
        inp = self.proc(text=[PROMPTS], images=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                        return_tensors="pt").to("cuda")
        out = self.model(**inp)
        side = max(h, w)  # OWLv2 pads to a square: boxes come back in the padded square's pixels
        r = self.proc.post_process_object_detection(out, threshold=MIN_SCORE,
                                                    target_sizes=torch.tensor([[side, side]], device="cuda"))[0]
        res = []
        for b, s, lab in zip(r["boxes"].tolist(), r["scores"].tolist(), r["labels"].tolist()):
            x1, y1, x2, y2 = max(0, b[0]), max(0, b[1]), min(w, b[2]), min(h, b[3])
            if x2 > x1 and y2 > y1:
                res.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1), round(s, 4), int(lab)])
        return res


class DenseReader:
    """Reads '<clip stem>@<frame>' keys from the original clip; sequential frames are decoded without seeking."""

    def __init__(self) -> None:
        self.clip, self.cap, self.pos = None, None, -1

    def read(self, key: str) -> np.ndarray:
        stem, idx = key.split("@")
        idx = int(idx)
        if stem != self.clip or idx != self.pos + 1:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(str(ROOT / "training_dataset" / f"{stem}.mp4"))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.clip = stem
        ok, img = self.cap.read()
        if not ok:
            raise RuntimeError(f"cannot read {key}")
        self.pos = idx
        return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="pool")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    samp = json.loads((TR / "data" / "sampling.json").read_text())
    assert a.set in ("pool", "dense"), "only the train/val pool or the train/val dense segments (never test frames)"
    if a.set == "pool":
        frames = [f"{p['clip'][:-4]}/{p['file']}" for p in samp["pool"]]
    else:  # every 25 fps frame of the train/val dense 10-s segments, keyed "<clip stem>@<frame index>"
        frames = [f"{clip[:-4]}@{i}" for clip, segs in samp["dense"].items() for s in segs
                  for i in range(s["start_frame"], s["end_frame"] + 1)]
    test_clips = {c[:-4] for c in json.loads((TR / "data" / "splits.json").read_text())["test"]}
    assert not any(f.split("/")[0].split("@")[0] in test_clips for f in frames), "test frames must never reach a voter"
    out = TR / "labels" / "raw_voters" / f"{a.set}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["key"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [f for f in frames if f not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(frames)} frames, {len(done)} done, {len(todo)} to run", flush=True)
    voters = {"dfine": DFine(), "rtmdet": RTMDet(), "owlv2": OWLv2()}
    reader = DenseReader()
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for k, key in enumerate(todo):
            img = reader.read(key) if "@" in key else cv2.imread(str(FR / key))
            rec = {"key": key, "h": img.shape[0], "w": img.shape[1]}
            for name, v in voters.items():
                rec[name] = v(img)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (k + 1) % 100 == 0 or k + 1 == len(todo):
                el = time.time() - t0
                rem = el / (k + 1) * (len(todo) - k - 1)
                print(f"{k + 1}/{len(todo)} ({100 * (k + 1) / len(todo):.0f} %) elapsed {el / 60:.1f} min, "
                      f"remaining ~{rem / 60:.0f} min, {el / (k + 1):.2f} s/frame", flush=True)


if __name__ == "__main__":
    main()
