"""Phase 1 data audit: per clip, numbers only (no frames are saved or viewed).

For every clip in training_dataset\\ (byte duplicates skipped): codec size / fps / duration, mean brightness (V of HSV),
foreground fraction (MOG2 on 1 fps, 240 px wide) and a crowd estimate = median / max person count from COCO D-FINE-M
(score > 0.4, 640 input) on one frame every 10 s. Writes training\\data\\audit.json.

    training\\.venv\\Scripts\\python.exe training\\tools\\audit_clips.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "training_dataset"
OUT = ROOT / "training" / "data" / "audit.json"
sys.path.insert(0, str(ROOT / "training" / "third_party" / "dfine"))


def dfine_m():
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(ROOT / "training/third_party/dfine/configs/dfine/dfine_hgnetv2_m_coco.yml"))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ck = torch.load(ROOT / "training/weights/dfine_m_coco.pth", map_location="cpu")
    cfg.model.load_state_dict(ck["ema"]["module"] if "ema" in ck else ck["model"])
    return cfg.model.deploy().cuda().eval(), cfg.postprocessor.deploy()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    model, post = dfine_m()
    seen: dict[str, str] = {}
    rows = []
    for p in sorted(DATA.glob("*.mp4")):
        h = sha256(p)
        if h in seen:
            rows.append({"clip": p.name, "duplicate_of": seen[h]})
            print(f"{p.name}: duplicate of {seen[h]}", flush=True)
            continue
        seen[h] = p.name
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, hgt = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
        mog = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=25, detectShadows=False)
        bright, fg, counts = [], [], []
        step = max(1, round(fps))
        for i in range(0, n, step):  # 1 fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                break
            small = cv2.resize(fr, (240, int(240 * hgt / w)))
            bright.append(float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 2].mean()))
            m = mog.apply(small)
            if len(bright) > 10:
                fg.append(float((m > 0).mean()))
            if (i // step) % 10 == 0:  # every 10 s
                rgb = cv2.cvtColor(cv2.resize(fr, (640, 640)), cv2.COLOR_BGR2RGB)
                x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda()
                with torch.no_grad():
                    lab, _, sc = post(model(x), torch.tensor([[w, hgt]], device="cuda"))
                counts.append(int(((lab[0] == 0) & (sc[0] > 0.4)).sum()))
        cap.release()
        r = {
            "clip": p.name, "sha256": h, "codec": fourcc.strip("\x00"), "width": w, "height": hgt,
            "fps": round(fps, 3), "frames": n, "seconds": round(n / fps, 1) if fps else None,
            "brightness_mean": round(float(np.mean(bright)), 1), "foreground_mean": round(float(np.mean(fg)), 4),
            "persons_median": float(np.median(counts)), "persons_max": int(max(counts)), "person_samples": len(counts),
        }
        rows.append(r)
        print(json.dumps(r), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
