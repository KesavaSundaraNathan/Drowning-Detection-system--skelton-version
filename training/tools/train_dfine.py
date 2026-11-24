"""Phase 6: fine-tune D-FINE from its COCO checkpoint through the queue (resumable).

Wraps third_party\\dfine\\train.py: registers the pool photometric augmentation (`PoolPhotometric`, pixel-level only, so
boxes are unchanged), seeds everything, and on `--resume` continues from <output_dir>\\last.pth instead of re-tuning.

    training\\.venv\\Scripts\\python.exe training\\tools\\train_dfine.py -c training\\configs\\dfine_l_v0.yml \\
        -t training\\weights\\dfine_l_coco.pth [--resume] [-u epochs=1 ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DFINE = ROOT / "training" / "third_party" / "dfine"
sys.path.insert(0, str(DFINE))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # dataloader workers (spawn) import this module by name

import albumentations as A  # noqa: E402
import numpy as np  # noqa: E402
import PIL.Image  # noqa: E402
import torchvision.transforms.v2 as T  # noqa: E402

from src.core import YAMLConfig, register, yaml_utils  # noqa: E402
from src.misc import dist_utils  # noqa: E402
from src.solver import TASKS  # noqa: E402


@register()
class PoolPhotometric(T.Transform):
    """HSV / brightness shifts, glare, motion blur, sensor noise and JPEG artefacts (MASTER_PROMPT Phase 6)."""

    _transformed_types = (PIL.Image.Image,)

    def __init__(self, p: float = 0.8) -> None:
        super().__init__()
        self.p = p
        self.aug = A.Compose([
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=25, val_shift_limit=20, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 1), src_radius=120, num_flare_circles_range=(2, 5), p=0.15),
            A.MotionBlur(blur_limit=(3, 9), p=0.25),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.25),
            A.ImageCompression(quality_range=(35, 90), p=0.4),
        ])

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if np.random.rand() >= self.p:
            return inpt
        return PIL.Image.fromarray(self.aug(image=np.asarray(inpt))["image"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-t", "--tuning", required=True, help="COCO checkpoint to fine-tune from")
    ap.add_argument("--resume", action="store_true", help="continue from <output_dir>/last.pth if it exists")
    ap.add_argument("-u", "--update", nargs="+", help="config overrides, e.g. epochs=1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-only", action="store_true")
    a = ap.parse_args()

    dist_utils.setup_distributed(0, "builtin", seed=a.seed)
    upd = yaml_utils.parse_cli(a.update) if a.update else {}
    cfg0 = YAMLConfig(a.config, **upd)
    last = Path(cfg0.output_dir) / "last.pth"
    kw: dict[str, Any] = {**upd, "seed": a.seed}
    if a.resume and last.exists():
        kw["resume"] = str(last)
        print(f"resuming from {last}")
    else:
        kw["tuning"] = a.tuning
    cfg = YAMLConfig(a.config, **kw)
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False  # weights come from the COCO checkpoint or last.pth
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.val() if a.test_only else solver.fit()
    dist_utils.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
