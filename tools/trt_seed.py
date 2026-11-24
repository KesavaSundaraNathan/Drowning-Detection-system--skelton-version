"""Seed a scratch TensorRT engine cache from engines built at install (Phase 6: tests and e2e run strict TensorRT).

    from tools import trt_seed
    trt_seed.seed(data_dir / "trt_cache", [det_model, pose_model])   # copies the matching engines, or raises

Engines are built once per GPU / driver / TensorRT version by `tools/build_trt_engines.py` into
<data-dir>/trt_cache. A test stack copies the engines whose file names start with each model's
`session.engine_prefix` (model stem + sha256) plus the timing cache, so it starts in seconds on TensorRT instead of
compiling for minutes - and a CUDA fallback is never measured. The source is LIFEGUARD_TRT_SEED_DIR, else
<clone>\\runtime\\trt_cache (this clone's own data dir). A model without a matching engine is an error: the box has
not run build_trt_engines.py since the model or the GPU stack changed.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from lifeguard.vision.inference.session import engine_prefix  # noqa: E402

DEFAULT_SEED_DIR = Path(__file__).resolve().parents[1] / "runtime" / "trt_cache"  # [rtmpose clone] its own data dir


def seed_dir() -> Path:
    return Path(os.environ.get("LIFEGUARD_TRT_SEED_DIR") or DEFAULT_SEED_DIR)


def seed(dst: Path, models: list[Path], src: Path | None = None) -> list[str]:
    """Copy each model's engines (+ profiles) and the timing caches from `src` into `dst`; the copied names.
    Raises RuntimeError when a model has no engine in `src`."""
    src = src or seed_dir()
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []
    for model in models:
        prefix = engine_prefix(str(model))
        files = sorted(src.glob(f"{prefix}_*")) if prefix and src.is_dir() else []
        if not any(f.suffix == ".engine" for f in files):
            missing.append(f"{model.name} (prefix {prefix})")
            continue
        for f in files:
            shutil.copy2(f, dst / f.name)
            copied.append(f.name)
    if missing:
        raise RuntimeError(
            f"no TensorRT engine in {src} for {', '.join(missing)}; build them first: "
            f"python tools\\build_trt_engines.py --data-dir {src.parent} --cameras 4 --fps 18"
        )
    for f in sorted(src.glob("*.timing")):
        shutil.copy2(f, dst / f.name)
        copied.append(f.name)
    return copied


def seed_stack_env(env: dict[str, str], data_dir: Path) -> None:
    """For a stack env on LIFEGUARD_INFERENCE_PROVIDER=tensorrt: seed <data_dir>/trt_cache with the env's detector and
    pose engines and point LIFEGUARD_TRT_CACHE_DIR at it. Other providers are left alone."""
    if env.get("LIFEGUARD_INFERENCE_PROVIDER") != "tensorrt":
        return
    from lifeguard.vision import config as vision_config

    models = [Path(env.get("LIFEGUARD_DETECTOR_MODEL") or vision_config.MODELS_DIR / vision_config.DEFAULT_DET_MODEL)]
    pose = env.get("LIFEGUARD_POSE_MODEL", "")
    if pose.lower() not in ("none", "off"):
        models.append(Path(pose or vision_config.MODELS_DIR / vision_config.DEFAULT_POSE_MODEL))
    cache = data_dir / "trt_cache"
    seed(cache, models)
    env["LIFEGUARD_TRT_CACHE_DIR"] = str(cache)
