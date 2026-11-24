"""Build (and cache) the TensorRT FP16 engines for RTMDet + RTMPose on THIS GPU, then measure them.

    python tools/build_trt_engines.py --data-dir C:\\ProgramData\\Life Guard --cameras 4 --fps 15 --people 8

Engines are GPU-, driver- and TensorRT-version-specific, so they are built on the target machine (the RTX 4060
laptop), never shipped. This runs ONNX Runtime's TensorRT execution provider exactly as the server does
(inference/session.py: FP16, engine + timing cache in <data-dir>/trt_cache, dynamic-batch profiles), so the
server's first start finds every engine already built instead of spending minutes compiling.

It then times each batch size and checks the 4-camera budget: per round, one detector call of `cameras` frames
plus one RTMPose call of `cameras x people` crops (pose on every person - the stride-2 saving is NOT counted,
admission must hold when every track is at risk). VRAM in use is read from nvidia-smi. Exit 1 if the budget
does not fit or a session does not land on TensorRT. Report: <data-dir>/trt_report.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404 - fixed argv nvidia-smi query
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from lifeguard.vision import config as vision_config  # noqa: E402
from lifeguard.vision.inference import models, pose, session  # noqa: E402


def vram_mib() -> int | None:
    try:
        out = subprocess.run(  # nosec B603 B607 - fixed argv
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return int(out.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def time_call(fn, repeats: int) -> float:
    fn()
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(t))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--det-model", type=Path, default=vision_config.MODELS_DIR / vision_config.DEFAULT_DET_MODEL)
    ap.add_argument("--pose-model", type=Path, default=vision_config.MODELS_DIR / vision_config.DEFAULT_POSE_MODEL)
    ap.add_argument("--cameras", type=int, default=4)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--people", type=int, default=8, help="people per camera frame for the budget")
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args(argv)

    session.set_trt_cache_dir(args.data_dir / "trt_cache")
    vram0 = vram_mib()
    t0 = time.perf_counter()
    bm = max(8, args.cameras)
    det = models.load_detector_model(str(args.det_model), "tensorrt", 640, trt_batch=(1, min(4, bm), bm))
    pm = models.load_pose_model(str(args.pose_model), "tensorrt")
    for m in (det, pm):
        if models.active_provider(m) != "TensorrtExecutionProvider":
            print(f"FAIL: {Path(m.onnx_model).name} runs on {models.active_provider(m)}, not TensorRT")
            return 1
    report: dict = {"trt_cache": str(session.trt_cache_dir()), "fp16": True, "detector": {}, "pose": {}}
    in_h, in_w = det.model_input_size
    det_ms: dict[int, float] = {}
    for b in sorted({1, min(4, bm), args.cameras, bm}):
        x = np.random.default_rng(b).standard_normal((b, 3, in_h, in_w)).astype(np.float32)
        det_ms[b] = time_call(lambda x=x: det.session.run(det.output_names, {det.input_name: x}), args.repeats)
        report["detector"][str(b)] = round(det_ms[b], 2)
    pw, ph = pm.model_input_size
    pose_ms: dict[int, float] = {}
    for b in pm.buckets:
        x = np.random.default_rng(b).standard_normal((b, 3, ph, pw)).astype(np.float32)
        pose_ms[b] = time_call(lambda x=x: pm.session.run(pm.output_names, {pm.input_name: x}), args.repeats)
        report["pose"][str(b)] = round(pose_ms[b], 2)
    report["build_and_measure_s"] = round(time.perf_counter() - t0, 1)

    crops = args.cameras * args.people
    pose_round = 0.0
    left = crops
    while left > 0:
        n = min(left, pm.max_batch)
        pose_round += pose_ms[pm.bucket_for(n)]
        left -= n
    round_ms = det_ms[args.cameras] + pose_round
    budget_ms = 1000.0 / args.fps
    vram1 = vram_mib()
    report["budget"] = {
        "cameras": args.cameras,
        "fps": args.fps,
        "people_per_camera": args.people,
        "detector_round_ms": round(det_ms[args.cameras], 2),
        "pose_round_ms": round(pose_round, 2),
        "gpu_round_ms": round(round_ms, 2),
        "budget_ms": round(budget_ms, 2),
        "fits": round_ms <= 0.8 * budget_ms,  # 20 % headroom for decode, tracking and behaviour
        "vram_mib_engines": (vram1 - vram0) if (vram0 is not None and vram1 is not None) else None,
    }
    args.data_dir.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "trt_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    # a quick end-to-end decode on the built engine, so a broken export cannot pass
    kp, _ = pose.run_pose(pm, np.full((720, 1280, 3), 120, np.uint8), np.array([[600, 200, 680, 420]], np.float32))
    if kp.shape != (1, 17, 2):
        print(f"FAIL: RTMPose decode returned {kp.shape}")
        return 1
    return 0 if report["budget"]["fits"] else 1


if __name__ == "__main__":
    sys.exit(main())
