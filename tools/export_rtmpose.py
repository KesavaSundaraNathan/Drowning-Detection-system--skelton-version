"""Export the shipped pose model, RTMPose-m COCO 256x192, from its MMPose checkpoint to ONNX - offline, once.

Runs in a throwaway venv built from requirements/pose_export.in (torch + mmpose + mmdeploy), NEVER on a
serving box. The server loads the resulting .onnx with ONNX Runtime; TensorRT FP16 engines are then built
from it on the target GPU by tools/build_trt_engines.py.

    python tools/export_rtmpose.py \\
        --checkpoint rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth \\
        --images build/v2_baseline/Datasets --register

Steps:
1. refuses any body7 checkpoint (owner rule: restricted non-commercial training data) and checks the file's
   sha256 starts with the 8-hex tag in its OpenMMLab file name (or equals --checkpoint-sha256);
2. exports with MMDeploy's ONNX Runtime SimCC path (pose-detection_simcc_onnxruntime_dynamic): input
   (B,3,256,192), outputs simcc_x (B,17,384) and simcc_y (B,17,512), dynamic batch, no post-processing in
   the graph (the server decodes SimCC itself, inference/pose.py);
3. validates against PyTorch (mmpose inference_topdown) on real frames with people (--images): RTMDet-free -
   the person boxes come from PyTorch's own pose-model input, i.e. the same top-down crop on both sides - every
   keypoint within --tol-px pixels, else it stops without registering;
4. writes <model>.ref.npz (crops + ONNX outputs) for regression tests on the target machine;
5. with --register, writes the sha256 and size into backend/models/MODELS.json (replacing "pending-export").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "backend" / "models"
OUT_NAME = "rtmpose-m_8xb256-420e_coco-256x192_dynbatch.onnx"
MODEL_CFG = "body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py"
INPUT_WH = (192, 256)
SIMCC_TOL = 1e-2  # max abs difference of the raw SimCC logits, ONNX Runtime vs PyTorch, same input
CONF_MIN = 0.3  # keypoints below this score are not compared against MMPose
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_checkpoint(path: Path, pinned_sha256: str | None = None) -> str | None:
    """None if acceptable, else the reason it is refused.

    The hash tag in OpenMMLab file names is not guaranteed to be a sha256 prefix of the hosted file, so when it
    does not match the owner can pin the file's full sha256 (printed in the refusal) with --checkpoint-sha256."""
    name = path.name.lower()
    if "body7" in name:
        return "body7 weights are never used (restricted non-commercial training data)"
    if "coco" not in name or "rtmpose-m" not in name:
        return "not an RTMPose-m COCO checkpoint"
    tag = re.search(r"-([0-9a-f]{8})_\d{8}\.pth$", name)
    if tag is None:
        return "no OpenMMLab hash tag (-xxxxxxxx_YYYYMMDD.pth) in the file name"
    actual = sha256(path)
    if actual.startswith(tag.group(1)):
        return None
    if pinned_sha256 and pinned_sha256.lower() == actual:
        print(f"note: tag {tag.group(1)} is not a sha256 prefix; accepted by the pinned sha256 {actual}")
        return None
    return (
        f"sha256 {actual} does not start with the file-name tag {tag.group(1)}. If this file was downloaded "
        f"from the official download.openmmlab.com URL, re-run with --checkpoint-sha256 {actual}"
    )


def deploy_config() -> dict:
    return {
        "codebase_config": {"type": "mmpose", "task": "PoseDetection", "export_postprocess": False},
        "backend_config": {"type": "onnxruntime"},
        "onnx_config": {
            "type": "onnx",
            "export_params": True,
            "keep_initializers_as_inputs": False,
            "opset_version": 11,
            "save_file": "end2end.onnx",
            "input_names": ["input"],
            "output_names": ["simcc_x", "simcc_y"],
            "input_shape": list(INPUT_WH),
            "optimize": True,
            "dynamic_axes": {"input": {0: "batch"}, "simcc_x": {0: "batch"}, "simcc_y": {0: "batch"}},
        },
    }


def sample_frames(src: Path, n: int) -> list[np.ndarray]:
    import cv2

    files = sorted(p for p in (src.rglob("*") if src.is_dir() else [src]) if p.suffix.lower() in VIDEO_EXT | IMAGE_EXT)
    frames: list[np.ndarray] = []
    for f in files:
        if f.suffix.lower() in IMAGE_EXT:
            img = cv2.imread(str(f))
            if img is not None:
                frames.append(img)
        else:
            cap = cv2.VideoCapture(str(f))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            for k in (total // 4, total // 2, 3 * total // 4):
                cap.set(cv2.CAP_PROP_POS_FRAMES, k)
                ok, img = cap.read()
                if ok:
                    frames.append(img)
            cap.release()
        if len(frames) >= n:
            break
    return frames[:n]


def person_boxes(img: np.ndarray, det_model: Path | None) -> np.ndarray:
    """Person boxes for validation: the server's own RTMDet ONNX when given, else the whole frame."""
    if det_model is None or not det_model.is_file():
        h, w = img.shape[:2]
        return np.array([[0, 0, w, h]], np.float32)
    sys.path.insert(0, str(ROOT / "backend"))
    from lifeguard.vision.inference import decode, models

    m = models.load_detector_model(str(det_model), "cpu", 640)
    boxes, scores, _ = decode.run_detector_people(m, img, score_thr=0.5)
    return boxes[scores >= 0.5].astype(np.float32)


def validate(onnx_path: Path, model_cfg: str, ckpt: Path, frames: list[np.ndarray], det: Path | None, tol: float):
    """Two checks on real frames with people.

    1. Export: the SAME normalised crop tensor through PyTorch and through the ONNX file; the raw SimCC outputs
       must agree (SIMCC_TOL) - this isolates the export from every preprocessing choice.
    2. Pipeline: the server's own crop + decode against MMPose's full inference_topdown. Compared on confident
       keypoints only (an occluded joint's argmax may flip on a one-bin tie) and measured in model-input
       pixels (192x256 crop), so the tolerance does not depend on how big the person is in the frame.
    """
    import onnxruntime as ort
    import torch
    from mmpose.apis import inference_topdown, init_model

    sys.path.insert(0, str(ROOT / "backend"))
    from lifeguard.vision.inference import models, pose

    torch_model = init_model(model_cfg, str(ckpt), device="cpu")
    # the RTMPose config averages the prediction with a mirrored pass (flip_test); the server runs one pass per
    # crop (half the cost, same as rtmlib), so the reference must too or the comparison measures flip, not code
    torch_model.test_cfg = {**dict(torch_model.test_cfg or {}), "flip_test": False}
    from lifeguard.vision.constants import POSE_INPUT_RGB

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # both channel orders are measured; the one the server is configured for (POSE_INPUT_RGB) must pass
    variants = {flag: models.PoseModel(str(onnx_path), sess, {}, rgb=flag) for flag in (False, True)}
    server = variants[bool(POSE_INPUT_RGB)]
    checked, worst_simcc = 0, 0.0
    worst_px = {False: 0.0, True: 0.0}
    crops, sx_all, sy_all = [], [], []
    for img in frames:
        boxes = person_boxes(img, det)
        if len(boxes) == 0:
            continue
        x, _, _ = pose.prepare(server, img, boxes)
        sx, sy = pose.infer(server, x)
        with torch.no_grad():
            tx, ty = torch_model(torch.from_numpy(x), None, mode="tensor")
        d = max(float(np.abs(tx.numpy() - sx).max()), float(np.abs(ty.numpy() - sy).max()))
        worst_simcc = max(worst_simcc, d)
        if d > SIMCC_TOL:
            raise SystemExit(f"FAIL: ONNX SimCC differs from PyTorch by {d:.4f} > {SIMCC_TOL} on the same input")
        ref = inference_topdown(torch_model, img, bboxes=boxes, bbox_format="xyxy")
        for flag, m in variants.items():
            xv, centers, scales = pose.prepare(m, img, boxes)
            svx, svy = pose.infer(m, xv)
            kpts, kscore = pose.decode(m, svx, svy, centers, scales)
            for i, sample in enumerate(ref):
                conf = kscore[i] >= CONF_MIN
                if conf.any():
                    err = np.abs(kpts[i] - sample.pred_instances.keypoints[0]).max(axis=1)[conf]
                    worst_px[flag] = max(worst_px[flag], float(err.max() * INPUT_WH[0] / scales[i, 0]))
        checked += len(ref)
        crops.append(x)
        sx_all.append(sx)
        sy_all.append(sy)
    if checked == 0:
        raise SystemExit("FAIL: no person found in the validation frames - pass --images with people in them")
    print(
        f"pipeline vs MMPose (worst confident keypoint, model-input px): "
        f"BGR-as-is {worst_px[False]:.2f}, RGB-flipped {worst_px[True]:.2f}; "
        f"server POSE_INPUT_RGB={bool(POSE_INPUT_RGB)}"
    )
    if worst_px[bool(POSE_INPUT_RGB)] > tol:
        other = worst_px[not bool(POSE_INPUT_RGB)]
        hint = (
            f" - the other channel order gives {other:.2f}: POSE_INPUT_RGB in thresholds.json should be "
            f"{str(not POSE_INPUT_RGB).lower()}"
            if other <= tol
            else ""
        )
        raise SystemExit(
            f"FAIL: server crop/decode differs from MMPose by "
            f"{worst_px[bool(POSE_INPUT_RGB)]:.2f} input-px > {tol}{hint}"
        )
    return (
        checked,
        worst_simcc,
        worst_px[bool(POSE_INPUT_RGB)],
        np.concatenate(crops),
        np.concatenate(sx_all),
        np.concatenate(sy_all),
    )


def register(path: Path) -> None:
    manifest_path = MODELS_DIR / "MODELS.json"
    d = json.loads(manifest_path.read_text(encoding="utf-8"))
    for m in d["models"]:
        if m["file"] == path.name:
            m["sha256"] = sha256(path)
            m["bytes"] = path.stat().st_size
            break
    else:
        raise SystemExit(f"{path.name} is not listed in MODELS.json")
    manifest_path.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True, help="frames with people: images, videos or a folder")
    ap.add_argument("--det-model", type=Path, default=None, help="RTMDet ONNX for validation boxes (optional)")
    ap.add_argument(
        "--checkpoint-sha256", default=None, help="full sha256 of the official download, if its tag differs"
    )
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument(
        "--tol-px", type=float, default=2.0, help="max server-vs-MMPose keypoint error, in model-input pixels"
    )
    ap.add_argument("--work-dir", type=Path, default=ROOT / "build" / "pose_export")
    ap.add_argument("--register", action="store_true", help="write sha256 + size into MODELS.json")
    args = ap.parse_args(argv)

    reason = check_checkpoint(args.checkpoint, args.checkpoint_sha256)
    if reason:
        print(f"FAIL: {args.checkpoint.name}: {reason}")
        return 1
    frames = sample_frames(args.images, args.frames)
    if not frames:
        print(f"FAIL: no readable images or videos under {args.images}")
        return 1

    import mmpose
    from mmdeploy.apis import torch2onnx
    from mmengine import Config
    from mmpose.utils import register_all_modules

    register_all_modules()
    model_cfg = str(Path(mmpose.__file__).parent / ".mim" / "configs" / MODEL_CFG)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    torch2onnx(
        frames[0],
        str(args.work_dir),
        "end2end.onnx",
        Config(deploy_config()),
        model_cfg,
        str(args.checkpoint),
        device="cpu",  # mmdeploy defaults to cuda:0; the export venv has the CPU build of torch
    )
    exported = args.work_dir / "end2end.onnx"
    checked, worst_simcc, worst_px, crops, sx, sy = validate(
        exported, model_cfg, args.checkpoint, frames, args.det_model, args.tol_px
    )

    out = MODELS_DIR / OUT_NAME
    out.write_bytes(exported.read_bytes())
    np.savez_compressed(out.with_suffix(".ref.npz"), crops=crops, simcc_x=sx, simcc_y=sy)
    print(
        f"OK {out.name}: {checked} people; SimCC max diff vs PyTorch {worst_simcc:.5f}; "
        f"server vs MMPose keypoints within {worst_px:.2f} input-px; sha256 {sha256(out)}"
    )
    if args.register:
        register(out)
        print("registered in backend/models/MODELS.json - next: python tools/verify_models.py, then")
        print("python tools/build_trt_engines.py --data-dir <data dir>   (on the RTX 4060)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
