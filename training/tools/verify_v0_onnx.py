"""Phase 10: stamp a v0 ONNX export with Life Guard's metadata and check it in the SERVING venv (repo `venv\\`).

1. metadata: lifeguard.det_format (detr_logits | xyxy_scores), lifeguard.task=detector, lifeguard.classes =
   ["swimmer", "person_out_of_water"]. The training class `person` means "completely out of the water" (labelling
   guide section 2), so it is exported under the app's out-of-water name: the live decoder then reads the class pair
   as the in-water vote (vision/inference/decode.person_class_ids) with no application change.
2. ONNX Runtime (CPU) vs the PyTorch reference from export_v0.py (max abs difference), and batch consistency (row 0
   of the batch-2 run equals the batch-1 run).
3. Life Guard's own loader + decoder (load_detector_model, run_detector_people) on real frames: boxes, scores and a
   finite in-water vote. Not registered in backend\\models\\MODELS.json (that is an application change: owner's call).

    $env:PYTHONPATH="backend"
    venv\\Scripts\\python.exe training\\tools\\verify_v0_onnx.py training\\work_dirs\\onnx\\dfine_l_v0.onnx
"""

# ruff: noqa: T201
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort

from lifeguard.vision.inference.decode import run_detector_people
from lifeguard.vision.inference.models import load_detector_model

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["swimmer", "person_out_of_water"]


def fail(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 2


def main() -> int:
    if len(sys.argv) < 2:
        return fail("usage: verify_v0_onnx.py MODEL.onnx [--stand-in]")
    path = Path(sys.argv[1])
    ref_path = Path(str(path) + ".ref.npz")
    if not path.is_file():
        return fail(f"ONNX file not found: {path}")
    if not ref_path.is_file():
        return fail(f"PyTorch reference not found: {ref_path} (written by export_v0.py next to the ONNX file)")

    # Validate everything before the file is rewritten with metadata.
    m = onnx.load(str(path))
    n_out = len(m.graph.output)
    fmt = "detr_logits" if n_out == 2 else "xyxy_scores"
    ref = np.load(str(ref_path))
    if fmt == "detr_logits":
        if "logits" not in ref or ref["logits"].shape[-1] != len(CLASSES):
            shape = ref["logits"].shape if "logits" in ref else "missing"
            return fail(f"{ref_path}: logits shape {shape}, expected last axis {len(CLASSES)} ({CLASSES})")
    elif n_out != 1:
        return fail(f"{path}: {n_out} graph outputs; expected 2 (detr_logits) or 1 (xyxy_scores)")

    meta = {"lifeguard.det_format": fmt, "lifeguard.task": "detector", "lifeguard.classes": json.dumps(CLASSES)}
    del m.metadata_props[:]
    for k, v in meta.items():
        e = m.metadata_props.add()
        e.key, e.value = k, v
    onnx.checker.check_model(m)
    onnx.save(m, str(path))

    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [o.name for o in s.get_outputs()]
    out2 = s.run(None, {"images": ref["x"]})
    out1 = s.run(None, {"images": ref["x"][:1]})
    ok = True
    if fmt == "detr_logits":  # DETR: queries may come out in another order, so match each ORT query to its PyTorch twin
        for i in range(len(ref["x"])):
            q_o = np.concatenate([out2[1][i], out2[0][i]], 1)  # (Q, 4 + C)
            q_t = np.concatenate([ref["boxes"][i], ref["logits"][i]], 1)
            d = np.abs(q_o[:, None, :] - q_t[None, :, :]).max(-1).min(1)  # each ORT query vs its nearest torch query
            ok_i = float(d.max()) < 2e-2
            ok &= ok_i
            print(f"image {i}: worst ORT query vs its nearest PyTorch query {d.max():.2e} {'ok' if ok_i else 'FAIL'}")
        for n, o2, o1 in zip(names, out2, out1, strict=True):
            d_b = float(np.abs(o2[:1] - o1).max())
            ok &= d_b < 1e-3
            print(f"{n}: max |batch2[0]-batch1| {d_b:.2e} {'ok' if d_b < 1e-3 else 'FAIL'}")
    else:  # anchor-based (fixed-order) layout: element-wise comparison
        for n, o2, o1 in zip(names, out2, out1, strict=True):
            d_ref = float(np.abs(o2 - ref[n]).max())
            d_b = float(np.abs(o2[:1] - o1).max())
            ref_ok = d_ref < 1e-2 * max(1.0, float(np.abs(ref[n]).max()))
            batch_ok = d_b < 1e-3 * max(1.0, float(np.abs(o1).max()))
            good = ref_ok and batch_ok
            ok &= good
            print(
                f"{n}: shape {o2.shape}, max |ORT-torch| {d_ref:.2e}, max |batch2[0]-batch1| {d_b:.2e} "
                f"{'ok' if good else 'FAIL'}"
            )

    if "--stand-in" in sys.argv:  # an untrained stand-in scores below the app's D-FINE floor; test the plumbing only
        import lifeguard.vision.inference.decode as dec

        dec.DET_DETR_MIN_SCORE = 0.0
    dm = load_detector_model(str(path), "cpu", 1024)
    frames_dir = ROOT / "training/data/frames"
    frames = sorted(frames_dir.glob("*/f00*.jpg"))[:3]
    if not frames:
        ok = False
        print(f"no frames matching {frames_dir}/*/f00*.jpg: Life Guard decode not tested FAIL")
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            ok = False
            print(f"Life Guard decode {f.parent.name}/{f.name}: cv2.imread could not read the frame FAIL")
            continue
        b, sc, w = run_detector_people(dm, img, score_thr=0.001, names=CLASSES)
        finite = bool(np.isfinite(w).all()) if len(w) else True
        ok &= finite
        print(
            f"Life Guard decode {f.parent.name}/{f.name}: {len(b)} boxes >= 0.001, top score "
            f"{(sc.max() if len(sc) else 0):.3f}, in-water vote finite: {finite}"
        )
    print(f"format {fmt}, detr={dm.detr}, input {dm.model_input_size}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
