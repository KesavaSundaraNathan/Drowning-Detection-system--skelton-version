"""Re-instantiate the RTMDet detector graph at a higher input resolution (plan Phase 7.2 lever, built Phase 8).

    python tools/make_input_size.py backend/models/<rtmdet>_dynbatch.onnx --size 960 [--check FRAME ...] [--register]

The plan's route was a dynamic-shape re-export from the mmdet checkpoint. That needs the mm* / torch stack
(new downloads) for a change that, in this graph, is purely structural: the network is fully convolutional
and the ONLY things tied to the 640 x 640 input are the anchor-point grids of the three head levels, baked in
by the exporter as constants:

  * per level (stride s = 8, 16, 32; n = 640 / s): a column (n, 1) and a row (1, n) of arange(n) * s, and
    the int64 shape [n, n] they are Expanded to;
  * the graph input dims (H, W) and the output anchor count (sum of n^2 = 8400).

This tool rewrites exactly those (n -> m = size / s) and refuses the graph if anything else looks
resolution-bound (an int64 constant holding an anchor count, a grid it cannot prove is arange * stride, an
input that is not N x 3 x 640 x 640). Weights are unchanged, so licence and data provenance are the source
model's. The batch axis stays dynamic.

`--check` runs onnxruntime (CPU) on real frames: the 640 model on a frame letterboxed to 640 and the new model
on the SAME 640 image upscaled to `size`. The network sees the same content at a larger scale, so every
confident person of the 640 model (score >= 0.5) must be found again at its upscaled place (IoU >= 0.6). A
wrong grid (stride, offset, order) misplaces every box and fails this immediately. It is a sanity check of the
graph edit, not an accuracy claim; accuracy is decided on the held-out split (docs/phase_reports/phase8).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SRC_SIZE = 640
ANCHORS = {8400, 6400, 1600, 400}


class SurgeryError(RuntimeError):
    pass


def resize_graph(model: Any, size: int) -> tuple[Any, list[str]]:
    import onnx
    from onnx import numpy_helper

    if size % 32 or size <= SRC_SIZE:
        raise SurgeryError(f"size must be a multiple of 32 above {SRC_SIZE}, got {size}")
    g = model.graph
    if len(g.input) != 1:
        raise SurgeryError("expected one graph input")
    dims = g.input[0].type.tensor_type.shape.dim
    if [d.dim_value for d in dims[1:]] != [3, SRC_SIZE, SRC_SIZE]:
        raise SurgeryError(f"input must be N x 3 x {SRC_SIZE} x {SRC_SIZE}")
    consts = {
        n.output[0]: (n, numpy_helper.to_array(n.attribute[0].t))
        for n in g.node
        if n.op_type == "Constant" and n.attribute and n.attribute[0].name == "value"
    }
    edits: list[str] = []
    strides: set[int] = set()
    grids: set[str] = set()
    for name, (node, v) in consts.items():
        if v.dtype.kind == "f" and v.ndim == 2 and 1 in v.shape and v.size >= 4:
            flat = v.ravel()
            s = float(flat[1] - flat[0])
            n = flat.size
            if not (s > 0 and n * s == SRC_SIZE and np.array_equal(flat, np.arange(n, dtype=v.dtype) * s)):
                raise SurgeryError(f"{name}: float grid constant {v.shape} is not arange(n) * stride")
            m = size // int(s)
            new = (np.arange(m, dtype=v.dtype) * s).reshape((m, 1) if v.shape[0] == n else (1, m))
            node.attribute[0].t.CopyFrom(numpy_helper.from_array(new, name=node.attribute[0].t.name))
            strides.add(int(s))
            grids.add(name)
            edits.append(f"{name}: arange({n})*{int(s)} {v.shape} -> {new.shape}")
    if strides != {8, 16, 32}:
        raise SurgeryError(f"expected grids for strides 8/16/32, found {sorted(strides)}")
    for node in g.node:
        if node.op_type == "Expand" and node.input[0] in grids:
            sname = node.input[1]
            snode, sv = consts[sname]
            s = SRC_SIZE // int(sv[0])
            new = np.array([size // s, size // s], np.int64)
            if not np.array_equal(sv, [SRC_SIZE // s] * 2):
                raise SurgeryError(f"{sname}: Expand shape {sv} is not the [n, n] grid")
            if snode.attribute[0].t.dims != [2] or not np.array_equal(numpy_helper.to_array(snode.attribute[0].t), new):
                snode.attribute[0].t.CopyFrom(numpy_helper.from_array(new, name=snode.attribute[0].t.name))
                edits.append(f"{sname}: Expand shape {sv.tolist()} -> {new.tolist()}")
    for name, (_, v) in consts.items():  # anything else holding an anchor count is not understood
        if v.dtype.kind == "i" and ANCHORS & {int(x) for x in v.ravel()}:
            raise SurgeryError(f"{name}: int constant {v.ravel()[:6]} holds an anchor count")
    for init in g.initializer:
        if init.data_type == onnx.TensorProto.INT64 and ANCHORS & set(numpy_helper.to_array(init).ravel().tolist()):
            raise SurgeryError(f"{init.name}: initializer holds an anchor count")
    dims[2].dim_value = dims[3].dim_value = size
    total = sum((size // s) ** 2 for s in strides)
    for out in g.output:
        od = out.type.tensor_type.shape.dim
        if len(od) >= 2 and od[1].dim_value == sum((SRC_SIZE // s) ** 2 for s in strides):
            od[1].dim_value = total
            edits.append(f"output {out.name}: anchors -> {total}")
    del g.value_info[:]
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model, edits


def _letterbox(img: np.ndarray, size: int) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    out = np.full((size, size, 3), 114, np.uint8)
    out[: int(h * r), : int(w * r)] = cv2.resize(img, (int(w * r), int(h * r)))
    return out


def _people(sess: Any, img: np.ndarray, thr: float) -> np.ndarray:
    """RTMDet `dets` (N, A, 4 + 80) xyxy + class scores -> person boxes (score >= thr) after NMS, input px."""
    import cv2

    mean, std = np.array([103.53, 116.28, 123.675], np.float32), np.array([57.375, 57.12, 58.395], np.float32)
    x = ((img.astype(np.float32) - mean) / std).transpose(2, 0, 1)[None]
    d = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    keep = d[:, 4] >= thr
    b, s = d[keep, :4], d[keep, 4]
    idx = cv2.dnn.NMSBoxes(
        [[float(v[0]), float(v[1]), float(v[2] - v[0]), float(v[3] - v[1])] for v in b], s.tolist(), thr, 0.45
    )
    return b[np.asarray(idx, int).reshape(-1)] if len(b) else b


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / u) if u > 0 else 0.0


def check(src: Path, out: Path, size: int, frames: list[Path]) -> dict[str, Any]:
    import cv2
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    a = ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    found = total = 0
    worst = 1.0
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            raise SystemExit(f"cannot read {f}")
        small = _letterbox(img, SRC_SIZE)
        big = cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR)
        ref = _people(a, small, 0.5)
        got = _people(b, big, 0.3) * (SRC_SIZE / size)
        for r in ref:
            total += 1
            best = max((_iou(r, g) for g in got), default=0.0)
            worst = min(worst, best)
            found += best >= 0.6
    return {
        "frames": len(frames),
        "people_640": total,
        "found_again": found,
        "min_iou": round(worst, 3),
        "ok": total > 0 and found == total,
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def register(src: Path, out: Path, size: int, edits: list[str], chk: dict[str, Any] | None) -> None:
    manifest_path = out.parent / "MODELS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = next(m for m in manifest["models"] if m["file"] == src.name)
    entry = {
        **{k: base[k] for k in ("role", "licence", "commercial_ok", "dev_only", "data_provenance", "source")},
        "file": out.name,
        "key": f"{base['key']}_{size}",
        "sha256": sha256(out),
        "bytes": out.stat().st_size,
        "note": f"{base['file']} re-instantiated at {size}x{size} input (Phase 8 resolution lever): anchor grids "
        f"and shapes only ({len(edits)} edits), weights unchanged. Check: {json.dumps(chk)}",
        "derived_from": {"file": src.name, "sha256": sha256(src), "tool": f"tools/make_input_size.py --size {size}"},
    }
    manifest["models"] = [m for m in manifest["models"] if m["file"] != out.name] + [entry]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    import onnx

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check", nargs="*", type=Path, default=None, help="real frames for the sanity check")
    ap.add_argument("--register", action="store_true", help="add the result to MODELS.json (next to the model)")
    a = ap.parse_args(argv)
    out = a.out or a.model.with_name(f"{a.model.stem}_{a.size}.onnx")
    try:
        model, edits = resize_graph(onnx.load(str(a.model)), a.size)
    except SurgeryError as exc:
        sys.stderr.write(f"REFUSED: {exc}\n")
        return 2
    meta = model.metadata_props.add()
    meta.key, meta.value = "lifeguard.derived_from", f"{a.model.name} sha256={sha256(a.model)} (input {a.size})"
    onnx.save(model, str(out))
    sys.stdout.write(f"wrote {out} ({len(edits)} edits)\n" + "".join(f"  {e}\n" for e in edits))
    chk = check(a.model, out, a.size, a.check) if a.check else None
    if chk is not None:
        sys.stdout.write(f"check: {json.dumps(chk)}\n")
        if not chk["ok"]:
            return 1
    if a.register:
        register(a.model, out, a.size, edits, chk)
        sys.stdout.write("registered in MODELS.json\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
