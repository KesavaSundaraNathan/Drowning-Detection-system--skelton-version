"""Re-export a detector ONNX with a DYNAMIC batch axis (plan Phase 4: batched GPU scheduler).

    python tools/make_dynamic_batch.py backend/models/<model>.onnx [--out <model>_dynbatch.onnx] [--check]

The plan's route was a re-export from the training checkpoint in a throwaway mm venv. That needs new
downloads (mmdet / mmdeploy / torch) for a change that is purely structural, so this tool edits the
exported graph instead - deterministically, without touching a single weight:

  * the graph input and every graph output get the symbolic leading dim "batch";
  * every Reshape whose constant target shape starts with a literal 1 (the batch the exporter baked
    in, e.g. [1, -1, 80]) gets 0 there instead - ONNX Reshape's "copy this dim from the input";
  * stale intermediate value_info shapes are dropped and the model is re-checked (onnx.checker,
    full shape inference).

The tool REFUSES a graph it does not understand (any other op with a hard-coded batch-1 constant,
an input that is not NCHW, a Reshape target computed at run time). `--check` then proves the edit on
onnxruntime (CPU): the batch-1 output is bit-identical to the original model's, and a batch of B
distinct frames equals B single-frame runs within float tolerance. Weights are unchanged, so the
licence and data provenance are the source model's (MODELS.json records `derived_from`).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

BATCH = "batch"
CHECK_BATCH = 4
ATOL = 1e-3  # cross-batch-size float tolerance on CPU (different GEMM blocking); batch 1 must be exact


class SurgeryError(RuntimeError):
    pass


def _const_values(graph: Any) -> dict[str, np.ndarray]:
    from onnx import numpy_helper

    values = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    values[node.output[0]] = numpy_helper.to_array(attr.t)
    return values


def make_dynamic(model: Any) -> tuple[Any, list[str]]:
    """Return (edited model, list of edits). Raises SurgeryError for graphs it cannot prove safe."""
    import onnx
    from onnx import numpy_helper

    graph = model.graph
    if len(graph.input) != 1:
        raise SurgeryError(f"expected exactly one graph input, found {len(graph.input)}")
    dims = graph.input[0].type.tensor_type.shape.dim
    if len(dims) != 4:
        raise SurgeryError("graph input is not NCHW")
    edits: list[str] = []
    consts = _const_values(graph)
    producers = {out: node for node in graph.node for out in node.output}
    for node in graph.node:
        if node.op_type != "Reshape":
            continue
        target = node.input[1]
        if target not in consts:
            raise SurgeryError(f"Reshape {node.name} has a run-time target shape; cannot prove batch safety")
        shape = consts[target]
        if shape.ndim == 1 and shape.size >= 2 and int(shape[0]) == 1:
            new = shape.copy()
            new[0] = 0  # copy the batch dim from the input tensor
            src = producers.get(target)
            tensor = numpy_helper.from_array(new.astype(np.int64), target)
            if src is not None and src.op_type == "Constant":
                for attr in src.attribute:
                    if attr.name == "value":
                        attr.t.CopyFrom(tensor)
            else:
                for i, init in enumerate(graph.initializer):
                    if init.name == target:
                        graph.initializer[i].CopyFrom(tensor)
            edits.append(f"Reshape {node.name}: {shape.tolist()} -> {new.tolist()}")
    # anything else with a literal leading 1 in a shape constant is unexplained -> refuse
    for node in graph.node:
        if node.op_type in ("Expand", "Tile", "ConstantOfShape"):
            arg = node.input[1] if len(node.input) > 1 else node.input[0]
            val = consts.get(arg)
            if val is not None and val.ndim == 1 and val.size >= 3 and int(val[0]) == 1:
                raise SurgeryError(f"{node.op_type} {node.name} has a batch-1 constant {val.tolist()}")
    dims[0].ClearField("dim_value")
    dims[0].dim_param = BATCH
    for out in graph.output:
        odims = out.type.tensor_type.shape.dim
        if not odims:
            raise SurgeryError(f"output {out.name} has no shape")
        odims[0].ClearField("dim_value")
        odims[0].dim_param = BATCH
    del graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    for out in inferred.graph.output:
        lead = out.type.tensor_type.shape.dim[0]
        if lead.dim_param != BATCH:
            raise SurgeryError(f"output {out.name} lost the batch axis after inference")
    return model, edits


def check_equivalence(original: Path, dynamic: Path, batch: int = CHECK_BATCH) -> dict[str, Any]:
    """onnxruntime CPU: batch-1 bit-identical; batch-B == B single runs within ATOL."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    a = ort.InferenceSession(str(original), opts, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(dynamic), opts, providers=["CPUExecutionProvider"])
    inp = a.get_inputs()[0]
    _, c, h, w = inp.shape
    rng = np.random.default_rng(1234)
    x = rng.normal(0.0, 1.0, (batch, int(c), int(h), int(w))).astype(np.float32)
    single = [a.run(None, {inp.name: x[i : i + 1]}) for i in range(batch)]
    b1 = b.run(None, {b.get_inputs()[0].name: x[:1]})
    exact = all(np.array_equal(p, q) for p, q in zip(single[0], b1, strict=True))
    batched = b.run(None, {b.get_inputs()[0].name: x})
    max_err = 0.0
    for k, out in enumerate(batched):
        ref = np.concatenate([s[k] for s in single], 0)
        if ref.shape != out.shape:
            raise SurgeryError(f"output {k}: batched shape {out.shape} != stacked singles {ref.shape}")
        max_err = max(max_err, float(np.max(np.abs(ref - out))))
    ok = exact and max_err <= ATOL
    return {"batch1_bit_identical": exact, "batch": batch, "max_abs_err": max_err, "atol": ATOL, "ok": ok}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    import onnx

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check", action="store_true", help="prove equivalence on onnxruntime (CPU)")
    args = ap.parse_args(argv)
    src: Path = args.model
    out: Path = args.out or src.with_name(src.stem + "_dynbatch.onnx")
    model = onnx.load(str(src))
    try:
        model, edits = make_dynamic(model)
    except SurgeryError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    for prop in model.metadata_props:
        if prop.key == "lifeguard.derived_from":
            break
    else:
        entry = model.metadata_props.add()
        entry.key, entry.value = "lifeguard.derived_from", f"{src.name} sha256={sha256(src)} (dynamic batch)"
    onnx.save(model, str(out))
    print(f"wrote {out} ({len(edits)} edits)")
    for e in edits:
        print("  " + e)
    print(f"sha256 {sha256(out)}  bytes {out.stat().st_size}")
    if args.check:
        res = check_equivalence(src, out)
        print(f"equivalence: {res}")
        if not res["ok"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
