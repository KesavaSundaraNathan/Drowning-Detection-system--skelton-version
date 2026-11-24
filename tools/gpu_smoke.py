"""GPU smoke test for the serving lock (audit #4: a clean install silently ran on the CPU).

    python tools/gpu_smoke.py            # exit 0 only if ONNX Runtime really ran a cuDNN conv on CUDA

Proves, on a machine with an NVIDIA GPU, that requirements/serving.lock alone is enough:
  * torch is NOT installed (v2 only worked because torch\\lib happened to ship the CUDA DLLs);
  * the check runs in a child process whose PATH has every directory containing a CUDA/cuDNN
    DLL removed and CUDA_PATH* unset, so only the nvidia-*-cu12 wheels can satisfy it;
  * the session's active provider must be CUDAExecutionProvider (ORT otherwise falls back to
    CPU with only a log line - exactly the silent failure being guarded against);
  * a Conv (cuDNN) result must match a NumPy reference.

Since Phase 3 the child uses the PRODUCTION provider guard (lifeguard.vision.inference: cuda_dlls +
session.make_session, which raises ProviderError instead of falling back), so this proves the guard
the server starts with - not a copy of it - works from the NVIDIA wheels alone.
Not part of hosted CI (no GPU there): run by `tools/ci.ps1 -Gpu`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess  # nosec B404  # re-runs this same script in a child process
import sys
from pathlib import Path

CUDA_DLLS = ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cufft64_11.dll", "cudnn64_9.dll")


def _sanitised_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("CUDA_PATH")}
    keep = []
    for entry in env.get("PATH", "").split(os.pathsep):
        p = Path(entry) if entry else None
        if p and p.is_dir() and any((p / dll).exists() for dll in CUDA_DLLS):
            continue  # system CUDA toolkit / driver copies: hide them
        keep.append(entry)
    env["PATH"] = os.pathsep.join(keep)
    env["LIFEGUARD_GPU_SMOKE_CHILD"] = "1"
    return env


def child() -> dict[str, object]:
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    from lifeguard.vision.inference import cuda_dlls, session  # the production provider guard

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 64, 64), dtype=np.float32)
    w = rng.standard_normal((8, 3, 3, 3), dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])],
        "smoke",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 64, 64])],
        [numpy_helper.from_array(w, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "smoke.onnx"
        onnx.save(model, str(path))
        try:
            sess, _info = session.make_session(str(path), "cuda")  # ProviderError unless CUDA is really active
        except session.ProviderError as exc:
            return {
                "error": str(exc),
                "active": [exc.active],
                "missing_dlls": exc.missing_dlls,
                "dll_dirs": [str(p) for p in cuda_dlls.wheel_dll_dirs()],
            }
        (y,) = sess.run(None, {"x": x})

    # NumPy reference convolution (padding 1, stride 1).
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3))  # (1,3,64,64,3,3)
    ref = np.einsum("bchwij,ocij->bohw", win, w)
    return {
        "ort": ort.__version__,
        "device": ort.get_device(),
        "available": ort.get_available_providers(),
        "active": sess.get_providers(),
        "max_abs_err": float(np.abs(y - ref).max()),
        "dll_dirs": [str(p) for p in cuda_dlls.wheel_dll_dirs()],
    }


def main() -> int:
    if os.environ.get("LIFEGUARD_GPU_SMOKE_CHILD") == "1":
        print(json.dumps(child()))
        return 0
    fails = []
    if importlib.util.find_spec("torch") is not None:
        fails.append("torch is installed in this environment - the serving env must not contain it")
    proc = subprocess.run(  # noqa: S603  # nosec B603  # runs this script itself, no shell
        [sys.executable, __file__],
        env=_sanitised_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr)
        fails.append(f"child process failed (exit {proc.returncode})")
        res: dict[str, object] = {}
    else:
        res = json.loads(proc.stdout.strip().splitlines()[-1])
        active = res.get("active") or []
        if not active or active[0] != "CUDAExecutionProvider":
            fails.append(f"active provider is {active} - CUDA did not load (silent CPU fallback)")
        if float(res.get("max_abs_err", 1e9)) > 1e-2:
            fails.append(f"CUDA conv differs from the NumPy reference by {res['max_abs_err']}")
        if not res.get("dll_dirs"):
            fails.append("no nvidia/*/bin DLL directories found - nvidia-*-cu12 wheels missing")
    for k, v in res.items():
        print(f"  {k}: {v}")
    for f in fails:
        print(f"FAIL  {f}")
    print("gpu smoke: OK" if not fails else f"gpu smoke: {len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
