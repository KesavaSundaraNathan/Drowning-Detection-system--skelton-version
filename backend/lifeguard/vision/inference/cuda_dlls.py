"""Make the CUDA 12 / cuDNN 9 DLLs of the NVIDIA wheels visible to onnxruntime-gpu on Windows.

onnxruntime-gpu's CUDA provider LoadLibrary()s cudart / cublas / cufft / cudnn by name; on a clean
Windows install they are not on PATH, and ORT then *silently* runs on the CPU (audit #4). The
serving lock installs them as `nvidia-*-cu12` wheels (nvidia/<lib>/bin); this module adds those
directories to the process's DLL search path (os.add_dll_directory + the process-local PATH, which
cuDNN 9 needs for the sub-libraries it loads lazily). Nothing outside this process changes.

[v3] From the v2 baseline detection/cuda_dlls.py. REMOVED: the torch/lib lookup (v2 only ran on the
GPU because a CUDA torch wheel happened to ship these DLLs - torch is not in the serving venv).
[2.5.x] The TensorRT lookup is back (owner decision 2026-10-14: TensorRT FP16 through ORT's TensorRT
execution provider; TensorRT is NVIDIA-licensed and optional, see THIRD_PARTY_NOTICES.md): the
`tensorrt-cu12-libs` wheel puts nvinfer_10.dll & co. in site-packages/tensorrt_libs. `missing_dlls()`
feeds the provider guard's FAULT message (vision/inference/session.py).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# The files ONNX Runtime >= 1.19 needs from CUDA 12 + cuDNN 9.
REQUIRED_DLLS = ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cufft64_11.dll", "cudnn64_9.dll")

_done = False
_added: list[str] = []


def wheel_dll_dirs() -> list[Path]:
    """nvidia/<lib>/bin directories of the installed nvidia-*-cu12 wheels, plus the TensorRT wheel's
    tensorrt_libs directory when `tensorrt-cu12-libs` is installed."""
    spec = importlib.util.find_spec("nvidia")
    roots = [Path(p) for p in (spec.submodule_search_locations or [])] if spec else []
    # CUDA 13 wheels (nvidia/cu13) put their DLLs one level deeper, in bin/x86_64.
    dirs = {d for r in roots for pat in ("*/bin", "*/bin/x86_64") for d in r.glob(pat) if d.is_dir()}
    trt = importlib.util.find_spec("tensorrt_libs")
    dirs |= {Path(p) for p in (trt.submodule_search_locations or [])} if trt else set()
    return sorted(d for d in dirs if d.is_dir())


def ensure() -> list[str]:
    """Register the wheel DLL directories once (idempotent; no-op off Windows). Returns the dirs added."""
    global _done
    if _done or sys.platform != "win32":
        _done = True
        return list(_added)
    _done = True
    for d in wheel_dll_dirs():
        if not any(d.glob("*.dll")):
            continue
        try:
            os.add_dll_directory(str(d))
        except OSError as exc:
            logger.warning("cannot add CUDA DLL directory %s: %s", d, exc)
            continue
        _added.append(str(d))
    if _added:
        os.environ["PATH"] = os.pathsep.join([*_added, os.environ.get("PATH", "")])
        logger.info("CUDA DLL directories registered for onnxruntime: %s", _added)
    return list(_added)


def missing_dlls() -> list[str]:
    """Which REQUIRED_DLLS cannot be found on the (patched) PATH. Empty off Windows."""
    if sys.platform != "win32":
        return []
    search = [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    return [n for n in REQUIRED_DLLS if not any((d / n).is_file() for d in search)]
