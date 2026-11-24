"""ONNX Runtime sessions with a PROVIDER GUARD (plan Phase 3; audit #4).

    sess, info = make_session(path, provider="cuda")      # ProviderError unless CUDA is really active
    sess, info = make_session(path, provider="auto")      # best available: TensorRT, CUDA, OpenVINO, CPU
    sess, info = make_session(path, "tensorrt", trt_batch=(1, 16, 32))   # FP16 engine, dynamic batch 1..32

An explicit provider ("tensorrt", "cuda", "openvino", "cpu") is strict: a session that does not run on it raises
`ProviderError` carrying the missing DLL list, and callers turn that into a startup failure + a FAULT
alert. The v2 baseline asked for CUDA and quietly ran on the CPU when the DLLs did not load; that
silent degradation is what the guard prevents.

TensorRT runs through ONNX Runtime's TensorRT execution provider (owner decision 2026-10-14; TensorRT is
NVIDIA-licensed, not open source - an optional install, never required): FP16, engines + timing cache
persisted under `trt_cache_dir()` (built on the target GPU on the first start, reused afterwards), and one
optimisation profile per dynamic-batch model covering (min, opt, max) batch. Nodes TensorRT cannot take run
on CUDA, the rest of the graph stays on TensorRT.

"auto" restores v2's ability to run without an NVIDIA GPU (Linux / WSL / CPU-only boxes) WITHOUT the
silence: the chosen provider is logged as a WARNING when it is not CUDA, reported as `info["active"]`
(health, startup audit, capacity cache), and the live capacity check still refuses any camera the
device cannot analyse at the target frame rate (CAPACITY_EXCEEDED), so a slower provider means fewer
admitted cameras, never silently dropped frames.

The CUDA provider options and session options are exactly the v2 baseline's
(detection/rtm_backend._provider_chain / _make_session), so the parity runs are numerically
comparable; CUDNN_CONV_ALGO_SEARCH comes from the versioned thresholds file because the cuDNN
algorithm choice can change float results.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from lifeguard.vision.constants import CUDNN_CONV_ALGO_SEARCH
from lifeguard.vision.inference import cuda_dlls

logger = logging.getLogger(__name__)

CONCRETE_PROVIDERS = ("tensorrt", "cuda", "openvino", "cpu")
PROVIDERS = ("auto", *CONCRETE_PROVIDERS)
AUTO_ORDER = CONCRETE_PROVIDERS
GPU_PROVIDERS = ("tensorrt", "cuda")
_ORT_NAME = {
    "tensorrt": "TensorrtExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "cpu": "CPUExecutionProvider",
}
# RTX 4060 laptop: 8 GB VRAM shared by the detector + pose engines and 4 streams' buffers
TRT_WORKSPACE_BYTES = 2 * 1024**3
_trt_cache_dir = Path(os.environ.get("LIFEGUARD_TRT_CACHE_DIR", "") or Path.home() / ".cache" / "lifeguard" / "trt")
_SHORT_NAME = {v: k for k, v in _ORT_NAME.items()}


class ProviderError(RuntimeError):
    """The requested execution provider is not the one the session runs on."""

    def __init__(self, path: str, requested: str, active: str, missing: list[str], detail: str = "") -> None:
        self.path, self.requested, self.active, self.missing_dlls = path, requested, active, missing
        msg = f"{Path(path).name}: requested {requested} but onnxruntime runs on {active}"
        if missing:
            msg += f"; missing CUDA/cuDNN DLLs: {', '.join(missing)}"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


def set_trt_cache_dir(path: Path) -> None:
    """Where TensorRT engines + timing caches live (the app sets <data_dir>/trt_cache)."""
    global _trt_cache_dir
    _trt_cache_dir = Path(path)


def trt_cache_dir() -> Path:
    return _trt_cache_dir


def _cuda_options(device_id: int) -> dict[str, Any]:
    return {
        "device_id": device_id,
        "cudnn_conv_algo_search": str(CUDNN_CONV_ALGO_SEARCH).upper(),
        "do_copy_in_default_stream": True,
        "arena_extend_strategy": "kSameAsRequested",
    }


@lru_cache(maxsize=32)
def _file_sha256(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def engine_prefix(path: str) -> str | None:
    """`<model stem>_<sha256[:16]>` for the TensorRT engine file names, so a replaced model with the same file name
    never loads a stale engine (ORT's own engine id hashes the graph's name and I/O, not its weights).
    None when the file does not exist (ORT then names the engines itself)."""
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return None
    stem = re.sub(r"[^A-Za-z0-9_]", "_", p.stem)
    return f"{stem}_{_file_sha256(str(p.resolve()), st.st_size, st.st_mtime_ns)[:16]}"


def trt_options(device_id: int = 0, profile: dict[str, str] | None = None, prefix: str | None = None) -> dict[str, Any]:
    """TensorRT EP options: FP16, persistent engine + timing cache, workspace cap, optional batch profile, optional
    engine file prefix (engine_prefix)."""
    cache = trt_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    opts: dict[str, Any] = {
        "device_id": device_id,
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(cache),
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str(cache),
        "trt_max_workspace_size": TRT_WORKSPACE_BYTES,
    }
    if prefix:
        opts["trt_engine_cache_prefix"] = prefix
    if profile:
        opts.update(profile)
    return opts


def trt_profile(input_name: str, chw: tuple[int, int, int], batch: tuple[int, int, int]) -> dict[str, str]:
    """trt_profile_{min,opt,max}_shapes for one dynamic-batch input, e.g. 'input:1x3x256x192'."""
    c, h, w = chw
    lo, opt, hi = batch
    return {
        f"trt_profile_{k}_shapes": f"{input_name}:{b}x{c}x{h}x{w}" for k, b in (("min", lo), ("opt", opt), ("max", hi))
    }


def _dynamic_input(path: str) -> tuple[str, tuple[int, int, int]] | None:
    """(input name, CHW) when the model's batch axis is dynamic (needs a TensorRT profile), else None."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    inp = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]).get_inputs()[0]
    shape = list(inp.shape)
    if isinstance(shape[0], int) and shape[0] > 0:
        return None
    if not all(isinstance(v, int) and v > 0 for v in shape[1:4]):
        return None
    return inp.name, (int(shape[1]), int(shape[2]), int(shape[3]))


def provider_chain(
    provider: str, device_id: int = 0, profile: dict[str, str] | None = None, prefix: str | None = None
) -> list[Any]:
    """ONNX Runtime provider list for an explicit provider (v2 options for CUDA)."""
    if provider == "tensorrt":
        return [
            ("TensorrtExecutionProvider", trt_options(device_id, profile, prefix)),
            ("CUDAExecutionProvider", _cuda_options(device_id)),  # nodes TensorRT cannot take
            "CPUExecutionProvider",
        ]
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "openvino":
        return [("OpenVINOExecutionProvider", {"device_type": "CPU"}), "CPUExecutionProvider"]
    if provider == "cuda":
        return [
            ("CUDAExecutionProvider", _cuda_options(device_id)),
            "CPUExecutionProvider",  # ORT places the few CUDA-unsupported nodes here, as in v2
        ]
    raise ValueError(f"provider must be one of {PROVIDERS}, got {provider!r}")


def available_providers() -> list[str]:
    """Concrete providers this onnxruntime build offers, in AUTO_ORDER."""
    import onnxruntime as ort

    names = set(ort.get_available_providers())
    return [p for p in AUTO_ORDER if _ORT_NAME[p] in names]


def short_name(ort_name: str) -> str:
    """'CUDAExecutionProvider' -> 'cuda' (unknown names pass through)."""
    return _SHORT_NAME.get(ort_name, ort_name)


def make_session(
    path: str, provider: str, device_id: int = 0, trt_batch: tuple[int, int, int] | None = None
) -> tuple[Any, dict[str, Any]]:
    """Create an InferenceSession on exactly `provider`, or raise ProviderError.

    provider="auto" tries every available provider in AUTO_ORDER and keeps the first that really runs.
    trt_batch = (min, opt, max) batch of the TensorRT optimisation profile for a dynamic-batch model."""
    if provider == "auto":
        return _make_auto_session(path, device_id, trt_batch)
    profile = None
    if provider == "tensorrt" and trt_batch:
        dyn = _dynamic_input(path)
        profile = trt_profile(dyn[0], dyn[1], trt_batch) if dyn else None
    prefix = engine_prefix(path) if provider == "tensorrt" else None
    chain = provider_chain(provider, device_id, profile, prefix)
    if provider in GPU_PROVIDERS:
        cuda_dlls.ensure()
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3  # node-placement chatter is not an error
    t0 = time.perf_counter()
    try:
        sess = ort.InferenceSession(path, so, providers=chain)
    except Exception as exc:  # no silent CPU retry (v2 retried on the CPU)
        raise ProviderError(
            path,
            provider,
            "none",
            cuda_dlls.missing_dlls() if provider in GPU_PROVIDERS else [],
            f"{type(exc).__name__}: {exc}",
        ) from exc
    active = sess.get_providers()[0]
    if active != _ORT_NAME[provider]:
        raise ProviderError(path, provider, active, cuda_dlls.missing_dlls() if provider in GPU_PROVIDERS else [])
    info = {"requested": provider, "active": active, "load_s": round(time.perf_counter() - t0, 2)}
    if profile:
        info["trt_profile"] = profile
    if prefix:
        info["trt_engine_prefix"] = prefix
    logger.info("%s: session on %s in %.2fs", Path(path).name, active, info["load_s"])
    return sess, info


def _make_auto_session(path: str, device_id: int, trt_batch: tuple[int, int, int] | None) -> tuple[Any, dict[str, Any]]:
    tried: list[str] = []
    last: ProviderError | None = None
    for candidate in available_providers():
        try:
            sess, info = make_session(path, candidate, device_id, trt_batch)
        except ProviderError as exc:
            tried.append(f"{candidate}: {exc}")
            last = exc
            continue
        info["requested"] = "auto"
        info["tried"] = tried
        if candidate not in GPU_PROVIDERS:
            logger.warning(
                "%s: provider auto -> %s (no working GPU%s); live capacity is measured on this device",
                Path(path).name,
                info["active"],
                f": {'; '.join(tried)}" if tried else "",
            )
        return sess, info
    if last is not None:
        raise ProviderError(path, "auto", "none", last.missing_dlls, "; ".join(tried))
    raise ProviderError(path, "auto", "none", [], "onnxruntime offers no usable execution provider")


def active_provider(model: Any) -> str:
    """The provider a session REALLY runs on (first in its list)."""
    try:
        return str(model.session.get_providers()[0])
    except (AttributeError, IndexError):
        return "unknown"
