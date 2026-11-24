"""Capacity cache (plan Phase 11 WP5, pre-registered in eval/site/levers_phase11.json "gates_wp5").

A restart re-uses the GPU batch-latency table measured by the previous start instead of re-running the
batch 1..batch_max sweep (3.1 s of a 9 s warm restart on the dev GPU):

* **What is cached:** ``batch_ms`` and, since [Phase 6] (SCHEMA 2), the CPU-side costs ``cpu`` = overhead_ms,
  pose_ms, pose_batch_ms. On a hit the overhead is still measured, as a spot check against the cached value (same
  tolerances); within them every cached number is reused, so a restart admits exactly what the measured start did
  (the per-start CPU noise made the limit flap 7/8 on TensorRT). Outside them the CPU costs are re-measured and
  written back.
* **Key:** everything the table depends on (``cache_key``): the detector and pose files' sha256, the input size, the
  provider (requested + active) and its options, the scheduler mode, the GPU (name, uuid, driver: nvidia-smi)
  or the CPU, the onnxruntime and app versions, batch_max, detector calls per frame (tiling) and the cameras'
  frame size. If the GPU identity cannot be read the cache is not used (``device_identity`` -> None).
* **Spot check** on a key hit: the median of SPOT_REPEATS batch-1 runs must lie within
  [cached * FASTER_MIN, cached * SLOWER_MAX]. Asymmetric on purpose: admission keeps a 10 % HEADROOM, so a GPU
  more than 10 % slower than the cache would over-admit; a faster one only wastes capacity (30 %).
  The tolerances are constants: configuration cannot loosen them.
* **Never blocks startup:** a missing, stale (> max age), corrupt or unreadable file, or a key mismatch,
  means a full measurement (the reason is reported); a failed write is logged. Writes are atomic
  (temp file + replace), so a crash mid-write leaves the previous file or none, never half of one.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import platform
import subprocess  # nosec B404  # fixed-argv nvidia-smi query, no shell
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = 2  # [Phase 6] 2: + cpu costs and the pose model in the key (a schema-1 file is a key mismatch)
FILE_NAME = "capacity_cache.json"
SLOWER_MAX = 1.10  # spot check: at most 10 % slower than the cache (= admission HEADROOM)
FASTER_MIN = 0.70  # ... and at most 30 % faster
SPOT_REPEATS = 5
NVIDIA_SMI_TIMEOUT_S = 5.0

_sha_cache: dict[tuple[str, int, int], str] = {}


def file_sha256(path: str | Path) -> str:
    """sha256 of a file (0.12 s for the 109 MB detector), memoised per (path, size, mtime) in this process."""
    p = Path(path)
    st = p.stat()
    memo = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if memo not in _sha_cache:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _sha_cache[memo] = h.hexdigest()
    return _sha_cache[memo]


def device_identity(provider: str) -> dict[str, Any] | None:
    """The device the latencies were measured on. CUDA: name, uuid and driver of GPU 0 from nvidia-smi
    (0.06 s), None when it cannot be read (the cache is then not used). CPU: processor name + core count."""
    if provider not in ("cuda", "tensorrt"):  # CPU / OpenVINO (CPU device) measured on this processor
        return {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(), "provider": provider}
    try:
        out = subprocess.run(  # nosec B603 B607 - fixed argv
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader", "--id=0"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_S,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        logger.warning("capacity cache: GPU identity unavailable (nvidia-smi) - cache not used", exc_info=True)
        return None
    parts = [p.strip() for p in out.splitlines()[0].split(",")] if out else []
    if len(parts) != 3 or not all(parts):
        logger.warning("capacity cache: unexpected nvidia-smi output %r - cache not used", out)
        return None
    return {"gpu": parts[0], "uuid": parts[1], "driver": parts[2]}


def cache_key(
    *,
    model_path: str,
    input_hw: tuple[int, int],
    provider: str,
    active_provider: str,
    provider_options: dict[str, Any],
    iobinding: bool,
    dynamic_batch: bool,
    batch_max: int,
    calls_per_frame: int,
    frame_size: tuple[int, int],
    device: dict[str, Any],
    pose_model_path: str | None = None,
) -> dict[str, Any]:
    """Every input the batch-latency table and the cached CPU costs depend on (pre-registered list + pose model)."""
    import onnxruntime as ort

    from lifeguard import __version__

    return {
        "schema": SCHEMA,
        "app_version": __version__,
        "detector_sha256": file_sha256(model_path),
        "input_hw": [int(input_hw[0]), int(input_hw[1])],
        "provider": provider,
        "active_provider": active_provider,
        "provider_options": {str(k): str(v) for k, v in sorted(provider_options.items())},
        "iobinding": bool(iobinding),
        "dynamic_batch": bool(dynamic_batch),
        "device": device,
        "onnxruntime": ort.__version__,
        "batch_max": int(batch_max),
        "calls_per_frame": int(calls_per_frame),
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "pose_sha256": file_sha256(pose_model_path) if pose_model_path else None,
    }


def load(
    path: Path, key: dict[str, Any], max_age_h: float, now: float | None = None
) -> tuple[dict[int, float] | None, str]:
    """(batch_ms, "hit") or (None, reason): missing | corrupt | key_mismatch:<fields> | stale."""
    now = time.time() if now is None else now
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, "missing"
    except OSError as exc:
        logger.warning("capacity cache unreadable (%s) - measuring", exc)
        return None, "corrupt"
    try:
        data = json.loads(raw.decode("utf-8"))
        stored_key = data["key"]
        batch_ms = {int(b): float(ms) for b, ms in data["batch_ms"].items()}
        measured_at = float(data["measured_at"])
        if not isinstance(stored_key, dict) or 1 not in batch_ms or any(ms <= 0 for ms in batch_ms.values()):
            raise ValueError("incomplete")
    except (ValueError, KeyError, TypeError, AttributeError, UnicodeDecodeError) as exc:
        logger.warning("capacity cache corrupt (%s: %s) - measuring", type(exc).__name__, exc)
        return None, "corrupt"
    if stored_key != key:
        fields = sorted(k for k in set(key) | set(stored_key) if key.get(k) != stored_key.get(k))
        return None, "key_mismatch:" + ",".join(fields)
    if not 0 <= now - measured_at <= max_age_h * 3600.0:
        return None, "stale"
    if set(batch_ms) != set(range(1, int(key["batch_max"]) + 1)):
        return None, "corrupt"
    return batch_ms, "hit"


def load_cpu(path: Path, key: dict[str, Any]) -> dict[str, Any] | None:
    """[Phase 6] The cached CPU costs {"overhead_ms", "pose_ms" (float | None), "pose_batch_ms" {int: float}} of a
    file whose key matches (call after a `load` hit), or None when absent or invalid (the CPU costs are then
    measured)."""
    try:
        data = json.loads(path.read_bytes().decode("utf-8"))
        if data["key"] != key:
            return None
        cpu = data["cpu"]
        overhead = float(cpu["overhead_ms"])
        pose = None if cpu["pose_ms"] is None else float(cpu["pose_ms"])
        pose_batch = {int(b): float(ms) for b, ms in cpu["pose_batch_ms"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, UnicodeDecodeError):
        return None
    if overhead <= 0 or (pose is not None and pose <= 0) or any(ms <= 0 for ms in pose_batch.values()):
        return None
    return {"overhead_ms": overhead, "pose_ms": pose, "pose_batch_ms": pose_batch}


def save(
    path: Path,
    key: dict[str, Any],
    batch_ms: dict[int, float],
    now: float | None = None,
    cpu: dict[str, Any] | None = None,
) -> bool:
    """Atomic write (temp file in the same directory + replace). False (logged) on any OS error. `cpu` = the CPU
    costs (load_cpu's shape); without it a later hit re-measures them."""
    now = time.time() if now is None else now
    data: dict[str, Any] = {
        "key": key,
        "batch_ms": {str(b): ms for b, ms in sorted(batch_ms.items())},  # exact: a hit admits what a measurement would
        "measured_at": now,
    }
    if cpu is not None:
        data["cpu"] = {
            "overhead_ms": cpu["overhead_ms"],
            "pose_ms": cpu["pose_ms"],
            "pose_batch_ms": {str(b): ms for b, ms in sorted(cpu["pose_batch_ms"].items())},
        }
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as fh:
            fh.write((json.dumps(data, indent=1, sort_keys=True) + "\n").encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
        return True
    except OSError:
        logger.warning("capacity cache not written to %s", path, exc_info=True)
        with contextlib.suppress(OSError):
            tmp.unlink()
        return False


def spot_ok(spot_ms: float, cached_ms: float) -> bool:
    """The pre-registered tolerance: not more than 10 % slower, not more than 30 % faster than the cache."""
    return cached_ms * FASTER_MIN <= spot_ms <= cached_ms * SLOWER_MAX
