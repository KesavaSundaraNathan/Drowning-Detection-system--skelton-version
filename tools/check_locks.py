"""Lock-file policy gate (plan Phase 1; audit #4 and #8).

    python tools/check_locks.py          # exit 1 on any violation
    python tools/check_locks.py --env    # also: this venv == dev.lock exactly

Rules
  1. Every requirement in every *.lock is pinned with `==` and carries at least one
     `--hash=sha256:` (so `pip install --require-hashes` can install it).
  2. serving.lock contains NO torch/torchvision/opencv-contrib/GUI opencv-python
     (inference is ONNX Runtime only; OpenCV is headless).
  3. serving.lock DOES contain onnxruntime-gpu and the NVIDIA CUDA/cuDNN runtime
     wheels, so a clean install can never fall back to CPU for lack of DLLs.
  4. dev.lock pins every serving package to the SAME version as serving.lock
     (tests run against exactly what production runs).
  5. training.lock is the only lock allowed to contain torch.
  6. (--env) the running interpreter contains EXACTLY dev.lock - no extras, no version drift.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQ = ROOT / "requirements"

SERVING_FORBIDDEN = {
    "torch",
    "torchvision",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
    "opencv-python",
    "ultralytics",
    "boxmot",
}
SERVING_REQUIRED = {
    "onnxruntime-gpu",
    "nvidia-cuda-runtime",  # [v7] CUDA 13 wheels (Blackwell); were the -cu12 builds
    "nvidia-cublas",
    "nvidia-cufft",
    "nvidia-cudnn-cu13",
    "opencv-python-headless",
}
_REQ_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(==|>=|<=|~=|!=|>|<)?\s*([^\s;\\]*)")


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(path: Path) -> tuple[dict[str, str], list[str]]:
    """Return ({package: version}, [problems]) for a pip-compile --generate-hashes file."""
    pins: dict[str, str] = {}
    problems: list[str] = []
    # Join backslash-continued lines into one logical requirement each.
    logical: list[str] = []
    buf = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(" #", 1)[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            if buf:
                logical.append(buf)
                buf = ""
            continue
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        logical.append(buf)
        buf = ""
    if buf:
        logical.append(buf)
    for entry in logical:
        text = entry.strip()
        if not text or (text.startswith(("-", "--")) and not text.startswith("--hash")):
            continue  # index options, -c/-r lines
        m = _REQ_LINE.match(text)
        if not m:
            continue
        name, op, ver = normalise(m.group(1)), m.group(3), m.group(4)
        if op != "==" or not ver:
            problems.append(f"{path.name}: {name} is not pinned with == ({text[:60]})")
            continue
        if "--hash=sha256:" not in text:
            problems.append(f"{path.name}: {name}=={ver} has no --hash")
        pins[name] = ver
    return pins, problems


def check() -> list[str]:
    fails: list[str] = []
    locks = {p.stem: p for p in sorted(REQ.glob("*.lock"))}
    for want in ("serving", "dev", "training"):
        if want not in locks:
            fails.append(f"requirements/{want}.lock is missing")
    parsed = {k: parse_lock(p) for k, p in locks.items()}
    for _k, (_pins, problems) in parsed.items():
        fails += problems

    serving = parsed.get("serving", ({}, []))[0]
    for bad in sorted(SERVING_FORBIDDEN & set(serving)):
        fails.append(f"serving.lock contains forbidden package {bad}")
    for need in sorted(SERVING_REQUIRED - set(serving)):
        fails.append(f"serving.lock is missing required package {need}")

    dev = parsed.get("dev", ({}, []))[0]
    for name, ver in sorted(serving.items()):
        if dev.get(name) != ver:
            fails.append(f"dev.lock {name}=={dev.get(name)} differs from serving.lock {name}=={ver}")

    for k, (pins, _p) in parsed.items():
        if k != "training" and ("torch" in pins or "torchvision" in pins):
            fails.append(f"{k}.lock contains torch — only training.lock may")
    return fails


def check_env(lock_name: str = "dev") -> list[str]:
    """The running interpreter's packages must be EXACTLY the given lock (no extras, no drift)."""
    import importlib.metadata as md

    want, _ = parse_lock(REQ / f"{lock_name}.lock")
    have = {normalise(d.metadata["Name"]): d.version for d in md.distributions()}
    fails = [
        f"env: {n}=={v} installed but {lock_name}.lock pins {want.get(n)}"
        for n, v in sorted(have.items())
        if n in want and want[n] != v
    ]
    fails += [f"env: {n} is in {lock_name}.lock but not installed" for n in sorted(set(want) - set(have))]
    fails += [
        f"env: {n}=={v} is installed but not in {lock_name}.lock" for n, v in sorted(have.items()) if n not in want
    ]
    return fails


def main(argv: list[str] | None = None) -> int:
    env = "--env" in (argv if argv is not None else sys.argv[1:])
    fails = check() + (check_env() if env else [])
    for f in fails:
        print(f"FAIL  {f}")
    counts = {p.stem: len(parse_lock(p)[0]) for p in sorted(REQ.glob("*.lock"))}
    print(f"locks {counts}: {'OK' if not fails else f'{len(fails)} violation(s)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
