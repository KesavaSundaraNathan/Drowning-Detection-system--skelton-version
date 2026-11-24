"""Compile / verify the hash-locked requirement files.

    python tools/lock.py                 # (re)compile serving.lock, dev.lock, training.lock
    python tools/lock.py --check         # CI: every lock is up to date with its .in file
    python tools/lock.py --upgrade-package fastapi   # bump one package everywhere

serving.lock / dev.lock: pip-tools (pip-compile). training.lock: uv (`uv pip compile`).
pip-tools 7.5.1 crashes (IndexError) and 7.6.1 downloads the ~2.5 GB wheel when hashing the
`+cu126` torch builds from the PyTorch index; uv uses the index's published sha256 and resolves in
seconds. uv runs with `--index-strategy unsafe-best-match` because its default (first index that
has a name wins) took stale mirrored copies from the PyTorch index (setuptools 78.1.0, with a
path-traversal advisory, instead of PyPI's current release). Every artifact is hash-pinned either way.

Existing pins are kept unless --upgrade-package is given, so re-running is a no-op when nothing
changed. --check relies on that: it re-resolves into a temporary copy and fails on any difference.
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import subprocess  # nosec B404  # only fixed-argv calls to pip-compile / uv, no shell
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQ = ROOT / "requirements"
PIPTOOLS_FLAGS = ["--quiet", "--generate-hashes", "--allow-unsafe", "--strip-extras", "--annotation-style=line"]
UV_FLAGS = [
    "--quiet",
    "--generate-hashes",
    "--emit-index-url",
    "--index-strategy",
    "unsafe-best-match",
    "--python-version",
    "3.12",
    "--python-platform",
    "x86_64-pc-windows-msvc",
]
ORDER = ["serving", "dev", "training"]  # dev.in constrains on serving.lock, so serving first


def _uv() -> str:
    exe = Path(sys.executable).parent / ("uv.exe" if sys.platform == "win32" else "uv")
    found = str(exe) if exe.exists() else shutil.which("uv")
    if not found:
        raise SystemExit("uv not found in this venv - install requirements/dev.lock first")
    return found


def _compile(name: str, output: Path, upgrade: list[str]) -> None:
    # Run from the repo root with RELATIVE paths: lock headers/annotations then read
    # "requirements/x.in" on every machine instead of embedding a local absolute path.
    src = f"requirements/{name}.in"
    out = output.relative_to(ROOT).as_posix() if output.is_relative_to(ROOT) else str(output)
    upgrades = [f"--upgrade-package={p}" for p in upgrade]
    if name == "training":
        cmd = [_uv(), "pip", "compile", *UV_FLAGS, *upgrades, f"--output-file={out}", src]
    else:
        cmd = [sys.executable, "-m", "piptools", "compile", *PIPTOOLS_FLAGS, *upgrades, f"--output-file={out}", src]
    subprocess.run(cmd, cwd=ROOT, check=True)  # noqa: S603  # nosec B603  # fixed argv, no shell


def _body(text: str) -> list[str]:
    # Headers record the command line (temp paths differ) and annotations name the .in path:
    # compare the requirements and hashes only.
    return [ln.rstrip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def check() -> int:
    stale = []
    for name in ORDER:
        lock = REQ / f"{name}.lock"
        if not lock.exists():
            stale.append(f"{name} (missing)")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / f"{name}.lock"
            shutil.copyfile(lock, out)  # keep existing pins, like a real re-lock
            _compile(name, out, [])
            got, want = _body(out.read_text(encoding="utf-8")), _body(lock.read_text(encoding="utf-8"))
        if got != want:
            stale.append(name)
            sys.stdout.writelines(
                difflib.unified_diff(
                    [w + "\n" for w in want],
                    [g + "\n" for g in got],
                    f"{name}.lock (committed)",
                    f"{name}.lock (re-resolved)",
                )
            )
    print(
        "locks up to date: " + ", ".join(ORDER)
        if not stale
        else f"STALE locks: {', '.join(stale)} - run python tools/lock.py"
    )
    return 1 if stale else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--upgrade-package", action="append", default=[])
    a = ap.parse_args(argv)
    if a.check:
        return check()
    for name in ORDER:
        print(f"compiling {name}.lock ...", flush=True)
        _compile(name, REQ / f"{name}.lock", a.upgrade_package)
    return 0


if __name__ == "__main__":
    sys.exit(main())
