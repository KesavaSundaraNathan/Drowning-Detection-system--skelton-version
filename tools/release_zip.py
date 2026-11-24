"""Versioned release zip (Phase 10, carried from Phase 8): what a site installs from, built and verified by CI.

    python tools/release_zip.py --out ci-artifacts/release            # build (needs frontend/dist)
    python tools/release_zip.py --verify ci-artifacts/release/lifeguard-<version>-<commit>.zip

Contents
  * every file git tracks at this tree, minus development-only paths (DEV_ONLY: phase reports, training,
    tests, Playwright specs, the dev evaluation split - the audit asked that internal material not ship);
  * frontend/dist (the prebuilt operator UI that Caddy serves; install.ps1 step 7 then needs no Node.js);
  * RELEASE_MANIFEST.json at the root: version, commit, dirty flag, the sha256 of every file in the zip, the
    tools manifest (deploy/windows/TOOLS.json: pinned Caddy / NSSM / ntfy / MediaMTX / FFmpeg) and the models
    manifest (backend/models/MODELS.json). Binaries and model weights are NOT in the zip: they are fetched
    separately and checked against those sha256s by install.ps1 (Get-Tool) and tools/verify_models.py.

The zip is reproducible: entries sorted, timestamps fixed to the commit time, no build-host paths, so the same
tree gives byte-identical zips. A <zip>.sha256 sidecar is written next to it. --verify re-hashes every entry
against the manifest and checks that nothing is missing, extra or development-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess  # nosec B404  # fixed-argv git calls
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "aquaguard-release/1"
MANIFEST = "RELEASE_MANIFEST.json"
DEV_ONLY = ("docs/phase_reports/", "training/", "tests/", "frontend/e2e/", "eval/")
REQUIRED = (
    "deploy/windows/install.ps1",
    "deploy/windows/services.ps1",
    "deploy/windows/TOOLS.json",
    "backend/models/MODELS.json",
    "requirements/serving.lock",
    "tools/verify_models.py",
    "tools/acceptance.ps1",
    "tools/start_lifeguard.ps1",
    "tools/stop_lifeguard.ps1",
    "tools/check_autostart.ps1",
    "frontend/dist/index.html",
    "docs/RUNBOOK.md",
    "docs/OPERATOR_MANUAL.md",
    "docs/SECURITY.md",
    "THIRD_PARTY_NOTICES.md",  # Apache-2.0 + OpenMMLab copyright notices for the shipped models
    "LICENSES/Apache-2.0.txt",
    "tools/build_trt_engines.py",
)


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)  # noqa: S603, S607  # nosec
    return out.stdout


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _version(root: Path) -> str:
    for line in (root / "backend" / "lifeguard" / "__init__.py").read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit("__version__ not found in backend/lifeguard/__init__.py")


def collect(root: Path) -> dict[str, bytes]:
    """Path (posix, relative) -> bytes, as committed to git (the working copy of tracked files) + frontend/dist."""
    files: dict[str, bytes] = {}
    for rel in _git(root, "ls-files", "-z").split("\0"):
        if not rel or rel.startswith(DEV_ONLY):
            continue
        path = root / rel
        if path.is_file():  # a tracked file deleted in the working copy is simply absent
            files[rel] = path.read_bytes()
    dist = root / "frontend" / "dist"
    if not (dist / "index.html").is_file():
        raise SystemExit("frontend/dist/index.html missing: run `npm run build` in frontend/ first")
    for path in sorted(dist.rglob("*")):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def build(root: Path, out_dir: Path) -> Path:
    commit = _git(root, "rev-parse", "HEAD").strip()
    dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=no").strip())
    committed = int(_git(root, "show", "-s", "--format=%ct", "HEAD").strip())
    version = _version(root)
    files = collect(root)
    manifest = {
        "format": FORMAT,
        "version": version,
        "commit": commit,
        "dirty": dirty,
        "commit_time": datetime.fromtimestamp(committed, UTC).isoformat(timespec="seconds"),
        "files": {rel: _sha256(files[rel]) for rel in sorted(files)},
        "tools": json.loads(files["deploy/windows/TOOLS.json"]),
        "models": json.loads(files["backend/models/MODELS.json"]),
        "not_included": "third-party binaries (deploy/windows/TOOLS.json) and model weights "
        "(backend/models/MODELS.json): fetch separately; both are sha256-checked at install",
    }
    stamp = datetime.fromtimestamp(max(committed, 315532800), UTC).timetuple()[:6]  # zip epoch is 1980
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"lifeguard-{version}-{commit[:7]}{'-dirty' if dirty else ''}.zip"
    target = out_dir / name
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        entries = {**files, MANIFEST: (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")}
        for rel in sorted(entries):
            info = zipfile.ZipInfo(rel, date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, entries[rel])
    (out_dir / (name + ".sha256")).write_text(f"{_sha256(target.read_bytes())}  {name}\n", encoding="ascii")
    return target


def verify(zip_path: Path) -> list[str]:
    """Problems found ([] = valid): every entry hashed against the manifest, nothing missing / extra / dev-only."""
    problems: list[str] = []
    side = zip_path.with_name(zip_path.name + ".sha256")
    if side.is_file() and side.read_text(encoding="ascii").split()[0] != _sha256(zip_path.read_bytes()):
        problems.append(f"{side.name} does not match the zip")
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        if MANIFEST not in names:
            return [*problems, f"{MANIFEST} missing"]
        m = json.loads(zf.read(MANIFEST))
        if m.get("format") != FORMAT:
            problems.append(f"unknown format {m.get('format')!r}")
        listed: dict[str, str] = m.get("files", {})
        for rel in sorted(names - {MANIFEST} - set(listed)):
            problems.append(f"extra file not in the manifest: {rel}")
        for rel, digest in sorted(listed.items()):
            if rel not in names:
                problems.append(f"missing: {rel}")
            elif _sha256(zf.read(rel)) != digest:
                problems.append(f"sha256 mismatch: {rel}")
            if rel.startswith(DEV_ONLY):
                problems.append(f"development-only file shipped: {rel}")
        for rel in REQUIRED:
            if rel not in listed:
                problems.append(f"required file missing: {rel}")
        if m.get("tools") is None or m.get("models") is None:
            problems.append("tools / models manifest missing from the release manifest")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "ci-artifacts" / "release")
    ap.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    ap.add_argument("--verify", type=Path, default=None, help="verify an existing release zip instead of building")
    a = ap.parse_args(argv)
    if a.verify is not None:
        problems = verify(a.verify)
        for p in problems:
            sys.stdout.write(f"  FAIL  {p}\n")
        sys.stdout.write(f"{a.verify.name}: {'INVALID' if problems else 'OK'}\n")
        return 1 if problems else 0
    z = build(a.root, a.out)
    problems = verify(z)
    with zipfile.ZipFile(z) as zf:
        m = json.loads(zf.read(MANIFEST))
    sys.stdout.write(
        f"{z.name}: {len(m['files'])} files, {z.stat().st_size / 1e6:.1f} MB, commit {m['commit'][:12]}"
        f"{' (DIRTY tree)' if m['dirty'] else ''}; verify {'OK' if not problems else 'FAILED'}\n"
    )
    for p in problems:
        sys.stdout.write(f"  FAIL  {p}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
