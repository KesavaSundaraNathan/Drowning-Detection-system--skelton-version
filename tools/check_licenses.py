"""Commercial-licence gate (plan Phase 1; ported from v2 scripts/check_licenses.py and extended).

    python tools/check_licenses.py                      # installed env + locks + npm + models
    python tools/check_licenses.py --no-npm --no-locks  # installed env + models only (offline)
    python tools/check_licenses.py --list               # print every package and licence

Fails (exit 1) on:
  * any Python package (installed, or pinned in requirements/*.lock) or npm package whose
    licence is GPL / AGPL / SSPL / non-commercial (CC-BY-NC, "non-commercial", ...),
    unless an `OR` alternative is permissive (dual licensing, e.g. "GPL-2.0 OR MIT");
  * a banned copyleft CV package (ultralytics, boxmot, yolov5, ...) whatever its metadata says;
  * a backend/models/MODELS.json entry missing licence / commercial_ok / dev_only / data_provenance / sha256;
  * a DEFAULT model (MODELS.json "defaults") that is not commercial_ok, or is dev_only
    (trained on footage without cleared rights - docs/phase_reports/phase0/evidence/RIGHTS.csv).
Warns (exit 0) on LGPL (review linking) and on missing licence metadata (verify by hand).
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import re
import shutil
import subprocess  # nosec B404  # only a fixed-argv npx call, no shell
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "backend" / "models" / "MODELS.json"
FRONTEND = ROOT / "frontend"
LOCKS = sorted((ROOT / "requirements").glob("*.lock"))
PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"

BANNED = {
    "ultralytics": "AGPL-3.0",
    "boxmot": "AGPL-3.0",
    "yolov5": "AGPL-3.0",
    "ultralytics-thop": "AGPL-3.0",
    "yolov10": "AGPL-3.0",
}
_FAIL = re.compile(
    r"(?<![A-Za-z])(A?GPL|GNU General Public|Affero|SSPL|Server Side Public|"
    r"non[- ]?commercial|CC[- ]BY[- ]NC|BUSL|Business Source)",
    re.I,
)
_LGPL = re.compile(r"LGPL|Lesser General Public|Library General Public", re.I)
_OR = re.compile(r"\s+OR\s+|\s*/\s*", re.I)  # explicit choice between licences
_AND = re.compile(r"\s*;\s*")  # several classifiers: ALL must pass
_PROPRIETARY = re.compile(r"Proprietary|EULA|Commercial License", re.I)
# Proprietary but free-of-charge, redistributable runtimes that the project explicitly accepted:
# ONNX Runtime's CUDA execution provider is the default (decision 2026-09-24; TensorRT stays opt-in).
# Anything proprietary NOT listed here is a WARN for human review.
REVIEWED_PROPRIETARY = {
    "nvidia-cuda-runtime-cu12": "NVIDIA CUDA EULA (redistributable runtime)",
    "nvidia-cublas-cu12": "NVIDIA CUDA EULA (redistributable runtime)",
    "nvidia-cufft-cu12": "NVIDIA CUDA EULA (redistributable runtime)",
    "nvidia-cuda-nvrtc-cu12": "NVIDIA CUDA EULA (redistributable runtime)",
    "nvidia-cudnn-cu12": "NVIDIA cuDNN SLA (redistributable runtime)",
    # [v7] the CUDA 13 stack that runs on Blackwell (sm_120); same NVIDIA redistributable runtimes
    "nvidia-cuda-runtime": "NVIDIA CUDA EULA (redistributable runtime)",
    "nvidia-cublas": "NVIDIA CUDA EULA (redistributable runtime)",
    "nvidia-cufft": "NVIDIA CUDA EULA (redistributable runtime)",
    "nvidia-curand": "NVIDIA CUDA EULA (redistributable runtime)",
    "nvidia-cuda-nvrtc": "NVIDIA CUDA EULA (redistributable runtime)",
    "nvidia-nvjitlink": "NVIDIA CUDA EULA (redistributable runtime)",
    "nvidia-cudnn-cu13": "NVIDIA cuDNN SLA (redistributable runtime)",
    "tensorrt-cu13-libs": "NVIDIA TensorRT SLA (redistributable runtime; needs human sign-off, was opt-in)",
}
# Packages that publish NO licence metadata, verified by hand against the project's LICENSE file.
REVIEWED_UNLABELLED = {
    "mypy-extensions": "MIT - github.com/python/mypy_extensions/blob/master/LICENSE (verified 2026-10-07)",
    "faster-coco-eval": "Apache-2.0 - github.com/MiXaiLL76/faster_coco_eval/blob/main/LICENSE (verified 2026-10-07)",
}
_REQUIRED_MODEL_FIELDS = ("file", "role", "sha256", "licence", "commercial_ok", "dev_only", "data_provenance")


@dataclass(frozen=True)
class Finding:
    level: str  # "FAIL" | "WARN" | "NOTE" (reviewed exception)
    message: str


def _classify_one(t: str) -> str:
    """One licence term (or an `A OR B` choice): ok if ANY alternative is permissive."""
    if not t or t.upper() in {"UNKNOWN", "NONE", "UNLICENSED"} or t.upper().startswith("SEE LICENSE"):
        return "warn"
    if _PROPRIETARY.search(t):
        return "warn"
    verdicts = []
    for alt in [a.strip("() ") for a in _OR.split(t) if a.strip("() ")] or [t]:
        if _LGPL.search(alt):
            verdicts.append("warn")
        elif _FAIL.search(alt):
            verdicts.append("fail")
        else:
            verdicts.append("ok")
    if "ok" in verdicts:
        return "ok"
    return "warn" if "warn" in verdicts else "fail"


def classify(text: str) -> str:
    """'ok' | 'warn' | 'fail'. `;`-separated parts (several classifiers) must ALL pass."""
    parts = [p for p in _AND.split((text or "").strip()) if p] or [""]
    verdicts = [_classify_one(p) for p in parts]
    if "fail" in verdicts:
        return "fail"
    return "warn" if "warn" in verdicts else "ok"


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _license_text(lic_expr: str | None, lic: str | None, classifiers: list[str]) -> str:
    """Best licence statement: SPDX License-Expression > classifiers (all must pass) > License field."""
    if lic_expr and lic_expr.strip():
        return lic_expr.strip()
    cls = [c.split("::")[-1].strip() for c in classifiers if c.startswith("License")]
    cls = [c for c in cls if c != "OSI Approved"] or cls  # bare "OSI Approved" says nothing
    if cls:
        return "; ".join(cls)
    # A free-text License field is often a whole licence body: keep its first line only.
    first = lic.strip().splitlines()[0] if lic and lic.strip() else ""
    return first[:200]


def check_package(name: str, text: str, origin: str) -> list[Finding]:
    low = _norm(name.split("==", 1)[0])
    if low in BANNED:
        return [Finding("FAIL", f"{origin} {name}: {BANNED[low]} - banned copyleft CV package")]
    verdict = classify(text)
    if verdict == "warn" and low in REVIEWED_PROPRIETARY and _PROPRIETARY.search(text or ""):
        return [Finding("NOTE", f"{origin} {name}: {REVIEWED_PROPRIETARY[low]} - reviewed exception")]
    if verdict == "warn" and low in REVIEWED_UNLABELLED and not (text or "").strip():
        return [Finding("NOTE", f"{origin} {name}: no metadata; {REVIEWED_UNLABELLED[low]}")]
    if verdict == "fail":
        return [Finding("FAIL", f"{origin} {name}: {text}")]
    if verdict == "warn":
        what = (
            "LGPL - review how it is linked/distributed"
            if _LGPL.search(text or "")
            else "no usable licence metadata - verify by hand"
        )
        return [Finding("WARN", f"{origin} {name}: {text or '-'} ({what})")]
    return []


# ── Python: installed environment ─────────────────────────────────────────────
def installed_packages() -> list[tuple[str, str]]:
    rows = set()
    for d in md.distributions():
        name = d.metadata.get("Name") or "?"
        text = _license_text(
            d.metadata.get("License-Expression"), d.metadata.get("License"), d.metadata.get_all("Classifier") or []
        )
        rows.add((name, text))
    return sorted(rows, key=lambda r: r[0].lower())


# ── Python: lock files via PyPI metadata (no install needed, e.g. training.lock) ──
def lock_pins(path: Path) -> list[tuple[str, str]]:
    pins = re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)", path.read_text(encoding="utf-8"), re.M)
    return [(n, v) for n, v in pins]


def pypi_license(name: str, version: str, timeout: float = 20.0) -> str:
    base = version.split("+", 1)[0]  # torch 2.5.1+cu121 -> PyPI 2.5.1
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or not re.fullmatch(r"[A-Za-z0-9._!-]+", base):
        raise ValueError(f"refusing to build a PyPI URL from {name!r}=={base!r}")
    url = PYPI_JSON.format(name=name, version=base)
    if not url.startswith("https://pypi.org/pypi/"):
        raise ValueError(f"refusing non-PyPI URL {url}")
    # scheme and host are fixed above; name/version are validated identifiers
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310  # nosec B310
        info = json.load(r)["info"]
    return _license_text(info.get("license_expression"), info.get("license"), info.get("classifiers") or [])


# ── npm ────────────────────────────────────────────────────────────────────────
def npm_packages(frontend: Path) -> list[tuple[str, str]]:
    npx = shutil.which("npx.cmd") or shutil.which("npx")
    if not npx:
        raise RuntimeError("npx not found - Node.js is required for the npm licence check")
    # npx resolves --no-install packages from the CURRENT directory's node_modules: run inside frontend/.
    out = subprocess.run(  # noqa: S603  # nosec B603  # fixed argv, no shell
        [npx, "--no-install", "license-checker-rseidelsohn", "--json", "--start", "."],
        cwd=frontend,
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
    )
    if out.returncode != 0:
        raise RuntimeError(f"license-checker-rseidelsohn exit {out.returncode}: {out.stderr.strip()[-300:]}")
    data = json.loads(out.stdout)
    root = json.loads((frontend / "package.json").read_text(encoding="utf-8"))
    own = f"{root.get('name')}@{root.get('version')}"
    rows = []
    for pkg, meta in data.items():
        if pkg == own:
            continue
        lic = meta.get("licenses", "")
        rows.append((pkg, " OR ".join(lic) if isinstance(lic, list) else str(lic)))
    return sorted(rows)


# ── models ─────────────────────────────────────────────────────────────────────
def check_manifest(manifest: dict) -> list[Finding]:
    out: list[Finding] = []
    if manifest.get("schema") != "aquaguard-models/3":
        out.append(Finding("FAIL", "MODELS.json: schema must be 'aquaguard-models/3'"))
    models = {m.get("file"): m for m in manifest.get("models", [])}
    for m in manifest.get("models", []):
        missing = [f for f in _REQUIRED_MODEL_FIELDS if f not in m]
        if missing:
            out.append(Finding("FAIL", f"MODELS.json {m.get('file', '?')}: missing {', '.join(missing)}"))
        elif classify(str(m["licence"])) == "fail":
            out.append(Finding("FAIL", f"MODELS.json {m['file']}: code/weights licence {m['licence']}"))
    defaults = manifest.get("defaults", {})
    for role, fname in defaults.items():
        m = models.get(fname)
        if m is None:
            out.append(Finding("FAIL", f"default {role} model {fname} is not listed in MODELS.json"))
            continue
        if m.get("commercial_ok") is not True:
            out.append(
                Finding("FAIL", f"default {role} model {fname} is not commercial_ok ({m.get('data_provenance', '?')})")
            )
        if m.get("dev_only") is not False:
            out.append(
                Finding(
                    "FAIL",
                    f"default {role} model {fname} is dev_only - trained on footage "
                    "without cleared rights; it may never be a shipped default",
                )
            )
    if "detector" not in defaults:
        out.append(Finding("WARN", "no default detector configured yet (expected until Phase 3)"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print every package and its licence")
    ap.add_argument("--no-locks", action="store_true", help="skip the PyPI lookup of requirements/*.lock")
    ap.add_argument("--no-npm", action="store_true", help="skip the frontend npm dependencies")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    findings: list[Finding] = []
    counts: dict[str, int] = {}

    rows = installed_packages()
    counts["python-installed"] = len(rows)
    for name, text in rows:
        findings += check_package(name, text, "installed")
        if a.list:
            print(f"  py   {name:34s} {text}")

    if not a.no_locks:
        origin: dict[tuple[str, str], str] = {}
        for lock in LOCKS:
            for name, ver in lock_pins(lock):
                origin.setdefault((name, ver), lock.name)

        def _lookup(key: tuple[str, str]) -> tuple[tuple[str, str], str | None, str | None]:
            try:
                return key, pypi_license(*key), None
            except Exception as exc:  # noqa: BLE001 - reported as a gate failure below, never swallowed
                return key, None, str(exc)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = sorted(pool.map(_lookup, origin))
        for (name, ver), text, err in results:
            if err is not None or text is None:
                findings.append(Finding("FAIL", f"{origin[(name, ver)]} {name}=={ver}: licence lookup failed ({err})"))
                continue
            findings += check_package(f"{name}=={ver}", text, origin[(name, ver)])
            if a.list:
                print(f"  lock {name + '==' + ver:34s} {text}")
        counts["python-locked"] = len(origin)

    if not a.no_npm:
        if not (FRONTEND / "node_modules").exists():
            findings.append(Finding("FAIL", "frontend/node_modules missing - run `npm ci` before the gate"))
        else:
            try:
                npm_rows = npm_packages(FRONTEND)
            except Exception as exc:  # noqa: BLE001 - a broken npm check is a gate failure, not a pass
                findings.append(Finding("FAIL", f"npm licence check could not run: {exc}"))
                npm_rows = []
            counts["npm"] = len(npm_rows)
            for name, text in npm_rows:
                findings += check_package(name, text, "npm")
                if a.list:
                    print(f"  npm  {name:34s} {text}")

    manifest = json.loads(a.manifest.read_text(encoding="utf-8")) if a.manifest.exists() else {}
    if not manifest:
        findings.append(Finding("FAIL", f"{a.manifest} missing or empty"))
    else:
        findings += check_manifest(manifest)
        counts["models"] = len(manifest.get("models", []))

    order = {"NOTE": 0, "WARN": 1, "FAIL": 2}
    for f in sorted(findings, key=lambda f: (order[f.level], f.message)):
        print(f"{f.level}  {f.message}")
    fails = [f for f in findings if f.level == "FAIL"]
    print(f"licence gate {counts}: {'OK' if not fails else f'{len(fails)} violation(s)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
