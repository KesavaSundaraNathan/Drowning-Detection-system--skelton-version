"""Vulnerability gate over every hash-locked requirement file (pip-audit), with no silent skips.

    python tools/audit_locks.py          # exit 1 on any known vulnerability or any un-audited package

pip-audit resolves advisories against PyPI. Packages pinned with a local version label
(torch==2.14.1+cu126 from the PyTorch index) are not on PyPI, so pip-audit SKIPS them and
still reports "No known vulnerabilities found". That is a fail-open hole for exactly the
packages that matter most in training (torch.load RCEs), so this wrapper re-audits every
skipped package at its public base version and fails if anything is still left un-audited.
"""

from __future__ import annotations

import json
import re
import subprocess  # nosec B404  # only fixed-argv calls to pip-audit, no shell
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCKS = sorted((ROOT / "requirements").glob("*.lock"))


def _pip_audit(args: list[str]) -> dict:
    cmd = [sys.executable, "-m", "pip_audit", *args, "--progress-spinner", "off", "-f", "json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603  # nosec B603  # fixed argv, no shell
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(f"pip-audit produced no report (exit {proc.returncode}): {proc.stderr.strip()[-500:]}")
    return json.loads(out)


def _lock_pins(lock: Path) -> dict[str, str]:
    text = lock.read_text(encoding="utf-8")
    return {
        re.sub(r"[-_.]+", "-", n).lower(): v
        for n, v in re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)", text, re.M)
    }


def audit() -> tuple[list[str], list[str], int]:
    fails: list[str] = []
    notes: list[str] = []
    audited = 0
    total = 0
    skipped: dict[str, str] = {}
    for lock in LOCKS:
        pins = _lock_pins(lock)
        total += len(pins)
        rep = _pip_audit(["-r", str(lock), "--require-hashes", "--disable-pip", "--no-deps"])
        for dep in rep.get("dependencies", []):
            if dep.get("skip_reason"):
                # pip-audit's JSON omits the version of a skipped package: take it from the lock.
                name = re.sub(r"[-_.]+", "-", dep["name"]).lower()
                ver = pins.get(name)
                if not ver:
                    fails.append(f"{lock.name}: {dep['name']} skipped and its pin is not in the lock")
                    continue
                skipped[dep["name"]] = ver
                continue
            audited += 1
            for v in dep.get("vulns", []):
                fails.append(
                    f"{lock.name}: {dep['name']}=={dep['version']} {v['id']} "
                    f"(fix: {', '.join(v.get('fix_versions') or []) or 'none published'})"
                )
    if skipped:
        base = {n: re.sub(r"\+.*$", "", v) for n, v in skipped.items() if v}
        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp) / "base.txt"
            req.write_text("".join(f"{n}=={v}\n" for n, v in base.items()), encoding="utf-8")
            rep = _pip_audit(["-r", str(req), "--no-deps", "--disable-pip"])
        for dep in rep.get("dependencies", []):
            if dep.get("skip_reason"):
                fails.append(f"{dep['name']}=={skipped.get(dep['name'])}: could not be audited ({dep['skip_reason']})")
                continue
            audited += 1
            notes.append(f"{dep['name']}=={skipped[dep['name']]} audited at its PyPI base version {dep['version']}")
            for v in dep.get("vulns", []):
                fails.append(
                    f"{dep['name']}=={skipped[dep['name']]} {v['id']} "
                    f"(fix: {', '.join(v.get('fix_versions') or []) or 'none published'})"
                )
        missing = set(base) - {d["name"] for d in rep.get("dependencies", [])}
        fails += [f"{n}=={skipped[n]}: missing from the base-version audit" for n in sorted(missing)]
    if audited != total:
        fails.append(f"accounting: {total} pins across locks but only {audited} were audited")
    return fails, notes, audited


def main() -> int:
    fails, notes, audited = audit()
    for n in notes:
        print(f"NOTE  {n}")
    for f in fails:
        print(f"FAIL  {f}")
    print(
        f"vulnerability audit over {[p.name for p in LOCKS]}: {audited} package pins audited, "
        f"{'OK' if not fails else f'{len(fails)} finding(s)'}"
    )
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
