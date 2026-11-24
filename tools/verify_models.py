"""Verify model files against backend/models/MODELS.json (sha256) - plan Phase 3 / Phase 8 install step.

    python tools/verify_models.py                 # exit 0 only if every DEFAULT model is present and intact
    python tools/verify_models.py --models-dir D  # e.g. an install's model directory

* every `defaults` model must exist and match its sha256 (else exit 1 - the server would refuse to
  start anyway, but the installer should fail first and say why);
* any other listed model that IS present must match its sha256 (a corrupt or swapped file is never
  "fine because it is optional");
* listed models that are absent and not defaults are reported, not failed (research comparisons).

[v3] The `--verify` half of the v2 baseline's scripts/setup_models.py, for the v3 manifest schema
(aquaguard-models/3). The download half is not ported: any new model download needs the product
owner's approval and a licence/lineage review first (plan), so nothing here fetches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "backend" / "models"


PENDING_EXPORT = "pending-export"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(models_dir: Path, manifest_path: Path | None = None) -> tuple[int, list[str]]:
    """(number of failures, report lines)."""
    manifest = json.loads((manifest_path or models_dir / "MODELS.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "aquaguard-models/3":
        return 1, ["FAIL MODELS.json: schema must be 'aquaguard-models/3'"]
    defaults = set(manifest.get("defaults", {}).values())
    bad, lines = 0, []
    for m in manifest.get("models", []):
        name = m["file"]
        p = models_dir / name
        role = "default" if name in defaults else "optional"
        if not p.is_file():
            if name in defaults:
                bad += 1
                lines.append(f"FAIL {name}: default model missing")
            else:
                lines.append(f"ABSENT {name} ({role})")
            continue
        if m["sha256"] == PENDING_EXPORT:
            bad += 1
            lines.append(f"FAIL {name}: sha256 not registered yet - run tools/export_rtmpose.py --register ({role})")
            continue
        got = sha256(p)
        if got != m["sha256"]:
            bad += 1
            lines.append(f"FAIL {name}: sha256 {got} != {m['sha256']} ({role})")
        else:
            lines.append(f"OK   {name} ({role})")
    for name in defaults - {m["file"] for m in manifest.get("models", [])}:
        bad += 1
        lines.append(f"FAIL {name}: listed in defaults but not in models")
    return bad, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--manifest", type=Path, default=None, help="default: <models-dir>/MODELS.json")
    a = ap.parse_args(argv)
    bad, lines = verify(a.models_dir, a.manifest)
    for line in lines:
        print(line)
    print(f"verify_models: {'OK' if not bad else f'{bad} failure(s)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
