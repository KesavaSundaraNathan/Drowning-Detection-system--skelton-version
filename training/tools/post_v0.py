"""Phase 6 wrap-up job (queued after the v0 training jobs): export -> verify -> evaluate -> report -> notify.

For D-FINE-L and RTMDet-L v0: pick the best checkpoint (D-FINE best_stg2 > best_stg1 > last; RTMDet best_coco_bbox_mAP_*),
export to ONNX (export_v0.py, training venv), stamp + check it with Life Guard's decoder (verify_v0_onnx.py, serving
venv), then evaluate on the REVIEWED val split against the app's current detectors (eval_v0.py) and write
training\\reports\\train_v0.md. Nothing is registered in backend\\models\\MODELS.json (owner's call).

    training\\.venv\\Scripts\\python.exe training\\tools\\post_v0.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "training"
WD = TR / "work_dirs"
TPY = str(TR / ".venv" / "Scripts" / "python.exe")
APY = str(ROOT / "venv" / "Scripts" / "python.exe")
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "backend"), "PYTHONIOENCODING": "utf-8"}
BASELINES = [ROOT / "backend/models/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c_dynbatch.onnx",
             ROOT / "backend/models/lifeguard_dfine_m_devtrain_v1_dynbatch.onnx"]


def best(kind: str, wd: Path) -> Path | None:
    if kind == "dfine":
        for n in ("best_stg2.pth", "best_stg1.pth", "last.pth"):
            if (wd / "dfine_l_v0" / n).exists():
                return wd / "dfine_l_v0" / n
        return None
    c = sorted((wd / "rtmdet_l_v0").glob("best_coco_bbox_mAP_epoch_*.pth"), key=lambda p: p.stat().st_mtime)
    return c[-1] if c else None


def run(cmd: list[str], env: dict | None = None) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(p.stdout[-3000:], p.stderr[-2000:] if p.returncode else "")
    return p.returncode, p.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=str(WD), help="folder holding dfine_l_v0 and rtmdet_l_v0 (tests use another)")
    ap.add_argument("--no-notify", action="store_true")
    ap.add_argument("--ann", default=str(TR / "data/coco/seed_r1/val.json"))
    ap.add_argument("--report", default=str(TR / "reports" / "train_v0.md"))
    a = ap.parse_args()
    wd = Path(a.work_dir)
    lines, models = [f"# v0 training report ({time.strftime('%Y-%m-%d %H:%M')})", ""], []
    for kind, cfg in (("dfine", "training/configs/dfine_l_v0.yml"), ("rtmdet", "training/configs/rtmdet_l_v0.py")):
        ck = best(kind, wd)
        if ck is None:
            lines.append(f"- {kind}: no checkpoint found, skipped")
            continue
        onnx = wd / "onnx" / f"{kind}_l_v0.onnx"
        rc, _ = run([TPY, "training/tools/export_v0.py", kind, "-c", cfg, "--checkpoint", str(ck), "--out", str(onnx)])
        rc2, out = (1, "") if rc else run([APY, "training/tools/verify_v0_onnx.py", str(onnx)], ENV)
        verdict = "PASS" if rc2 == 0 else "FAIL"
        lines.append(f"- {kind}-L v0: checkpoint `{ck}` -> `{onnx}`; export rc {rc}, "
                     f"Life Guard decode check **{verdict}**")
        if rc2 == 0:
            models.append(str(onnx))
    rc, out = run([APY, "training/tools/eval_v0.py", "--ann", a.ann,
                   *[x for m in [*map(str, BASELINES), *models] for x in ("--model", m)],
                   "--out", str(wd / "onnx" / "eval_val_v0.json")], ENV)
    lines += ["", "## Val (reviewed seed frames; never the test split)", "",
              *[ln for ln in out.splitlines() if ln.startswith("|") or "frames," in ln], "",
              "Baselines: the app's current RTMDet-m (COCO) and the dev-only D-FINE-M. Curves use no score floor.",
              "Not registered in backend\\models\\MODELS.json; the owner decides (Phase 10)."]
    Path(a.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not a.no_notify:
        subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(TR / "tools/notify.ps1"),
                       "-Title", "Life Guard training", "-Message", "v0 done: training\\reports\\train_v0.md"], check=False)
    print(json.dumps({"models": models, "eval_rc": rc}))
    return 0 if models and rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
