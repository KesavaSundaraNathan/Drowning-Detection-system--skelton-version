"""Wrap-up for the hard-negative fine-tune (queued after train_dfine_l_v01): export -> verify -> evaluate -> rule -> report.

Applies rules 1-3 of training\\reports\\hardneg_r1_rule.md (written before training) to v0 vs v0.1 on the reviewed val
split; rule 4 (the owner's upload replay) is checked afterwards in the app. Writes training\\reports\\train_v01.md.

    training\\.venv\\Scripts\\python.exe training\\tools\\post_v01.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from post_v0 import APY, ENV, TPY, TR, WD, run  # noqa: E402

V0 = WD / "onnx" / "dfine_l_v0.onnx"
V01 = WD / "onnx" / "dfine_l_v01.onnx"


def main() -> int:
    ck = next((WD / "dfine_l_v01" / n for n in ("best_stg2.pth", "best_stg1.pth", "last.pth") if (WD / "dfine_l_v01" / n).exists()), None)
    if ck is None:
        print("no v0.1 checkpoint")
        return 1
    lines = [f"# v0.1 (hard negatives) report ({time.strftime('%Y-%m-%d %H:%M')})", "",
             "Decision rule: `training/reports/hardneg_r1_rule.md` (written before training).", ""]
    rc, _ = run([TPY, "training/tools/export_v0.py", "dfine", "-c", "training/configs/dfine_l_v01.yml", "--checkpoint", str(ck),
                 "--out", str(V01)])
    rc2 = 1 if rc else run([APY, "training/tools/verify_v0_onnx.py", str(V01)], ENV)[0]
    lines.append(f"- checkpoint `{ck.name}` -> `{V01}`; export rc {rc}, Life Guard decode check **{'PASS' if rc2 == 0 else 'FAIL'}**")
    if rc2:
        Path(TR / "reports/train_v01.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 1
    out_json = WD / "onnx" / "eval_val_v01.json"
    rc, out = run([APY, "training/tools/eval_v0.py", "--ann", str(TR / "data/coco/seed_r1/val.json"),
                   "--model", str(V0), "--model", str(V01), "--out", str(out_json)], ENV)
    r0, r1 = json.loads(out_json.read_text())
    checks = [
        ("1 safety: recall@0.5fppi >= v0 - 0.02", r1["recall@0.5fppi"] >= r0["recall@0.5fppi"] - 0.02),
        ("1 safety: small_recall@0.5fppi >= v0 - 0.03", (r1["small_recall@0.5fppi"] or 0) >= (r0["small_recall@0.5fppi"] or 0) - 0.03),
        ("2 distractors: fp/frame at score 0.5 down >= 30 %", r1["fp_per_frame@s0.5"] <= 0.7 * r0["fp_per_frame@s0.5"]),
        ("3 AP50 >= 0.52", r1["AP50"] >= 0.52),
    ]
    lines += ["", "## Val (50 owner-reviewed frames, never the test split)", "",
              *[ln for ln in out.splitlines() if ln.startswith("|") or "frames," in ln], "", "## Rule", "",
              "| check | v0 | v0.1 | pass |", "|---|---|---|---|"]
    keys = ["recall@0.5fppi", "small_recall@0.5fppi", "fp_per_frame@s0.5", "AP50"]
    for (name, ok), k in zip(checks, keys):
        lines.append(f"| {name} | {r0[k]:.3f} | {r1[k]:.3f} | {'yes' if ok else '**no**'} |")
    verdict = "PASS (rules 1-3; rule 4 = upload replay still to check)" if all(ok for _, ok in checks) else "FAIL"
    if not (checks[0][1] and checks[1][1]):
        verdict = "FAIL on safety (rule 1): v0.1 rejected"
    lines += ["", f"**Verdict: {verdict}**"]
    Path(TR / "reports/train_v01.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(TR / "tools/notify.ps1"),
                    "-Title", "Life Guard training", "-Message", f"v0.1: {verdict}"], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
