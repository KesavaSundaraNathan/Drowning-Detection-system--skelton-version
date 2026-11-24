"""Wrap-up for v0.2: export -> verify -> evaluate v0 / v0.1 / v0.2 on the CLEANED val -> rule (hardneg_r1_rule.md addendum) -> report.

    training\\.venv\\Scripts\\python.exe training\\tools\\post_v02.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from post_v0 import APY, ENV, TPY, TR, WD, run  # noqa: E402

M = {n: WD / "onnx" / f"dfine_l_{n}.onnx" for n in ("v0", "v01", "v02")}


def main() -> int:
    ck = next((WD / "dfine_l_v02" / n for n in ("best_stg2.pth", "best_stg1.pth", "last.pth") if (WD / "dfine_l_v02" / n).exists()), None)
    if ck is None:
        print("no v0.2 checkpoint")
        return 1
    lines = [f"# v0.2 report ({time.strftime('%Y-%m-%d %H:%M')})", "",
             "Rule: `training/reports/hardneg_r1_rule.md` addendum (written before the cleaned-val scoring). Val = `val_clean.json`.", ""]
    rc, _ = run([TPY, "training/tools/export_v0.py", "dfine", "-c", "training/configs/dfine_l_v02.yml", "--checkpoint", str(ck),
                 "--out", str(M["v02"])])
    rc2 = 1 if rc else run([APY, "training/tools/verify_v0_onnx.py", str(M["v02"])], ENV)[0]
    lines.append(f"- checkpoint `{ck.name}` -> `{M['v02']}`; export rc {rc}, decode check **{'PASS' if rc2 == 0 else 'FAIL'}**")
    if rc2:
        (TR / "reports/train_v02.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 1
    out_json = WD / "onnx" / "eval_valclean_v02.json"
    rc, out = run([APY, "training/tools/eval_v0.py", "--ann", str(TR / "data/coco/seed_r1/val_clean.json"),
                   *[x for m in M.values() for x in ("--model", str(m))], "--out", str(out_json)], ENV)
    r0, r1, r2 = json.loads(out_json.read_text())
    checks = [
        ("1 recall@0.5fppi >= v0 - 0.02", r2["recall@0.5fppi"] >= r0["recall@0.5fppi"] - 0.02),
        ("1 small_recall@0.5fppi >= v0 - 0.03", (r2["small_recall@0.5fppi"] or 0) >= (r0["small_recall@0.5fppi"] or 0) - 0.03),
        ("2 fp/frame at score 0.5 down >= 30 % vs v0", r2["fp_per_frame@s0.5"] <= 0.7 * r0["fp_per_frame@s0.5"]),
        ("3 AP50 >= v0 - 0.02", r2["AP50"] >= r0["AP50"] - 0.02),
    ]
    keys = ["recall@0.5fppi", "small_recall@0.5fppi", "fp_per_frame@s0.5", "AP50"]
    lines += ["", "## Cleaned val (50 frames, owner-cleaned)", "", *[ln for ln in out.splitlines() if ln.startswith("|") or "frames," in ln],
              "", "## Rule (v0.2 vs v0)", "", "| check | v0 | v0.1 | v0.2 | pass |", "|---|---|---|---|---|"]
    for (name, ok), k in zip(checks, keys):
        lines.append(f"| {name} | {r0[k]:.3f} | {r1[k]:.3f} | {r2[k]:.3f} | {'yes' if ok else '**no**'} |")
    verdict = "PASS rules 1-3 (rule 4 = upload replay still to check)" if all(ok for _, ok in checks) else "FAIL"
    lines += ["", f"**Verdict: {verdict}**"]
    (TR / "reports/train_v02.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(TR / "tools/notify.ps1"),
                    "-Title", "Life Guard training", "-Message", f"v0.2: {verdict}"], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
