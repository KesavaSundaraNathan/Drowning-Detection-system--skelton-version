"""Wading rule A/B (docs/evidence/wading/rule.md, pre-registered): eval/heldout_v1 dev split, High, tensorrt, A = thresholds
as shipped (keys off), B = candidate (keys on), for the default detector and for v0. Writes docs/evidence/wading/ab.md with the
verdict of the safety and false-alert checks (the replay check is run separately).

    venv\\Scripts\\python.exe training\\tools\\wading_ab.py          # SERVING venv, repo root, GPU free
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training" / "work_dirs" / "wading"
CAND = OUT / "candidate_wading.json"
V0 = ROOT / "backend" / "models" / "lifeguard_dfine_l_pool_v0.onnx"
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "backend"), "PYTHONIOENCODING": "utf-8"}
DETS = {"rtmdet_m": None, "dfine_l_v0": V0}


def run(det: str, variant: str) -> dict:
    name = f"{det}_{variant}"
    out = OUT / name
    if not (out / "report.json").exists():
        cmd = [str(ROOT / "venv/Scripts/python.exe"), "-m", "lifeguard.eval.evaluate", "--eval-dir", "eval/heldout_v1",
               "--split", "dev", "--mode", "High", "--provider", "tensorrt", "--name", name, "--out", str(out),
               "--trace", str(out / "t.trace.jsonl.gz")]
        if variant == "B":
            cmd += ["--thresholds", str(CAND)]
        if DETS[det]:
            cmd += ["--det-model", str(DETS[det])]
        p = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (OUT / f"{name}.log").write_text(p.stdout[-6000:] + p.stderr[-4000:], encoding="utf-8")
        if p.returncode:
            raise SystemExit(f"{name} failed rc {p.returncode}; see {OUT / (name + '.log')}")
    return json.loads((out / "report.json").read_text())


def events(r: dict) -> dict:
    return {(c["clip_id"], e["id"]): e for c in r["clips"] for e in c["events"]}


def fa(r: dict) -> int:
    return sum(len(c["false_alerts"]) for c in r["clips"])


def main() -> int:
    lines = ["# Wading rule A/B (dev split, High, TensorRT)", "", "Rule: `docs/evidence/wading/rule.md` (pre-registered).", ""]
    ok_all = True
    for det in DETS:
        a, b = run(det, "A"), run(det, "B")
        ea, eb = events(a), events(b)
        lost, late = [], []
        for k, e in ea.items():
            if e["detected"]:
                f = eb.get(k)
                if f is None or not f["detected"]:
                    lost.append(k)
                elif f["time_to_alert_s"] - e["time_to_alert_s"] > 2.0:
                    late.append((k, e["time_to_alert_s"], f["time_to_alert_s"]))
        fa_a, fa_b = fa(a), fa(b)
        ha, hb = a["headline"], b["headline"]
        ok = not lost and not late and fa_b <= fa_a
        ok_all &= ok
        lines += [f"## Detector {det}", "", "| | A (off) | B (on) |", "|---|---|---|",
                  f"| event recall | {ha['event_recall']} | {hb['event_recall']} |",
                  f"| time to alert median (s) | {ha['time_to_alert_median_s']} | {hb['time_to_alert_median_s']} |",
                  f"| false alerts total | {fa_a} | {fa_b} |",
                  f"| false alerts standing in water | {ha['false_alerts_standing_in_water']} | {hb['false_alerts_standing_in_water']} |",
                  f"| false alerts normal swimming | {ha['false_alerts_normal_swimming']} | {hb['false_alerts_normal_swimming']} |",
                  f"| false alerts no person | {ha['false_alerts_no_person']} | {hb['false_alerts_no_person']} |", "",
                  f"Events A detected but B lost: {lost or 'none'}. Alerts more than 2 s later in B: {late or 'none'}.",
                  f"**{det}: {'PASS' if ok else 'FAIL'}** (safety + false alerts)", ""]
    lines.append(f"**Verdict (safety + false-alert checks): {'PASS' if ok_all else 'FAIL'}.** The replay check is run separately.")
    (ROOT / "docs/evidence/wading/ab.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
