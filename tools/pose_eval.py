"""Held-out re-evaluation after restoring pose: pose OFF (the 2.4.0 pipeline) vs pose ON (top-down RTMPose-m COCO).

    python tools/pose_eval.py --eval-dir eval\\heldout_v1 [--modes Medium High] [--splits dev test] [--provider auto]

Runs the existing harness (python -m lifeguard.eval.evaluate) once per (split, mode, arm) in its own process, on the
same detector and thresholds, with only the pose model switched (`--pose-model none` for the OFF arm). Writes
docs/phase_reports/phase12/evidence/pose_ab.json and prints the table: events caught / total, event recall against
v3's measured baseline (TEST, Medium, no zones: 1 of 5 = 0.2), false alerts, time to alert, latency.

Nothing is rounded up and nothing is tuned on TEST: decisions are taken on DEV; TEST only reports. The clips
(build/v2_baseline/Datasets, see eval/heldout_v1/annotations) and the exported pose model are on the owner's machine.
Refuses to run while the pose model's sha256 is still "pending-export" (tools/export_rtmpose.py --register).
"""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404 - this repo's own python, fixed argv, no shell
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from tools.verify_models import verify  # noqa: E402

BASELINE = {"split": "test", "mode": "Medium", "zones": False, "detected": 1, "total": 5, "recall": 0.2}
EVIDENCE = ROOT / "docs" / "phase_reports" / "phase12" / "evidence" / "pose_ab.json"
ARMS = {"pose_off": "none", "pose_on": ""}


def arm_argv(eval_dir: Path, split: str, mode: str, arm: str, out: Path, provider: str, zones: bool) -> list[str]:
    argv = [sys.executable, "-m", "lifeguard.eval.evaluate", "--eval-dir", str(eval_dir), "--split", split]
    argv += ["--mode", mode, "--name", out.name, "--out", str(out), "--provider", provider]
    if ARMS[arm]:
        argv += ["--pose-model", ARMS[arm]]
    if zones:
        argv.append("--zones")
    return argv


def summarise(report: dict[str, Any], split: str) -> dict[str, Any]:
    s = (report.get("splits") or {}).get(split) or {}
    ev = s.get("events") or {}
    fa = s.get("false_alerts") or {}
    perf = report.get("performance") or {}
    return {
        "events_detected": ev.get("detected"),
        "events_total": ev.get("total"),
        "event_recall": ev.get("recall"),
        "by_type": ev.get("by_type"),
        "time_to_alert_median_s": (ev.get("time_to_alert_s") or {}).get("median"),
        "false_alerts_per_camera_hour": fa.get("per_camera_hour"),
        "false_alerts_by_bucket": {k: v.get("alerts") for k, v in (fa.get("by_bucket") or {}).items()},
        "latency_p50_ms": (perf.get("latency") or {}).get("p50_ms"),
        "latency_p95_ms": (perf.get("latency") or {}).get("p95_ms"),
        "pose_model": ((report.get("meta") or {}).get("tree") or {}).get("pose_model"),
    }


def verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recall on TEST/Medium with pose ON against the v3 baseline (1/5). Reported, never rounded up."""
    on = next((r for r in rows if r["split"] == "test" and r["mode"] == "Medium" and r["arm"] == "pose_on"), None)
    if on is None or on["event_recall"] is None:
        return {"measured": False, "baseline": BASELINE}
    return {
        "measured": True,
        "baseline": BASELINE,
        "pose_on": f"{on['events_detected']}/{on['events_total']}",
        "recall": on["event_recall"],
        "improved": on["event_recall"] > BASELINE["recall"],
        "note": "TEST has 5 events / 0.069 camera-hours: release-gate criterion 8 (>= 20 events, >= 1 h) cannot "
        "pass on heldout_v1 whatever the result",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "eval" / "heldout_v1")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"], choices=("dev", "test"))
    ap.add_argument("--modes", nargs="+", default=["Medium", "High"], choices=("Low", "Medium", "High"))
    ap.add_argument("--provider", default="auto", choices=("auto", "tensorrt", "cuda", "openvino", "cpu"))
    ap.add_argument("--zones", action="store_true", help="use the annotated water polygons as zones")
    ap.add_argument("--work-dir", type=Path, default=ROOT / "build" / "pose_eval")
    ap.add_argument("--evidence", type=Path, default=EVIDENCE)
    args = ap.parse_args(argv)
    # every run below starts with cwd=backend: a relative path typed in the owner's folder must not be re-read there
    args.eval_dir = args.eval_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    args.evidence = args.evidence.resolve()
    if not (args.eval_dir / "manifest.json").is_file():
        print(f"FAIL: no manifest.json in {args.eval_dir} - pass the held-out folder, e.g. --eval-dir eval\\heldout_v1")
        return 1

    bad, lines = verify(ROOT / "backend" / "models")
    if bad:
        print("\n".join(lines))
        print("FAIL: models not verified - export and register the pose model first (tools/export_rtmpose.py)")
        return 1
    rows: list[dict[str, Any]] = []
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    for split in args.splits:
        for mode in args.modes:
            for arm in ARMS:
                out = args.work_dir / f"{stamp}_{split}_{mode}_{arm}"
                cmd = arm_argv(args.eval_dir, split, mode, arm, out, args.provider, args.zones)
                print(f"== {split} {mode} {arm}: {' '.join(cmd[2:])}", flush=True)
                subprocess.run(cmd, cwd=ROOT / "backend", check=True)  # noqa: S603  # nosec B603
                report = json.loads((out / "report.json").read_text(encoding="utf-8"))
                row = {"split": split, "mode": mode, "arm": arm, "report": str(out), **summarise(report, split)}
                if row["events_total"] is None:
                    print(f"FAIL: nothing measured for {split} {mode} {arm} - see {out / 'report.md'}")
                    return 1
                rows.append(row)
    doc = {"format": "aquaguard-pose-ab/1", "created_at": stamp, "args": vars(args) | {"eval_dir": str(args.eval_dir)}}
    doc["rows"] = rows
    doc["verdict"] = verdict(rows)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n{'split':6} {'mode':7} {'arm':9} {'events':>8} {'recall':>7} {'FA/h':>6} {'TTA s':>6}")
    for r in rows:
        ev = f"{r['events_detected']}/{r['events_total']}"
        print(
            f"{r['split']:6} {r['mode']:7} {r['arm']:9} {ev:>8} {r['event_recall']!s:>7} "
            f"{r['false_alerts_per_camera_hour']!s:>6} {r['time_to_alert_median_s']!s:>6}"
        )
    print(f"\nverdict: {json.dumps(doc['verdict'])}\nevidence: {args.evidence}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
