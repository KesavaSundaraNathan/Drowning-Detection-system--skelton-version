"""Sprint evaluation: clip-level drowning verdicts on eval/sprint_v1 (owner-approved plan 2026-11-12).

The owner's complaint is the per-clip verdict ("a clear drowning clip ends up non-drowning"), so the primary measure
here is clip-level: every drowning clip must raise at least one alert, every normal clip none. Clips with an event
window also score alert-in-window and time-to-alert. Each clip runs through the harness's `run_clip` - the upload
worker's path: ClipClock time base and analysis rate, SourcePipeline, end-of-clip flush.

    python -m lifeguard.eval.sprint --clips eval/sprint_v1/clips.json --split dev --out <dir> \
        [--thresholds candidate.json] [--det-model model.onnx] [--provider tensorrt] [--compare <old report.json>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PRE_S, POST_S = 1.0, 3.0  # alert-in-window tolerance around [onset, end]


def _parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--split", default="dev", choices=("dev", "test", "all"))
    ap.add_argument("--only", default="", help="comma-separated clip ids")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", default="High", choices=("Low", "Medium", "High"))
    ap.add_argument("--thresholds", type=Path, default=None)
    ap.add_argument("--det-model", type=Path, default=None)
    ap.add_argument("--provider", default=None, choices=("auto", "tensorrt", "cuda", "openvino", "cpu"))
    ap.add_argument("--compare", type=Path, default=None, help="an earlier report.json: print a before/after table")
    return ap.parse_args(argv)


def _bootstrap(args: argparse.Namespace) -> None:
    """Thresholds and detector BEFORE anything imports the vision constants (as evaluate.py does)."""
    if args.thresholds is not None:
        os.environ["LIFEGUARD_THRESHOLDS_FILE"] = str(args.thresholds.resolve())
    from lifeguard.eval import runner
    from lifeguard.vision import config as vision_config

    base = vision_config.from_env()
    vision_config.configure(
        vision_config.VisionConfig(
            str(args.det_model.resolve()) if args.det_model else base.det_model_path,
            args.provider or base.provider,
            base.pose_model_path,
        )
    )
    runner.set_tree(runner.Tree.detect(None))


def score_clip(clip: dict[str, Any], alerts: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    """Clip-level verdict from its alerts (pure: unit-tested)."""
    out: dict[str, Any] = {
        "id": clip["id"],
        "label": clip["label"],
        "split": clip["split"],
        "duration_s": round(duration, 1),
        "alerts": alerts,
        "n_alerts": len(alerts),
    }
    if clip["label"] == "drowning":
        out["detected"] = bool(alerts)
        if "onset" in clip and "end" in clip:
            lo, hi = float(clip["onset"]) - PRE_S, float(clip["end"]) + POST_S
            hits = [a for a in alerts if lo <= a["t"] <= hi]
            out["in_window"] = bool(hits)
            out["time_to_alert_s"] = round(max(0.0, hits[0]["t"] - float(clip["onset"])), 2) if hits else None
            out["outside_window"] = len(alerts) - len(hits)
    else:
        out["false_alerts"] = len(alerts)
    return out


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    drown = [r for r in rows if r["label"] == "drowning"]
    normal = [r for r in rows if r["label"] == "normal"]
    windowed = [r for r in drown if "in_window" in r]
    tta = sorted(r["time_to_alert_s"] for r in windowed if r.get("time_to_alert_s") is not None)
    normal_h = sum(r["duration_s"] for r in normal) / 3600.0
    fa = sum(r["false_alerts"] for r in normal)
    return {
        "drowning_clips": len(drown),
        "drowning_detected": sum(r["detected"] for r in drown),
        "windowed_clips": len(windowed),
        "alert_in_window": sum(r["in_window"] for r in windowed),
        "median_time_to_alert_s": tta[len(tta) // 2] if tta else None,
        "normal_clips": len(normal),
        "normal_hours": round(normal_h, 3),
        "normal_clips_with_alert": sum(1 for r in normal if r["false_alerts"]),
        "false_alerts": fa,
        "false_alerts_per_hour": round(fa / normal_h, 1) if normal_h else None,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    from lifeguard.eval import runner

    doc = json.loads(args.clips.read_text(encoding="utf-8"))
    root = args.clips.resolve().parent.parent.parent  # eval/sprint_v1/clips.json -> repo root
    only = {c for c in args.only.split(",") if c}
    clips = [
        c for c in doc["clips"] if (args.split == "all" or c["split"] == args.split) and (not only or c["id"] in only)
    ]
    adapter = runner.make_adapter("pipeline", args.mode)
    adapter.warmup()
    rows = []
    for c in clips:
        path = root / c["video"]
        ns = SimpleNamespace(
            clip_id=c["id"],
            video_path=path,
            fps=None,
            frame_count=None,
            camera_id=None,
            water_polygons=None,
            annotated=set(),
            labelled=False,
        )
        t0 = time.perf_counter()
        run = runner.run_clip(adapter, ns)
        alerts = [
            {
                "t": round(float(a.t), 2),
                "track": int(a.track_id),
                "scenario": a.scenario,
                "ghost": bool(a.ghost),
                "gate": getattr(a, "gate", None),
            }
            for a in run.alerts
        ]
        duration = run.frames / run.fps if run.fps else 0.0
        row = score_clip(c, alerts, duration)
        row["tracks"] = len(run.track_spans)
        row["wall_s"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
        verdict = (
            ("DETECTED" if row["detected"] else "MISSED") if c["label"] == "drowning" else f"FA={row['false_alerts']}"
        )
        _say(f"{c['id']:24s} {c['label']:8s} {verdict:9s} alerts={len(alerts):3d} tracks={row['tracks']:4d}")
    return {"format": "aquaguard-sprint-report/1", "split": args.split, "rows": rows, "summary": summarise(rows)}


def compare(new: dict[str, Any], old: dict[str, Any]) -> str:
    lines = [f"{'metric':28s} {'before':>10s} {'after':>10s}"]
    for k, v in new["summary"].items():
        lines.append(f"{k:28s} {old['summary'].get(k)!s:>10s} {v!s:>10s}")
    by_old = {r["id"]: r for r in old["rows"]}
    lines.append("")
    for r in new["rows"]:
        o = by_old.get(r["id"])
        if o is None:
            continue
        key = "detected" if r["label"] == "drowning" else "false_alerts"
        if o.get(key) != r.get(key) or o.get("in_window") != r.get("in_window"):
            lines.append(
                f"  {r['id']:24s} {key}: {o.get(key)} -> {r.get(key)}"
                + (f"   in_window: {o.get('in_window')} -> {r.get('in_window')}" if "in_window" in r else "")
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    _bootstrap(args)
    report = _run(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    _say(json.dumps(report["summary"], indent=1))
    if args.compare is not None:
        _say(compare(report, json.loads(args.compare.read_text(encoding="utf-8"))))
    return 0


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
