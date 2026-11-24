"""Decision-level comparison of two evaluation reports (report.json).

    python -m lifeguard.eval.compare <reference report.json> <candidate report.json>

[v3] Ported from docs/phase_reports/phase0/compare_reports.py (the Phase 0 exit check, frozen there
byte-exact as evidence) under the full lint / type gates; logic unchanged.

Two classes of field:

* DETERMINISTIC — everything the pipeline decides (detections → tracks → alerts,
  labelled-split recall/precision/IDF1/event recall, per-clip alert diagnostics,
  the seeded synthetic behaviour suite). These must be EXACTLY equal.
* TIMING — wall-clock measurements (latency, fps, real-time factor, cold start,
  and the whole live-stream section, whose analysed fraction depends on how fast
  the GPU kept up). These vary with GPU temperature and load, so they are
  reported side by side, not asserted.

Exit code 0 = every deterministic field identical; 1 = at least one differs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Field names that are wall-clock measurements wherever they appear.
TIMING_KEYS = {
    "latency",
    "stages",
    "decode",
    "pipeline_fps",
    "realtime_factor",
    "wall_s",
    "first_result_s",
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "max_ms",
    "p99_over_p50",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "cold_start_to_first_frame_s",
    "upload_realtime_factor",
    "live_analysed_fraction",
    "warmup_s",
    "elapsed_s",
}
# Whole top-level sections that are run metadata or timing-only.
TIMING_SECTIONS = {"meta", "performance", "startup", "live", "upload_path"}
# Targets that are derived from timing fields.
TIMING_TARGETS = {"p99_le_2x_p50", "cold_start_lt_5s", "upload_ge_realtime", "live_every_frame"}


def diff(a: Any, b: Any, path: str, out: list[tuple[str, Any, Any]], timing: list[tuple[str, Any, Any]]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else k
            if k in TIMING_KEYS:
                timing.append((p, a.get(k), b.get(k)))
                continue
            if k not in a or k not in b:
                out.append((p, a.get(k, "<missing>"), b.get(k, "<missing>")))
                continue
            diff(a[k], b[k], p, out, timing)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path + ".<len>", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, timing)
    elif a != b:
        out.append((path, a, b))


def compare(
    orig: dict[str, Any], rerun: dict[str, Any]
) -> tuple[list[tuple[str, Any, Any]], list[tuple[str, Any, Any]], int]:
    det: list[tuple[str, Any, Any]] = []
    timing: list[tuple[str, Any, Any]] = []
    checked = 0
    for section in sorted(set(orig) | set(rerun)):
        if section in TIMING_SECTIONS:
            timing.append((section, "<section>", "<section>"))
            continue
        if section == "targets":
            for name in sorted(set(orig[section]) | set(rerun[section])):
                if name in TIMING_TARGETS:
                    timing.append(
                        (
                            f"targets.{name}",
                            orig[section].get(name, {}).get("value"),
                            rerun[section].get(name, {}).get("value"),
                        )
                    )
                else:
                    diff(orig[section].get(name), rerun[section].get(name), f"targets.{name}", det, timing)
                    checked += 1
            continue
        if section not in orig or section not in rerun:
            det.append((section, section in orig, section in rerun))
            continue
        diff(orig[section], rerun[section], section, det, timing)
        checked += 1
    return det, timing, checked


def count_leaves(x: Any, skip_timing: bool = True) -> int:
    if isinstance(x, dict):
        return sum(count_leaves(v) for k, v in x.items() if not (skip_timing and k in TIMING_KEYS))
    if isinstance(x, list):
        return sum(count_leaves(v) for v in x)
    return 1


def main() -> int:
    orig = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    rerun = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    det, timing, _ = compare(orig, rerun)
    leaves = sum(count_leaves(orig[s]) for s in orig if s not in TIMING_SECTIONS)
    print(f"original: {sys.argv[1]}\nrerun:    {sys.argv[2]}")
    print(f"deterministic leaf values compared: {leaves}")
    if det:
        print(f"DETERMINISTIC DIFFERENCES: {len(det)}")
        for p, x, y in det[:200]:
            print(f"  {p}: {x!r} -> {y!r}")
    else:
        print("DETERMINISTIC DIFFERENCES: 0  (every decision-level field identical)")
    print("\nheadline (original -> rerun):")
    for k in orig.get("headline", {}):
        x, y = orig["headline"][k], rerun.get("headline", {}).get(k)
        tag = "timing" if k in TIMING_KEYS else ("SAME" if x == y else "DIFF")
        print(f"  {k:38} {x!s:>10} -> {y!s:<10} [{tag}]")
    return 1 if det else 0


if __name__ == "__main__":
    sys.exit(main())
