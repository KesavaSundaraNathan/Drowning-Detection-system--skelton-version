"""Frame-by-frame parity of two harness traces (plan Phase 3 exit: boxes, ids and alerts match).

    python -m lifeguard.eval.trace <reference.trace.jsonl.gz> <candidate.trace.jsonl.gz> [--box-tol 1 --conf-tol 1e-4]

A trace (lifeguard.eval.runner.run_clip(trace=...)) holds, for every frame of every clip, every
reported person [id, x1, y1, x2, y2, conf, state, zone_state, in_water] and every alert
[id, scenario, ghost, first, x1, y1, x2, y2].

EXACT: the clips and frames present, each frame's set of track ids (tentative boxes, id -1, are
paired in box order), every person's state / zone state / in-water flag, and every alert's track,
scenario, ghost and first flags.
TOLERANCE: box coordinates (pixels; boxes are integer-clipped, so a float difference upstream can
move a rounding by 1 px) and detector confidence. The output counts how many values were EXACTLY
equal, so "within tolerance" is always quantified. Exit 0 = parity, 1 = any difference.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceDiff:
    frames: int = 0
    people: int = 0
    alerts: int = 0
    boxes_exact: int = 0
    conf_exact: int = 0
    max_box_delta: float = 0.0
    max_conf_delta: float = 0.0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def read(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _people_by_id(people: list[list[Any]]) -> dict[Any, list[Any]]:
    out: dict[Any, list[Any]] = {}
    tentative = sorted((p for p in people if p[0] < 0), key=lambda p: (p[1], p[2], p[3], p[4]))
    for i, p in enumerate(tentative):
        out[("tentative", i)] = p
    for p in people:
        if p[0] >= 0:
            if p[0] in out:
                raise ValueError(f"duplicate track id {p[0]} in one frame")
            out[p[0]] = p
    return out


def compare(ref: Path, cand: Path, box_tol: float = 1.0, conf_tol: float = 1e-4, max_problems: int = 200) -> TraceDiff:
    d = TraceDiff()

    def problem(msg: str) -> None:
        if len(d.problems) < max_problems:
            d.problems.append(msg)
        elif len(d.problems) == max_problems:
            d.problems.append("... (more differences not listed)")

    a_iter, b_iter = read(ref), read(cand)
    while True:
        a, b = next(a_iter, None), next(b_iter, None)
        if a is None and b is None:
            break
        if a is None or b is None:
            problem(f"trace length differs after {d.frames} frames ({'reference' if a is None else 'candidate'} ended)")
            break
        where = f"{a.get('clip')}#{a.get('f')}"
        if (a.get("clip"), a.get("f")) != (b.get("clip"), b.get("f")):
            problem(f"frame order differs: {where} vs {b.get('clip')}#{b.get('f')}")
            break
        d.frames += 1
        pa, pb = _people_by_id(a["people"]), _people_by_id(b["people"])
        if set(pa) != set(pb):
            problem(f"{where}: track ids {sorted(map(str, pa))} vs {sorted(map(str, pb))}")
        for k in sorted(set(pa) & set(pb), key=str):
            x, y = pa[k], pb[k]
            d.people += 1
            deltas = [abs(float(x[i]) - float(y[i])) for i in range(1, 5)]
            d.max_box_delta = max(d.max_box_delta, *deltas)
            d.boxes_exact += all(v == 0.0 for v in deltas)
            if max(deltas) > box_tol:
                problem(f"{where} id {k}: box {x[1:5]} vs {y[1:5]}")
            dc = abs(float(x[5]) - float(y[5]))
            d.max_conf_delta = max(d.max_conf_delta, dc)
            d.conf_exact += dc == 0.0
            if dc > conf_tol:
                problem(f"{where} id {k}: conf {x[5]} vs {y[5]}")
            if x[6:] != y[6:]:
                problem(f"{where} id {k}: state/zone/in_water {x[6:]} vs {y[6:]}")
        aa = sorted(a["alerts"], key=lambda r: (r[0], r[1]))
        ab = sorted(b["alerts"], key=lambda r: (r[0], r[1]))
        d.alerts += len(aa)
        if [r[:4] for r in aa] != [r[:4] for r in ab]:
            problem(f"{where}: alerts {[r[:4] for r in aa]} vs {[r[:4] for r in ab]}")
        else:
            for ra, rb in zip(aa, ab, strict=True):
                if max(abs(float(ra[i]) - float(rb[i])) for i in range(4, 8)) > box_tol:
                    problem(f"{where}: alert box {ra[4:]} vs {rb[4:]}")
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--box-tol", type=float, default=1.0)
    ap.add_argument("--conf-tol", type=float, default=1e-4)
    args = ap.parse_args(argv)
    d = compare(args.reference, args.candidate, args.box_tol, args.conf_tol)
    out = sys.stdout
    out.write(f"reference: {args.reference}\ncandidate: {args.candidate}\n")
    out.write(f"frames compared: {d.frames}\nperson observations compared: {d.people}\nalerts compared: {d.alerts}\n")
    if d.people:
        out.write(
            f"boxes bit-identical: {d.boxes_exact}/{d.people}  (max |delta| {d.max_box_delta:g} px, "
            f"tolerance {args.box_tol:g})\n"
        )
        out.write(
            f"confidences bit-identical: {d.conf_exact}/{d.people}  (max |delta| {d.max_conf_delta:.3g}, "
            f"tolerance {args.conf_tol:g})\n"
        )
    out.write(f"DIFFERENCES: {len(d.problems)}\n")
    for p in d.problems:
        out.write(f"  {p}\n")
    return 0 if d.ok else 1


if __name__ == "__main__":
    sys.exit(main())
