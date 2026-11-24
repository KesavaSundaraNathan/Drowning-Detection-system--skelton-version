"""Site session report (Phase 11 WP6; rows and lever rules pre-registered in eval/site/levers_phase11.json "gates_wp6").

    python -m lifeguard.eval.site_report --session <DataDir>/site_session/<id> [--stage final --labels <dir>]

PROVISIONAL (on the day) computes every row its inputs allow from the drill log, the checks and the replay traces;
FINAL adds the rows that need the 1 fps box labels and the alert review. A row whose inputs are missing is PENDING -
never PASS. Times: drill marks are server wall clock; replay times are session time t = wall - origin_wall.

Performer identification (D5 / D5b, no boxes needed): the performer is a person (global id) seen in the water in the
3 s before the 'under' mark and NOT seen from 1.5 s after it until 'surfaced' (the rescuer and a D5b passer stay
visible). Only a SUBMERSION for such a person counts - a SUBMERSION for the passer would not prove the watch held.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
import subprocess  # nosec B404  # fixed-argv ffmpeg clip cutting, no shell
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PASS, FAIL, PENDING, INFO = "PASS", "FAIL", "PENDING", "INFO"
SUBMERSION = "SUBMERSION"
SUBMERSION_MAX_S = 10.0
D5_RUNS_REQUIRED = 10
HANDOFF_GAP_S = 1.0
ENTRY_WINDOW_S = 10.0
PERFORMER_BEFORE_S = 3.0
PERFORMER_GONE_AFTER_S = 1.5


@dataclass
class Row:
    id: str
    criterion: str
    pass_level: str
    verdict: str
    value: Any = None
    stage: str = "PROVISIONAL"
    detail: dict[str, Any] = field(default_factory=dict)


# ── inputs ───────────────────────────────────────────────────────────────────────────────────────────────────
def runs_of(log: dict[str, Any], *codes: str) -> list[dict[str, Any]]:
    """Completed runs of the given drills, each with {event: [wall, ...]}."""
    out = []
    for r in log.get("runs", []):
        if r["code"] in codes and r.get("outcome", "completed") == "completed":
            marks: dict[str, list[float]] = {}
            for m in r["marks"]:
                marks.setdefault(m["event"], []).append(float(m["wall"]))
            out.append({**r, "m": marks})
    return out


def read_trace(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def observations(trace: Iterable[dict[str, Any]]) -> list[tuple[float, str, int]]:
    """(t, camera, global id) of every confirmed in-water track with a global id, in time order."""
    obs = [
        (float(row["t"]), str(row["cam"]), int(tr[1]))
        for row in trace
        for tr in row["tracks"]
        if tr[0] is not None and tr[0] >= 0 and tr[1] is not None and tr[7]
    ]
    return sorted(obs)


def in_window(obs: list[tuple[float, str, int]], a: float, b: float) -> list[tuple[float, str, int]]:
    return [o for o in obs if a <= o[0] <= b]


# ── criteria (pure) ──────────────────────────────────────────────────────────────────────────────────────────
def performer_ids(obs: list[tuple[float, str, int]], under: float, surfaced: float) -> set[int]:
    before = {g for t, _, g in in_window(obs, under - PERFORMER_BEFORE_S, under)}
    later = {
        g for t, _, g in in_window(obs, under + PERFORMER_GONE_AFTER_S, max(under + PERFORMER_GONE_AFTER_S, surfaced))
    }
    return before - later


def submersion_runs(
    runs: list[dict[str, Any]], obs: list[tuple[float, str, int]], alerts: list[dict[str, Any]], origin: float
) -> list[dict[str, Any]]:
    """Per D5 / D5b run: performer ids, the first audible SUBMERSION for a performer within 10 s, its latency."""
    out = []
    for r in runs:
        under = r["m"]["under"][0] - origin
        surfaced = r["m"].get("surfaced", [under + origin + 15.0])[0] - origin
        perf = performer_ids(obs, under, surfaced)
        hits = sorted(
            a["t"]
            for a in alerts
            if a["scenario"] == SUBMERSION
            and a["audible"]
            and a["t"] is not None
            and under <= a["t"] <= under + SUBMERSION_MAX_S
            and a.get("global_id") in perf
        )
        incidents = {
            a.get("global_id")
            for a in alerts
            if a["scenario"] == SUBMERSION
            and a["t"] is not None
            and under <= a["t"] <= surfaced + 5.0
            and a.get("global_id") in perf
        }
        out.append(
            {
                "run": f"{r['code']}#{r['run']}",
                "position": r.get("position", ""),
                "performer_ids": sorted(perf),
                "detected": bool(hits),
                "latency_s": round(hits[0] - under, 2) if hits else None,
                "incidents": len(incidents),
            }
        )
    return out


def handoffs(obs: list[tuple[float, str, int]], a: float, b: float) -> dict[str, Any]:
    """Lone-swimmer window: a handoff = an observation by a camera that did not see the swimmer in the previous
    HANDOFF_GAP_S; kept = its global id equals the id the OTHER cameras last saw (gates_wp6 C2a)."""
    win = in_window(obs, a, b)
    last_seen: dict[str, float] = {}
    total = kept = 0
    latest: tuple[float, str, int] | None = None
    for t, cam, g in win:
        new_here = t - last_seen.get(cam, -1e9) > HANDOFF_GAP_S
        if new_here and latest is not None and latest[1] != cam:
            total += 1
            kept += int(latest[2] == g)
        last_seen[cam] = t
        latest = (t, cam, g)
    ids = {g for _, _, g in win}
    minutes = max(1e-9, (b - a) / 60.0)
    return {
        "handoffs": total,
        "kept": kept,
        "retention": round(kept / total, 4) if total else None,
        "distinct_ids": len(ids),
        "breaks_per_min": round((len(ids) - 1) / minutes, 3) if ids else None,
    }


def entries(obs: list[tuple[float, str, int]], marks: list[float]) -> list[float | None]:
    """D3: per entry mark, seconds to the first NEW global id in the water (None = missed within 10 s)."""
    out: list[float | None] = []
    for e in marks:
        known = {g for t, _, g in in_window(obs, e - ENTRY_WINDOW_S, e)}
        first = next((t for t, _, g in in_window(obs, e, e + ENTRY_WINDOW_S) if g not in known), None)
        out.append(round(first - e, 2) if first is not None else None)
    return out


def p95(values: list[float]) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(0.95 * (len(s) - 1)))]


def alerts_in(
    alerts: list[dict[str, Any]], windows: list[tuple[float, float]], *, audible_only: bool = True
) -> list[dict[str, Any]]:
    return [
        a
        for a in alerts
        if a["t"] is not None and (a["audible"] or not audible_only) and any(lo <= a["t"] <= hi for lo, hi in windows)
    ]


def same_camera_duplicates(trace: Iterable[dict[str, Any]]) -> int:
    """Frames where one camera shows two tracks with the same global id."""
    n = 0
    for row in trace:
        gids = [tr[1] for tr in row["tracks"] if tr[1] is not None and tr[0] is not None and tr[0] >= 0]
        n += len(gids) != len(set(gids))
    return n


# ── lever rules (pre-registered) ─────────────────────────────────────────────────────────────────────────────
def decide_target_fps(live_run: dict[str, Any] | None) -> Row:
    if not live_run:
        return Row("LEVER_target_fps", "target_analysis_fps stays 15", "all MONITORED, >= 0.99, 0 dropped", PENDING)
    ok = (
        bool(live_run.get("all_monitored"))
        and float(live_run.get("min_analysed_fraction", 0)) >= 0.99
        and int(live_run.get("dropped", 1)) == 0
    )
    return Row(
        "LEVER_target_fps",
        "target_analysis_fps stays 15 (never lowered automatically)",
        "all MONITORED, >= 0.99, 0 dropped",
        PASS if ok else FAIL,
        {k: live_run.get(k) for k in ("all_monitored", "min_analysed_fraction", "dropped", "cameras")},
        detail={}
        if ok
        else {"options": ["fewer cameras per GPU", "a stronger GPU", "cameras at a lower fps, re-validated"]},
    )


def decide_gating(off: dict[str, Any] | None, on: dict[str, Any] | None) -> Row:
    """WP2 rule (63f90c3): ON iff normal false alarms/cam-h ON < OFF, 0 drill alarms lost, D5 10/10 in both,
    first sounding alert per drill ON - OFF <= 0.5 s. Inputs None -> PENDING (the lever stays OFF)."""
    lvl = "fa ON < OFF; 0 drill alarms lost; D5 10/10 both; first alert ON-OFF <= 0.5 s"
    if not off or not on or off.get("normal_fa_per_cam_h") is None or on.get("normal_fa_per_cam_h") is None:
        return Row(
            "LEVER_RESPONSIBILITY_GATING", "RESPONSIBILITY_GATING", lvl, PENDING, "OFF (until decided)", stage="FINAL"
        )
    lost = sorted(set(off["drill_audible_ids"]) - set(on["drill_audible_ids"]))
    delay = max((on["first_alert"].get(d, 1e9) - t for d, t in off["first_alert"].items()), default=0.0)
    win = (
        on["normal_fa_per_cam_h"] < off["normal_fa_per_cam_h"]
        and not lost
        and off["d5_detected"] == on["d5_detected"] == off["d5_runs"] >= D5_RUNS_REQUIRED
        and delay <= 0.5
    )
    return Row(
        "LEVER_RESPONSIBILITY_GATING",
        "RESPONSIBILITY_GATING",
        lvl,
        PASS if win else FAIL,
        "ON" if win else "OFF",
        stage="FINAL",
        detail={
            "fa_off": off["normal_fa_per_cam_h"],
            "fa_on": on["normal_fa_per_cam_h"],
            "lost": lost,
            "worst_delay_s": delay,
        },
    )


def decide_multiview(off: dict[str, Any] | None, on: dict[str, Any] | None) -> Row:
    """WP3 site rule (e595224): needs time_to_first_detection on labels -> FINAL; the other clauses are reported."""
    lvl = (
        "TTFD median & p95 ON <= OFF (one strictly lower), breaks ON <= OFF, fa ON <= OFF, "
        "no false SUBMERSION added, dup ids 0"
    )
    if not off or not on or off.get("ttfd_median") is None or on.get("ttfd_median") is None:
        clauses = {k: (off or {}).get(k) for k in ("same_camera_duplicates",)} | {
            "on_" + k: (on or {}).get(k) for k in ("same_camera_duplicates",)
        }
        return Row(
            "LEVER_MULTIVIEW_CONFIRM",
            "MULTIVIEW_CONFIRM",
            lvl,
            PENDING,
            "OFF (until decided)",
            stage="FINAL",
            detail=clauses,
        )
    strictly = on["ttfd_median"] < off["ttfd_median"] or on["ttfd_p95"] < off["ttfd_p95"]
    win = (
        on["ttfd_median"] <= off["ttfd_median"]
        and on["ttfd_p95"] <= off["ttfd_p95"]
        and strictly
        and on["breaks_per_min"] <= off["breaks_per_min"]
        and on["normal_fa_per_cam_h"] is not None
        and off["normal_fa_per_cam_h"] is not None
        and on["normal_fa_per_cam_h"] <= off["normal_fa_per_cam_h"]
        and on["false_submersions"] <= off["false_submersions"]
        and on["same_camera_duplicates"] == 0
    )
    return Row(
        "LEVER_MULTIVIEW_CONFIRM",
        "MULTIVIEW_CONFIRM",
        lvl,
        PASS if win else FAIL,
        "ON" if win else "OFF",
        stage="FINAL",
    )


# ── one arm's drill facts ────────────────────────────────────────────────────────────────────────────────────
def arm_facts(session: Path, arm: str, log: dict[str, Any], review: dict[str, str] | None) -> dict[str, Any] | None:
    base = session / "replay" / "drills" / arm
    if not (base / "summary.json").exists():
        return None
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    origin = float(summary["origin_wall"])
    trace = read_trace(base / "trace.jsonl.gz")
    alerts = json.loads((base / "alerts.json").read_text(encoding="utf-8"))
    obs = observations(trace)
    sub = submersion_runs(runs_of(log, "D5"), obs, alerts, origin)
    d5b = submersion_runs(runs_of(log, "D5b"), obs, alerts, origin)
    drill_windows = {
        f"{r['code']}#{r['run']}": (
            min(w for v in r["m"].values() for w in v) - origin,
            max(w for v in r["m"].values() for w in v) - origin + 15.0,
        )
        for r in runs_of(log, "D1", "D2", "D3", "D4", "D5", "D5b", "D6", "D7", "D7s")
        if r["m"]
    }
    first_alert = {}
    for d, (lo, hi) in drill_windows.items():
        ts = [a["t"] for a in alerts_in(alerts, [(lo, hi)])]
        if ts:
            first_alert[d] = min(ts)
    crossing = [
        (min(v[0] for v in r["m"].values()) - origin, max(v[-1] for v in r["m"].values()) - origin)
        for r in runs_of(log, "D2", "D4", "D7")
    ]
    false_sub = [a for a in alerts_in(alerts, crossing) if a["scenario"] == SUBMERSION]
    lone = [
        h
        for r in runs_of(log, "D2")
        if "start" in r["m"] and "end" in r["m"]
        for h in [handoffs(obs, r["m"]["start"][0] - origin, r["m"]["end"][0] - origin)]
    ]
    entry_marks = [w - origin for r in runs_of(log, "D3") for w in r["m"].get("entry", [])]
    ent = entries(obs, entry_marks)
    normal = normal_rate(session, arm, review)
    d7 = [
        handoffs(obs, r["m"]["start"][0] - origin, r["m"]["end"][0] - origin)
        for r in runs_of(log, "D7")
        if "start" in r["m"] and "end" in r["m"]
    ]
    return {
        "origin": origin,
        "d5": sub,
        "d5b": d5b,
        "d7": d7,
        "d7s": submersion_runs(runs_of(log, "D7s"), obs, alerts, origin),
        "d5_runs": len(sub),
        "d5_detected": sum(s["detected"] for s in sub),
        "drill_audible_ids": sorted({str(a.get("global_id")) for a in alerts_in(alerts, list(drill_windows.values()))}),
        "first_alert": first_alert,
        "false_submersions": len(false_sub),
        "handoffs": lone,
        "entries": ent,
        "d6_false": len(
            alerts_in(
                alerts,
                [
                    (r["m"]["start"][0] - origin, r["m"]["end"][0] - origin)
                    for r in runs_of(log, "D6")
                    if "start" in r["m"] and "end" in r["m"]
                ],
            )
        ),
        "d6_camera_hours": sum(
            (r["m"]["end"][0] - r["m"]["start"][0]) for r in runs_of(log, "D6") if "start" in r["m"] and "end" in r["m"]
        )
        / 3600.0
        * max(1, len(summary.get("cameras", {}))),
        "same_camera_duplicates": same_camera_duplicates(trace),
        "normal_fa_per_cam_h": normal,
        "breaks_per_min": max((h["breaks_per_min"] or 0.0 for h in lone), default=None),
        "ttfd_median": None,  # FINAL (labels)
        "ttfd_p95": None,
    }


def normal_rate(session: Path, arm: str, review: dict[str, str] | None) -> float | None:
    """False alerts per camera-hour in the reviewed normal session (None: not recorded / not reviewed)."""
    base = session / "replay" / "normal" / arm
    if review is None or not (base / "alerts.json").exists():
        return None
    alerts = [a for a in json.loads((base / "alerts.json").read_text(encoding="utf-8")) if a["audible"]]
    keys = [alert_key("normal", a) for a in alerts]
    if any(k not in review for k in keys):
        return None  # not every alert reviewed yet
    recs = json.loads((session / "recordings.json").read_text(encoding="utf-8"))["blocks"].get("normal", [])
    cam_h = sum((r["end_wall"] - r["start_wall"]) for r in recs if "end_wall" in r) / 3600.0
    false = sum(1 for k in keys if review[k].upper() == "FALSE")
    return round(false / cam_h, 3) if cam_h > 0 else None


def alert_key(block: str, a: dict[str, Any]) -> str:
    return f"{block}|{a['cam']}|{a['t']:.2f}|{a['scenario']}|{a.get('global_id')}"


def review_sheet(session: Path, block: str = "normal") -> Path | None:
    """CSV of every audible alert of the block (OFF + GATING + MULTIVIEW, deduplicated) for the reviewer: verdict TRUE /
    FALSE; the clip column names the clip to watch."""
    rows: dict[str, dict[str, Any]] = {}
    for arm in ("OFF", "GATING", "MULTIVIEW"):
        f = session / "replay" / block / arm / "alerts.json"
        if not f.exists():
            continue
        origin = json.loads((session / "replay" / block / arm / "summary.json").read_text(encoding="utf-8"))[
            "origin_wall"
        ]
        for a in json.loads(f.read_text(encoding="utf-8")):
            if a["audible"] and a["t"] is not None:
                k = alert_key(block, a)
                r = rows.setdefault(
                    k,
                    {
                        **a,
                        "arms": [],
                        "wall": datetime.fromtimestamp(origin + a["t"], UTC).isoformat(timespec="seconds"),
                    },
                )
                r["arms"].append(arm)
    if not rows:
        return None
    out = session / "report" / f"review_{block}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["key", "arms", "camera", "wall_utc", "scenario", "global_id", "verdict"])
        for k, r in sorted(rows.items(), key=lambda kv: kv[1]["t"]):
            w.writerow([k, ";".join(r["arms"]), r["cam"], r["wall"], r["scenario"], r.get("global_id"), ""])
    return out


def load_review(session: Path, block: str = "normal") -> dict[str, str] | None:
    f = session / "report" / f"review_{block}_completed.csv"
    if not f.exists():
        return None
    with f.open(encoding="utf-8") as fh:
        return {r["key"]: r["verdict"].strip() for r in csv.DictReader(fh) if r.get("verdict", "").strip()}


# ── clips ────────────────────────────────────────────────────────────────────────────────────────────────────
def extract_clip(
    ffmpeg: Path,
    recording: Path,
    start_wall: float,
    at_wall: float,
    out: Path,
    before: float = 5.0,
    after: float = 15.0,
) -> Path | None:
    """A -c copy cut around a failing event (input seek is relative to the recording start)."""
    ss = max(0.0, at_wall - start_wall - before)
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{ss:.2f}", "-i", str(recording)]
    argv += ["-t", f"{before + after:.2f}", "-c", "copy", str(out)]
    done = subprocess.run(argv, capture_output=True, timeout=120, check=False)  # noqa: S603  # nosec B603
    return out if done.returncode == 0 and out.exists() and out.stat().st_size > 0 else None


def recording_continuity(session: Path) -> Row:
    """Every recorder restart / extra part / give-up (site/recorder.py supervision). The replay uses each camera's
    main (longest) part only, so footage in other parts is listed here, not silently left out."""
    if not (session / "recordings.json").exists():
        return Row("recording_continuity", "every camera recorded the whole block", "no camera given up", PENDING)
    reg = json.loads((session / "recordings.json").read_text(encoding="utf-8"))
    issues = [
        {
            "block": block,
            "camera": r.get("camera"),
            "restarts": r.get("restarts", 0),
            "main": r.get("file"),
            "other_parts": [
                p["file"] for p in r.get("parts", []) if p.get("file") != r.get("file") and "start_wall" in p
            ],
            "gave_up": r.get("gave_up", False),
        }
        for block, recs in reg.get("blocks", {}).items()
        for r in recs
        if r.get("restarts") or r.get("gave_up") or "start_wall" not in r
    ]
    lost = [i for i in issues if i["gave_up"]]
    return Row(
        "recording_continuity",
        "every camera recorded the whole block (restarts / extra parts not replayed)",
        "no camera given up",
        FAIL if lost else INFO,
        issues or "no restart",
    )


# ── the report ───────────────────────────────────────────────────────────────────────────────────────────────
def build(session: Path, *, stage: str = "PROVISIONAL", ffmpeg: Path | None = None) -> dict[str, Any]:
    log = (
        json.loads((session / "drill_log.json").read_text(encoding="utf-8"))
        if (session / "drill_log.json").exists()
        else {"runs": []}
    )
    checks = (
        {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (session / "checks").glob("*.json")}
        if (session / "checks").exists()
        else {}
    )
    review = load_review(session)
    off = arm_facts(session, "OFF", log, review)
    gating = arm_facts(session, "GATING", log, review)
    multiview = arm_facts(session, "MULTIVIEW", log, review)
    rows: list[Row] = []

    # C1
    rows.append(
        Row(
            "C1a_in_water_recall",
            "in-water recall (D1, D6)",
            ">= 0.95",
            PENDING,
            stage="FINAL",
            detail={"needs": "box labels"},
        )
    )
    live = checks.get("live_run")
    if live:
        ok = (
            bool(live.get("all_monitored"))
            and float(live.get("min_analysed_fraction", 0)) >= 0.99
            and int(live.get("dropped", 1)) == 0
        )
        rows.append(
            Row(
                "C1b_analysed_fraction",
                "every camera analysed at 15 fps",
                ">= 0.99, MONITORED, 0 dropped",
                PASS if ok else FAIL,
                live.get("min_analysed_fraction"),
                detail=live,
            )
        )
    else:
        rows.append(
            Row("C1b_analysed_fraction", "every camera analysed at 15 fps", ">= 0.99, MONITORED, 0 dropped", PENDING)
        )
    # C2
    if off and off["handoffs"]:
        tot = sum(h["handoffs"] for h in off["handoffs"])
        kept = sum(h["kept"] for h in off["handoffs"])
        ret = kept / tot if tot else None
        rows.append(
            Row(
                "C2a_handoff_retention",
                "global id kept per handoff (D2 lone swimmer)",
                ">= 0.95",
                PENDING if ret is None else PASS if ret >= 0.95 else FAIL,
                ret,
                detail={"handoffs": tot, "kept": kept},
            )
        )
    else:
        rows.append(Row("C2a_handoff_retention", "global id kept per handoff (D2 lone swimmer)", ">= 0.95", PENDING))
    rows.append(
        Row(
            "C2b_id_switch_in_alert",
            "id switches inside an alert",
            "0",
            PENDING,
            stage="FINAL",
            detail={"needs": "box labels"},
        )
    )
    if off and off["d5"] + off["d5b"]:
        per = [s["incidents"] for s in off["d5"] + off["d5b"] if s["detected"]]
        rows.append(
            Row(
                "C2c_incidents_per_person",
                "incidents per D5 / D5b run",
                "1",
                PASS if per and all(n == 1 for n in per) else FAIL,
                per,
            )
        )
    else:
        rows.append(Row("C2c_incidents_per_person", "incidents per D5 / D5b run", "1", PENDING))
    # C3
    if off and off["entries"]:
        got = [e for e in off["entries"] if e is not None]
        n_missed = sum(e is None for e in off["entries"])
        med = statistics.median(got) if got else None
        pp = p95(got) if got else None
        ok = med is not None and pp is not None and med <= 1.0 and pp <= 2.0 and n_missed == 0
        rows.append(
            Row(
                "C3a_time_to_first_detection",
                "entry -> first confirmed track (D3, event level)",
                "median <= 1 s, p95 <= 2 s",
                PASS if ok else FAIL,
                {"median": med, "p95": pp, "missed": n_missed},
            )
        )
    else:
        rows.append(
            Row(
                "C3a_time_to_first_detection",
                "entry -> first confirmed track (D3)",
                "median <= 1 s, p95 <= 2 s",
                PENDING,
            )
        )
    if off and off["handoffs"]:
        bpm = off["breaks_per_min"]
        ok = bpm is not None and bpm <= 0.5 and off["false_submersions"] == 0
        rows.append(
            Row(
                "C3b_track_breaks",
                "breaks per person-minute (D2) + false SUBMERSION (D2/D4/D7)",
                "<= 0.5 and 0",
                PASS if ok else FAIL,
                {"breaks_per_min": bpm, "false_submersions": off["false_submersions"]},
            )
        )
    else:
        rows.append(Row("C3b_track_breaks", "breaks per person-minute + false SUBMERSION", "<= 0.5 and 0", PENDING))
    if off and off["d5"]:
        det = off["d5_detected"]
        ok = det == off["d5_runs"] >= D5_RUNS_REQUIRED and all(
            s["latency_s"] is not None and s["latency_s"] <= SUBMERSION_MAX_S for s in off["d5"]
        )
        rows.append(
            Row(
                "C3c_submersion",
                "SUBMERSION for the performer <= 10 s (D5)",
                "10/10",
                PASS if ok else FAIL,
                f"{det}/{off['d5_runs']}",
                detail={"runs": off["d5"]},
            )
        )
    else:
        rows.append(Row("C3c_submersion", "SUBMERSION for the performer <= 10 s (D5)", "10/10", PENDING))
    if off and off["d5b"]:
        missed_runs = [s["run"] for s in off["d5b"] if not s["detected"]]
        rows.append(
            Row(
                "C3c_D5b_watch_clearing",
                "SUBMERSION for the performer with a passer-by (D5b)",
                "every run",
                PASS if not missed_runs else FAIL,
                f"{len(off['d5b']) - len(missed_runs)}/{len(off['d5b'])}",
                detail={
                    "runs": off["d5b"],
                    "finding": "WP3 safety finding confirmed on site: a pre-registered lever proposal follows"
                    if missed_runs
                    else "",
                },
            )
        )
    else:
        rows.append(Row("C3c_D5b_watch_clearing", "SUBMERSION with a passer-by (D5b)", "every run", PENDING))
    # C4
    fa = off["normal_fa_per_cam_h"] if off else None
    rows.append(
        Row(
            "C4_false_alerts",
            "false alerts per camera-hour (D6 + reviewed normal session)",
            "<= 1.0",
            PENDING if fa is None else PASS if fa <= 1.0 else FAIL,
            fa,
            stage="FINAL",
            detail={"d6_false_alerts": off["d6_false"] if off else None},
        )
    )
    # levers
    rows.append(decide_target_fps(live))
    cap = checks.get("capacity")
    if cap:
        admits = bool(cap.get("tiling_on_admits_all"))
        rows.append(
            Row(
                "LEVER_DET_TILING",
                "DET_TILING: capacity with tiling (computed) + recall rises (labels)",
                "admits all at 15 fps AND recall ON > OFF",
                PENDING if admits else FAIL,
                "off",
                stage="FINAL",
                detail=cap,
            )
        )
    rows.append(decide_gating(off, gating))
    rows.append(decide_multiview(off, multiview))
    if off:
        rows.append(
            Row(
                "GLARE_HOLD_recheck",
                "D7 glare passes + D7s submersion in glare (GLARE_HOLD OFF)",
                "reported; a missed D7s SUBMERSION keeps the lever OFF",
                INFO,
                {"d7_passes": off["d7"], "d7s_submersions": off["d7s"]},
            )
        )
    # clips for failed submersion runs
    clips: list[str] = []
    if ffmpeg is not None and off:
        recs = json.loads((session / "recordings.json").read_text(encoding="utf-8"))["blocks"].get("drills", [])
        for s in off["d5"] + off["d5b"]:
            if s["detected"]:
                continue
            run = next(r for r in runs_of(log, "D5", "D5b") if f"{r['code']}#{r['run']}" == s["run"])
            for rec in recs:
                c = extract_clip(
                    ffmpeg,
                    session / "recordings" / "drills" / rec["file"],
                    rec["start_wall"],
                    run["m"]["under"][0],
                    session / "report" / "clips" / f"{s['run'].replace('#', '_')}_{rec['camera']}.mkv",
                )
                if c is not None:
                    clips.append(c.name)
    rows.append(recording_continuity(session))
    sheet = review_sheet(session)
    rep: dict[str, Any] = {
        "stage": stage,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": [asdict(r) for r in rows],
        "clips": clips,
        "review_sheet": sheet.name if sheet else None,
        "review_done": review is not None,
    }
    if stage == "FINAL":
        for r in rep["rows"]:
            if r["verdict"] == PENDING:
                r["detail"]["final_missing"] = True
    return rep


def _cell(value: Any) -> str:
    return json.dumps(value, default=str)[:120].replace("|", "/")


def write(session: Path, rep: dict[str, Any]) -> Path:
    out = session / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "site_session_report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    lines = [
        f"# Site session report ({rep['stage']})",
        "",
        f"Generated {rep['generated_utc']}.",
        "",
        "| Row | Criterion | Pass level | Verdict | Value |",
        "|---|---|---|---|---|",
    ]
    for r in rep["rows"]:
        lines.append(f"| {r['id']} | {r['criterion']} | {r['pass_level']} | **{r['verdict']}** | {_cell(r['value'])} |")
    lines += [
        "",
        f"Review sheet: {rep['review_sheet'] or '-'} (reviewed: {rep['review_done']}).",
        f"Clips of failures: {', '.join(rep['clips']) or '-'}.",
    ]
    (out / "site_session_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out / "site_session_report.md"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--stage", default="PROVISIONAL", choices=("PROVISIONAL", "FINAL"))
    ap.add_argument("--ffmpeg", type=Path, default=None, help="pinned ffmpeg for failure clips")
    args = ap.parse_args(argv)
    rep = build(args.session, stage=args.stage, ffmpeg=args.ffmpeg)
    path = write(args.session, rep)
    sys.stdout.write(f"{path}\n")
    return 1 if any(r["verdict"] == FAIL for r in rep["rows"]) else 0


if __name__ == "__main__":
    sys.exit(main())
