"""
[v3] Ported verbatim from the v2 baseline evaluation/report.py (imports only).

evaluation/report.py — turn clip runs into report.json + report.md.

Headline numbers (`headline`) are the ones every later change is compared on:
in-water recall, precision, ID switches, event recall, time-to-alert, false
alerts per camera-hour by bucket, throughput, latency and cold start.
`compare()` renders a before/after table against a saved baseline report.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from lifeguard.eval.annotations import FALSE_ALERT_BUCKETS, ClipAnnotation
from lifeguard.eval.metrics import (
    ClipDetectionTracking,
    activity_exposure,
    entry_times,
    latency_summary,
    person_sequences_metrics,
    pool_entry_metrics,
    score_events,
)
from lifeguard.eval.runner import ClipRun

TARGETS = {
    "in_water_recall": 0.95,
    "false_alerts_normal_or_standing": 0,
    "p99_over_p50_max": 2.0,
    "cold_start_s_max": 5.0,
    "upload_realtime_factor_min": 1.0,
}


def _r(x: float | None, n: int = 4) -> float | None:
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), n)


def score_clip(clip: ClipAnnotation, run: ClipRun, iou_thr: float = 0.3) -> dict[str, Any]:
    out: dict[str, Any] = {
        "clip_id": clip.clip_id,
        "split": clip.split,
        "labelled": clip.labelled,
        "camera_id": clip.camera_id,
        "conditions": sorted(clip.conditions),
        "fps": run.fps,
        "frames": run.frames,
        "analysed_frames": run.analysed,
        "analysed_fraction": _r(run.analysed / run.frames if run.frames else None),
        "duration_s": _r(run.frames / run.fps if run.fps else None, 2),
        "wall_s": _r(run.wall_s, 2),
        "notes": run.notes,
        "latency": latency_summary(run.step_ms),
        "stages": {k: latency_summary(v) for k, v in run.stage_ms.items()},
        "decode": latency_summary(run.decode_ms),
        "pipeline_fps": _r(run.analysed / (sum(run.step_ms) / 1000.0) if run.step_ms else None, 2),
        "realtime_factor": _r((run.frames / run.fps) / run.wall_s if run.wall_s and run.fps else None, 3),
        "first_result_s": _r(run.first_result_s, 3),
    }
    alerts_first = [a for a in run.alerts if a.first]
    spans = [s[1] - s[0] for s in run.track_spans.values()]
    out["diagnostics"] = {
        "alerts": len(run.alerts),
        "alert_episodes": len(alerts_first),
        "alerts_by_scenario": dict(Counter(a.scenario for a in run.alerts)),
        "alerts_per_hour": _r(len(run.alerts) / (run.frames / run.fps / 3600.0) if run.frames else None, 1),
        "confirmed_track_ids": len(run.track_spans),
        "mean_track_life_s": _r(float(np.mean(spans)) if spans else None, 2),
        "median_track_life_s": _r(float(np.median(spans)) if spans else None, 2),
        "mean_people_per_frame": _r(float(np.mean(run.people_per_frame)) if run.people_per_frame else None, 2),
        "teacher": run.teacher or None,
    }
    if not clip.labelled:
        return out

    acc = ClipDetectionTracking()
    for f in sorted(set(run.people) & clip.annotated):
        acc.add_frame(f, clip.boxes_by_frame.get(f, []), run.people[f], clip.conditions_at(f), iou_thr)
    out["detection"] = acc.detection()
    out["tracking"] = acc.tracking(run.fps)
    # [Phase 11 WP3] entry detection + in-water consistency
    out["entry"] = person_sequences_metrics(acc.gt_seq, run.fps)
    out["_entry_times"] = entry_times(acc.gt_seq, run.fps)  # dropped before serialisation
    ev = score_events(clip, run.alerts, acc.pred_to_gt(), people=run.people)
    out["events"] = ev["events"]
    out["false_alerts"] = ev["false_alerts"]
    out["alert_groups"] = ev.get("alert_groups", {"notified": 0, "events": 0})
    if "continuity" in ev:
        out["continuity"] = ev["continuity"]
    out["exposure_person_s"] = activity_exposure(clip)
    out["_acc"] = acc  # dropped before serialisation
    return out


def _continuity(vic: list[dict[str, int]]) -> dict[str, Any] | None:
    """[Phase 3] Whole-clip subject continuity (continuity-only clips, eval/underwater_v1); None when there are none."""
    if not vic:
        return None
    pairs = sum(v["pairs"] for v in vic)
    keyframes = sum(v["keyframes"] for v in vic)
    ids = [v["distinct_ids"] for v in vic if v["matched"]]
    return {
        "clips": len(vic),
        "keyframes": keyframes,
        "matched_rate": _r(sum(v["matched"] for v in vic) / keyframes) if keyframes else None,
        "id_kept_rate": _r(sum(v["kept"] for v in vic) / pairs) if pairs else None,
        "pairs": pairs,
        "kept": sum(v["kept"] for v in vic),
        "distinct_ids_median": _r(float(np.median(ids)), 1) if ids else None,
        "distinct_ids_max": max(ids) if ids else None,
    }


def aggregate(clips: list[dict[str, Any]], split: str) -> dict[str, Any]:
    sel = [c for c in clips if c["split"] == split and c["labelled"]]
    agg: dict[str, Any] = {"split": split, "clips": len(sel)}
    if not sel:
        return agg
    hours = sum(c["duration_s"] or 0 for c in sel) / 3600.0
    recall: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    prec: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    water_conf: Counter = Counter()
    zone_conf: Counter = Counter()
    crowd: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])  # key → [sum abs err, frames]
    for c in sel:
        zone_conf.update(c["detection"].get("zone_state_confusion", {}))
        for k, v in (c["detection"].get("crowd_count") or {}).items():
            if isinstance(v, dict) and v.get("frames"):
                crowd[k][0] += v["mae"] * v["frames"]
                crowd[k][1] += v["frames"]
        for k, v in c["detection"]["in_water_recall"].items():
            recall[k][0] += v["tp"]
            recall[k][1] += v["fn"]
        for k, v in c["detection"]["precision"].items():
            prec[k][0] += v["preds"]
            prec[k][1] += v["fp"]
        water_conf.update(c["detection"]["water_state_confusion"])
    agg["camera_hours"] = _r(hours, 3)
    agg["in_water_recall"] = {
        k: {"recall": _r(tp / (tp + fn)) if tp + fn else None, "n": tp + fn} for k, (tp, fn) in sorted(recall.items())
    }
    agg["precision"] = {
        k: {"precision": _r(1 - fp / n) if n else None, "preds": n} for k, (n, fp) in sorted(prec.items())
    }
    agg["water_state_confusion"] = dict(water_conf)
    agg["zone_state_confusion"] = dict(zone_conf)
    zn = sum(zone_conf.values())
    agg["zone_state_accuracy"] = (
        _r(sum(v for k, v in zone_conf.items() if k.split("_pred_")[0][3:] == k.split("_pred_")[1]) / zn)
        if zn
        else None
    )
    agg["crowd_count_mae"] = {k: {"mae": _r(s / n, 3), "frames": int(n)} for k, (s, n) in crowd.items() if n}
    agg["tracking"] = {
        "id_switches": sum(c["tracking"]["id_switches"] for c in sel),
        "fragmentations": sum(c["tracking"]["fragmentations"] for c in sel),
        "gt_tracks": sum(c["tracking"]["gt_tracks"] for c in sel),
        "idf1_mean": _r(
            float(np.mean([c["tracking"]["idf1"] for c in sel if c["tracking"]["idf1"] is not None]))
            if any(c["tracking"]["idf1"] is not None for c in sel)
            else None
        ),
    }
    gaps: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for c in sel:
        for k, v in c["tracking"]["gap_continuity"].items():
            gaps[k][0] += v["kept"]
            gaps[k][1] += v["gaps"]
    agg["tracking"]["gap_continuity"] = {k: {"rate": _r(a / b) if b else None, "gaps": b} for k, (a, b) in gaps.items()}
    agg["entry"] = pool_entry_metrics(sel)  # [Phase 11 WP3]

    events = [e for c in sel for e in c["events"]]
    det = [e for e in events if e["detected"]]
    tta = [e["time_to_alert_s"] for e in det]
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for e in events:
        by_type[e["type"]][0] += int(e["detected"])
        by_type[e["type"]][1] += 1
    agg["events"] = {
        "total": len(events),
        "detected": len(det),
        "recall": _r(len(det) / len(events)) if events else None,
        "by_type": {k: {"detected": a, "total": b, "recall": _r(a / b)} for k, (a, b) in by_type.items()},
        "time_to_alert_s": {
            "median": _r(float(np.median(tta)), 2) if tta else None,
            "p90": _r(float(np.percentile(tta, 90)), 2) if tta else None,
            "max": _r(max(tta), 2) if tta else None,
        },
    }
    # [Phase 3 Step 0] strict window credit + the victim's track continuity at the GT keyframes
    win = [e for e in events if e.get("detected_in_window")]
    tta_w = [e["time_to_alert_in_window_s"] for e in win]
    vic = [e["victim"] for e in events if e.get("victim")]
    pairs = sum(v["pairs"] for v in vic)
    keyframes = sum(v["keyframes"] for v in vic)
    ids = [v["distinct_ids"] for v in vic if v["matched"]]
    agg["events"]["in_window"] = {
        "detected": len(win),
        "recall": _r(len(win) / len(events)) if events else None,
        "time_to_alert_median_s": _r(float(np.median(tta_w)), 2) if tta_w else None,
    }
    agg["events"]["victim"] = {
        "keyframes": keyframes,
        "matched_rate": _r(sum(v["matched"] for v in vic) / keyframes) if keyframes else None,
        "id_kept_rate": _r(sum(v["kept"] for v in vic) / pairs) if pairs else None,
        "pairs": pairs,
        "distinct_ids_median": _r(float(np.median(ids)), 1) if ids else None,
        "distinct_ids_max": max(ids) if ids else None,
    }
    agg["continuity"] = _continuity([c["continuity"] for c in sel if c.get("continuity")])
    # [Phase 5] alerts per event (alerts/events.py grouping) and notified alerts per detected GT event
    n_alerts = sum(c.get("alert_groups", {}).get("notified", 0) for c in sel)
    n_events = sum(c.get("alert_groups", {}).get("events", 0) for c in sel)
    per_true = [e["notified_alerts"] for e in events if e.get("detected") and "notified_alerts" in e]
    agg["alerts_per_event"] = {
        "notified_alerts": n_alerts,
        "events": n_events,
        "ratio": _r(n_alerts / n_events) if n_events else None,
        "per_detected_gt_event_median": _r(float(np.median(per_true)), 1) if per_true else None,
    }
    fa = [a for c in sel for a in c["false_alerts"]]
    counts = Counter(a["bucket"] for a in fa)
    episodes = Counter(a["bucket"] for a in fa if a["first"])
    exposure: dict[str, float] = defaultdict(float)
    for c in sel:
        for k, v in c["exposure_person_s"].items():
            exposure[k] += v
    notified = [a for a in fa if a.get("notified", True)]
    by_clip: dict[str, dict[str, Any]] = {}
    for c in sel:
        cfa = c["false_alerts"]
        if cfa:
            by_clip[c["clip_id"]] = {
                "alerts": len(cfa),
                "notified": sum(bool(a.get("notified", True)) for a in cfa),
                "people": len({a["track_id"] for a in cfa}),
            }
    agg["false_alerts"] = {
        "total": len(fa),
        "per_camera_hour": _r(len(fa) / hours, 2) if hours else None,
        "notified": len(notified),  # [Phase 3 Step 0] what a live camera would have sent (rising edges + cooldown)
        "people": sum(v["people"] for v in by_clip.values()),  # distinct tracks falsely alerted, per clip
        "by_clip": by_clip,
        "by_bucket": {
            b: {
                "alerts": counts.get(b, 0),
                "episodes": episodes.get(b, 0),
                "per_camera_hour": _r(counts.get(b, 0) / hours, 2) if hours else None,
                "exposure_person_hours": _r(exposure.get(b, 0.0) / 3600.0, 3),
            }
            for b in FALSE_ALERT_BUCKETS
        },
    }
    return agg


def perf(clips: list[dict[str, Any]], runs: list[ClipRun]) -> dict[str, Any]:
    ms = [m for r in runs for m in r.step_ms]
    stages: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        for k, v in r.stage_ms.items():
            stages[k].extend(v)
    frames = sum(r.frames for r in runs)
    analysed = sum(r.analysed for r in runs)
    src_s = sum(r.frames / r.fps for r in runs if r.fps)
    wall = sum(r.wall_s for r in runs)
    return {
        "frames": frames,
        "analysed": analysed,
        "analysed_fraction": _r(analysed / frames if frames else None),
        "latency": latency_summary(ms),
        "stages": {k: latency_summary(v) for k, v in stages.items()},
        "pipeline_fps": _r(analysed / (sum(ms) / 1000.0) if ms else None, 2),
        "source_seconds": _r(src_s, 1),
        "wall_seconds": _r(wall, 1),
        "realtime_factor": _r(src_s / wall if wall else None, 3),
    }


def headline(report: dict[str, Any]) -> dict[str, Any]:
    t = report.get("splits", {}).get("test", {}) or {}
    d = report.get("splits", {}).get("dev", {}) or {}
    src = t if t.get("clips") else d
    p = report.get("performance", {})
    h: dict[str, Any] = {"labelled_split_used": "test" if t.get("clips") else ("dev" if d.get("clips") else None)}
    if src.get("clips"):
        h["in_water_recall"] = src["in_water_recall"].get("all", {}).get("recall")
        h["precision"] = src["precision"].get("all", {}).get("precision")
        h["id_switches"] = src["tracking"]["id_switches"]
        h["idf1"] = src["tracking"]["idf1_mean"]
        h["event_recall"] = src["events"]["recall"]
        h["time_to_alert_median_s"] = src["events"]["time_to_alert_s"]["median"]
        h["event_recall_in_window"] = src["events"]["in_window"]["recall"]
        h["time_to_alert_in_window_median_s"] = src["events"]["in_window"]["time_to_alert_median_s"]
        h["victim_id_kept_rate"] = src["events"]["victim"]["id_kept_rate"]
        h["victim_distinct_ids_median"] = src["events"]["victim"]["distinct_ids_median"]
        h["false_alerts_notified"] = src["false_alerts"]["notified"]
        cont = src.get("continuity") or {}
        h["continuity_id_kept_rate"] = cont.get("id_kept_rate")
        ape = src.get("alerts_per_event") or {}
        h["alerts_per_event"] = ape.get("ratio")
        h["notified_alerts_per_detected_event_median"] = ape.get("per_detected_gt_event_median")
        h["continuity_distinct_ids_median"] = cont.get("distinct_ids_median")
        h["continuity_matched_rate"] = cont.get("matched_rate")
        h["false_alert_people"] = src["false_alerts"]["people"]
        h["false_alerts_per_camera_hour"] = src["false_alerts"]["per_camera_hour"]
        for b in FALSE_ALERT_BUCKETS:
            h[f"false_alerts_{b}"] = src["false_alerts"]["by_bucket"][b]["alerts"]
    h["analysed_fraction"] = p.get("analysed_fraction")
    h["latency_p50_ms"] = p.get("latency", {}).get("p50_ms")
    h["latency_p95_ms"] = p.get("latency", {}).get("p95_ms")
    h["latency_p99_ms"] = p.get("latency", {}).get("p99_ms")
    h["p99_over_p50"] = p.get("latency", {}).get("p99_over_p50")
    h["pipeline_fps"] = p.get("pipeline_fps")
    h["realtime_factor"] = p.get("realtime_factor")
    if report.get("startup"):
        h["cold_start_to_first_frame_s"] = report["startup"].get("process_to_first_result_s")
    if report.get("upload_path"):
        h["upload_realtime_factor"] = report["upload_path"].get("realtime_factor")
    if report.get("live"):
        h["live_streams"] = report["live"].get("streams")
        h["live_analysed_fraction"] = report["live"].get("analysed_fraction_min")
    if report.get("behavior_suite"):
        bs = report["behavior_suite"]
        h["synthetic_recall"] = bs.get("synthetic", {}).get("summary", {}).get("recall")
        h["synthetic_false_alerts"] = bs.get("synthetic", {}).get("summary", {}).get("false_alerts")
        h["standing_suite_false_alerts"] = bs.get("standing_suite", {}).get("false_alerts")
    unl = [c for c in report.get("clips", []) if not c["labelled"]]
    if unl:
        h["unlabelled_alerts_per_hour"] = _r(
            sum(c["diagnostics"]["alerts"] for c in unl) / max(sum(c["duration_s"] or 0 for c in unl) / 3600.0, 1e-9), 1
        )
        tp = sum((c["diagnostics"].get("teacher") or {}).get("found_by_pipeline", 0) for c in unl)
        tn = sum((c["diagnostics"].get("teacher") or {}).get("teacher_people", 0) for c in unl)
        h["unlabelled_teacher_agreement"] = _r(tp / tn) if tn else None
    return h


def targets(report: dict[str, Any]) -> dict[str, Any]:
    h = report["headline"]
    out: dict[str, Any] = {}

    def chk(name: str, ok: bool | None, value: Any, target: Any) -> None:
        out[name] = {
            "value": value,
            "target": target,
            "status": "not measured" if ok is None else ("MET" if ok else "NOT MET"),
        }

    r = h.get("in_water_recall")
    chk("in_water_recall", None if r is None else r >= TARGETS["in_water_recall"], r, ">= 0.95")
    fa = [
        h.get("false_alerts_normal_swimming"),
        h.get("false_alerts_standing_in_water"),
        h.get("false_alerts_out_of_water"),
    ]
    fa_ok = None if any(v is None for v in fa) else sum(fa) == 0
    chk("no_alerts_on_normal_or_standing", fa_ok, fa, "0 / 0 / 0")
    q = h.get("p99_over_p50")
    chk("p99_le_2x_p50", None if q is None else q <= TARGETS["p99_over_p50_max"], q, "<= 2.0")
    cs = h.get("cold_start_to_first_frame_s")
    chk("cold_start_lt_5s", None if cs is None else cs < TARGETS["cold_start_s_max"], cs, "< 5 s")
    up = h.get("upload_realtime_factor")
    chk("upload_ge_realtime", None if up is None else up >= TARGETS["upload_realtime_factor_min"], up, ">= 1.0x")
    af = h.get("analysed_fraction")
    chk("every_frame_analysed", None if af is None else af >= 0.999, af, "1.0")
    lf = h.get("live_analysed_fraction")
    chk("live_every_frame", None if lf is None else lf >= 0.999, lf, "1.0 on every stream")
    return out


def build(
    clips_scored: list[dict[str, Any]], runs: list[ClipRun], meta: dict[str, Any], extra: dict[str, Any]
) -> dict[str, Any]:
    rep: dict[str, Any] = {"meta": meta}
    rep["clips"] = [{k: v for k, v in c.items() if not k.startswith("_")} for c in clips_scored]
    rep["splits"] = {s: aggregate(clips_scored, s) for s in ("dev", "test")}
    rep["performance"] = perf(clips_scored, runs)
    rep.update(extra)
    rep["headline"] = headline(rep)
    rep["targets"] = targets(rep)
    return rep


# ═════════════════════════════════════════════════════════════════════════════
# Markdown
# ═════════════════════════════════════════════════════════════════════════════
def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return " / ".join(_fmt(x) for x in v)
    return str(v)


def to_markdown(rep: dict[str, Any], baseline: dict[str, Any] | None = None) -> str:
    m = rep["meta"]
    L: list[str] = [f"# Life Guard pipeline evaluation — {m.get('name', '')}", ""]
    L.append(f"- Tree under test: `{m.get('backend_dir')}` (adapter `{m.get('adapter')}`)")
    L.append(
        f"- Mode: {m.get('mode')} · zones: {m.get('zones')} · device: {m.get('device')} · "
        f"providers: {m.get('providers')}"
    )
    L.append(f"- Run at {m.get('started_at')} on {m.get('host')}")
    if m.get("warnings"):
        for w in m["warnings"]:
            L.append(f"- **Warning:** {w}")
    L.append("")
    L.append("## Headline" + (" (vs baseline)" if baseline else ""))
    L.append("")
    bh = (baseline or {}).get("headline", {})
    L.append("| metric | value |" + (" baseline | change |" if baseline else ""))
    L.append("|---|---|" + ("---|---|" if baseline else ""))
    for k, v in rep["headline"].items():
        row = f"| {k} | {_fmt(v)} |"
        if baseline:
            b = bh.get(k)
            d = (v - b) if isinstance(v, (int, float)) and isinstance(b, (int, float)) else None
            row += f" {_fmt(b)} | {_fmt(round(d, 4) if d is not None else None)} |"
        L.append(row)
    L.append("")
    L.append("## Targets")
    L.append("")
    L.append("| target | value | required | status |")
    L.append("|---|---|---|---|")
    for k, t in rep["targets"].items():
        L.append(f"| {k} | {_fmt(t['value'])} | {t['target']} | **{t['status']}** |")
    L.append("")

    for split in ("test", "dev"):
        s = rep["splits"].get(split, {})
        if not s.get("clips"):
            continue
        L.append(f"## Labelled `{split}` split — {s['clips']} clip(s), {s['camera_hours']} camera-hours")
        L.append("")
        L.append("### In-water person recall")
        L.append("| bucket | recall | n |")
        L.append("|---|---|---|")
        for k, v in s["in_water_recall"].items():
            L.append(f"| {k} | {_fmt(v['recall'])} | {v['n']} |")
        L.append("")
        L.append("### Precision")
        L.append("| bucket | precision | predictions |")
        L.append("|---|---|---|")
        for k, v in s["precision"].items():
            L.append(f"| {k} | {_fmt(v['precision'])} | {v['preds']} |")
        L.append("")
        if s.get("zone_state_confusion"):
            L.append(f"### Zone state (in water / edge / deck) — accuracy {_fmt(s.get('zone_state_accuracy'))}")
            for k, v in sorted(s["zone_state_confusion"].items()):
                L.append(f"- {k}: {v}")
            L.append("")
        if s.get("crowd_count_mae"):
            L.append("### Person count error per frame (confirmed tracks vs visible GT)")
            for k, v in s["crowd_count_mae"].items():
                L.append(f"- {k}: MAE {_fmt(v['mae'])} over {v['frames']} frames")
            L.append("")
        tr = s["tracking"]
        L.append(
            f"### Tracking — ID switches {tr['id_switches']}, fragmentations {tr['fragmentations']}, "
            f"IDF1 {_fmt(tr['idf1_mean'])}, GT tracks {tr['gt_tracks']}"
        )
        for k, v in tr["gap_continuity"].items():
            L.append(f"- same ID after an invisible gap {k}: {_fmt(v['rate'])} ({v['gaps']} gaps)")
        L.append("")
        ev = s["events"]
        L.append(
            f"### Drowning events — recall {_fmt(ev['recall'])} ({ev['detected']}/{ev['total']}), "
            f"time-to-alert median {_fmt(ev['time_to_alert_s']['median'])} s, "
            f"p90 {_fmt(ev['time_to_alert_s']['p90'])} s"
        )
        for k, v in ev["by_type"].items():
            L.append(f"- {k}: {v['detected']}/{v['total']}")
        iw, vc = ev.get("in_window") or {}, ev.get("victim") or {}
        if iw:
            L.append(
                f"- inside the event window, one alert per event: {iw['detected']}/{ev['total']}, "
                f"time-to-alert median {_fmt(iw['time_to_alert_median_s'])} s"
            )
        if vc:
            L.append(
                f"- victim at GT keyframes: matched {_fmt(vc['matched_rate'])} of {vc['keyframes']}, same id kept "
                f"{_fmt(vc['id_kept_rate'])} of {vc['pairs']} consecutive pairs, distinct ids per event median "
                f"{_fmt(vc['distinct_ids_median'])} (max {_fmt(vc['distinct_ids_max'])})"
            )
        ct = s.get("continuity") or {}
        if ct:
            L.append(
                f"- subject over whole clips ({ct['clips']} continuity clips): matched {_fmt(ct['matched_rate'])} of "
                f"{ct['keyframes']} keyframes, same id kept {_fmt(ct['id_kept_rate'])} ({ct['kept']}/{ct['pairs']} "
                f"pairs), distinct ids per clip median {_fmt(ct['distinct_ids_median'])} "
                f"(max {_fmt(ct['distinct_ids_max'])})"
            )
        L.append("")
        fa = s["false_alerts"]
        L.append(f"### False alerts — {fa['total']} ({_fmt(fa['per_camera_hour'])} per camera-hour)")
        L.append("| bucket | alerts | episodes | per camera-hour | person-hours of exposure |")
        L.append("|---|---|---|---|---|")
        for b, v in fa["by_bucket"].items():
            L.append(
                f"| {b} | {v['alerts']} | {v['episodes']} | {_fmt(v['per_camera_hour'])} | "
                f"{_fmt(v['exposure_person_hours'])} |"
            )
        if fa.get("by_clip"):
            L.append("")
            L.append(
                f"Notified (as a live camera would): {fa['notified']}; distinct people falsely alerted: {fa['people']}"
            )
            L.append("")
            L.append("| clip | alerts | notified | people |")
            L.append("|---|---|---|---|")
            for cid, v in sorted(fa["by_clip"].items(), key=lambda kv: -kv[1]["alerts"]):
                L.append(f"| {cid} | {v['alerts']} | {v['notified']} | {v['people']} |")
        L.append("")

    p = rep["performance"]
    L.append("## Throughput and latency (clip runs)")
    L.append("")
    L.append(f"- Frames: {p['frames']} decoded, {p['analysed']} analysed (fraction {_fmt(p['analysed_fraction'])})")
    L.append(f"- Pipeline fps (analysed frames / processing time): {_fmt(p['pipeline_fps'])}")
    L.append(f"- Real-time factor (source seconds / wall seconds, incl. decode): {_fmt(p['realtime_factor'])}")
    lat = p["latency"]
    L.append(
        f"- Per-analysed-frame latency: p50 {_fmt(lat.get('p50_ms'))} ms, p95 {_fmt(lat.get('p95_ms'))} ms, "
        f"p99 {_fmt(lat.get('p99_ms'))} ms, max {_fmt(lat.get('max_ms'))} ms "
        f"(p99/p50 = {_fmt(lat.get('p99_over_p50'))})"
    )
    for k, v in p["stages"].items():
        L.append(f"  - stage `{k}`: p50 {_fmt(v.get('p50_ms'))} ms, p99 {_fmt(v.get('p99_ms'))} ms")
    L.append("")
    if rep.get("startup"):
        st = rep["startup"]
        L.append(f"## Cold start — process start → first analysed frame: {_fmt(st.get('process_to_first_result_s'))} s")
        for k, v in st.items():
            if k != "process_to_first_result_s":
                L.append(f"- {k}: {_fmt(v)}")
        L.append("")
    if rep.get("upload_path"):
        up = rep["upload_path"]
        L.append(
            f"## Upload path (with Model View render + VP8 encode) — real-time factor {_fmt(up.get('realtime_factor'))}"
        )
        L.append(
            f"- latency p50 {_fmt(up.get('latency', {}).get('p50_ms'))} ms, "
            f"p99 {_fmt(up.get('latency', {}).get('p99_ms'))} ms; analysed fraction "
            f"{_fmt(up.get('analysed_fraction'))}"
        )
        L.append("")
    if rep.get("live"):
        lv = rep["live"]
        L.append(f"## Live cameras — {lv.get('streams')} stream(s) of `{lv.get('clip')}` for {lv.get('seconds')} s")
        L.append("")
        L.append("| camera | source fps | frames decoded | frames analysed | analysed fraction | dropped | max lag s |")
        L.append("|---|---|---|---|---|---|---|")
        for c in lv.get("cameras", []):
            L.append(
                f"| {c['camera']} | {_fmt(c.get('source_fps'))} | {c.get('decoded')} | {c.get('analysed')} | "
                f"{_fmt(c.get('analysed_fraction'))} | {c.get('dropped')} | {_fmt(c.get('max_lag_s'))} |"
            )
        for n in lv.get("notes", []):
            L.append(f"- {n}")
        L.append("")
    if rep.get("behavior_suite"):
        bs = rep["behavior_suite"]
        L.append("## Behaviour suite (synthetic + operator recordings — NOT a substitute for real footage)")
        L.append("")
        syn = bs.get("synthetic", {})
        if syn:
            s = syn.get("summary", {})
            L.append(
                f"- Synthetic ({syn.get('seeds')} seeds × {len(syn.get('scenarios', {}))} scenarios, "
                f"{syn.get('rate_hz')} Hz): recall {_fmt(s.get('recall'))}, false alerts "
                f"{_fmt(s.get('false_alerts'))}, early alerts {_fmt(s.get('early_alerts'))}, "
                f"median latency {_fmt(s.get('median_latency_s'))} s"
            )
            L.append("")
            L.append("| scenario | label | runs | alerted | false | latency s |")
            L.append("|---|---|---|---|---|---|")
            for k, v in syn.get("scenarios", {}).items():
                L.append(
                    f"| {k} | {v.get('label')} | {v.get('runs')} | {v.get('alerted', v.get('detected'))} | "
                    f"{v.get('false_alerts')} | {_fmt(v.get('median_latency_s'))} |"
                )
            L.append("")
        ss = bs.get("standing_suite")
        if ss:
            L.append(
                f"- No-zone standing / deck suite: {ss.get('false_alerts')} false alert(s) over "
                f"{ss.get('people')} people, {ss.get('person_minutes')} person-minutes"
            )
            for k, v in ss.get("by_scenario", {}).items():
                L.append(f"  - {k}: {v}")
            L.append("")
        rv = bs.get("recordings")
        if rv:
            L.append(f"- Operator recordings (labelled): {json.dumps(rv.get('summary', rv))}")
            L.append("")

    unl = [c for c in rep["clips"] if not c["labelled"]]
    if unl:
        L.append("## Unlabelled clips — diagnostics only (no ground truth)")
        L.append("")
        L.append(
            "| clip | s | analysed | people/frame | track ids | median life s | alerts | alerts/h | "
            "teacher agreement | p50 ms | p99 ms |"
        )
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for c in unl:
            dg = c["diagnostics"]
            tt = (dg.get("teacher") or {}).get("teacher_agreement")
            L.append(
                f"| {c['clip_id']} | {_fmt(c['duration_s'])} | {_fmt(c['analysed_fraction'])} | "
                f"{_fmt(dg['mean_people_per_frame'])} | {dg['confirmed_track_ids']} | "
                f"{_fmt(dg['median_track_life_s'])} | {dg['alerts']} {json.dumps(dg['alerts_by_scenario'])} | "
                f"{_fmt(dg['alerts_per_hour'])} | {_fmt(tt)} | {_fmt(c['latency'].get('p50_ms'))} | "
                f"{_fmt(c['latency'].get('p99_ms'))} |"
            )
        L.append("")
        L.append(
            "Teacher agreement = share of people found by high-resolution tiled inference of the stock "
            "detector (conf ≥ 0.5) that the pipeline also reported on the same frame. It is a proxy for "
            "recall on small/distant people, not recall: the teacher shares the model's blind spots."
        )
        L.append("")
    return "\n".join(L)


def _json_default(o: Any) -> Any:
    """Sets → sorted lists, numpy scalars / arrays → Python values. Returning `o`
    itself for anything else (the old fallback) made json raise "Circular
    reference detected" and lost the whole report at the very end of a run."""
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "tolist"):  # numpy scalar or array
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def save(rep: dict[str, Any], out_dir: Path, baseline: dict[str, Any] | None = None) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    j = out_dir / "report.json"
    j.write_text(json.dumps(rep, indent=2, default=_json_default), encoding="utf-8")
    md = out_dir / "report.md"
    md.write_text(to_markdown(rep, baseline), encoding="utf-8")
    return j, md
