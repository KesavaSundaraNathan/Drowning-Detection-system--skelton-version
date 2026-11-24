"""
Run the FULL Life Guard pipeline over an evaluation set and write report.{json,md}.

    python -m lifeguard.eval.evaluate --eval-dir <eval> --name run1                      # this (v3) tree
    python -m lifeguard.eval.evaluate --eval-dir <eval> --name v2 --backend-dir <v2>/backend   # a v2 tree
    ... --trace run1.trace.jsonl.gz     # per-frame people + alerts, for lifeguard.eval.trace
    ... --thresholds candidate.json     # measure a proposed thresholds version (v3 tree)

--backend-dir points the harness at a v2 checkout (the frozen baseline): the SAME harness code then
measures both trees (plan Phase 3 parity). Run a v2 tree with a Python that has the v2 dependencies
(its own venv) and PYTHONPATH=<this repo>/backend.

[v3] From the v2 baseline scripts/evaluate_pipeline.py: argument names, report sections and the
cold-start / upload-path / behaviour-suite procedures are unchanged. New: --det-model / --provider
(v3 tree; provider guard, no "auto"), --thresholds, --recordings-dir (v2 read the harness tree's own
storage/behavior_recordings), --trace. --eval-dir is required (v3 ships no footage). The live
benchmark drives the v3 LiveRuntime (Phase 4: batched GPU scheduler, admission bypassed for A/B).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import platform
import subprocess  # nosec B404  # re-runs this module as a cold-start probe, fixed argv, no shell
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_T_PROCESS = time.time()
logger = logging.getLogger("lifeguard.eval")


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--backend-dir", type=Path, default=None, help="a v2 backend to measure instead of this tree")
    ap.add_argument("--name", default=datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S"))
    ap.add_argument("--out", type=Path, default=None, help="report dir (default <eval-dir>/reports/<name>)")
    ap.add_argument("--adapter", default="auto", choices=("auto", "pipeline"))
    ap.add_argument("--mode", default="Medium", choices=("Low", "Medium", "High"))
    ap.add_argument("--split", default="all", choices=("all", "dev", "test"))
    ap.add_argument("--clips", default="", help="comma-separated clip ids (default: all in manifest)")
    ap.add_argument("--no-clips", action="store_true", help="skip clip runs (only the other sections)")
    ap.add_argument("--max-seconds", type=float, default=None, help="cap each clip")
    ap.add_argument("--zones", action="store_true", help="use the annotated water polygons as zones")
    ap.add_argument(
        "--zones-file",
        type=Path,
        default=None,
        help="JSON {clip_id: [[x, y], ...] normalised}: drawn water zones, as a camera's drawn WATER zone (live)",
    )
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--teacher", action="store_true", help="teacher agreement on unlabelled clips")
    ap.add_argument("--teacher-every", type=int, default=15)
    ap.add_argument("--startup", action="store_true", help="measure cold start in a fresh process")
    ap.add_argument("--startup-repeats", type=int, default=2)
    ap.add_argument("--upload-path", default="", help="clip id to time with render + encode (upload path)")
    ap.add_argument("--live-streams", type=int, default=0, help="N concurrent live cameras (file sources)")
    ap.add_argument("--live-clip", default="", help="clip id used for the live benchmark")
    ap.add_argument("--live-seconds", type=float, default=60.0)
    ap.add_argument("--behavior-suite", action="store_true")
    ap.add_argument("--behavior-seeds", type=int, default=8)
    ap.add_argument(
        "--recordings-dir", type=Path, default=None, help="operator-labelled alert recordings for the behaviour suite"
    )
    ap.add_argument("--compare", type=Path, default=None, help="baseline report.json for a before/after table")
    ap.add_argument(
        "--merge-into",
        type=Path,
        default=None,
        help="write the sections this run produced into an existing report.json (and re-render it)",
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="set an environment variable for this run (v2 trees read their settings from it)",
    )
    ap.add_argument("--det-model", type=Path, default=None, help="v3 tree: detector .onnx")
    ap.add_argument(
        "--provider",
        default=None,
        choices=("auto", "tensorrt", "cuda", "openvino", "cpu"),
        help="v3 tree: execution provider",
    )
    ap.add_argument("--pose-model", default=None, help="v3 tree: pose .onnx, or 'none' for the pose-off arm")
    ap.add_argument("--thresholds", type=Path, default=None, help="v3 tree: thresholds file to evaluate")
    ap.add_argument("--trace", type=Path, default=None, help="write per-frame people + alerts (.jsonl.gz)")
    ap.add_argument("--startup-probe", default="", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def _bootstrap(args: argparse.Namespace) -> Any:
    """Select the tree under test BEFORE anything imports it (env overrides first)."""
    for kv in args.set:
        k, _, v = kv.partition("=")
        os.environ[k.strip()] = v.strip()
    if args.thresholds is not None:
        os.environ["LIFEGUARD_THRESHOLDS_FILE"] = str(args.thresholds.resolve())
    from lifeguard.eval import runner

    tree = runner.Tree.detect(args.backend_dir)
    if tree.layout == "v2":
        os.chdir(args.backend_dir)  # as the v2 harness did (relative paths in v2 code)
    else:
        from lifeguard.vision import config as vision_config

        base = vision_config.from_env()
        pose = base.pose_model_path if args.pose_model is None else vision_config.pose_model_from(args.pose_model)
        vision_config.configure(
            vision_config.VisionConfig(
                str(args.det_model.resolve()) if args.det_model else base.det_model_path,
                args.provider or base.provider,
                str(Path(pose).resolve()) if pose else None,
            )
        )
    runner.set_tree(tree)
    return tree


def _device_info(tree: Any) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import onnxruntime as ort

        info["providers"] = ort.get_available_providers()
        info["ort"] = ort.__version__
    except ImportError as exc:
        info["providers"] = f"unavailable: {exc}"
    info["tree"] = tree.describe()
    return info


def startup_probe(video: str, adapter_kind: str, mode: str) -> None:
    """Child process: import -> warm up -> first analysed frame. Prints JSON timings."""
    t0 = time.time()
    import cv2

    from lifeguard.eval.runner import make_adapter, tree

    t_import = time.time()
    ad = make_adapter(adapter_kind, mode)
    ad.warmup()
    t_warm = time.time()
    cap = cv2.VideoCapture(video)
    _ok, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    h, w = frame.shape[:2]
    ad.start_clip("startup", fps, w, h, None)
    ad.step(0, 0.0, frame)
    t_first = time.time()
    ad.step(1, 1.0 / fps, frame)
    t_second = time.time()
    ad.end_clip()
    sys.stdout.write(
        "STARTUP_JSON "
        + json.dumps(
            {
                "t_process": _T_PROCESS,
                "t_main": t0,
                "imports_s": t_import - t0,
                "warmup_s": t_warm - t_import,
                "first_frame_s": t_first - t_warm,
                "second_frame_s": t_second - t_first,
                "t_first": t_first,
                "adapter": ad.name,
                "providers": tree().active_providers(),
            }
        )
        + "\n"
    )


def _passthrough(args: argparse.Namespace) -> list[str]:
    out = [a for kv in args.set for a in ("--set", kv)]
    for flag, val in (
        ("--backend-dir", args.backend_dir),
        ("--det-model", args.det_model),
        ("--provider", args.provider),
        ("--pose-model", args.pose_model),
        ("--thresholds", args.thresholds),
    ):
        if val is not None:
            out += [flag, str(Path(val).resolve()) if isinstance(val, Path) else str(val)]
    return out


def measure_startup(args: argparse.Namespace, video: str) -> dict[str, Any]:
    runs = []
    for _ in range(max(1, args.startup_repeats)):
        t_spawn = time.time()
        cmd = [
            sys.executable,
            "-m",
            "lifeguard.eval.evaluate",
            "--eval-dir",
            str(args.eval_dir),
            "--startup-probe",
            video,
            "--adapter",
            args.adapter,
            "--mode",
            args.mode,
            *_passthrough(args),
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)  # noqa: S603  # nosec B603  # fixed argv
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("STARTUP_JSON ")), None)
        if line is None:
            return {"error": (out.stderr or out.stdout)[-2000:]}
        d = json.loads(line[len("STARTUP_JSON ") :])
        d["process_to_first_result_s"] = d["t_first"] - t_spawn
        d["interpreter_start_s"] = d["t_main"] - t_spawn
        runs.append(d)
    # the FIRST run can include one-time work (engine build, OS file cache); report the warm-cache run
    best = runs[-1]
    return {
        "process_to_first_result_s": round(best["process_to_first_result_s"], 2),
        "interpreter_start_s": round(best["interpreter_start_s"], 2),
        "imports_s": round(best["imports_s"], 2),
        "model_load_and_warmup_s": round(best["warmup_s"], 2),
        "first_frame_s": round(best["first_frame_s"], 3),
        "second_frame_s": round(best["second_frame_s"], 3),
        "first_run_process_to_first_result_s": round(runs[0]["process_to_first_result_s"], 2),
        "runs": len(runs),
        "providers": best.get("providers"),
    }


def live_bench(args: argparse.Namespace, tree: Any, video: str, clip_id: str) -> dict[str, Any]:
    """N file-backed cameras paced at source fps through the tree's live worker."""
    if tree.layout == "v3":
        return live_bench_v3(args, video, clip_id)
    import threading  # noqa: F401  # the v2 workers start threads

    import services.alert_writer as aw  # type: ignore[import-not-found]
    from detection.camera_stream import CameraSource, CameraWorker  # type: ignore[import-not-found]
    from detection.live_pipeline import CameraDetectionWorker  # type: ignore[import-not-found]

    aw.persist_alert = lambda *a, **k: None  # never write alerts from a benchmark
    cams = []
    for i in range(args.live_streams):
        cid = f"bench-{i}"
        src = CameraSource(cid, f"bench {i}", "file", video)
        cw = CameraWorker(src, loop_file=True)
        dw = CameraDetectionWorker(
            cid, f"bench {i}", cw, args.mode, geometry_loader=lambda c, w, h: None, publish_frames=True
        )
        cams.append((cid, cw, dw))
    for _, cw, _dw in cams:
        cw.start()
    for _, _cw, dw in cams:
        dw.start()
    warm_deadline = time.time() + 120
    while time.time() < warm_deadline and not all(getattr(dw, "frames_seen", 0) > 0 for _, _, dw in cams):
        time.sleep(0.2)
    base = {cid: (cw.status.frames_read, _analysed(dw), cw.frames_dropped) for cid, cw, dw in cams}
    lags = {cid: 0.0 for cid, _, _ in cams}
    t0 = time.time()
    while time.time() - t0 < args.live_seconds:
        time.sleep(0.5)
        for cid, _cw, dw in cams:
            snap = dw.snapshot() if hasattr(dw, "snapshot") else {}
            lags[cid] = max(lags[cid], float(snap.get("lag_s") or 0.0))
    out = []
    for cid, cw, dw in cams:
        r0, a0, d0 = base[cid]
        dec = cw.status.frames_read - r0
        ana = _analysed(dw) - a0
        snap = dw.snapshot() if hasattr(dw, "snapshot") else {}
        out.append(
            {
                "camera": cid,
                "source_fps": cw.status.source_fps,
                "decoded": dec,
                "analysed": ana,
                "analysed_fraction": round(ana / dec, 4) if dec else None,
                "dropped": (cw.frames_dropped - d0) + int(snap.get("frames_dropped_delta", 0) or 0),
                "max_lag_s": round(lags[cid], 2) if lags[cid] else snap.get("lag_s"),
                "inference_ms": snap.get("inference_ms"),
                "behind_realtime": snap.get("behind_realtime"),
            }
        )
    for _, cw, dw in cams:
        dw.stop()
        cw.stop()
    stop_rt = getattr(sys.modules.get("detection.live_runtime"), "shutdown", None)
    if callable(stop_rt):
        stop_rt()
    fr = [c["analysed_fraction"] for c in out if c["analysed_fraction"] is not None]
    return {
        "streams": args.live_streams,
        "clip": clip_id,
        "seconds": args.live_seconds,
        "cameras": out,
        "analysed_fraction_min": min(fr) if fr else None,
        "notes": [],
    }


def live_bench_v3(args: argparse.Namespace, video: str, clip_id: str) -> dict[str, Any]:
    """v3 (Phase 4): N file cameras through the LiveRuntime - one batched GPU scheduler.

    Admission control is bypassed (enforce_capacity=False) so the same N as the v2 run is measured;
    the runtime's own computed camera limit is reported next to the result."""
    from lifeguard.core.config import load_settings
    from lifeguard.live.runtime import CameraSpec, LiveRuntime
    from lifeguard.vision import config as vision_config

    vc = vision_config.current()
    cfg = load_settings(env="test", data_dir=Path(tempfile.mkdtemp(prefix="lifeguard-bench-")), bcrypt_rounds=4)
    rt = LiveRuntime(cfg, vc.det_model_path, vc.provider, lambda f, c: None, lambda c, f, a, i, m=None: None)
    rt.start()
    for i in range(args.live_streams):
        rt.start_camera(CameraSpec(f"bench-{i}", f"bench {i}", video, "file", args.mode), enforce_capacity=False)
    cams = [rt._cams[f"bench-{i}"] for i in range(args.live_streams)]
    warm_deadline = time.time() + 120
    while time.time() < warm_deadline and not all(c.worker.analysed > 0 for c in cams):
        time.sleep(0.2)
    base = {c.spec.camera_id: (c.capture.frames_read, c.worker.analysed, c.queue.dropped) for c in cams}
    lags = {c.spec.camera_id: 0.0 for c in cams}
    t0 = time.time()
    while time.time() - t0 < args.live_seconds:
        time.sleep(0.5)
        for c in cams:
            lags[c.spec.camera_id] = max(lags[c.spec.camera_id], c.queue.lag_s())
    out: list[dict[str, Any]] = []
    for c in cams:
        r0, a0, d0 = base[c.spec.camera_id]
        dec = c.capture.frames_read - r0
        ana = c.worker.analysed - a0
        out.append(
            {
                "camera": c.spec.camera_id,
                "source_fps": c.capture.fps,
                "decoded": dec,
                "analysed": ana,
                "analysed_fraction": round(ana / dec, 4) if dec else None,
                "dropped": c.queue.dropped - d0,
                "max_lag_s": round(lags[c.spec.camera_id], 2),
                "inference_ms": round(c.worker.proc_ms_ewma, 2),
                "behind_realtime": c.lagging,
                "hw_accel": c.capture.hw_accel,
            }
        )
    status = rt.status()
    rt.stop()
    fr: list[float] = [float(c["analysed_fraction"]) for c in out if c["analysed_fraction"] is not None]
    return {
        "streams": args.live_streams,
        "clip": clip_id,
        "seconds": args.live_seconds,
        "cameras": out,
        "analysed_fraction_min": min(fr) if fr else None,
        "scheduler": {k: status["scheduler"][k] for k in ("rounds", "frames", "mean_batch", "batch_hist")},
        "capacity": status["capacity"],
        "notes": ["v3 LiveRuntime: one batched GPU scheduler; admission bypassed to match the v2 stream count"],
    }


def _analysed(dw: Any) -> int:
    for attr in ("frames_analysed", "_inferences"):
        if hasattr(dw, attr):
            return int(getattr(dw, attr))
    return 0


def behavior_suite(args: argparse.Namespace, tree: Any) -> dict[str, Any]:
    """Synthetic scenario suite + a no-zone standing/deck suite + labelled operator recordings."""
    bc = tree.module("suite")
    simulate_scene = tree.module("simulation").simulate_scene
    out: dict[str, Any] = {}
    seeds = list(range(args.behavior_seeds))
    rate = getattr(bc, "default_rate_hz", None)
    rate = rate(args.mode) if callable(rate) else None
    out["synthetic"] = bc.evaluate(args.mode, seeds, rate_hz=rate, crowd_scenes=6)
    rate_used = out["synthetic"]["rate_hz"]

    # No zones drawn = the common deployment. Deck people and people standing in shallow water must
    # not alert when the whole frame counts as water.
    standing = ("deck_walker", "deck_edge", "standing_shallow", "treading", "resting_wall")
    fa, people, secs, by = 0, 0, 0.0, {}
    for scen in standing:
        n_fa = 0
        for seed in seeds:
            for noise, miss in ((1.0, 0.02), (2.5, 0.10)):  # clean / poor footage
                sc = simulate_scene(
                    [scen], seed=5000 + seed, duration=60.0, rate_hz=rate_used, noise=noise, miss_rate=miss
                )
                sc.geometry = None
                run = bc.run_scene(sc, args.mode)
                people += 1
                secs += sc.duration
                if run.fires:
                    n_fa += 1
        by[scen] = f"{n_fa}/{2 * len(seeds)} runs alerted"
        fa += n_fa
    out["standing_suite"] = {
        "false_alerts": fa,
        "people": people,
        "person_minutes": round(secs / 60.0, 1),
        "by_scenario": by,
        "zones": "none (whole frame = water)",
        "noise_levels": "1.0/2% miss and 2.5/10% miss",
    }
    if args.recordings_dir is None:
        out["recordings"] = {"summary": {"recordings": 0}, "note": "no --recordings-dir given"}
    else:
        recs = bc.load_recordings(args.recordings_dir.resolve(), labelled_only=True)
        out["recordings"] = {
            "summary": bc.evaluate_recordings(recs) if recs else {"recordings": 0},
            "note": "operator-labelled alert recordings from storage/behavior_recordings",
        }
    return out


class _TraceWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = gzip.open(path, "wt", encoding="utf-8")  # noqa: SIM115  # closed by close()
        self.records = 0

    def __call__(self, rec: dict[str, Any]) -> None:
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.records += 1

    def close(self) -> None:
        self._f.close()


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    args.eval_dir = args.eval_dir.resolve()
    if args.backend_dir is not None:
        args.backend_dir = args.backend_dir.resolve()
    # _bootstrap() may chdir into the tree under test: resolve every other path argument first.
    for name in ("out", "compare", "merge_into", "trace", "recordings_dir", "det_model", "thresholds"):
        if getattr(args, name, None) is not None:
            setattr(args, name, getattr(args, name).resolve())
    tree = _bootstrap(args)
    if args.startup_probe:
        startup_probe(args.startup_probe, args.adapter, args.mode)
        return 0

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    from lifeguard.eval import report as R
    from lifeguard.eval.annotations import load_manifest
    from lifeguard.eval.runner import Teacher, make_adapter, run_clip

    clips = load_manifest(args.eval_dir)
    if args.clips:
        want = set(args.clips.split(","))
        clips = [c for c in clips if c.clip_id in want]
    if args.split != "all":
        clips = [c for c in clips if c.split == args.split or not c.labelled]
    by_id = {c.clip_id: c for c in clips}
    for c in clips:
        if not c.video_path.exists():
            logger.error("[eval] missing video for %s: %s", c.clip_id, c.video_path)
    clips = [c for c in clips if c.video_path.exists()]

    adapter = make_adapter(args.adapter, args.mode)
    meta: dict[str, Any] = {
        "name": args.name,
        "backend_dir": str(args.backend_dir or ""),
        "adapter": adapter.name,
        "mode": args.mode,
        "zones": f"drawn ({args.zones_file})" if args.zones_file else ("annotated" if args.zones else "none"),
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": f"{platform.node()} {platform.platform()} py{platform.python_version()}",
        "argv": sys.argv[1:],
        "overrides": args.set,
        "warnings": [],
    }
    meta.update(_device_info(tree))
    labelled = [c for c in clips if c.labelled]
    if not labelled:
        meta["warnings"].append(
            "No labelled clips in the eval set - detection recall, precision, ID switches, "
            "event recall and false-alert rates CANNOT be measured. Only performance, "
            "diagnostics and the synthetic behaviour suite are reported."
        )

    t0 = time.time()
    adapter.warmup()
    meta["warmup_s"] = round(time.time() - t0, 2)
    meta["active_providers"] = tree.active_providers()
    teacher = Teacher() if args.teacher else None
    tracer = _TraceWriter(args.trace) if args.trace else None

    drawn_zones: dict[str, Any] = json.loads(args.zones_file.read_text()) if args.zones_file else {}
    scored, runs = [], []
    try:
        if not args.no_clips:
            for c in clips:
                print(f"[eval] {c.clip_id} ({'labelled' if c.labelled else 'unlabelled'}, {c.split}) ...", flush=True)
                run = run_clip(
                    adapter,
                    c,
                    max_seconds=args.max_seconds,
                    use_zones=args.zones,
                    drawn_zones=drawn_zones.get(c.clip_id),
                    teacher=teacher if not c.labelled else None,
                    teacher_every=args.teacher_every,
                    trace=tracer,
                )
                runs.append(run)
                scored.append(R.score_clip(c, run, args.iou))
                lat = scored[-1]["latency"]
                print(
                    f"       {run.frames} frames, {run.analysed} analysed, p50 {lat.get('p50_ms')} ms, "
                    f"p99 {lat.get('p99_ms')} ms, alerts {len(run.alerts)}",
                    flush=True,
                )
    finally:
        if tracer is not None:
            tracer.close()
            meta["trace"] = {"path": str(args.trace), "records": tracer.records}

    extra: dict[str, Any] = {}
    if args.startup and clips:
        print("[eval] cold start ...", flush=True)
        extra["startup"] = measure_startup(args, str(clips[0].video_path))
    if args.upload_path and args.upload_path in by_id:
        print("[eval] upload path (render + encode) ...", flush=True)
        ad2 = make_adapter(args.adapter, args.mode, render=True)
        ad2.warmup()
        r2 = run_clip(ad2, by_id[args.upload_path], max_seconds=args.max_seconds)
        s2 = R.score_clip(by_id[args.upload_path], r2, args.iou)
        extra["upload_path"] = {
            "clip": args.upload_path,
            "realtime_factor": s2["realtime_factor"],
            "latency": s2["latency"],
            "analysed_fraction": s2["analysed_fraction"],
            "stages": s2["stages"],
        }
    if args.live_streams:
        cid = args.live_clip or (clips[0].clip_id if clips else "")
        print(f"[eval] live benchmark: {args.live_streams} stream(s) of {cid} ...", flush=True)
        extra["live"] = live_bench(args, tree, str(by_id[cid].video_path), cid)
    if args.behavior_suite:
        print("[eval] behaviour suite ...", flush=True)
        extra["behavior_suite"] = behavior_suite(args, tree)

    baseline = json.loads(args.compare.read_text(encoding="utf-8")) if args.compare else None
    if args.merge_into:
        rep = json.loads(args.merge_into.read_text(encoding="utf-8"))
        rep.update(extra)
        rep.setdefault("meta", {}).setdefault("merged_sections", []).append(
            {"sections": sorted(extra), "at": meta["started_at"], "argv": meta["argv"]}
        )
        rep["headline"] = R.headline(rep)
        rep["targets"] = R.targets(rep)
        j, _md = R.save(rep, args.merge_into.parent, baseline)
        print(f"[eval] merged {sorted(extra)} into {j}")
        return 0
    rep = R.build(scored, runs, meta, extra)
    out_dir = args.out or (args.eval_dir / "reports" / args.name)
    j, md = R.save(rep, out_dir, baseline)
    print(f"[eval] wrote {j}\n[eval] wrote {md}")
    for k, t in rep["targets"].items():
        print(f"   {k:32s} {t['status']:12s} value={t['value']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
