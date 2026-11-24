"""Site replay (Phase 11 WP6, decision 3; arms pre-registered in eval/site/levers_phase11.json "gates_wp6".replay_arms).

    python -m lifeguard.eval.site_replay --session <DataDir>/site_session/<id> --block drills
        [--arms OFF,GATING,MULTIVIEW]

Runs a session block's synchronised recordings through the REAL LiveRuntime with "replay" sources (live/replay.py):
every camera on the session clock, lockstep, nothing dropped, zones and calibration built by the live geometry loader
from the session's geometry snapshot (no secrets in it), zone schedules evaluated at the RECORDING time.

Arms run in separate processes (the switches are module constants read from the thresholds file): each gets a complete
thresholds file with every Phase 11 switch pinned. The OFF arm runs the detector (GPU) and records every detector call
(camera, call number, an input digest, the boxes / scores / water votes); GATING and MULTIVIEW replay that tape - a
call whose input digest differs aborts the arm, so the arms differ only by their switch.

Per arm: trace.jsonl.gz (per processed frame: camera, session time, tracks with local + global id, box, in-water) and
alerts.json (every fire: camera, time, scenario, global id, audible / muted reason), summary.json.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import subprocess  # nosec B404  # fixed-argv ffmpeg calls on session recordings, no shell
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PHASE11_SWITCHES = (
    "POOL_HANDOFF_GUARD",
    "POOL_ID_REPAIR",
    "POOL_RANK_BY_DISTANCE",
    "RESPONSIBILITY_GATING",
    "MULTIVIEW_CONFIRM",
    "GLARE_HOLD",
    "DET_TILING",
    "DET_ZOOM",
)
ARMS: dict[str, dict[str, Any]] = {
    "OFF": {
        "POOL_HANDOFF_GUARD": True,
        "POOL_ID_REPAIR": True,
        "POOL_RANK_BY_DISTANCE": False,
        "RESPONSIBILITY_GATING": False,
        "MULTIVIEW_CONFIRM": False,
        "GLARE_HOLD": False,
        "DET_TILING": "off",
        "DET_ZOOM": False,
    },
}
ARMS["GATING"] = {**ARMS["OFF"], "RESPONSIBILITY_GATING": True}
ARMS["MULTIVIEW"] = {**ARMS["OFF"], "MULTIVIEW_CONFIRM": True}
RECORDING_ARM = "OFF"


class ReplayMismatchError(RuntimeError):
    """A replayed detector call does not match the recorded one: the arms would not be comparable."""


# ── arm thresholds ────────────────────────────────────────────────────────────────────────────────────────────
def write_arm_thresholds(arm: str, out: Path, base: Path | None = None) -> Path:
    """A complete thresholds file = the shipped one with every Phase 11 switch pinned for `arm`."""
    from lifeguard.vision import thresholds as th

    src = base or th.DEFAULT_FILE
    doc = json.loads(Path(src).read_text(encoding="utf-8"))
    pins = ARMS[arm]
    seen = set()
    for section in doc["values"].values():
        for key in list(section):
            if key in pins:
                section[key] = pins[key]
                seen.add(key)
    if seen != set(pins):
        raise ValueError(f"switches not in the thresholds file: {sorted(set(pins) - seen)}")
    version = f"{doc['version']}+site-arm-{arm.lower()}"
    doc["version"] = version
    doc["changelog"].append(
        {
            "version": version,
            "date": "replay",
            "change": f"WP6 site replay arm {arm}: " + json.dumps(pins, sort_keys=True),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(doc, indent=2) + "\n").encode("utf-8"))
    th.parse(out.read_bytes(), str(out))  # refuse a file the runtime would refuse
    return out


# ── detection tape ────────────────────────────────────────────────────────────────────────────────────────────
def input_digest(img: np.ndarray) -> str:
    sub = np.ascontiguousarray(img[::8, ::8])
    return hashlib.blake2b(sub.tobytes() + repr(img.shape).encode(), digest_size=12).hexdigest()


class DetectionTape:
    """Records / replays `decode.run_detector_people` per camera (BatchedDetModel.client = camera id)."""

    def __init__(self, folder: Path, mode: str) -> None:
        if mode not in ("record", "replay"):
            raise ValueError(mode)
        self.folder, self.mode = folder, mode
        self._lock = threading.Lock()
        self.calls: dict[str, list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]] = {}
        self.cursor: dict[str, int] = {}
        self.mismatch: str | None = None
        if mode == "replay":
            self._load()

    def install(self) -> None:
        from lifeguard.vision.inference import decode

        real = decode.run_detector_people
        tape = self

        def run_detector_people(dm: Any, img: np.ndarray, *args: Any, **kwargs: Any) -> Any:
            cam = str(getattr(dm, "client", "?"))
            digest = input_digest(img)
            if tape.mode == "record":
                boxes, scores, water = real(dm, img, *args, **kwargs)
                with tape._lock:
                    tape.calls.setdefault(cam, []).append(
                        (
                            digest,
                            np.asarray(boxes, np.float32),
                            np.asarray(scores, np.float32),
                            np.asarray(water, np.float32),
                        )
                    )
                return boxes, scores, water
            with tape._lock:
                i = tape.cursor.get(cam, 0)
                recorded = tape.calls.get(cam, [])
                if i >= len(recorded) or recorded[i][0] != digest:
                    tape.mismatch = f"camera {cam} call {i}: input differs from the recorded run"
                    raise ReplayMismatchError(tape.mismatch)
                tape.cursor[cam] = i + 1
                _, b, s, w = recorded[i]
            return b.copy(), s.copy(), w.copy()

        decode.run_detector_people = run_detector_people  # type: ignore[assignment]

    def save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        for cam, calls in self.calls.items():
            n = np.array([len(c[1]) for c in calls], np.int32)
            np.savez_compressed(
                self.folder / f"{cam}.npz",
                digests=np.array([c[0] for c in calls]),
                counts=n,
                boxes=np.concatenate([c[1].reshape(-1, 4) for c in calls]) if calls else np.zeros((0, 4), np.float32),
                scores=np.concatenate([c[2].reshape(-1) for c in calls]) if calls else np.zeros(0, np.float32),
                water=np.concatenate([c[3].reshape(-1) for c in calls]) if calls else np.zeros(0, np.float32),
            )

    def _load(self) -> None:
        for f in sorted(self.folder.glob("*.npz")):
            d = np.load(f)
            ends = np.cumsum(d["counts"])
            starts = ends - d["counts"]
            self.calls[f.stem] = [
                (str(dg), d["boxes"][a:b], d["scores"][a:b], d["water"][a:b])
                for dg, a, b in zip(d["digests"], starts, ends, strict=True)
            ]

    def unused(self) -> dict[str, int]:
        return {c: len(v) - self.cursor.get(c, 0) for c, v in self.calls.items() if len(v) != self.cursor.get(c, 0)}


# ── session ───────────────────────────────────────────────────────────────────────────────────────────────────
def load_block(session: Path, block: str) -> dict[str, Any]:
    meta = json.loads((session / "session.json").read_text(encoding="utf-8"))
    recs = json.loads((session / "recordings.json").read_text(encoding="utf-8"))["blocks"][block]
    usable = [r for r in recs if "start_wall" in r]
    if not usable:
        raise ValueError(f"block {block!r} has no usable recording")
    sync_file = session / "sync.json"
    offsets = (
        json.loads(sync_file.read_text(encoding="utf-8")).get("applied_offsets_s", {}) if sync_file.exists() else {}
    )
    return {
        "meta": meta,
        "recordings": usable,
        "origin_wall": min(r["start_wall"] for r in usable),
        "offsets": offsets,
        "folder": session / "recordings" / block,
    }


# recording wall time per camera, for zone schedules (monitoring_active) during the replay
_REPLAY_NOW: dict[str, float] = {}


def replay_geometry_cls() -> Any:
    from lifeguard.vision.scene.geometry import ZoneGeometry

    class ReplayZoneGeometry(ZoneGeometry):
        def monitoring_active(self, hit: Any, now: datetime | None = None) -> bool:
            wall = _REPLAY_NOW.get(self.camera_id)
            if now is None and wall is not None:
                now = datetime.fromtimestamp(wall).astimezone()
            return bool(super().monitoring_active(hit, now))

    return ReplayZoneGeometry


# ── one arm (runs in its own process) ─────────────────────────────────────────────────────────────────────────
def run_arm(session: Path, block: str, arm: str, out: Path, *, det_model: str | None, provider: str) -> dict[str, Any]:
    from lifeguard.app.live_startup import make_geometry_loader
    from lifeguard.core.config import load_settings
    from lifeguard.db.session import make_database
    from lifeguard.live.replay import ReplayCoordinator, ReplayFeed
    from lifeguard.live.runtime import CameraSpec, LiveRuntime
    from lifeguard.vision import config as vision_config
    from lifeguard.vision import thresholds as th

    active = th.active()
    if not active.version.endswith(f"+site-arm-{arm.lower()}"):
        raise RuntimeError(f"arm {arm} must run with its own thresholds file (got {active.version})")
    data = load_block(session, block)
    tape = DetectionTape(session / "replay" / block / "detections", "record" if arm == RECORDING_ARM else "replay")
    tape.install()
    det = det_model or str(vision_config.MODELS_DIR / vision_config.DEFAULT_DET_MODEL)
    vision_config.configure(vision_config.VisionConfig(det, provider))
    work = out / "runtime"
    cfg = load_settings(env="test", data_dir=work, bcrypt_rounds=4, capacity_cache=False, capacity_overhead_ms=5.0)
    trace_path = out / "trace.jsonl.gz"
    trace = gzip.open(trace_path, "wt", encoding="utf-8")  # noqa: SIM115 - closed in finally
    alerts: list[dict[str, Any]] = []
    faults: list[dict[str, Any]] = []
    last_t: dict[str, float] = {}
    lock = threading.Lock()

    def on_start(cid: str, frame: Any) -> None:
        last_t[cid] = frame.t
        _REPLAY_NOW[cid] = data["origin_wall"] + frame.t

    def on_frame(cid: str, frame: Any, worker: Any) -> None:
        dets = getattr(worker, "last_detections", None)
        scene = getattr(getattr(worker, "pipeline", None), "scene", None)
        tracks = []
        if dets is not None and getattr(dets, "frame_index", None) == frame.index:
            for p in dets.tracks:
                ts = scene.tracks.get(p.track_id) if scene is not None and p.track_id >= 0 else None
                hit = getattr(ts, "hit", None)
                x1, y1, x2, y2 = (round(float(v), 1) for v in p.bbox)
                tracks.append(
                    [
                        p.track_id,
                        None,  # [Phase 5] was the pool-wide global id
                        x1,
                        y1,
                        x2,
                        y2,
                        round(float(p.confidence), 3),
                        bool(getattr(hit, "in_water", False)),
                    ]
                )
        with lock:
            trace.write(json.dumps({"cam": cid, "t": round(frame.t, 4), "i": frame.index, "tracks": tracks}) + "\n")

    def fire_sink(camera_id: str, fire: Any, audible: bool, image: Any, muted_reason: str | None = None) -> None:
        cid = camera_id
        with lock:
            alerts.append(
                {
                    "cam": cid,
                    "t": last_t.get(cid),
                    "scenario": fire.result.scenario,
                    "track_id": fire.track_id,
                    "global_id": None,  # [Phase 5] no cross-camera identity
                    "first": fire.first,
                    "ghost": fire.ghost,
                    "audible": bool(audible),
                    "muted_reason": muted_reason,
                    "zone": fire.zone_name,
                }
            )

    def fault_sink(f: Any, cid: str | None) -> None:
        with lock:
            faults.append({"code": f.code, "camera": cid, "t": last_t.get(cid or ""), "message": f.message})

    rt = LiveRuntime(cfg, det, provider, fault_sink, fire_sink)
    snapshot = make_database(f"sqlite:///{(session / 'geometry.sqlite').as_posix()}")
    coord = ReplayCoordinator(step_s=1.0 / 15.0)
    geo_cls = replay_geometry_cls()
    t0 = time.monotonic()
    try:
        rt.start(expected_cameras=len(data["recordings"]))
        for r in data["recordings"]:
            cid = r["camera_id"]
            feed = ReplayFeed(
                r["start_wall"],
                data["origin_wall"],
                coord,
                offset_s=float(data["offsets"].get(cid, 0.0)),
                on_frame=on_frame,
                on_start=on_start,
            )
            spec = CameraSpec(
                cid,
                r["camera"],
                str(data["folder"] / r["file"]),
                "replay",
                geometry_loader=make_geometry_loader(snapshot, cid, geometry_cls=geo_cls),
                replay=feed,
            )
            rt.start_camera(spec, enforce_capacity=False)
        while not (coord.idle() and all(c.capture.replay_finished for c in rt._cams.values())):
            if tape.mismatch:
                raise ReplayMismatchError(tape.mismatch)
            time.sleep(0.2)
        time.sleep(0.5)  # the last frames' alerts reach the sinks
    finally:
        rt.stop()
        snapshot.dispose()
        trace.close()
    if tape.mismatch:
        raise ReplayMismatchError(tape.mismatch)
    if arm == RECORDING_ARM:
        tape.save()
    elif tape.unused():
        raise ReplayMismatchError(f"recorded detector calls not replayed: {tape.unused()}")
    summary = {
        "arm": arm,
        "block": block,
        "switches": ARMS[arm],
        "thresholds": active.describe(),
        "cameras": {c: {"frames": None} for c in last_t},
        "origin_wall": data["origin_wall"],
        "offsets_s": data["offsets"],
        "alerts": len(alerts),
        "faults": faults,
        "seconds": round(time.monotonic() - t0, 1),
        "detector": "recorded" if arm == RECORDING_ARM else "replayed",
    }
    (out / "alerts.json").write_text(json.dumps(alerts, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


# ── orchestration ─────────────────────────────────────────────────────────────────────────────────────────────
def run_arms(
    session: Path, block: str, arms: list[str], *, det_model: str | None = None, provider: str = "auto"
) -> dict[str, Any]:
    """OFF first (records the detector), then the others in parallel (replay the tape). Each arm its own process."""
    base = session / "replay" / block
    results: dict[str, Any] = {}

    def launch(arm: str) -> subprocess.Popen[bytes]:
        out = base / arm
        thr = write_arm_thresholds(arm, out / "thresholds.json")
        import lifeguard

        pkg_root = str(Path(lifeguard.__file__).resolve().parents[1])  # the arm imports THIS lifeguard
        path = os.pathsep.join([pkg_root, *([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])])
        env = {**os.environ, "LIFEGUARD_THRESHOLDS_FILE": str(thr), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": path}
        argv = [sys.executable, "-m", "lifeguard.eval.site_replay", "--session", str(session), "--block", block]
        argv += ["--arm", arm, "--provider", provider, *(["--det-model", det_model] if det_model else [])]
        with (out / "arm.log").open("wb") as log:  # the child keeps its own handle
            return subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)  # noqa: S603  # nosec B603

    order = [RECORDING_ARM] + [a for a in arms if a != RECORDING_ARM]
    if RECORDING_ARM not in arms and not (base / "detections").exists():
        raise ValueError("the OFF arm (detector recording) has to run first")
    first = launch(RECORDING_ARM) if RECORDING_ARM in arms else None
    if first is not None and first.wait() != 0:
        raise RuntimeError(f"arm OFF failed: see {base / 'OFF' / 'arm.log'}")
    rest = {a: launch(a) for a in order[1:]}
    for a, p in rest.items():
        if p.wait() != 0:
            raise RuntimeError(f"arm {a} failed: see {base / a / 'arm.log'}")
    for a in arms:
        results[a] = json.loads((base / a / "summary.json").read_text(encoding="utf-8"))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--block", default="drills")
    ap.add_argument("--arm", default=None, help="run ONE arm in this process (needs its thresholds file)")
    ap.add_argument("--arms", default="OFF,GATING,MULTIVIEW")
    ap.add_argument("--provider", default="auto", choices=("auto", "tensorrt", "cuda", "openvino", "cpu"))
    ap.add_argument("--det-model", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.arm:
        run_arm(
            args.session,
            args.block,
            args.arm,
            args.session / "replay" / args.block / args.arm,
            det_model=args.det_model,
            provider=args.provider,
        )
        return 0
    res = run_arms(
        args.session,
        args.block,
        [a.strip() for a in args.arms.split(",") if a.strip()],
        det_model=args.det_model,
        provider=args.provider,
    )
    sys.stdout.write(
        json.dumps({a: {k: v[k] for k in ("alerts", "seconds", "detector")} for a, v in res.items()}) + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
