r"""Validate the LIVE (RTSP) path on labelled clips: MediaMTX + FFmpeg -> rtsp://127.0.0.1:<port>/<clip> -> LiveRuntime.

Each clip of eval/sprint_v1/clips.json is published once, in real time, by FFmpeg (`-re`; video copied when it is
H.264, else re-encoded with libx264 at up to 1920 px wide) to a private MediaMTX (loopback only, RTSP over TCP,
everything else off). The REAL live runtime (capture thread, frame queue, batched GPU scheduler, behaviour stage,
alert sink) analyses the stream as it would a camera, and the alerts it raises are scored with the same clip-level
rules as the upload path (lifeguard.eval.sprint): every drowning clip must alert, every normal clip must not. Also
reported per clip: the most people tracked at once, distinct track ids, analysed fraction and worst lag - the
owner's manual checks (more than 2 people at once, steady ids) as numbers.

Cost: one stream at a time, in real time, so a run takes the footage's duration; `--max-seconds` caps long clips.

    $env:PYTHONPATH="backend"; venv\Scripts\python.exe tools\rtsp_validate.py --split dev --out <dir> \
        [--det-model model.onnx] [--only ip_5491,ip_5495] [--max-seconds 120]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess  # nosec B404 - fixed argv to the local MediaMTX / FFmpeg binaries, no shell
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MTX_DIR = ROOT / "build" / "tools" / "mediamtx"
X264 = ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-g", "30", "-bf", "0", "-pix_fmt", "yuv420p"]
SCALE = ["-vf", "scale=min(1920\\,iw):-2"]
MTX_CONFIG = [
    "logLevel: warn", "rtsp: true", "rtspAddress: 127.0.0.1:{port}", "rtspTransports: [tcp]", "rtmp: false",
    "hls: false", "webrtc: false", "srt: false", "api: false", "metrics: false", "pprof: false", "playback: false",
    "moq: false", "paths:", "  all_others:",
]  # fmt: skip


def _tool(name: str, hint: Path | None = None) -> str:
    found = shutil.which(name)
    if found:
        return found
    if hint is not None:
        hits = sorted(hint.rglob(f"{name}.exe")) + sorted(hint.rglob(name))
        if hits:
            return str(hits[0])
    raise SystemExit(f"{name} not found (PATH, {hint})")


def _codec(ffprobe: str, video: Path) -> str:
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0"]
    out = subprocess.run([*cmd, str(video)], capture_output=True, text=True, check=False)  # noqa: S603  # nosec B603 - fixed argv, local binaries
    return out.stdout.strip()


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, default=ROOT / "eval" / "sprint_v1" / "clips.json")
    ap.add_argument("--split", default="dev", choices=("dev", "test", "all"))
    ap.add_argument("--only", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", default="High", choices=("Low", "Medium", "High"))
    ap.add_argument("--det-model", type=Path, default=None)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--port", type=int, default=8654, help="RTSP port of the private MediaMTX (not the owner's 8554)")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="cap each clip (0 = whole clip)")
    a = ap.parse_args(argv)

    from lifeguard.core.config import load_settings
    from lifeguard.eval.sprint import score_clip, summarise
    from lifeguard.live.runtime import CameraSpec, LiveRuntime
    from lifeguard.vision import config as vision_config

    ffmpeg, ffprobe, mtx = _tool("ffmpeg"), _tool("ffprobe"), _tool("mediamtx", MTX_DIR)
    work = Path(tempfile.mkdtemp(prefix="lifeguard-rtsp-val-"))
    (work / "mediamtx.yml").write_text("\n".join(MTX_CONFIG).format(port=a.port), encoding="ascii")
    quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    server = subprocess.Popen([mtx, str(work / "mediamtx.yml")], cwd=work, **quiet)  # noqa: S603  # nosec B603 - fixed argv, local binaries
    time.sleep(2.0)

    base = vision_config.from_env()
    det = str(a.det_model.resolve()) if a.det_model else base.det_model_path
    cfg = load_settings(env="test", data_dir=work / "data", bcrypt_rounds=4)
    fires: list[tuple[str, Any, float]] = []

    def fire_sink(camera_id: str, fire: Any, audible: bool, image: Any, muted_reason: str | None = None) -> None:
        fires.append((camera_id, fire, time.monotonic()))

    rt = LiveRuntime(cfg, det, a.provider or base.provider, lambda f, c: None, fire_sink)
    rt.start()
    only = {c for c in a.only.split(",") if c}
    doc = json.loads(a.clips.read_text(encoding="utf-8"))
    clips = [c for c in doc["clips"] if (a.split == "all" or c["split"] == a.split) and (not only or c["id"] in only)]
    rows: list[dict[str, Any]] = []
    try:
        for c in clips:
            video = ROOT / c["video"]
            url = f"rtsp://127.0.0.1:{a.port}/{c['id']}"
            vcodec = ["-c:v", "copy"] if _codec(ffprobe, video) == "h264" else [*X264, *SCALE]
            limit = ["-t", str(a.max_seconds)] if a.max_seconds > 0 else []
            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-re", "-i", str(video), *limit, "-an", *vcodec]
            pub = subprocess.Popen([*cmd, "-f", "rtsp", "-rtsp_transport", "tcp", url], **quiet)  # noqa: S603  # nosec B603 - fixed argv, local binaries
            time.sleep(1.5)
            cam_id = f"val-{c['id']}"
            rt.start_camera(CameraSpec(cam_id, c["id"], url, "rtsp", a.mode), enforce_capacity=False)
            cam = rt._cams[cam_id]
            t_first: float | None = None
            max_people, ids, max_lag = 0, set(), 0.0
            while pub.poll() is None:
                time.sleep(0.1)
                if t_first is None and cam.worker.analysed > 0:
                    t_first = time.monotonic()
                det_now = cam.worker.last_detections
                if det_now is not None:
                    people = [p.track_id for p in det_now.tracks if p.track_id >= 0]
                    max_people = max(max_people, len(people))
                    ids.update(people)
                max_lag = max(max_lag, cam.queue.lag_s())
            time.sleep(2.0)  # the frames still in the queue
            read, analysed = cam.capture.frames_read, cam.worker.analysed
            rt.stop_camera(cam_id)
            t0 = t_first if t_first is not None else time.monotonic()
            mine = [(f, at - t0) for cid, f, at in fires if cid == cam_id]
            alerts = [
                {
                    "t": round(max(0.0, at), 2),
                    "track": int(f.track_id),
                    "scenario": f.result.scenario,
                    "ghost": bool(f.ghost),
                    "onset_t": None if f.onset_t is None or f.t is None else round(max(0.0, at - (f.t - f.onset_t)), 2),
                }
                for f, at in mine
                if getattr(f, "notify", True)
            ]
            ongoing = sum(1 for f, _ in mine if not getattr(f, "notify", True))
            row = score_clip(c, alerts, read / (cam.capture.fps or 25.0))
            row.update(
                max_people_at_once=max_people,
                track_ids=len(ids),
                frames_read=read,
                analysed=analysed,
                analysed_fraction=round(analysed / read, 3) if read else None,
                max_lag_s=round(max_lag, 2),
                ongoing_updates=ongoing,
            )
            rows.append(row)
            hit = "DETECTED" if row.get("detected") else "MISSED"
            verdict = hit if c["label"] == "drowning" else f"FA={row['false_alerts']}"
            _say(
                f"{c['id']:22s} {c['label']:8s} {verdict:9s} alerts={len(alerts):3d} ongoing={ongoing:3d} "
                f"people_max={max_people:2d} ids={len(ids):3d} analysed={row['analysed_fraction']} lag={max_lag:.1f}s"
            )
    finally:
        rt.stop()
        server.terminate()
    report = {
        "format": "aquaguard-rtsp-validation/1",
        "split": a.split,
        "detector": Path(det).name,
        "mode": a.mode,
        "rows": rows,
        "summary": summarise(rows),
    }
    report["summary"]["max_people_at_once"] = max((r["max_people_at_once"] for r in rows), default=0)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "rtsp_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    _say(json.dumps(report["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
