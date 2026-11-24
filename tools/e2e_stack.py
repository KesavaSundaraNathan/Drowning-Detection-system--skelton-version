"""Browser end-to-end stack for Playwright (Phase 7): the real API in the demo profile, a scratch data dir.

Started by frontend/playwright.config.ts as a `webServer`; it replaces the manual real-browser walkthrough
of the Phase 6 report (section 3.1). What it does:

  * a fresh data dir under build/e2e/<run>/ - `lifeguard keys init` (DPAPI), `db upgrade`, `admin create`;
  * the API (uvicorn, 127.0.0.1:<api-port>, LIFEGUARD_ENV=demo, Secure cookies - Chromium treats
    http://localhost as a secure context);
  * a private MediaMTX on 127.0.0.1:<rtsp-port> with FFmpeg looping the `web_lane_pool_1080p` eval clip into
    rtsp://127.0.0.1:<rtsp-port>/cam0 (Life Guard is RTSP-only). MediaMTX and FFmpeg come from LIFEGUARD_MEDIAMTX /
    LIFEGUARD_FFMPEG, PATH or build/tools; without them (hosted CI) the cameras get the same URL and simply do not
    connect, which the operator flows do not need (`--live` refuses to start without them);
  * the live runtime is always on (with it off the UI correctly shows NOT MONITORING over everything);
    `--live` puts the detector on strict TensorRT (engines seeded by tools/trt_seed.py; CI `-Gpu`, cameras are
    started and analysed), without it on the CPU provider, so the operator flows run on any machine (no camera is
    started there);
  * a loopback-only control endpoint (<control-port>) the tests use to crash and restart the API:
        POST /api/kill   - TerminateProcess on the API (as a crash would; no graceful shutdown)
        POST /api/start  - start it again on the same data dir
        POST /alerts/inject - [Phase 11] raise alerts through the real alert writer (tools/e2e_inject.py,
                           body {"fires": [{"camera": <name>, "global_id": <int|null>, "scenario": ...}]})
        GET  /status     - {"api": "running" | "stopped", "data_dir": ...}

The admin credentials are test values from frontend/e2e/admin.fixture.json (also read by e2e/fixtures.ts), passed in
the environment (E2E_ADMIN_USER / E2E_ADMIN_PASSWORD; unset: read from that file). The stack stops when this process
is terminated.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess  # nosec B404  # fixed-argv calls to this repo's own python, no shell
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import trt_seed  # noqa: E402

ADMIN_FIXTURE = ROOT / "frontend" / "e2e" / "admin.fixture.json"  # test-only e2e login (also used by Playwright)
POSE_MODEL = ROOT / "backend" / "models" / "rtmpose-m_8xb256-420e_coco-256x192_dynbatch.onnx"
CLIP = ROOT / "build" / "v2_baseline" / "backend" / "storage" / "uploads" / "5696203ffd4b445d9508dcb788477f2f.mp4"
CLIP_NAME = "web_lane_pool_1080p.mp4"
TOOLS = ROOT / "build" / "tools"
MTX_CONFIG = [
    "logLevel: warn", "rtsp: true", "rtspAddress: 127.0.0.1:{port}", "rtspTransports: [tcp]", "rtmp: false",
    "hls: false", "webrtc: false", "srt: false", "api: false", "metrics: false", "pprof: false", "playback: false",
    "moq: false", "paths:", "  all_others:",
]  # fmt: skip


def _tool(env_var: str, name: str) -> str | None:
    explicit = os.environ.get(env_var)
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which(name)
    if found:
        return found
    hits = sorted((TOOLS / name).rglob(f"{name}.exe")) + sorted((TOOLS / name).rglob(name))
    return next((str(h) for h in hits if h.is_file()), None)


def _log(msg: str) -> None:
    sys.stderr.write(f"[e2e-stack] {msg}\n")
    sys.stderr.flush()


def _synthetic_clip(path: Path, frames: int = 50) -> None:
    import cv2
    import numpy as np

    out = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 25.0, (640, 360))
    for i in range(frames):
        img = np.full((360, 640, 3), (180, 120, 40), np.uint8)
        cv2.circle(img, (40 + i * 10, 180), 20, (230, 230, 230), -1)
        out.write(img)
    out.release()
    _log(f"eval clip not present: wrote a synthetic {frames}-frame clip to {path}")


class Stack:
    def __init__(self, data: Path, api_port: int, live: bool, rtsp_port: int) -> None:
        self.data, self.api_port, self.live, self.rtsp_port = data, api_port, live, rtsp_port
        self.proc: subprocess.Popen[bytes] | None = None
        self.rtsp: list[subprocess.Popen[bytes]] = []
        self.lock = threading.Lock()
        media = data / "media"
        media.mkdir(parents=True, exist_ok=True)
        if CLIP.is_file():
            shutil.copyfile(CLIP, media / CLIP_NAME)
        elif live:
            raise SystemExit(f"--live needs the eval clip {CLIP} (build/v2_baseline, see Phase 3 report)")
        else:  # hosted CI has no eval footage: the operator flows only need a valid video file
            _synthetic_clip(media / CLIP_NAME)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("LIFEGUARD_")}
        self.env.update(
            {
                "PYTHONPATH": str(ROOT / "backend"),
                "LIFEGUARD_ENV": "demo",
                "LIFEGUARD_DATA_DIR": str(data),
                "LIFEGUARD_LIVE_ENABLED": "true",
                "LIFEGUARD_VISION_STARTUP_CHECK": "true",
                "LIFEGUARD_INFERENCE_PROVIDER": "tensorrt" if live else "cpu",
                # pose ON when the exported RTMPose model is installed, else an explicit no-pose run (hosted CI)
                "LIFEGUARD_POSE_MODEL": "" if POSE_MODEL.is_file() else "none",
                "LIFEGUARD_SUPERVISOR_EXPECTED": "false",
                "LIFEGUARD_ESCALATE_AFTER_S": "5",
            }
        )
        if not live:
            # The camera registered by the operator flows is started on every API (re)start; at the product's 18 fps
            # target a CPU detector cannot carry it, admission refuses it (CAPACITY_EXCEEDED) and NOT MONITORING stays
            # up - correct, but it is not what the restart test checks. 1 fps fits one camera on a CPU
            # (detector + pose).
            self.env["LIFEGUARD_TARGET_ANALYSIS_FPS"] = "1"
        trt_seed.seed_stack_env(self.env, data)

    def init(self, user: str, password: str) -> None:
        for argv in (["keys", "init"], ["db", "upgrade"]):
            subprocess.run([sys.executable, "-m", "lifeguard", *argv], env=self.env, cwd=ROOT, check=True)  # noqa: S603  # nosec B603
        subprocess.run(  # noqa: S603  # nosec B603
            [sys.executable, "-m", "lifeguard", "admin", "create", "--username", user, "--password-stdin"],
            env=self.env,
            cwd=ROOT,
            check=True,
            input=password.encode(),
        )

    def start_rtsp(self) -> None:
        """MediaMTX + one FFmpeg publisher looping the clip into rtsp://127.0.0.1:<rtsp-port>/cam0."""
        mtx, ffmpeg = _tool("LIFEGUARD_MEDIAMTX", "mediamtx"), _tool("LIFEGUARD_FFMPEG", "ffmpeg")
        if not (mtx and ffmpeg):
            if self.live:
                raise SystemExit("--live needs MediaMTX and FFmpeg (LIFEGUARD_MEDIAMTX / _FFMPEG or build/tools)")
            _log("MediaMTX/FFmpeg not found: the RTSP test camera will not connect (fine without --live)")
            return
        cfg = self.data / "mediamtx.yml"
        cfg.write_text(chr(10).join(MTX_CONFIG).format(port=self.rtsp_port) + chr(10), encoding="ascii")
        log = (self.data / "rtsp.out.log").open("ab")
        self.rtsp.append(subprocess.Popen([mtx, str(cfg)], cwd=self.data, stdout=log, stderr=subprocess.STDOUT))  # noqa: S603  # nosec B603
        deadline = time.monotonic() + 20
        while not _listening(self.rtsp_port):
            if time.monotonic() > deadline:
                raise RuntimeError(f"MediaMTX did not open :{self.rtsp_port}; see {self.data / 'rtsp.out.log'}")
            time.sleep(0.2)
        clip = self.data / "media" / CLIP_NAME
        argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", str(clip), "-an"]
        argv += ["-c:v", "copy", "-f", "rtsp", "-rtsp_transport", "tcp", f"rtsp://127.0.0.1:{self.rtsp_port}/cam0"]
        self.rtsp.append(subprocess.Popen(argv, cwd=self.data, stdout=log, stderr=subprocess.STDOUT))  # noqa: S603  # nosec B603
        _log(f"RTSP test camera on rtsp://127.0.0.1:{self.rtsp_port}/cam0")

    def stop_rtsp(self) -> None:
        for p in reversed(self.rtsp):
            if p.poll() is None:
                p.kill()
                p.wait(10)

    def start(self) -> None:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return
            out = (self.data / "api.out.log").open("ab")
            argv = [sys.executable, "-m", "uvicorn", "--factory", "lifeguard.app.main:create_app"]
            argv += ["--host", "127.0.0.1", "--port", str(self.api_port), "--workers", "1", "--no-access-log"]
            self.proc = subprocess.Popen(argv, env=self.env, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)  # noqa: S603  # nosec B603
        deadline = time.monotonic() + 300  # the runtime measures capacity at startup (slow on the CPU)
        while time.monotonic() < deadline:
            if self.ready():
                _log(f"API ready on :{self.api_port}")
                return
            if self.proc.poll() is not None:
                break
            time.sleep(0.5)
        raise RuntimeError(f"API did not become ready; see {self.data / 'api.out.log'}")

    def kill(self) -> None:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.kill()  # TerminateProcess: no shutdown hooks, like a crash
                self.proc.wait(30)

    def ready(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.api_port}/api/v1/health/ready", timeout=2) as r:  # nosec B310  # fixed loopback http URL
                return bool(r.status == 200)
        except (urllib.error.URLError, OSError):
            return False

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def inject(self, body: bytes) -> dict[str, object]:
        r = subprocess.run(  # noqa: S603  # nosec B603
            [sys.executable, str(ROOT / "tools" / "e2e_inject.py")],
            env=self.env,
            cwd=ROOT,
            input=body,
            capture_output=True,
            check=True,
            timeout=120,
        )
        return dict(json.loads(r.stdout.decode().strip().splitlines()[-1]))


def _listening(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _control(stack: Stack, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: dict[str, object]) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/status":
                self._reply(200, {"api": "running" if stack.running() else "stopped", "data_dir": str(stack.data)})
            else:
                self._reply(404, {"detail": "not_found"})

        def do_POST(self) -> None:
            try:
                if self.path == "/alerts/inject":
                    n = int(self.headers.get("Content-Length") or 0)
                    self._reply(200, stack.inject(self.rfile.read(n)))
                    return
                if self.path == "/api/kill":
                    stack.kill()
                elif self.path == "/api/start":
                    stack.start()
                else:
                    self._reply(404, {"detail": "not_found"})
                    return
            except Exception as exc:  # noqa: BLE001 - reported to the test, which fails with it
                self._reply(500, {"detail": str(exc)})
                return
            self._reply(200, {"api": "running" if stack.running() else "stopped"})

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--control-port", type=int, default=8099)
    ap.add_argument("--live", action="store_true", help="detector on CUDA (needs the GPU); default CPU")
    ap.add_argument("--rtsp-port", type=int, default=18554, help="private MediaMTX for the RTSP test camera")
    args = ap.parse_args()
    user, password = os.environ.get("E2E_ADMIN_USER"), os.environ.get("E2E_ADMIN_PASSWORD")
    if not user or not password:  # [Phase 6] the tracked test-only fixture (also read by frontend/e2e/fixtures.ts)
        fixture = json.loads(ADMIN_FIXTURE.read_text(encoding="utf-8"))
        user, password = fixture["username"], fixture["password"]
    run = datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
    data = ROOT / "build" / "e2e" / run
    data.mkdir(parents=True)
    stack = Stack(data, args.api_port, args.live, args.rtsp_port)
    if stack.ready() or any(_listening(p) for p in (args.api_port, args.control_port, args.rtsp_port)):
        # a stale stack from an earlier run would answer instead of this one: the tests would be void
        _log(f"port {args.api_port}, {args.control_port} or {args.rtsp_port} is in use - stop the stale stack first")
        return 3
    stack.init(user, password)
    server: ThreadingHTTPServer | None = None
    try:
        stack.start_rtsp()
        stack.start()  # first: Playwright waits for the control endpoint, so it must mean "API ready"
        server = _control(stack, args.control_port)
        _log(f"control on :{args.control_port}; data {data}")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.shutdown()
        stack.kill()  # also when start() failed: never leave an API behind on the port
        stack.stop_rtsp()
    return 0


if __name__ == "__main__":
    sys.exit(main())
