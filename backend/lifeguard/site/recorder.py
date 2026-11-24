"""Synchronised multi-camera recording for the site session (Phase 11 WP6).

One FFmpeg per camera, `-c copy` (no re-encode), RTSP over TCP, with **wall-clock arrival timestamps kept
absolute** (`-use_wallclock_as_timestamps 1 -copyts`): every packet's pts is the server's UTC epoch time at
arrival - the same instant the live capture stamps a frame with. A recording's `start_time` is therefore an absolute
time, and a frame's time is start_time + its position. Measured (WP6.md): four recorders of ONE stream, started up to
2.5 s apart, give the same barcode frame the same time within 2 ms. Differences between REAL cameras (encoder /
network latency) are measured with the SYNC flash (site_report).

Recorders stop gracefully ('q' on stdin; `-t` cannot be used: with -copyts it compares absolute timestamps). FFmpeg's
stderr - which can echo the input URL - goes through the Redactor into the block's log. After a block: per file
start / end time, frames, arrival gaps > GAP_S, size, sha256, and a DATA_PROTOCOL 8 RIGHTS.csv row.

Supervised (found in the full integration suite, then measured: 1 recorder start in 30 fails on a first RTSP
packet without a timestamp, "Can't write packet with unknown timestamp"; -fflags +genpts and a setts filter did
not prevent it): a recorder that exits on its own is restarted into a new part file (never over
an existing one), so one bad packet never costs a camera's drill recording.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import itertools
import json
import logging
import re
import shutil
import subprocess  # nosec B404  # fixed-argv ffmpeg recorders, no shell
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lifeguard.site.sources import SiteCamera

logger = logging.getLogger(__name__)

GAP_S = 1.0
USABLE_BYTES = 4096  # a part smaller than this recorded nothing
SUPERVISE_S = 0.5
RESTART_DELAY_S = 0.5
MAX_RESTARTS = 5  # per camera and block
HEADROOM = 1.5  # disk needed = cameras x bitrate x seconds x HEADROOM
RIGHTS_COLUMNS = [
    "file",
    "sha256",
    "bytes",
    "source_evidence",
    "watermark",
    "rights_holder",
    "licence_on_file",
    "rights_status",
    "personal_data",
    "allowed_use",
    "trained_model_status",
    "incident_group",
    "byte_identical_copies_in_storage_uploads",
    "notes",
]


class RecorderError(RuntimeError):
    pass


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "camera"


def disk_needed_bytes(cameras: int, seconds: float, bitrate_kbps: float = 8000.0) -> int:
    return int(cameras * bitrate_kbps * 1000 / 8 * seconds * HEADROOM)


def check_disk(folder: Path, cameras: int, seconds: float, bitrate_kbps: float = 8000.0) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free
    need = disk_needed_bytes(cameras, seconds, bitrate_kbps)
    return {"free_gb": round(free / 1e9, 1), "needed_gb": round(need / 1e9, 1), "ok": free >= need}


@dataclass
class _Part:
    path: Path
    proc: subprocess.Popen[bytes]
    started_mono: float
    drain: threading.Thread | None = None


@dataclass
class _Rec:
    camera: SiteCamera
    parts: list[_Part] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    restarts: int = 0
    gave_up: bool = False


class BlockRecorder:
    """Records every camera of one block (the drills, or the normal session) until stop(), SUPERVISED: a recorder that
    exits on its own is restarted into a new part file, whether it produced nothing usable (the measured start-up
    failure: a first RTSP packet without a timestamp, 1 start in 30) or failed later. A file is never overwritten.
    At most MAX_RESTARTS restarts per camera."""

    def __init__(self, ffmpeg: Path, ffprobe: Path, folder: Path, block: str, redact: Callable[[str], str]) -> None:
        self.ffmpeg, self.ffprobe, self.folder, self.block, self.redact = ffmpeg, ffprobe, folder, block, redact
        self.recs: list[_Rec] = []
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._supervisor: threading.Thread | None = None

    # ── processes ──
    def _path(self, rec: _Rec, index: int) -> Path:
        base = f"{safe_name(self.block)}_{safe_name(rec.camera.name)}"
        return self.folder / (f"{base}.mkv" if index == 0 else f"{base}_part{index + 1}.mkv")

    def _argv(self, rec: _Rec, path: Path) -> list[str]:
        argv = [str(self.ffmpeg), "-hide_banner", "-nostats", "-loglevel", "warning", "-rtsp_transport", "tcp"]
        argv += ["-use_wallclock_as_timestamps", "1", "-i", rec.camera.url, "-map", "0:v:0", "-c", "copy", "-copyts"]
        # a Matroska cluster at least every second, written through at once: a recorder that dies hard loses ~1 s
        return [*argv, "-cluster_time_limit", "1000", "-flush_packets", "1", "-y", str(path)]

    def _launch(self, rec: _Rec, path: Path) -> None:
        argv = self._argv(rec, path)
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)  # noqa: S603  # nosec B603
        part = _Part(path, proc, time.monotonic())
        part.drain = threading.Thread(target=self._drain, args=(rec, proc), daemon=True)
        part.drain.start()
        rec.parts.append(part)

    def _drain(self, rec: _Rec, proc: subprocess.Popen[bytes]) -> None:
        if proc.stderr is None:
            return
        for raw in proc.stderr:
            rec.log_lines.append(self.redact(raw.decode("utf-8", "replace").rstrip()))

    @staticmethod
    def _close(part: _Part) -> None:
        if part.drain is not None:
            part.drain.join(5)
        for pipe in (part.proc.stdin, part.proc.stderr):
            if pipe is not None:
                with contextlib.suppress(OSError):
                    pipe.close()

    def start(self, cameras: list[SiteCamera]) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        for cam in cameras:
            rec = _Rec(cam)
            self._launch(rec, self._path(rec, 0))
            self.recs.append(rec)
        self._supervisor = threading.Thread(target=self._supervise, name="recorder-supervisor", daemon=True)
        self._supervisor.start()

    def _supervise(self) -> None:
        while not self._stopping.wait(SUPERVISE_S):
            with self._lock:
                for rec in self.recs:
                    try:
                        self._check(rec)
                    except Exception as exc:  # the supervisor must outlive any one camera's problem
                        rec.events.append({"wall": time.time(), "event": "supervisor_error", "error": repr(exc)})
                        logger.exception("recorder supervisor: %s", rec.camera.name)

    def _check(self, rec: _Rec) -> None:
        part = rec.parts[-1]
        rc = part.proc.poll()
        if rc is None or self._stopping.is_set() or rec.gave_up:
            return
        self._close(part)
        usable = part.path.is_file() and part.path.stat().st_size >= USABLE_BYTES
        last = rec.log_lines[-1] if rec.log_lines else ""
        if rec.restarts >= MAX_RESTARTS:
            rec.gave_up = True
            rec.events.append({"wall": time.time(), "event": "gave_up", "exit_code": rc, "last_log": last})
            return
        rec.restarts += 1
        # NEVER overwrite: a hard-killed recorder's footage may still look unusable on disk (found: a kill after 4 s
        # left a header-only file that a same-file retry would have destroyed). Every restart writes a new part; the
        # description marks unusable parts and never picks one as the main file.
        path = self._path(rec, len(rec.parts))
        kind = "new_part" if usable else "startup_retry"
        rec.events.append({"wall": time.time(), "event": kind, "exit_code": rc, "last_log": last, "file": path.name})
        time.sleep(RESTART_DELAY_S)
        self._launch(rec, path)

    def alive(self) -> dict[str, bool]:
        return {r.camera.name: bool(r.parts) and r.parts[-1].proc.poll() is None for r in self.recs}

    def stop(self, timeout: float = 30.0) -> list[dict[str, Any]]:
        self._stopping.set()
        if self._supervisor is not None:
            self._supervisor.join(SUPERVISE_S * 4 + RESTART_DELAY_S + 5)
        with self._lock:
            for r in self.recs:
                p = r.parts[-1].proc
                if p.poll() is None and p.stdin is not None:
                    try:
                        p.stdin.write(b"q")
                        p.stdin.flush()
                    except OSError:
                        pass
            for r in self.recs:
                for part in r.parts:
                    try:
                        part.proc.wait(timeout)
                    except subprocess.TimeoutExpired:
                        part.proc.kill()
                        part.proc.wait(10)
                        r.log_lines.append("recorder killed after a stop timeout")
                    self._close(part)
        out = [self._describe(r) for r in self.recs]
        log = self.folder / f"{safe_name(self.block)}_recorders.log"
        log.write_text(
            "\n".join(f"[{r.camera.name}] {line}" for r in self.recs for line in r.log_lines) + "\n", encoding="utf-8"
        )
        return out

    def _describe(self, r: _Rec) -> dict[str, Any]:
        """The camera's main recording (the longest usable part) + every part + the restart events."""
        parts = []
        for part in r.parts:
            d: dict[str, Any] = {"file": part.path.name, "exit_code": part.proc.returncode}
            if part.path.is_file() and part.path.stat().st_size >= USABLE_BYTES:
                d.update(describe_recording(part.path, self.ffprobe))
            else:
                d["error"] = "no usable recording"
            parts.append(d)
        usable = [p for p in parts if "start_wall" in p]
        info: dict[str, Any] = {"camera_id": r.camera.camera_id, "camera": r.camera.name}
        if usable:
            info.update(max(usable, key=lambda p: p["frames"]))
        else:
            info.update({"file": r.parts[-1].path.name if r.parts else "", "error": "no usable recording"})
        info.update({"parts": parts, "restarts": r.restarts, "events": r.events, "gave_up": r.gave_up})
        return info


def describe_recording(path: Path, ffprobe: Path) -> dict[str, Any]:
    """Absolute start / end, frames, arrival gaps, size, sha256 of a wall-clock recording."""
    argv = [str(ffprobe), "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time:format=start_time"]
    argv += ["-of", "json", str(path)]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=600, check=False)  # noqa: S603  # nosec B603
        data = json.loads(done.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:  # never lose the OTHER cameras' descriptions
        return {"error": f"ffprobe could not read the recording ({type(exc).__name__})"}
    pts = sorted(float(p["pts_time"]) for p in data.get("packets", []) if p.get("pts_time") not in (None, "N/A"))
    if len(pts) < 2:
        return {"error": "fewer than 2 frames"}
    gaps = [(a, round(b - a, 3)) for a, b in itertools.pairwise(pts) if b - a > GAP_S]
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {
        "start_wall": pts[0],
        "end_wall": pts[-1],
        "frames": len(pts),
        "fps": round((len(pts) - 1) / (pts[-1] - pts[0]), 3),
        "gaps": [{"at_wall": a, "seconds": s} for a, s in gaps],
        "bytes": path.stat().st_size,
        "sha256": h.hexdigest(),
    }


def write_rights_rows(
    path: Path, recordings: list[dict[str, Any]], *, site: str, session_id: str, dsa_ref: str
) -> None:
    """DATA_PROTOCOL 8 inventory rows. rights_status stays 'pending' until the agreement is countersigned."""
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        if new:
            w.writerow(RIGHTS_COLUMNS)
        for r in recordings:
            if "sha256" not in r:
                continue
            w.writerow(
                [
                    r["file"],
                    r["sha256"],
                    r["bytes"],
                    f"production camera {r['camera']} RTSP, site session {session_id}",
                    "no",
                    f"{site} operator",
                    f"yes ({dsa_ref})",
                    "pending",
                    "yes - adults in swimwear",
                    "test",
                    "",
                    session_id,
                    "",
                    "WP6 site session recording (adults only, attested per run)",
                ]
            )
