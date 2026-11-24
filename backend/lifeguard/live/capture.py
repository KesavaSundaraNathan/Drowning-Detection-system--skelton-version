"""Camera capture (plan Phase 4): OpenCV FFmpeg with hardware decoding, stall detection, reconnect.

One thread per camera decodes frames into the camera's FrameQueue.

* **Hardware decoding.** ``CAP_PROP_HW_ACCELERATION=VIDEO_ACCELERATION_ANY`` (D3D11VA on Windows).
  OpenCV falls back to software decoding when no hardware decoder fits; the mode actually used is
  reported in ``status()["hw_accel"]`` (0 = software).
* **Stall detection.** ``last_frame_age_s`` is the age of the newest decoded frame. The runtime's
  monitor compares it against ``STALL_S``: a stall raises a camera FAULT and calls
  ``request_reconnect()``. ``CAP_PROP_READ_TIMEOUT_MSEC = max(STALL_S, first_frame_timeout_s)``
  bounds a blocked read, so a reconnect request is always honoured within that time (see the
  first-frame grace below).
* **Reconnect** with exponential backoff (``initial * 2^(n-1)``, capped). The wait function is
  injectable, so the backoff is tested deterministically (replaces v2's sleep-based racy test).
* **File sources** (demo profile, soak/bench) are paced to the source fps and loop at the end;
  their stream time is ``index / fps``, exactly what the evaluation harness uses.
* **Stream clock** [Phase 11]: network sources stamp frames with the runtime-wide clock passed in by
  the runtime (one origin for every camera, monotonic across reconnects): cross-camera identity
  compares the timestamps of different cameras.
* **RTSP over TCP** [Phase 11 WP4]: ``force_rtsp_tcp()`` runs before the first open. UDP loses packets
  of a 4 MP H.265 stream even on loopback (measured: MediaMTX "RTP packets lost", HEVC "Could not find
  ref"). OpenCV's default ``rtsp_flags=prefer_tcp`` silently falls back to UDP; forced TCP does not.
* **First-frame grace** [Phase 11 WP4]: after each (re)open a camera has ``first_frame_timeout_s``
  (12 s) to deliver its first frame. FFmpeg may wait for the next keyframe inside the open (H.264, 8 s
  GOP: 4-6 s measured) or the first read, and the stall monitor does not count that wait as a stall
  (``in_first_frame_grace``). OpenCV cannot change a capture's read timeout after the open
  (``set(CAP_PROP_READ_TIMEOUT_MSEC)`` returns False) and a read that timed out leaves the capture
  dead, so the read timeout is ``max(STALL_S, first_frame_timeout_s)``. The cost: a blocked read on a
  black-holed stream returns after up to 12 s instead of STALL_S. The CAMERA_STALLED FAULT is not
  delayed: the monitor raises it at STALL_S.
* **Analysis rate cap** [v4]: a network source faster than ``max_fps`` (settings.target_analysis_fps) is
  decimated at capture by decode index (25 -> 15 fps keeps 3 frames of every 5, independent of network jitter).
  ``fps`` is then the rate frames are QUEUED at (what the pipeline and admission use), ``source_fps`` the
  stream's own. Measured on the site NVR: 4 sub-streams at 25 fps oversubscribed the server (two cameras ran
  2.4 s behind); capped at 15 fps every camera is analysed in full. File and replay sources are never decimated
  (the evaluation harness needs every frame).
* **Refused opens** [v4]: an open that fails at once while another stream of the same host is up is reported as
  the device refusing it (measured: a TP-Link NVR answers the 2nd main-stream SETUP with "509 Bandwidth Limit
  Exceeded"), so the FAULT names the likely cause instead of a bare stall.
* **Short opens** [Phase 6 A2]: OpenCV's FFmpeg backend opens ONE capture at a time per process (a global lock
  around the whole open; measured: 16 threads' opens finished 4.0 s apart, with or without GPU load or hardware
  decoding). The open used to run FFmpeg's full stream probe (~1.6 s on a 15 fps H.265 1 s-GOP stream, ~4 s on a
  2 s-GOP H.264 one), so the 8th camera's first frame came after the 12 s grace (CAMERA_STALLED). ``probesize``
  defaults to ``PROBESIZE`` (an operator value in OPENCV_FFMPEG_CAPTURE_OPTIONS wins): opens take ~0.4 s. A short
  probe sees no frame timing (the stream reports the 90 kHz RTP clock), so the capture measures the source fps from
  the first ``FPS_MEASURE_FRAMES`` frames' timestamps (``fps_source="measured"``); until then, and when the
  timestamps give no plausible rate, it uses camera_fps_fallback.
* **Unreachable hosts** [Phase 6 A2]: the open holds that process-wide lock for its whole timeout, so one
  black-holed camera blocked every other camera's (re)open for 12 s (measured: a healthy camera opened at 16 s).
  ``open_capture`` first makes a plain TCP connection (``PRECONNECT_TIMEOUT_S``) outside the lock; an unreachable
  host fails at once and backs off without touching OpenCV.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import math
import os
import socket
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from typing import Any, Protocol
from urllib.parse import urlsplit

import cv2
import numpy as np

from lifeguard.live.frames import Frame, FrameQueue

logger = logging.getLogger(__name__)

# [Phase 11 WP5] an implausible stream fps (e.g. 90000, the RTP clock, from a stream without frame-rate timing)
# falls back to settings.camera_fps_fallback (the site cameras' configured fps), not to a fixed 25
FPS_FALLBACK = 15.0
FPS_PLAUSIBLE = (1.0, 240.0)
OPEN_TIMEOUT_S = 10.0
FIRST_FRAME_TIMEOUT_S = 12.0  # settings.first_frame_timeout_s
FFMPEG_OPTIONS_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
REFUSED_OPEN_S = 1.5  # an open failing faster than this never reached a stream (refused, not a timeout)
PROBESIZE = 65536  # [Phase 6 A2] FFmpeg stream-probe bytes (default 5 MB): measured 4.0 s -> 0.4 s per open
FPS_MEASURE_FRAMES = 15  # frames whose timestamps give the source fps after a short probe
PRECONNECT_TIMEOUT_S = 2.0
RTSP_DEFAULT_PORT = 554
DECIMATE_TOLERANCE = 1.05  # a source this close to max_fps is not decimated (15.3 fps stays 15.3)

# capture states
CONNECTING = "CONNECTING"
STREAMING = "STREAMING"
RECONNECTING = "RECONNECTING"
STOPPED = "STOPPED"


class VideoSource(Protocol):
    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def get(self, prop: int) -> float: ...
    def release(self) -> None: ...


Opener = Callable[[str, str, float], VideoSource]


def merge_ffmpeg_options(current: str | None) -> str:
    """OpenCV's FFmpeg capture options (``key;value|key;value``) with ``rtsp_transport`` forced to tcp and
    ``probesize`` defaulted to PROBESIZE (an operator's value wins); every other option is kept, in order."""
    items = [item for item in (current or "").split("|") if item.strip()]
    kept = [item for item in items if item.partition(";")[0].strip() != "rtsp_transport"]
    if not any(item.partition(";")[0].strip() == "probesize" for item in kept):
        kept.append(f"probesize;{PROBESIZE}")
    return "|".join(["rtsp_transport;tcp", *kept])


def _set_msvcrt_env(name: str, value: str) -> bool:
    """Write `name` into msvcrt.dll's environment and read it back. OpenCV's FFmpeg plugin
    (opencv_videoio_ffmpeg*.dll, a MinGW build) imports msvcrt.dll, whose getenv reads its own copy of the
    environment taken when msvcrt loaded. os.environ (the UCRT) never reaches that copy (measured: with only
    os.environ set, the reader fell back to UDP)."""
    try:
        crt = ctypes.CDLL("msvcrt")
        crt._putenv_s.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        crt._putenv_s.restype = ctypes.c_int
        crt.getenv.argtypes = [ctypes.c_char_p]
        crt.getenv.restype = ctypes.c_char_p
        if crt._putenv_s(name.encode(), value.encode()) != 0:
            return False
        return bool(crt.getenv(name.encode()) == value.encode())
    except (OSError, AttributeError):
        return False


_tcp_lock = threading.Lock()
_tcp_status: dict[str, Any] | None = None


def force_rtsp_tcp() -> dict[str, Any]:
    """Force RTSP over TCP for every OpenCV FFmpeg capture of this process. Idempotent; it runs before the first
    open (LiveRuntime.start, open_capture). Returns {"rtsp_transport", "options", "verified"} for the live status:
    `verified` = the value the FFmpeg plugin will read is the forced one."""
    global _tcp_status
    with _tcp_lock:
        if _tcp_status is not None:
            return _tcp_status
        before = os.environ.get(FFMPEG_OPTIONS_ENV)
        value = merge_ffmpeg_options(before)
        if before and "rtsp_transport" in before and "rtsp_transport;tcp" not in before:
            logger.warning("%s=%r: rtsp_transport overridden to tcp", FFMPEG_OPTIONS_ENV, before)
        os.environ[FFMPEG_OPTIONS_ENV] = value  # the UCRT + the process block (child processes)
        verified = _set_msvcrt_env(FFMPEG_OPTIONS_ENV, value) if sys.platform == "win32" else True
        if not verified:
            logger.error("could not force RTSP over TCP: OpenCV's FFmpeg may fall back to UDP (packet loss)")
        _tcp_status = {"rtsp_transport": "tcp", "options": value, "verified": verified}
        return _tcp_status


def preconnect(source: str, timeout_s: float = PRECONNECT_TIMEOUT_S) -> None:
    """Raise OSError when an RTSP source's host:port does not accept a TCP connection within timeout_s. Runs
    before OpenCV's process-wide open lock, so an unreachable camera never blocks the others' opens."""
    try:
        parts = urlsplit(source)
        host, port = parts.hostname, parts.port or RTSP_DEFAULT_PORT
    except ValueError as exc:
        raise OSError(f"bad source URL: {exc}") from None
    if not host:
        raise OSError("source URL has no host")
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            pass
    except OSError as exc:
        raise OSError(f"{host}:{port} unreachable ({type(exc).__name__}, {timeout_s:g} s pre-connect)") from None


def open_capture(
    source: str, source_type: str, read_timeout_s: float, *, open_timeout_s: float = OPEN_TIMEOUT_S
) -> VideoSource:
    """OpenCV FFmpeg capture with hardware decoding requested and RTSP forced to TCP. An RTSP host that does not
    accept TCP raises OSError before the open."""
    force_rtsp_tcp()
    if source_type == "rtsp":
        preconnect(source)
    params = [
        cv2.CAP_PROP_HW_ACCELERATION,
        cv2.VIDEO_ACCELERATION_ANY,
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        int(open_timeout_s * 1000),
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        max(1000, int(read_timeout_s * 1000)),
    ]
    cap: Any = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
    return cap  # type: ignore[no-any-return]


_hosts_lock = threading.Lock()
_streaming_hosts: Counter[str] = Counter()  # host -> captures of this process currently streaming from it


def source_host(source: str) -> str | None:
    """Host of a network source (no credentials), or None."""
    try:
        return urlsplit(source).hostname
    except ValueError:
        return None


def streaming_from(host: str | None) -> int:
    with _hosts_lock:
        return _streaming_hosts[host] if host else 0


def keep_frame(index: int, ratio: float) -> bool:
    """Decimation by decode index: keep frame `index` when floor(index * ratio) advances (ratio = kept / source)."""
    if ratio >= 1.0 or index == 0:
        return True
    return math.floor(index * ratio + 1e-9) > math.floor((index - 1) * ratio + 1e-9)


def backoff_delay(failures: int, initial_s: float, max_s: float) -> float:
    """Delay before reconnect attempt number `failures` (1-based): initial * 2^(n-1), capped at max."""
    if failures < 1:
        return 0.0
    return float(min(max_s, initial_s * (2 ** min(failures - 1, 30))))


class CaptureWorker:
    def __init__(
        self,
        camera_id: str,
        source: str,
        source_type: str,
        queue: FrameQueue,
        stall_s: float = 5.0,
        *,
        opener: Opener | None = None,
        wait: Callable[[float], bool] | None = None,
        backoff_initial_s: float = 0.5,
        backoff_max_s: float = 30.0,
        loop_file: bool = True,
        realtime: bool = True,
        on_open: Callable[[float, tuple[int, int]], None] | None = None,
        clock: Callable[[float], float] | None = None,
        first_frame_timeout_s: float = FIRST_FRAME_TIMEOUT_S,
        fps_fallback: float = FPS_FALLBACK,
        replay: Any = None,
        gated: bool = False,
        gate_hold_s: float = 60.0,
        max_fps: float | None = None,
    ) -> None:
        if source_type not in ("rtsp", "file", "replay"):
            raise ValueError(f"unknown source type {source_type!r}")
        self.camera_id = camera_id
        self._source = source  # may hold credentials: never logged, never returned by status()
        self.source_type = source_type
        self.queue = queue
        self.stall_s = stall_s
        # [Phase 11 WP4] first-frame grace: the open and the first read may each wait for a keyframe
        self.first_frame_timeout_s = first_frame_timeout_s
        self.read_timeout_s = max(stall_s, first_frame_timeout_s)
        self.opener: Opener = opener or functools.partial(
            open_capture, open_timeout_s=max(OPEN_TIMEOUT_S, first_frame_timeout_s)
        )
        self._stop = threading.Event()
        self._reconnect = threading.Event()
        self.wait: Callable[[float], bool] = wait or self._stop.wait
        self.backoff_initial_s = backoff_initial_s
        self.backoff_max_s = backoff_max_s
        self.loop_file = loop_file
        self.realtime = realtime
        self.on_open = on_open
        # [Phase 11] runtime-wide stream clock (monotonic -> runtime seconds) for network sources, so every
        # camera's tracker time is comparable (cross-camera identity) and keeps rising across reconnects
        self.clock = clock
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # [Phase 11 WP7] pre-open: a gated capture opens and reads the stream (keeping the TCP session drained) but
        # DISCARDS every frame - nothing is queued - until the runtime admits the camera and opens the gate
        self._gate = threading.Event()
        if not gated:
            self._gate.set()
        self.gate_hold_s = gate_hold_s
        self.gated_frames = 0  # frames read and discarded while the gate was closed
        self.preopened_at: float | None = None  # monotonic time of the first frame read behind the closed gate
        self.preopen_adopted = False  # opened by the PreOpener before admission and taken over by the runtime
        # status
        self.state = STOPPED
        self.epoch = 0
        self.fps_fallback = fps_fallback
        self.replay = replay  # [Phase 11 WP6] live/replay.ReplayFeed for source_type "replay" (eval only)
        self.replay_finished = False
        self.fps = fps_fallback  # rate frames are queued at (after the max_fps cap)
        self.source_fps = fps_fallback  # the stream's own rate
        self.max_fps = max_fps if source_type == "rtsp" else None  # [v4] analysis rate cap, network sources only
        # [dynamic fps] the admission floor (settings.target_analysis_fps); max_fps moves between it and the ceiling
        self.floor_fps = self.max_fps
        self._rate_version = 0  # bumped by set_rate: the read loop recomputes its decimation ratio
        self.decimated = 0  # frames read and not queued because of max_fps
        self.host = source_host(source) if source_type == "rtsp" else None
        self.fps_source = "fallback"  # "stream" once the source reports a plausible fps
        self.reported_fps: float | None = None
        self.frame_size: tuple[int, int] | None = None
        self.hw_accel = 0
        self.frames_read = 0
        self.reconnects = 0
        self.failures = 0  # consecutive failed opens / reads
        self.last_error: str | None = None
        self.started_at = time.monotonic()
        self.last_frame_at: float | None = None
        self.backoff_log: list[float] = []  # every reconnect delay (tests, status)
        self._eof_loop = False
        self.attempt_started_at: float | None = None  # start of the current (re)open
        self.awaiting_first_frame = False  # an open is in progress, or opened and no frame yet
        self.first_frame_s: float | None = None  # open start -> first frame, last (re)open
        self.first_frame_at: float | None = None  # [Phase 11 WP5] monotonic time of the very first frame

    # ── control ───────────────────────────────────────────────────────────────
    def start(self) -> None:
        self._stop.clear()
        self.started_at = time.monotonic()
        self._thread = threading.Thread(target=self.run, name=f"capture-{self.camera_id}", daemon=True)
        self._thread.start()

    @property
    def gated(self) -> bool:
        return not self._gate.is_set()

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def signal_stop(self) -> None:
        """Ask the thread to end without waiting (stop() joins): lets many captures be signalled first."""
        self._stop.set()
        self._reconnect.set()

    def open_gate(self) -> None:
        """Admitted: from the next frame on, frames are queued (the queue and clock were attached before)."""
        self._gate.set()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._reconnect.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self.state = STOPPED

    def request_reconnect(self, reason: str) -> None:
        """Drop the current connection at the next frame boundary (the stall monitor calls this)."""
        logger.warning("[capture:%s] reconnect requested: %s", self.camera_id, reason)
        self.last_error = reason
        self._reconnect.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def last_frame_age_s(self, now: float | None = None) -> float:
        """Seconds since the newest decoded frame (since start when none has arrived yet)."""
        now = time.monotonic() if now is None else now
        ref = self.last_frame_at if self.last_frame_at is not None else self.started_at
        return max(0.0, now - ref)

    def in_first_frame_grace(self, now: float | None = None) -> bool:
        """[Phase 11 WP4] True from the start of each (re)open until its first frame, for at most
        first_frame_timeout_s: waiting for a keyframe is not a stall. Backoff pauses between attempts are not
        grace (a camera that cannot be opened still stalls at STALL_S)."""
        started = self.attempt_started_at
        if not self.awaiting_first_frame or started is None:
            return False
        now = time.monotonic() if now is None else now
        return now - started < self.first_frame_timeout_s

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "source_type": self.source_type,
            "fps": round(self.fps, 3),
            "source_fps": round(self.source_fps, 3),
            "decimated": self.decimated,
            "rate_cap_fps": self.max_fps,
            "fps_source": self.fps_source,
            "reported_fps": self.reported_fps,
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "hw_accel": self.hw_accel,
            "frames_read": self.frames_read,
            "reconnects": self.reconnects,
            "last_frame_age_s": round(self.last_frame_age_s(), 3),
            "last_error": self.last_error,
            "epoch": self.epoch,
            "awaiting_first_frame": self.awaiting_first_frame,
            "first_frame_s": None if self.first_frame_s is None else round(self.first_frame_s, 3),
            "first_frame_timeout_s": self.first_frame_timeout_s,
            "preopened": self.preopened_at is not None,  # a frame had already arrived behind the gate at admission
            "preopen_adopted": self.preopen_adopted,  # the open started before admission (frame may still be pending)
            "gated": self.gated,
        }

    # ── thread body ───────────────────────────────────────────────────────────
    def run(self) -> None:
        if self.source_type == "replay":
            self._run_replay()
            self.state = STOPPED
            return
        while not self._stop.is_set():
            self.state = CONNECTING if self.epoch == 0 else RECONNECTING
            self.attempt_started_at = time.monotonic()
            self.awaiting_first_frame = True
            cap = self._open()
            if cap is None:
                self.awaiting_first_frame = False
                self._backoff()
                continue
            try:
                self._stream(cap)
            finally:
                self.awaiting_first_frame = False
                try:
                    cap.release()
                except Exception:
                    logger.debug("[capture:%s] release failed", self.camera_id, exc_info=True)
            if not self._stop.is_set() and not self._eof_loop:
                self.reconnects += 1
                self._backoff()
        self.state = STOPPED

    def _open(self) -> VideoSource | None:
        t0 = time.monotonic()
        try:
            cap = self.opener(self._source, self.source_type, self.read_timeout_s)
            if not cap.isOpened():
                raise OSError("source did not open")
        except Exception as exc:  # noqa: BLE001 - any open failure -> logged, backoff, retry
            self.failures += 1
            self.last_error = f"open failed: {type(exc).__name__}: {exc}"
            others = streaming_from(self.host)
            if others and time.monotonic() - t0 < REFUSED_OPEN_S:
                self.last_error += (
                    f" - refused at once while {others} other stream(s) of this camera/NVR are open: the device's"
                    " stream or bandwidth limit (use its sub-stream, or raise the limit on the device)"
                )
            logger.warning("[capture:%s] %s (attempt %d)", self.camera_id, self.last_error, self.failures)
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.reported_fps = fps
        if FPS_PLAUSIBLE[0] <= fps <= FPS_PLAUSIBLE[1]:
            self._set_source_fps(fps, "stream")
        else:
            self._set_source_fps(self.fps_fallback, "fallback")
            logger.info(
                "[capture:%s] source reports %g fps (short probe / no frame timing): camera_fps_fallback %g fps "
                "until measured from the first frames",
                self.camera_id,
                fps,
                self.fps_fallback,
            )
        self.hw_accel = int(cap.get(cv2.CAP_PROP_HW_ACCELERATION) or 0)
        self.epoch += 1
        self._reconnect.clear()
        logger.info(
            "[capture:%s] opened (epoch %d, %.2f fps, hw_accel=%d)", self.camera_id, self.epoch, self.fps, self.hw_accel
        )
        return cap

    def _set_source_fps(self, fps: float, source: str) -> None:
        """Source rate and the queued rate after the max_fps cap."""
        self.source_fps, self.fps_source = fps, source
        self.fps = self._capped(fps, self.max_fps)
        if self.fps < fps:
            logger.info("[capture:%s] source %.2f fps: analysing %.2f fps (decimated)", self.camera_id, fps, self.fps)

    @staticmethod
    def _capped(source_fps: float, cap: float | None) -> float:
        return cap if cap and source_fps > cap * DECIMATE_TOLERANCE else source_fps

    @property
    def required_fps(self) -> float:
        """[dynamic fps] The rate this camera NEEDS (what admission counts): the queued rate at the floor cap. A camera
        running above the floor (live/rate.py) can always be stepped back down to it."""
        return self._capped(self.source_fps, self.floor_fps)

    def set_rate(self, fps: float) -> bool:
        """[dynamic fps] Move the analysis cap (never below the floor). True when the queued rate changed."""
        if self.max_fps is None or self.floor_fps is None:
            return False
        cap = max(float(self.floor_fps), float(fps))
        if abs(cap - self.max_fps) < 1e-6:
            return False
        before = self.fps
        self.max_fps = cap
        self.fps = self._capped(self.source_fps, cap)
        self._rate_version += 1
        return abs(self.fps - before) > 1e-6

    def _measure_fps(self, stamps: list[float]) -> bool:
        """Source fps from the first frames' stream timestamps (ms). True when a plausible rate replaced the
        fallback; an implausible one keeps camera_fps_fallback (logged, as before the short probe)."""
        span = stamps[-1] - stamps[0]
        fps = (len(stamps) - 1) * 1000.0 / span if span > 0 else 0.0
        if FPS_PLAUSIBLE[0] <= fps <= FPS_PLAUSIBLE[1]:
            self._set_source_fps(round(fps, 3), "measured")
            logger.info("[capture:%s] source fps measured from timestamps: %.2f", self.camera_id, self.source_fps)
            return True
        logger.warning(
            "[capture:%s] implausible fps (reported %g, timestamps %.3g): using camera_fps_fallback %g fps",
            self.camera_id,
            self.reported_fps or 0.0,
            fps,
            self.fps_fallback,
        )
        return False

    def _stream(self, cap: VideoSource) -> None:
        counted = False
        try:
            counted = self._stream_frames(cap)
        finally:
            if counted and self.host:
                with _hosts_lock:
                    _streaming_hosts[self.host] -= 1
                    if _streaming_hosts[self.host] <= 0:
                        del _streaming_hosts[self.host]

    def _stream_frames(self, cap: VideoSource) -> bool:
        """Read until stop / reconnect / failure. Returns True when this capture was counted as streaming from its
        host (the caller uncounts it)."""
        self._eof_loop = False
        index = 0
        kept = 0  # frames queued since this open (the Frame index the pipeline sees)
        ratio = self.fps / self.source_fps if self.source_fps > 0 else 1.0
        rate_version = self._rate_version
        opened = time.monotonic()
        notified = False
        counted = False
        # [Phase 6 A2] a short probe leaves the fps unknown: measure it from the first frames' timestamps
        stamps: list[float] | None = [] if self.source_type == "rtsp" and self.fps_source == "fallback" else None
        while not self._stop.is_set() and not self._reconnect.is_set():
            ok, img = cap.read()
            if not ok or img is None or img.size == 0:
                if self.source_type == "file" and self.loop_file and index > 0:
                    self._eof_loop = True  # clean end of file: reopen at once, no backoff
                    return counted
                self.failures += 1
                if index == 0 and self.attempt_started_at is not None:
                    waited = time.monotonic() - self.attempt_started_at
                    self.last_error = (
                        f"no first frame {waited:.1f} s after (re)connecting "
                        f"(first_frame_timeout_s={self.first_frame_timeout_s:g}; keyframe interval too long?)"
                    )
                else:
                    self.last_error = "read failed / stream ended"
                logger.warning("[capture:%s] %s after %d frames", self.camera_id, self.last_error, index)
                return counted
            now = time.monotonic()
            if not notified:
                self.frame_size = (int(img.shape[1]), int(img.shape[0]))
                if self.on_open is not None:
                    self.on_open(self.fps, self.frame_size)
                notified = True
                if self.host:
                    with _hosts_lock:
                        _streaming_hosts[self.host] += 1
                    counted = True
            self.state = STREAMING
            self.failures = 0
            if stamps is not None:
                stamps.append(float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0))
                if len(stamps) >= FPS_MEASURE_FRAMES:
                    if self._measure_fps(stamps):
                        ratio = self.fps / self.source_fps
                    stamps = None
            if not self._gate.is_set():  # [WP7] pre-opened, not admitted yet: drain, never queue
                self.gated_frames += 1
                self.last_frame_at = now
                self.awaiting_first_frame = False
                if self.preopened_at is None:
                    self.preopened_at = now
                    if self.attempt_started_at is not None:
                        self.first_frame_s = now - self.attempt_started_at
                    logger.info("[capture:%s] pre-opened (gate closed), first frame behind it", self.camera_id)
                if now - self.preopened_at > self.gate_hold_s:
                    logger.warning(
                        "[capture:%s] pre-open gate never opened within %g s: closing", self.camera_id, self.gate_hold_s
                    )
                    self._stop.set()
                    return counted
                index += 1
                continue
            if rate_version != self._rate_version:  # [dynamic fps] the cap moved: decimate at the new rate
                rate_version = self._rate_version
                ratio = self.fps / self.source_fps if self.source_fps > 0 else 1.0
            if not keep_frame(index, ratio):  # [v4] above max_fps: not queued, but the stream is alive
                self.decimated += 1
                self.last_frame_at = now
                index += 1
                continue
            if self.source_type == "file":
                t = index / self.fps
            else:
                t = self.clock(now) if self.clock is not None else now - opened
            self.queue.put(Frame(img, kept, t, now, self.epoch))
            kept += 1
            self.last_frame_at = now
            if self.first_frame_at is None:
                self.first_frame_at = now
            if self.awaiting_first_frame:  # after last_frame_at: the monitor never sees a stale age out of grace
                self.awaiting_first_frame = False
                if self.attempt_started_at is not None:
                    self.first_frame_s = now - self.attempt_started_at
                    logger.info("[capture:%s] first frame %.2f s after open", self.camera_id, self.first_frame_s)
            self.frames_read += 1
            index += 1
            if self.source_type == "file" and self.realtime:
                delay = opened + index / self.fps - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
        return counted

    def _run_replay(self) -> None:
        """[Phase 11 WP6] A recording on the session clock (live/replay.py): every frame waits its lockstep turn,
        nothing is dropped or paced, no reconnect / loop; the coordinator is told when the file ends."""
        feed = self.replay
        if feed is None:
            raise ValueError("replay source without a ReplayFeed")
        self.state = CONNECTING
        cap = self._open()
        index = 0
        try:
            if cap is None:
                return
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok or img is None or img.size == 0:
                    break
                t = feed.stamp(float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0)
                if not feed.coordinator.wait_turn(self.camera_id, t, self._stop):
                    break
                now = time.monotonic()
                if index == 0:
                    self.frame_size = (int(img.shape[1]), int(img.shape[0]))
                    if self.on_open is not None:
                        self.on_open(self.fps, self.frame_size)
                    self.first_frame_at = now
                    self.awaiting_first_frame = False
                self.state = STREAMING
                self.queue.put(Frame(img, index, t, now, self.epoch))
                self.last_frame_at = now
                self.frames_read += 1
                index += 1
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    logger.debug("[capture:%s] release failed", self.camera_id, exc_info=True)
            feed.coordinator.finish(self.camera_id)
            self.replay_finished = True
            logger.info("[capture:%s] replay finished after %d frames", self.camera_id, index)

    def _backoff(self) -> None:
        delay = backoff_delay(max(self.failures, 1), self.backoff_initial_s, self.backoff_max_s)
        self.backoff_log.append(delay)
        self.wait(delay)
