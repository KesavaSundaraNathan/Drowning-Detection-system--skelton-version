"""Camera check against docs/CAMERA_SETUP.md (Phase 11 WP6; rules pre-registered in eval/site/levers_phase11.json
"gates_wp6".session_checks.camera_check).

Two measurements per camera, then a verdict per rule (PASS / WARN / FAIL with the reason):

* ``probe_stream``: the pinned ffprobe reads a PACKET window (default 20 s) over RTSP/TCP - codec, profile, size,
  the stream's declared frame rates, the keyframe flags (-> measured GOP), pts order in decode order (-> B-frame
  reordering), packet sizes (-> bitrate) and the MEASURED fps from the packet timestamps (not the declared one:
  OpenCV reported 90000 for a stream without timing, WP4 finding 4.2).
* ``probe_opencv``: the live capture's own open path (``live.capture.open_capture``, TCP forced): open time,
  first-frame time, hardware decoding, the fps OpenCV reports and whether the fallback would be used.

Nothing in a result contains the URL; ffprobe's stderr goes through the caller's Redactor.
"""

from __future__ import annotations

import itertools
import json
import statistics
import subprocess  # nosec B404  # fixed-argv ffprobe over the camera stream, no shell
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
WINDOW_S = 20.0
FPS_TOL = 0.5
FIRST_FRAME_MAX_S = 3.0


@dataclass
class Check:
    check: str
    verdict: str
    value: Any
    expected: str
    why: str = ""


def _rate(text: str | None) -> float | None:
    try:
        f = Fraction(str(text))
    except (ValueError, ZeroDivisionError):
        return None
    return float(f) if f > 0 else None


def probe_stream(
    url: str, ffprobe: Path, *, seconds: float = WINDOW_S, redact: Callable[[str], str] = lambda s: s
) -> dict[str, Any]:
    """ffprobe packet window over RTSP/TCP -> the measured stream facts (no URL in the result)."""
    argv = [str(ffprobe), "-v", "error", "-rtsp_transport", "tcp", "-read_intervals", f"%+{seconds:g}"]
    argv += ["-select_streams", "v:0", "-show_entries"]
    argv += [
        "stream=codec_name,profile,width,height,r_frame_rate,avg_frame_rate,has_b_frames:packet=pts_time,dts_time,flags,size"
    ]
    argv += ["-of", "json", url]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=seconds + 30, check=False)  # noqa: S603  # nosec B603
    except subprocess.TimeoutExpired:
        return {"error": f"ffprobe timed out after {seconds + 30:g} s"}
    if done.returncode != 0:
        return {"error": redact(done.stderr.strip()[-500:]) or f"ffprobe exit {done.returncode}"}
    try:
        data = json.loads(done.stdout)
    except ValueError:
        return {"error": "ffprobe output unreadable"}
    return summarise_packets(data)


def summarise_packets(data: dict[str, Any]) -> dict[str, Any]:
    """ffprobe JSON (streams + packets) -> codec, size, declared + measured fps, keyframe gaps, B-frames, bitrate."""
    streams = data.get("streams") or [{}]
    st = streams[0]
    pkts = [p for p in data.get("packets") or [] if p.get("pts_time") not in (None, "N/A")]
    pts = [float(p["pts_time"]) for p in pkts]
    keys = sorted(float(p["pts_time"]) for p in pkts if "K" in str(p.get("flags", "")))
    gaps = [round(b - a, 3) for a, b in itertools.pairwise(keys)]
    span = (max(pts) - min(pts)) if len(pts) > 1 else 0.0
    measured_fps = (len(pts) - 1) / span if span > 0 else None
    reordered = any(b < a for a, b in itertools.pairwise(pts))  # decode order != display order
    size = sum(int(p.get("size") or 0) for p in pkts)
    return {
        "codec": st.get("codec_name"),
        "profile": st.get("profile"),
        "size": [st.get("width"), st.get("height")],
        "declared_fps": _rate(st.get("r_frame_rate")),
        "timing_fps": _rate(st.get("avg_frame_rate")),  # 0/0 = the stream carries no frame-rate timing
        "measured_fps": round(measured_fps, 3) if measured_fps else None,
        "packets": len(pts),
        "window_s": round(span, 3),
        "keyframes": len(keys),
        "keyframe_gaps_s": gaps,
        # B-frames = REORDERING (pts going back in decode order). ffprobe's has_b_frames is only the declared reorder
        # depth: over RTSP it read 1 for a P-only low-delay H.265 stream (measured), so it is reported, not judged
        "b_frames": reordered,
        "declared_reorder_depth": int(st.get("has_b_frames") or 0),
        "bitrate_kbps": round(size * 8 / span / 1000) if span > 0 else None,
    }


def probe_opencv(url: str, *, frames: int = 15) -> dict[str, Any]:
    """The live capture's open path: open + first frame time, hw decoding, the fps OpenCV reports."""
    import cv2

    from lifeguard.live.capture import FIRST_FRAME_TIMEOUT_S, FPS_PLAUSIBLE, OPEN_TIMEOUT_S, open_capture

    t0 = time.monotonic()
    cap = open_capture(url, "rtsp", FIRST_FRAME_TIMEOUT_S, open_timeout_s=max(OPEN_TIMEOUT_S, FIRST_FRAME_TIMEOUT_S))
    try:
        if not cap.isOpened():
            return {"opened": False, "open_s": round(time.monotonic() - t0, 2)}
        open_s = time.monotonic() - t0
        ok, img = cap.read()
        first_s = time.monotonic() - t0
        n = int(ok)
        while ok and n < frames:
            ok, _ = cap.read()
            n += int(ok)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return {
            "opened": True,
            "open_s": round(open_s, 2),
            "first_frame_s": round(first_s, 2) if img is not None else None,
            "frames_read": n,
            "frame_size": [int(img.shape[1]), int(img.shape[0])] if img is not None else None,
            "hw_accel": int(cap.get(cv2.CAP_PROP_HW_ACCELERATION) or 0),
            "reported_fps": fps,
            "fps_source": "stream" if FPS_PLAUSIBLE[0] <= fps <= FPS_PLAUSIBLE[1] else "fallback",
        }
    finally:
        cap.release()


def verdict(stream: dict[str, Any], cv: dict[str, Any], expect: dict[str, Any]) -> list[Check]:
    """The pre-registered rules. `expect`: codec, size [w, h], fps, gop_s (site_profile.json "camera")."""
    if stream.get("error"):
        return [Check("stream", FAIL, stream["error"], "the stream can be read over RTSP/TCP")]
    fps_e, gop_e, frame = float(expect["fps"]), float(expect["gop_s"]), 1.0 / float(expect["fps"])
    out: list[Check] = []

    codec = stream.get("codec")
    out.append(
        Check(
            "codec",
            PASS if codec == expect["codec"] else FAIL,
            codec,
            expect["codec"],
            "" if codec == expect["codec"] else "CAMERA_SETUP: H.265 main stream",
        )
    )
    size = stream.get("size")
    out.append(
        Check(
            "resolution",
            PASS if size == list(expect["size"]) else FAIL,
            size,
            f"{expect['size'][0]}x{expect['size'][1]}",
        )
    )

    mfps = stream.get("measured_fps")
    ok_fps = mfps is not None and abs(mfps - fps_e) <= FPS_TOL
    out.append(
        Check(
            "fps_measured",
            PASS if ok_fps else FAIL,
            mfps,
            f"{fps_e:g} +- {FPS_TOL:g}",
            "" if ok_fps else "camera frame rate must be 15",
        )
    )

    gaps = stream.get("keyframe_gaps_s") or []
    if len(gaps) < 2:
        out.append(
            Check(
                "gop",
                FAIL,
                gaps,
                f"{gop_e:g} s",
                f"fewer than 2 GOPs in {stream.get('window_s')} s: I-frame interval too long",
            )
        )
    else:
        off = [g for g in gaps if abs(g - gop_e) > frame + 1e-6]
        regular = max(gaps) - min(gaps) <= frame + 1e-6
        if not off:
            out.append(Check("gop", PASS, statistics.median(gaps), f"{gop_e:g} s"))
        elif regular:
            out.append(
                Check(
                    "gop",
                    FAIL,
                    statistics.median(gaps),
                    f"{gop_e:g} s",
                    "I-frame interval must equal the frame rate (1 s)",
                )
            )
        else:
            out.append(
                Check(
                    "gop",
                    FAIL,
                    [min(gaps), max(gaps)],
                    f"{gop_e:g} s",
                    "irregular keyframes: H.265+ / smart codec / dynamic GOP must be off",
                )
            )

    out.append(Check("b_frames", FAIL if stream.get("b_frames") else PASS, bool(stream.get("b_frames")), "none"))
    timing = stream.get("timing_fps")
    out.append(
        Check(
            "frame_rate_timing",
            PASS if timing else WARN,
            timing,
            "declared in the stream",
            "" if timing else "no frame-rate timing: Life Guard uses camera_fps_fallback",
        )
    )

    if not cv.get("opened"):
        out.append(Check("opencv_open", FAIL, cv, "opens over TCP", "the live capture path cannot open the stream"))
        return out
    ff = cv.get("first_frame_s")
    out.append(
        Check(
            "first_frame",
            PASS if ff is not None and ff <= FIRST_FRAME_MAX_S else FAIL,
            ff,
            f"<= {FIRST_FRAME_MAX_S:g} s",
        )
    )
    out.append(
        Check(
            "hw_decode",
            PASS if cv.get("hw_accel") else WARN,
            cv.get("hw_accel"),
            "hardware decoding",
            "" if cv.get("hw_accel") else "software decoding: more CPU per camera",
        )
    )
    fsrc = cv.get("fps_source")
    out.append(
        Check(
            "opencv_fps",
            PASS if fsrc == "stream" else WARN,
            cv.get("reported_fps"),
            "plausible",
            "" if fsrc == "stream" else "implausible fps reported: camera_fps_fallback used",
        )
    )
    return out


def overall(checks: list[Check]) -> str:
    verdicts = {c.verdict for c in checks}
    return FAIL if FAIL in verdicts else WARN if WARN in verdicts else PASS


def check_camera(
    url: str,
    ffprobe: Path,
    expect: dict[str, Any],
    *,
    redact: Callable[[str], str] = lambda s: s,
    seconds: float = WINDOW_S,
) -> dict[str, Any]:
    stream = probe_stream(url, ffprobe, seconds=seconds, redact=redact)
    cv = probe_opencv(url) if not stream.get("error") else {"opened": False, "skipped": "stream unreadable"}
    checks = verdict(stream, cv, expect)
    return {"verdict": overall(checks), "checks": [asdict(c) for c in checks], "stream": stream, "opencv": cv}
