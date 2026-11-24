"""1440p H.265 test clips for the camera-robustness tests (plan Phase 11 WP4). No download: the pinned LGPL FFmpeg
(build/tools/ffmpeg, already approved) has the encoders.

    python tools/make_hevc_clips.py [--out build/test_media/hevc] [--source CLIP] [--seconds 20] [--encoder libkvazaar]

Writes 2560x1440 @ 15 fps clips (the site's 4 MP bullets) and a manifest.json (sha256, measured keyframe interval,
encoder, FFmpeg version):

  hevc_1440p15_gop1s.mkv   H.265, keyframe every 1 s  (docs/CAMERA_SETUP.md: I-frame interval = fps)
  hevc_1440p15_gop8s.mkv   H.265, keyframe every 8 s  (a badly configured / "H.265+" camera)
  h264_1440p15_gop8s.mkv   H.264, keyframe every 8 s  (the control where FFmpeg's open blocks until the keyframe)

Encoders: libkvazaar (BSD-3, CPU, default) or hevc_nvenc (NVIDIA GPU); H.264 uses libopenh264 (BSD-2). Every clip
has no frame reordering and no scene-cut keyframes, like a camera stream (`hevc_b`: a deliberately reordering
H.265 stream for the camera check, WP6). The keyframe interval is MEASURED
with ffprobe after encoding, and a clip whose interval is not the requested one is refused, so a test never runs
on a clip that does not have the GOP it claims.

`--source` is the picture content (default: the v2 baseline clip used by the soak/chaos harness when present,
otherwise FFmpeg's synthetic testsrc2). The content does not matter for these tests; the codec, size, fps and
GOP do.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess  # nosec B404  # fixed-argv calls to the pinned ffmpeg, no shell
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "build" / "test_media" / "hevc"
DEFAULT_SOURCE = (
    ROOT / "build" / "v2_baseline" / "backend" / "storage" / "uploads" / "5696203ffd4b445d9508dcb788477f2f.mp4"
)
WIDTH, HEIGHT, FPS = 2560, 1440, 15
BITRATE = 6_000_000  # a 4 MP H.265 camera's usual main-stream rate
CSV = ("-of", "csv=p=0")
CLIPS: tuple[tuple[str, str, float], ...] = (
    ("hevc_1440p15_gop1s.mkv", "hevc", 1.0),
    ("hevc_1440p15_gop8s.mkv", "hevc", 8.0),
    ("h264_1440p15_gop8s.mkv", "h264", 8.0),
)


class ClipError(RuntimeError):
    pass


def find_tool(name: str, env_var: str) -> Path | None:
    """`env_var`, else the pinned build under build/tools (ffmpeg.exe / ffprobe.exe / mediamtx.exe)."""
    env = os.environ.get(env_var)
    if env and Path(env).is_file():
        return Path(env)
    exe = f"{name}.exe" if sys.platform == "win32" else name
    hits = sorted((ROOT / "build" / "tools").glob(f"*/*/**/{exe}"))
    return hits[-1] if hits else None


def encoder_args(codec: str, encoder: str, gop: int) -> list[str]:
    if codec == "h264":
        return ["-c:v", "libopenh264", "-g", str(gop), "-b:v", str(BITRATE)]
    if codec == "hevc_b":
        # [WP6] a camera WITH B-frame reordering (kvazaar's hierarchical GOP of 8; the intra period must be a
        # multiple of it): the camera check must flag it
        period = max(8, round(gop / 8) * 8)
        return ["-c:v", "libkvazaar", "-kvazaar-params", f"preset=ultrafast,gop=8,period={period},bitrate={BITRATE}"]
    if encoder == "libkvazaar":
        # gop=0: no hierarchical GOP, no reordering (low delay: kvazaar codes B slices that reference only past
        # frames, pts == dts); an intra period need not be a multiple of 8
        return ["-c:v", "libkvazaar", "-kvazaar-params", f"preset=ultrafast,gop=0,period={gop},bitrate={BITRATE}"]
    if encoder == "hevc_nvenc":
        return ["-c:v", "hevc_nvenc", "-g", str(gop), "-bf", "0", "-no-scenecut", "1", "-b:v", str(BITRATE)]
    raise ClipError(f"unknown H.265 encoder {encoder!r} (libkvazaar or hevc_nvenc)")


def keyframe_times(ffprobe: Path, clip: Path) -> list[float]:
    out = subprocess.run(  # noqa: S603  # nosec B603 - fixed argv, pinned binary
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time,flags",
            *CSV,
            str(clip),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    times = []
    for line in out.splitlines():
        pts, _, flags = line.partition(",")
        if "K" in flags and pts not in ("", "N/A"):
            times.append(float(pts))
    return times


def make_clip(
    ffmpeg: Path,
    out: Path,
    codec: str,
    gop_s: float,
    *,
    seconds: float = 20.0,
    source: Path | None = None,
    encoder: str = "libkvazaar",
) -> dict[str, Any]:
    """Encode one clip and verify its keyframe interval; returns its manifest entry."""
    gop = round(gop_s * FPS)
    src = (
        ["-stream_loop", "-1", "-i", str(source)]
        if source is not None
        else ["-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}"]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", *src, "-t", f"{seconds:g}", "-an"]
    argv += ["-vf", f"fps={FPS},scale={WIDTH}:{HEIGHT}:flags=bicubic", "-pix_fmt", "yuv420p"]
    argv += [*encoder_args(codec, encoder, gop), str(out)]
    done = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603  # nosec B603
    if done.returncode != 0 or not out.is_file():
        raise ClipError(f"ffmpeg failed for {out.name}: {done.stderr.strip()[-400:]}")
    ffprobe = ffmpeg.with_name(ffmpeg.name.replace("ffmpeg", "ffprobe"))
    keys = keyframe_times(ffprobe, out)
    gaps = [round(b - a, 3) for a, b in itertools.pairwise(keys)]
    if not keys or keys[0] != 0.0 or not gaps or any(abs(g - gop_s) > 1.5 / FPS for g in gaps):
        raise ClipError(f"{out.name}: keyframes at {keys[:8]} - not every {gop_s:g} s")
    return {
        "file": out.name,
        "codec": codec,
        "encoder": "libopenh264" if codec == "h264" else encoder,
        "size": [WIDTH, HEIGHT],
        "fps": FPS,
        "seconds": seconds,
        "gop_s": gop_s,
        "measured_keyframe_gaps_s": sorted(set(gaps)),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "source": "testsrc2" if source is None else source.name,
    }


# ── [WP6] barcode clip: every frame carries its own index (recording-sync tests) ──────────────────────────────────
BARCODE_BITS = 16
BARCODE_SIZE = (640, 360)
_BLOCK = 36


def barcode_frame(index: int, size: tuple[int, int] = BARCODE_SIZE) -> Any:
    """A grey frame with `index` as BARCODE_BITS black / white blocks along the top (MSB first)."""
    import numpy as np

    w, h = size
    img = np.full((h, w, 3), 128, np.uint8)
    for bit in range(BARCODE_BITS):
        on = (index >> (BARCODE_BITS - 1 - bit)) & 1
        x = 8 + bit * _BLOCK
        img[8 : 8 + _BLOCK - 6, x : x + _BLOCK - 6] = 255 if on else 0
    return img


def read_barcode(img: Any) -> int:
    """The frame index written by barcode_frame (block centres, threshold at mid grey)."""
    value = 0
    for bit in range(BARCODE_BITS):
        cx, cy = 8 + bit * _BLOCK + (_BLOCK - 6) // 2, 8 + (_BLOCK - 6) // 2
        value = (value << 1) | int(img[cy - 4 : cy + 4, cx - 4 : cx + 4].mean() > 128)
    return value


def make_barcode_clip(ffmpeg: Path, out: Path, *, seconds: float = 30.0, gop_s: float = 1.0) -> dict[str, Any]:
    """640x360 @ 15 fps H.265 (libkvazaar, 1 s GOP) whose frame i shows i as a barcode; raw frames piped in."""
    w, h = BARCODE_SIZE
    n = round(seconds * FPS)
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24"]
    argv += ["-s", f"{w}x{h}", "-r", str(FPS), "-i", "pipe:0", "-pix_fmt", "yuv420p"]
    argv += [
        "-c:v",
        "libkvazaar",
        "-kvazaar-params",
        f"preset=ultrafast,gop=0,period={round(gop_s * FPS)},qp=22",
        str(out),
    ]
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: S603  # nosec B603
    if proc.stdin is None:
        raise ClipError("ffmpeg stdin unavailable")
    try:
        for i in range(n):
            proc.stdin.write(barcode_frame(i).tobytes())
    finally:
        proc.stdin.close()
    err = ""
    if proc.stderr is not None:
        with proc.stderr:
            err = proc.stderr.read().decode(errors="replace")
    if proc.wait(120) != 0 or not out.is_file():
        raise ClipError(f"barcode clip failed: {err.strip()[-300:]}")
    return {"file": out.name, "frames": n, "fps": FPS, "size": [w, h], "gop_s": gop_s}


def make_all(
    out_dir: Path, *, seconds: float = 20.0, source: Path | None = None, encoder: str = "libkvazaar"
) -> dict[str, Any]:
    ffmpeg = find_tool("ffmpeg", "LIFEGUARD_FFMPEG")
    if ffmpeg is None:
        raise ClipError("FFmpeg not found: set LIFEGUARD_FFMPEG or unpack the pinned build under build/tools/ffmpeg")
    version = subprocess.run(  # noqa: S603  # nosec B603
        [str(ffmpeg), "-version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    clips = [
        make_clip(ffmpeg, out_dir / name, codec, gop_s, seconds=seconds, source=source, encoder=encoder)
        for name, codec, gop_s in CLIPS
    ]
    manifest = {"ffmpeg": version, "clips": clips}
    (out_dir / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--source", type=Path, default=None, help="picture content (default: v2 clip or testsrc2)")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--encoder", choices=("libkvazaar", "hevc_nvenc"), default="libkvazaar")
    args = ap.parse_args(argv)
    source = args.source or (DEFAULT_SOURCE if DEFAULT_SOURCE.is_file() else None)
    try:
        manifest = make_all(args.out, seconds=args.seconds, source=source, encoder=args.encoder)
    except ClipError as exc:
        print(f"ERROR: {exc}")
        return 1
    for c in manifest["clips"]:
        w, h = c["size"]
        print(f"{c['file']}: {c['codec']} {w}x{h} @ {c['fps']} fps, keyframe gaps {c['measured_keyframe_gaps_s']} s")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
