"""[2.20.0] Hard-negative mining: crops of every false alert raised on normal footage, with metadata, for retraining.

Runs normal (no-drowning) clips through the upload path (`runner.run_clip`: ClipClock, SourcePipeline, end-of-clip
flush), so every alert it raises is a false alert by construction. For each one it saves a crop of the alerted box
and a metadata line, ready for the detector hard-negative round (training/tools/hardneg.py) or a verifier:

    hard_negatives/<run>/
        manifest.jsonl                       one HardNegativeRecord per crop (JSON lines)
        <clip>/<clip>_f000123_t7_SUBMERSION.jpg

    $env:PYTHONPATH="backend"
    python -m lifeguard.eval.hard_negatives --clips eval/sprint_v1/clips.json --split dev --out hard_negatives \
        [--det-model backend/models/<detector>.onnx] [--provider tensorrt] [--full-frame]
    python -m lifeguard.eval.hard_negatives --video "Red Rescue/clip.mp4" --camera-id rr_cam1 --out hard_negatives

Safety rules (a drowning victim must never be taught as "not a person"):
  * only clips labelled `normal` are mined; the frozen `test` split is never read;
  * a `--video` is mined only with `--i-confirm-normal` (no drowning anywhere in it);
  * crops are candidates, not labels: `verified` is false until a person has looked at them. Many are real
    swimmers (a correct detection with a wrong behaviour verdict) and belong in a behaviour-negative set, not in
    the detector's background set - the manifest keeps scenario, detector score and track history so the
    two can be told apart.

For a vanished-track (SUBMERSION ghost) alert the crop is taken at the track's last sighting, where the object that
was mistaken for a swimmer is still in view, not at the alert frame where it is gone.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FORMAT = "aquaguard-hardneg/1"


@dataclass(frozen=True)
class HardNegativeRecord:
    """One mined false alert (one line of manifest.jsonl)."""

    crop: str  # path relative to the run directory
    clip_id: str
    camera_id: str
    video: str
    t: float  # clip seconds of the crop frame
    frame: int  # frame index of the crop frame
    alert_t: float  # clip seconds of the alert (differs from t for a ghost)
    track_id: int
    scenario: str
    ghost: bool
    det_conf: float | None  # detector score of the box on the crop frame
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels
    frame_w: int
    frame_h: int
    gate: dict[str, float] | None  # track history at the alert (scene/temporal_gate.py)
    detector: str
    thresholds_version: str
    full_frame: str | None = None
    verified: bool = False
    format: str = FORMAT


@dataclass
class _Pending:
    """An alert waiting for its frame in the second pass."""

    frame: int
    t: float
    alert_t: float
    track_id: int
    scenario: str
    ghost: bool
    det_conf: float | None
    bbox: tuple[float, float, float, float]
    gate: dict[str, float] | None


@dataclass
class _Sightings:
    """Per-track last sighting, collected from the run's trace callback."""

    last: dict[int, tuple[int, float, tuple[float, float, float, float], float]] = field(default_factory=dict)
    pending: list[_Pending] = field(default_factory=list)

    def __call__(self, rec: dict[str, Any]) -> None:
        f, t = int(rec["f"]), float(rec["t"])
        for p in rec.get("people", []):
            tid, x1, y1, x2, y2, conf = int(p[0]), *(float(v) for v in p[1:6])
            self.last[tid] = (f, t, (x1, y1, x2, y2), conf)


class HardNegativeWriter:
    """Writes crops + manifest lines; at most one crop per track per `min_interval_s`, `max_per_clip` per clip."""

    def __init__(
        self,
        run_dir: Path,
        *,
        pad: float = 0.15,
        min_interval_s: float = 2.0,
        max_per_clip: int = 300,
        full_frame: bool = False,
        detector: str = "",
        thresholds_version: str = "",
    ) -> None:
        self.run_dir = run_dir
        self.pad, self.min_interval_s, self.max_per_clip = pad, min_interval_s, max_per_clip
        self.full_frame = full_frame
        self.detector, self.thresholds_version = detector, thresholds_version
        run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = run_dir / "manifest.jsonl"
        self.written = 0

    def select(self, pending: Iterable[_Pending]) -> list[_Pending]:
        """Thin out repeats of one track (an alert re-fires while it lasts) and cap the clip."""
        out: list[_Pending] = []
        last_t: dict[int, float] = {}
        for p in sorted(pending, key=lambda q: q.alert_t):
            if p.alert_t - last_t.get(p.track_id, -1e9) < self.min_interval_s:
                continue
            last_t[p.track_id] = p.alert_t
            out.append(p)
        return out[: self.max_per_clip]

    def write(self, clip_id: str, camera_id: str, video: Path, frame: np.ndarray, p: _Pending) -> HardNegativeRecord:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = p.bbox
        px, py = (x2 - x1) * self.pad, (y2 - y1) * self.pad
        cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        cx2, cy2 = min(w, round(x2 + px)), min(h, round(y2 + py))
        if cx2 <= cx1 or cy2 <= cy1:
            raise ValueError(f"empty crop for track {p.track_id} at frame {p.frame}: {p.bbox}")
        sub = Path(_safe(clip_id))
        (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        stem = f"{_safe(clip_id)}_f{p.frame:06d}_t{p.track_id}_{p.scenario}"
        crop_rel = sub / f"{stem}.jpg"
        _imwrite(self.run_dir / crop_rel, frame[cy1:cy2, cx1:cx2])
        full_rel: Path | None = None
        if self.full_frame:
            full_rel = sub / f"{stem}_frame.jpg"
            _imwrite(self.run_dir / full_rel, frame)
        rec = HardNegativeRecord(
            crop=crop_rel.as_posix(),
            clip_id=clip_id,
            camera_id=camera_id,
            video=str(video),
            t=round(p.t, 3),
            frame=p.frame,
            alert_t=round(p.alert_t, 3),
            track_id=p.track_id,
            scenario=p.scenario,
            ghost=p.ghost,
            det_conf=None if p.det_conf is None else round(p.det_conf, 4),
            bbox=tuple(round(v, 1) for v in p.bbox),  # type: ignore[arg-type]
            frame_w=w,
            frame_h=h,
            gate=p.gate,
            detector=self.detector,
            thresholds_version=self.thresholds_version,
            full_frame=full_rel.as_posix() if full_rel is not None else None,
        )
        with self.manifest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec)) + "\n")
        self.written += 1
        return rec


def mine_clip(
    adapter: Any, clip: SimpleNamespace, writer: HardNegativeWriter, camera_id: str
) -> list[HardNegativeRecord]:
    """Run one normal clip; every alert is a false alert. Returns the records written."""
    from lifeguard.eval import runner

    seen = _Sightings()
    run = runner.run_clip(adapter, clip, trace=seen)
    for a in run.alerts:
        sighting = seen.last.get(a.track_id)
        box: tuple[float, float, float, float]
        conf: float | None
        if a.ghost and sighting is not None:
            f, t, box, conf = sighting  # where the "vanished" object was last in view
        else:
            f, t = a.frame, a.t
            box = (float(a.bbox[0]), float(a.bbox[1]), float(a.bbox[2]), float(a.bbox[3]))
            conf = sighting[3] if sighting is not None and sighting[0] == a.frame else None
        seen.pending.append(
            _Pending(f, t, float(a.t), int(a.track_id), a.scenario, bool(a.ghost), conf, box, getattr(a, "gate", None))
        )
    todo = writer.select(seen.pending)
    if not todo:
        return []
    by_frame: dict[int, list[_Pending]] = {}
    for p in todo:
        by_frame.setdefault(p.frame, []).append(p)
    out: list[HardNegativeRecord] = []
    cap = cv2.VideoCapture(str(clip.video_path))
    try:
        idx, last = 0, max(by_frame)
        while idx <= last and cap.grab():
            if idx in by_frame:
                ok, frame = cap.retrieve()
                if ok:
                    for p in by_frame[idx]:
                        try:
                            out.append(writer.write(clip.clip_id, camera_id, Path(clip.video_path), frame, p))
                        except (ValueError, OSError) as exc:
                            logger.warning("hard negative skipped: %s", exc)
            idx += 1
    finally:
        cap.release()
    return out


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _imwrite(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OSError(f"could not encode {path}")
    path.write_bytes(buf.tobytes())  # imencode + write: unicode-safe paths on Windows


def normal_clips(doc: dict[str, Any], split: str, only: set[str]) -> list[dict[str, Any]]:
    """The clips this tool may mine: label `normal`, never the test split."""
    if split == "test":
        raise ValueError("the test split is never mined (it stays unseen until the final evaluation)")
    return [
        c
        for c in doc["clips"]
        if c["label"] == "normal"
        and c["split"] != "test"
        and (split == "all" or c["split"] == split)
        and (not only or c["id"] in only)
    ]


# ── CLI ────────────────────────────────────────────────────────────────────
def _parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--clips", type=Path, help="an lifeguard eval clips.json (eval/sprint_v1/clips.json)")
    src.add_argument("--video", type=Path, help="one normal video (needs --i-confirm-normal)")
    ap.add_argument("--split", default="dev", choices=("dev", "train", "all"))
    ap.add_argument("--only", default="", help="comma-separated clip ids")
    ap.add_argument("--camera-id", default="", help="camera id recorded for --video (default: file stem)")
    ap.add_argument("--i-confirm-normal", action="store_true", help="--video contains no drowning")
    ap.add_argument("--out", type=Path, default=Path("hard_negatives"))
    ap.add_argument("--run", default=datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S"))
    ap.add_argument("--mode", default="High", choices=("Low", "Medium", "High"))
    ap.add_argument("--thresholds", type=Path, default=None)
    ap.add_argument("--det-model", type=Path, default=None)
    ap.add_argument("--provider", default=None, choices=("auto", "tensorrt", "cuda", "openvino", "cpu"))
    ap.add_argument("--full-frame", action="store_true", help="also save the whole frame (detector retraining)")
    ap.add_argument("--min-interval", type=float, default=2.0, help="seconds between crops of one track")
    ap.add_argument("--max-per-clip", type=int, default=300)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.video is not None and not args.i_confirm_normal:
        logger.error("--video needs --i-confirm-normal: crops from a clip with a drowning would teach a victim as junk")
        return 2
    if args.thresholds is not None:
        os.environ["LIFEGUARD_THRESHOLDS_FILE"] = str(args.thresholds.resolve())
    from lifeguard.eval import runner
    from lifeguard.vision import config as vision_config
    from lifeguard.vision.thresholds import active

    base = vision_config.from_env()
    det_path = str(args.det_model.resolve()) if args.det_model else base.det_model_path
    vision_config.configure(vision_config.VisionConfig(det_path, args.provider or base.provider, base.pose_model_path))
    runner.set_tree(runner.Tree.detect(None))

    if args.clips is not None:
        doc = json.loads(args.clips.read_text(encoding="utf-8"))
        root = args.clips.resolve().parent.parent.parent
        only = {c for c in args.only.split(",") if c}
        jobs = [(c["id"], c.get("camera", c["id"]), root / c["video"]) for c in normal_clips(doc, args.split, only)]
    else:
        jobs = [(args.video.stem, args.camera_id or args.video.stem, args.video)]

    writer = HardNegativeWriter(
        args.out / args.run,
        min_interval_s=args.min_interval,
        max_per_clip=args.max_per_clip,
        full_frame=args.full_frame,
        detector=Path(det_path).name,
        thresholds_version=str(active().version),
    )
    adapter = runner.make_adapter("pipeline", args.mode)
    adapter.warmup()
    for clip_id, camera_id, video in jobs:
        t0 = time.perf_counter()
        ns = SimpleNamespace(
            clip_id=clip_id,
            video_path=video,
            fps=None,
            frame_count=None,
            camera_id=camera_id,
            water_polygons=None,
            annotated=set(),
            labelled=False,
        )
        recs = mine_clip(adapter, ns, writer, camera_id)
        logger.info("%-24s %4d crops  (%.0f s)", clip_id, len(recs), time.perf_counter() - t0)
    logger.info("%d hard-negative crops -> %s", writer.written, writer.manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
