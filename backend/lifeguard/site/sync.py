"""Inter-camera sync from the SYNC flash (Phase 11 WP6, gates_wp6.session_checks.sync).

Recording timestamps are the server's arrival times (site/recorder.py), so two cameras differ by their encoder +
network latency. The SYNC drill puts one light flash in view of every camera; in each recording the flash is the
largest frame-to-frame rise of mean brightness within +-SEARCH_S of the logged mark. offset = the camera's flash time -
the median over cameras. Offsets are APPLIED in the replay only if the largest exceeds MAX_FRAMES frames; either way
they are reported. A camera where no clear rise is found (< MIN_RISE grey levels) gets no offset and is reported.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import numpy as np

SEARCH_S = 2.0
MIN_RISE = 12.0
MAX_FRAMES = 2


def flash_time(recording: Path, start_wall: float, mark_wall: float) -> dict[str, Any]:
    """Wall time of the largest brightness rise within +-SEARCH_S of the mark."""
    import cv2

    cap = cv2.VideoCapture(str(recording))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, (mark_wall - SEARCH_S - start_wall) * 1000.0))
        prev: float | None = None
        best: tuple[float, float] | None = None
        while True:
            ok, img = cap.read()
            if not ok:
                break
            wall = start_wall + float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if wall > mark_wall + SEARCH_S:
                break
            level = float(np.mean(img[::4, ::4]))
            if prev is not None and wall >= mark_wall - SEARCH_S:
                rise = level - prev
                if best is None or rise > best[0]:
                    best = (rise, wall)
            prev = level
    finally:
        cap.release()
    if best is None or best[0] < MIN_RISE:
        return {"found": False, "rise": None if best is None else round(best[0], 1)}
    return {"found": True, "rise": round(best[0], 1), "wall": best[1]}


def offsets(recordings: list[dict[str, Any]], folder: Path, mark_wall: float, fps: float = 15.0) -> dict[str, Any]:
    per: dict[str, dict[str, Any]] = {}
    for r in recordings:
        if "start_wall" not in r:
            continue
        per[r["camera_id"]] = {"camera": r["camera"], **flash_time(folder / r["file"], r["start_wall"], mark_wall)}
    found = {cid: v["wall"] for cid, v in per.items() if v.get("found")}
    if len(found) < 2:
        return {
            "mark_wall": mark_wall,
            "per_camera": per,
            "offsets_s": {},
            "applied_offsets_s": {},
            "verdict": "NO_FLASH",
        }
    med = statistics.median(found.values())
    offs = {cid: round(w - med, 3) for cid, w in found.items()}
    worst_frames = max(abs(o) for o in offs.values()) * fps
    applied = offs if worst_frames > MAX_FRAMES else {}
    return {
        "mark_wall": mark_wall,
        "per_camera": per,
        "offsets_s": offs,
        "max_offset_frames": round(worst_frames, 2),
        "applied_offsets_s": applied,
        "verdict": "APPLIED" if applied else "WITHIN_TOLERANCE",
        "missing": sorted(set(per) - set(found)),
    }
