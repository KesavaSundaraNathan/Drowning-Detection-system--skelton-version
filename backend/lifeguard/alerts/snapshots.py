"""Alert snapshot files: naming, JPEG writing, and PATH CONTAINMENT.

Every alert stores up to three JPEGs under the snapshots directory (annotated Model View frame, raw
frame, annotated crop). A stored path is only ever served or deleted when it resolves to a regular
file INSIDE that directory - a tampered or legacy row pointing at `..\\..\\Windows\\...` or another
volume is refused, never followed.

[v3] From the v2 baseline services/alert_writer.py: recommended_action, _write_jpeg (as write_jpeg),
the snapshot file names and _inside_snapshots_dir / snapshot_path_for / delete_snapshots (as
resolve_snapshot / delete_snapshots) - logic unchanged, the snapshots directory is a parameter
instead of v2's global settings, and failures are logged with their stack. The DB half of
persist_alert is Phase 5 (alert outbox).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from lifeguard.vision.constants import RECOMMENDED_ACTIONS
from lifeguard.vision.scene import alert_reason

logger = logging.getLogger(__name__)

SNAPSHOT_JPEG_QUALITY = 85
KINDS = ("annotated", "raw", "thumbnail")


def recommended_action(priority: str, scenario: str) -> str:
    first = alert_reason.REASON_ACTIONS.get(alert_reason.reason_code(scenario))
    base = RECOMMENDED_ACTIONS.get(priority, "Monitor immediately.")
    return f"{first} {base}" if first else base


def snapshot_paths(snapshots_dir: Path, alert_id: int) -> dict[str, Path]:
    """The three files of one alert (v2 names)."""
    return {
        "annotated": snapshots_dir / f"alert_{alert_id}.jpg",
        "raw": snapshots_dir / f"alert_{alert_id}_raw.jpg",
        "thumbnail": snapshots_dir / f"alert_{alert_id}_thumb.jpg",
    }


def write_jpeg(path: Path, img: np.ndarray) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_JPEG_QUALITY])
        if not ok:
            return False
        path.write_bytes(buf.tobytes())
        return True
    except (OSError, cv2.error):
        logger.warning("[alert] snapshot write failed %s", path, exc_info=True)
        return False


def resolve_snapshot(stored: str | None, snapshots_dir: Path) -> Path | None:
    """The stored path, only if it is a regular file INSIDE `snapshots_dir`."""
    if not stored:
        return None
    try:
        p = Path(stored).resolve()
        p.relative_to(snapshots_dir.resolve())
    except (ValueError, OSError):
        return None
    return p if p.is_file() else None


def delete_snapshots(stored: Iterable[str | None], snapshots_dir: Path) -> int:
    """Delete an alert's snapshot files - ONLY files inside `snapshots_dir`. Returns the number removed."""
    removed = 0
    for value in stored:
        p = resolve_snapshot(value, snapshots_dir)
        if p is not None:
            try:
                p.unlink()
                removed += 1
            except OSError:
                logger.warning("[alert] could not delete snapshot %s", p, exc_info=True)
    return removed
