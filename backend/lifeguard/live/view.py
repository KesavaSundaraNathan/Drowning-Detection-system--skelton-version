"""Operator view of a live camera (plan Phase 6: "MJPEG and snapshots load same-origin").

The analysis worker keeps a REFERENCE to the newest frame and the newest detections (no copy, no
encoding on the hot path). Encoding happens only while somebody watches, at most `mjpeg_fps`
frames per second per viewer, downscaled to `mjpeg_width`, in a worker thread.

Annotation (the "Model View"): every tracked person gets a box and its track id; tentative
detections are grey. [Phase 5] A person whose alert is raised right now is drawn red and thicker, labelled
"#3 ALERT" (track highlight; cameras are independent, so there is no pool-wide id). The camera's monitoring
state is NOT burned into the image - the UI draws the NOT MONITORING overlay from the live status, so a
stale frame can never look healthy. [2.35.0] A person whose track found no box this frame (coasting,
TRACK_COAST_SECONDS) is drawn as a dashed light-grey box at their last observed position, labelled "#3 ?" - display
only, never scored.

[rtmpose clone] Skeleton overlay: the RTMPose keypoints the pipeline already computed for each tracked person are
drawn over the box (live/skeleton.py; `skeleton=False` turns it off). Display only: no inference here, nothing on the
capture / analysis path, and the cost is paid only while somebody watches.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from lifeguard.live import skeleton as skel

CONFIRMED = (60, 200, 60)  # BGR green
TENTATIVE = (160, 160, 160)
COASTING = (200, 200, 200)  # [2.35.0] a track carried without a box this frame: dashed, light grey
ALERTING = (40, 40, 230)  # BGR red: the person's alert is raised
BOXES_HEADER = "X-Lifeguard-Boxes"  # [view modes] overlay=client: this frame's boxes (boxes_json)


def track_label(t: Any) -> str:
    """'#3': the camera's own track id (Phase 5: no pool-wide global id)."""
    return f"#{t.track_id}"


def annotate(
    image: np.ndarray,
    detections: Any | None,
    width: int,
    alerting: frozenset[int] = frozenset(),
    poses: list[tuple[int, np.ndarray]] | None = None,
) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, width / float(w))
    out = (
        cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    )
    if detections is None:
        return out
    thick = max(1, out.shape[1] // 480)
    if poses:  # [rtmpose clone] under the boxes and labels, so they stay readable
        skel.draw(out, poses, scale, thick)
    for t in getattr(detections, "tracks", []) or []:
        x1, y1, x2, y2 = (int(v * scale) for v in t.bbox)
        hot = t.track_id >= 0 and t.track_id in alerting
        colour = ALERTING if hot else (TENTATIVE if getattr(t, "tentative", False) else CONFIRMED)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thick * 2 if hot else thick)
        if t.track_id >= 0:
            label = f"{track_label(t)} ALERT" if hot else track_label(t)
            cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * thick, colour, thick)
    for c in getattr(detections, "coasting", None) or []:  # [2.35.0] carried without a box: dashed, never scored
        x1, y1, x2, y2 = (int(v * scale) for v in c["bbox"])
        _dashed_rect(out, (x1, y1), (x2, y2), COASTING, thick)
        cv2.putText(
            out,
            f"#{c['track_id']} under" if c.get("followed") else f"#{c['track_id']} ?",  # [2.40.0] SUBMERGED_FOLLOW
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45 * thick,
            COASTING,
            thick,
        )
    return out


def _dashed_rect(
    img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], colour: tuple[int, int, int], thick: int
) -> None:
    dash = max(4, 6 * thick)
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, 2 * dash):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), colour, thick)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), colour, thick)
    for y in range(y1, y2, 2 * dash):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), colour, thick)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), colour, thick)


def encode_jpeg(image: np.ndarray, quality: int = 70) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def render(view: tuple[Any, ...] | None, width: int, camera: str = "", skeleton: bool = True) -> bytes | None:
    """`view` = (image, detections) or (image, detections, alerting track ids) from LiveRuntime.latest_view.
    [rtmpose clone] `skeleton`: draw each person's RTMPose keypoints (`camera` keys the pose memory)."""
    if view is None:
        return None
    image, detections = view[0], view[1]
    alerting = frozenset(view[2]) if len(view) > 2 and view[2] else frozenset()
    poses = skel.MEMORY.poses(camera, detections) if skeleton else None
    return encode_jpeg(annotate(image, detections, width, alerting, poses))


def boxes_json(
    detections: Any | None,
    alerting: frozenset[int],
    scale: float,
    size: tuple[int, int],
    poses: list[tuple[int, np.ndarray]] | None = None,
) -> str:
    """[view modes] The boxes annotate() would draw, as compact JSON in the scaled frame's pixels, for a browser that
    draws them itself (Cameras page view modes: Full / Clean / Alerts only). Display only.
    {"w", "h", "t": [[id, x1, y1, x2, y2, tentative 0/1, alerting 0/1], ...],
     "c": [[id, x1, y1, x2, y2, followed 0/1], ...]}
    [rtmpose clone] "k": [[id, x0, y0, c0, ..., x16, y16, c16], ...] - the skeletons (live/skeleton.to_json), only
    when `poses` is given."""
    import json

    tracks = []
    for t in getattr(detections, "tracks", []) or []:
        x1, y1, x2, y2 = (int(v * scale) for v in t.bbox)
        hot = int(t.track_id >= 0 and t.track_id in alerting)
        tracks.append([int(t.track_id), x1, y1, x2, y2, int(bool(getattr(t, "tentative", False))), hot])
    coast = []
    for c in getattr(detections, "coasting", None) or []:
        x1, y1, x2, y2 = (int(v * scale) for v in c["bbox"])
        coast.append([int(c["track_id"]), x1, y1, x2, y2, int(bool(c.get("followed")))])
    doc: dict[str, Any] = {"w": size[0], "h": size[1], "t": tracks, "c": coast}
    if poses is not None:
        ids = [int(t.track_id) for t in getattr(detections, "tracks", []) or []]
        doc["k"] = skel.to_json(poses, ids, scale)
    return json.dumps(doc, separators=(",", ":"))


def render_client(
    view: tuple[Any, ...] | None, width: int, camera: str = "", skeleton: bool = True
) -> tuple[bytes, str] | None:
    """[view modes] (clean JPEG without boxes, boxes_json of the same frame) - the browser draws the boxes, so one
    stream serves every view mode and switching mode never reconnects. Same frame, scale and detections as render()."""
    if view is None:
        return None
    image, detections = view[0], view[1]
    alerting = frozenset(view[2]) if len(view) > 2 and view[2] else frozenset()
    w = image.shape[1]
    scale = min(1.0, width / float(w))
    out = annotate(image, None, width)
    jpg = encode_jpeg(out)
    if jpg is None:
        return None
    poses = skel.MEMORY.poses(camera, detections) if skeleton else None
    return jpg, boxes_json(detections, alerting, scale, (int(out.shape[1]), int(out.shape[0])), poses)
