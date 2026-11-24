"""
[v3] Ported verbatim from the v2 baseline detection/model_view.py (imports only).

detection/model_view.py — "What the model sees" overlay (Phase 3 / 4).

One renderer for every Model View surface so they all look identical:
  * live MJPEG `?view=model` for each camera (RTSP / webcam / file),
  * the annotated snapshot stored with every alert,
  * the annotated peak frame of an uploaded video.
(The uploaded-video player draws the same thing client-side from the stored
timeline — frontend ModelViewOverlay uses the same colours and labels.)

Drawn: water/shallow/deck polygon outlines, every tracked person's box, track
id, pose skeleton, foot point, state and label with risk %, detected floatables,
submerged-swimmer markers, a header with inference stats, a colour legend and a
red frame border while any alert is active.

Colours: Red = drowning scenario · Yellow = deck boundary warning ·
Green = normal swimming · Orange = caution · Grey = deck / bystander.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import cv2
import numpy as np

from lifeguard.vision.constants import (
    BLACKOUT_CRITICAL_STATE,
    BLACKOUT_WATCH_STATE,
    KEYPOINT_CONF_THRESHOLD,
    PERSON_STATE_COLORS_BGR,
    POSE_SKELETON,
)

ZONE_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "WATER": (255, 160, 40),
    "SHALLOW": (255, 220, 90),
    "DECK": (150, 150, 150),
}
FLOAT_COLOR = (255, 80, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

LEGEND = (
    ("DROWNING", "Drowning"),
    ("CAUTION", "Caution"),
    ("SWIMMER", "Normal"),
    ("DECK_EDGE", "Deck edge"),
    ("DECK", "Deck"),
)


def person_caption(item: dict[str, Any]) -> str:
    """The text drawn above a person's box (mirrored by the frontend overlay)."""
    tid = item.get("track_id", "?")
    state = item.get("state", "SWIMMER")
    label = item.get("label", "")
    pct = int(round(float(item.get("risk", 0.0)) * 100))
    if state == "DECK":
        return f"#{tid} DECK"
    if state == "DECK_EDGE":
        return f"#{tid} DECK | EDGE WARNING"
    if state == "TENTATIVE":
        return "#? detecting"
    if state == "ANALYZING":
        return f"#{tid} SWIMMER | analyzing"
    if state in ("SUBMERGED", BLACKOUT_WATCH_STATE, BLACKOUT_CRITICAL_STATE):
        el = item.get("elapsed")
        secs = f" {el:.1f}s" if isinstance(el, (int, float)) else ""
        if state == BLACKOUT_CRITICAL_STATE:
            return f"#{tid} DROWNING ALERT | SUBMERGED{secs} | {pct}%"
        if state == BLACKOUT_WATCH_STATE and not item.get("alert"):
            return f"#{tid} BLACKOUT WATCH{secs}"
        return f"#{tid} SUBMERGED{secs} | {pct}%" if item.get("alert") else f"#{tid} LOST{secs}"
    if state == "DROWNING":  # [Stage 4] one alert name: never the rule's type
        return f"#{tid} DROWNING ALERT {pct}%"
    if state == "CAUTION":
        return f"#{tid} CAUTION {pct}%"
    return f"#{tid} SWIMMER | {label} {pct}%"


def _text(canvas: np.ndarray, text: str, org: tuple[int, int], scale: float, color, thick: int = 1) -> None:
    cv2.putText(canvas, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _label_box(canvas: np.ndarray, text: str, x: int, y: int, scale: float, bg, fg=(0, 0, 0)) -> None:
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    h, w = canvas.shape[:2]
    x = int(max(0, min(w - tw - 6, x)))
    y0 = int(max(th + 6, y))
    cv2.rectangle(canvas, (x, y0 - th - 6), (x + tw + 6, y0), bg, -1)
    _text(canvas, text, (x + 3, y0 - 4), scale, fg)


def _dashed_rect(
    canvas: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color, thick: int = 2, dash: int = 8
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(canvas, (x, y1), (min(x + dash, x2), y1), color, thick)
        cv2.line(canvas, (x, y2), (min(x + dash, x2), y2), color, thick)
    for y in range(y1, y2, dash * 2):
        cv2.line(canvas, (x1, y), (x1, min(y + dash, y2)), color, thick)
        cv2.line(canvas, (x2, y), (x2, min(y + dash, y2)), color, thick)


def draw_skeleton(canvas: np.ndarray, kps: Sequence[Sequence[float]], color, thick: int = 1) -> None:
    thr = KEYPOINT_CONF_THRESHOLD
    for a, b in POSE_SKELETON:
        if a < len(kps) and b < len(kps) and kps[a][2] >= thr and kps[b][2] >= thr:
            cv2.line(
                canvas, (int(kps[a][0]), int(kps[a][1])), (int(kps[b][0]), int(kps[b][1])), color, thick, cv2.LINE_AA
            )
    for k in kps:
        if k[2] >= thr:
            cv2.circle(canvas, (int(k[0]), int(k[1])), max(2, thick + 1), color, -1, cv2.LINE_AA)


def render_model_view(
    frame: np.ndarray,
    items: Iterable[dict[str, Any]],
    *,
    zones: Iterable[Any] | None = None,
    header: str = "",
    floatables: Iterable[Any] | None = None,
    legend: bool = True,
) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    fs = max(0.38, min(0.9, 0.5 * w / 1280.0 + 0.1))
    thick = 2 if w >= 960 else 1
    items = list(items)

    # ── zones ────────────────────────────────────────────────────────────────
    zones = list(zones or [])
    if zones:
        overlay = canvas.copy()
        for z in zones:
            color = ZONE_COLORS_BGR.get(getattr(z, "zone_type", "WATER"), (200, 200, 200))
            cv2.fillPoly(overlay, [z.contour], color)
        cv2.addWeighted(overlay, 0.14, canvas, 0.86, 0, canvas)
        for z in zones:
            color = ZONE_COLORS_BGR.get(z.zone_type, (200, 200, 200))
            cv2.polylines(canvas, [z.contour], True, color, thick, cv2.LINE_AA)
            cx, cy = z.contour.reshape(-1, 2).mean(axis=0).astype(int)
            _text(
                canvas,
                f"{z.name} [{z.zone_type}] x{z.sensitivity_multiplier:g}",
                (int(cx) - 60, int(cy)),
                fs * 0.9,
                color,
            )

    # ── floatables ───────────────────────────────────────────────────────────
    for f in floatables or []:
        x1, y1, x2, y2 = f.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), FLOAT_COLOR, 1)
        _text(canvas, f.class_name, (x1, max(12, y1 - 3)), fs * 0.8, FLOAT_COLOR)

    # ── people (drowning drawn last so it is on top) ─────────────────────────
    order = {
        "DECK": 0,
        "TENTATIVE": 0,
        "ANALYZING": 1,
        "SWIMMER": 2,
        "DECK_EDGE": 3,
        "CAUTION": 4,
        BLACKOUT_WATCH_STATE: 5,
        "SUBMERGED": 5,
        BLACKOUT_CRITICAL_STATE: 6,
        "DROWNING": 6,
    }
    for it in sorted(items, key=lambda i: order.get(i.get("state", ""), 2)):
        state = it.get("state", "SWIMMER")
        color = PERSON_STATE_COLORS_BGR.get(state, (255, 255, 255))
        x1, y1, x2, y2 = (int(v) for v in it["bbox"])
        if it.get("ghost"):
            _dashed_rect(canvas, (x1, y1), (x2, y2), color, thick + 1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.drawMarker(canvas, (cx, cy), color, cv2.MARKER_TILTED_CROSS, 18, thick + 1)
        else:
            cv2.rectangle(
                canvas, (x1, y1), (x2, y2), color, thick + (2 if state in ("DROWNING", BLACKOUT_CRITICAL_STATE) else 0)
            )
            if it.get("keypoints"):
                draw_skeleton(canvas, it["keypoints"], color, 1 if thick == 1 else 2)
            fx, fy = it.get("foot", ((x1 + x2) / 2, y2))
            cv2.circle(canvas, (int(fx), int(fy)), 4, color, -1, cv2.LINE_AA)
        _label_box(canvas, person_caption(it), x1, y1, fs, color)
        zone = it.get("zone")
        if zone and zone not in ("Whole Frame",):
            _text(canvas, zone, (x1 + 2, min(h - 4, y2 + int(16 * fs / 0.5))), fs * 0.85, color)

    # ── header ───────────────────────────────────────────────────────────────
    if header:
        bar_h = int(22 * fs / 0.5)
        cv2.rectangle(canvas, (0, 0), (w, bar_h), (0, 0, 0), -1)
        _text(canvas, header, (6, bar_h - int(6 * fs / 0.5)), fs, (255, 255, 255))

    # ── legend ───────────────────────────────────────────────────────────────
    if legend:
        sq = int(10 * fs / 0.5)
        pad = 6
        x = pad
        y = h - pad - sq
        widths = [cv2.getTextSize(t, FONT, fs * 0.85, 1)[0][0] for _, t in LEGEND]
        total = sum(widths) + len(LEGEND) * (sq + 14) + pad
        cv2.rectangle(canvas, (0, y - pad), (min(w, total), h), (0, 0, 0), -1)
        for (state, text), tw in zip(LEGEND, widths):
            cv2.rectangle(canvas, (x, y), (x + sq, y + sq), PERSON_STATE_COLORS_BGR[state], -1)
            _text(canvas, text, (x + sq + 4, y + sq - 1), fs * 0.85, (255, 255, 255))
            x += sq + 14 + tw

    if any(it.get("alert") for it in items):
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), PERSON_STATE_COLORS_BGR["DROWNING"], 6)
    return canvas


def crop_around(frame: np.ndarray, bbox: Sequence[int], margin: float = 1.2, min_size: int = 160) -> np.ndarray:
    """A crop centred on `bbox` (for alert thumbnails)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(min_size, int(max(x2 - x1, y2 - y1) * (1 + 2 * margin)))
    sx1 = int(max(0, min(w - 1, cx - side / 2)))
    sy1 = int(max(0, min(h - 1, cy - side / 2)))
    sx2 = int(max(sx1 + 1, min(w, cx + side / 2)))
    sy2 = int(max(sy1 + 1, min(h, cy + side / 2)))
    return frame[sy1:sy2, sx1:sx2].copy()
