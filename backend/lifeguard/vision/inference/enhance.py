# mypy: disable-error-code="assignment, attr-defined"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""
[v3] Ported verbatim from the v2 baseline detection/enhance.py (imports only).

detection/enhance.py — optional input conditioning for poor CCTV (Phase 1.2).

Every option here is OFF by default (config/settings.py → DET_ENHANCE,
DET_TILING). The spec is explicit: each one is kept only if the eval set
shows it helps, so they exist as switches the harness can A/B:

    python scripts/evaluate_pipeline.py --name clahe --set DET_ENHANCE=clahe

  clahe    contrast-limited histogram equalisation on the L channel (LAB):
           lifts faint, low-contrast swimmers in murky or night footage
  denoise  3×3 bilateral filter: removes compression speckle without
           smearing edges
  glare    soft-knee compression of the brightest luminance so a sun glint
           does not swamp the detector's normalisation

They run on the ALREADY RESIZED detector input (≈640 px), so the cost is
~1–3 ms, not the 10+ ms full-resolution filtering would take.

Tiling: `tile_windows()` lays native-resolution windows (detector input size,
with overlap) over a region of interest — the pool — so small, distant
swimmers are seen at full resolution instead of being downscaled 3×.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import cv2
import numpy as np


def parse_modes(spec: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                m.strip().lower()
                for m in (spec or "").split(",")
                if m.strip() and m.strip().lower() not in ("none", "off")
            }
        )
    )


@lru_cache(maxsize=4)
def _clahe(clip: float, grid: int):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))


@lru_cache(maxsize=2)
def _glare_lut(knee: int) -> np.ndarray:
    x = np.arange(256, dtype=np.float32)
    y = np.where(x <= knee, x, knee + (x - knee) * 0.35)
    return np.clip(y, 0, 255).astype(np.uint8)


def apply(img: np.ndarray, modes: Sequence[str], clahe_clip: float = 2.0) -> np.ndarray:
    """Return a conditioned copy of `img` (BGR uint8). `modes` from parse_modes()."""
    if not modes:
        return img
    out = img
    if "denoise" in modes:
        out = cv2.bilateralFilter(out, 5, 30, 5)
    if "clahe" in modes or "glare" in modes:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0]
        if "glare" in modes:
            l = cv2.LUT(l, _glare_lut(200))
        if "clahe" in modes:
            l = _clahe(float(clahe_clip), 8).apply(l)
        lab[:, :, 0] = l
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out


def tile_windows(
    roi: tuple[int, int, int, int], tile: int, overlap: float, frame_w: int, frame_h: int
) -> list[tuple[int, int, int, int]]:
    """Native-resolution windows (x1, y1, x2, y2) of size ≤ tile covering `roi`."""
    x1, y1, x2, y2 = (int(v) for v in roi)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)
    step = max(1, int(tile * (1.0 - overlap)))

    def starts(a: int, b: int) -> list[int]:
        if b - a <= tile:
            return [max(0, min(a, b - tile))]
        s = list(range(a, b - tile + 1, step))
        if s[-1] + tile < b:
            s.append(b - tile)
        return s

    out = []
    for yy in starts(y1, y2):
        for xx in starts(x1, x2):
            out.append((xx, yy, min(xx + tile, frame_w), min(yy + tile, frame_h)))
    return out


def merge_boxes(
    boxes: np.ndarray, scores: np.ndarray, iou: float = 0.6, iomin: float = 0.75, return_index: bool = False
):
    """
    Class-agnostic greedy merge over full-frame + tile detections (best score
    wins). A box is suppressed by a better one when their IoU > `iou` OR when
    most of the smaller box lies inside the other (intersection / smaller
    area > `iomin`). Plain IoU-NMS alone let a tile's partial "half swimmer"
    box survive next to the full-frame box of the same person (IoU ~0.4),
    which is exactly the duplicate-track / ghost-alert spike measured in
    eval/reports/coco_only_tiled.
    """
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    scores = np.asarray(scores, np.float32).reshape(-1)
    if len(boxes) == 0:
        keep = np.zeros(0, int)
        return (boxes, scores, keep) if return_index else (boxes, scores)
    area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    order = np.argsort(-scores, kind="stable")
    alive = np.ones(len(boxes), bool)
    out = boxes.copy()
    keep = []
    for i in order:
        if not alive[i]:
            continue
        keep.append(i)
        alive[i] = False
        ix1 = np.maximum(boxes[i, 0], boxes[:, 0])
        iy1 = np.maximum(boxes[i, 1], boxes[:, 1])
        ix2 = np.minimum(boxes[i, 2], boxes[:, 2])
        iy2 = np.minimum(boxes[i, 3], boxes[:, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        union = area[i] + area - inter
        ov = inter / np.maximum(union, 1e-6)
        contain = inter / np.maximum(np.minimum(area[i], area), 1e-6)
        hit = alive & ((ov > iou) | (contain > iomin))
        # A kept box that lies inside a suppressed, larger box is a fragment of
        # the same person (tile cut, partial occlusion): it keeps its score but
        # takes the enclosing extent, so the track sees the whole body.
        enclosing = hit & (contain > iomin) & (area > area[i])
        for j in np.where(enclosing)[0]:
            out[i, :2] = np.minimum(out[i, :2], boxes[j, :2])
            out[i, 2:] = np.maximum(out[i, 2:], boxes[j, 2:])
        alive &= ~hit
    keep = np.asarray(keep, int)
    return (out[keep], scores[keep], keep) if return_index else (out[keep], scores[keep])


def cut_by_tile(
    boxes: np.ndarray, tile: tuple[int, int, int, int], frame_w: int, frame_h: int, margin: int = 3
) -> np.ndarray:
    """
    Mask of tile-local boxes (tile pixel coords) that touch an edge of the tile
    that is NOT a frame edge — i.e. a person cut in half by the tile window.
    With 20 % overlap the neighbouring tile or the full-frame pass sees them
    whole, so the fragment is dropped rather than merged.
    """
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    tx1, ty1, tx2, ty2 = tile
    w, h = tx2 - tx1, ty2 - ty1
    cut = np.zeros(len(boxes), bool)
    if tx1 > 0:
        cut |= boxes[:, 0] <= margin
    if ty1 > 0:
        cut |= boxes[:, 1] <= margin
    if tx2 < frame_w:
        cut |= boxes[:, 2] >= w - margin
    if ty2 < frame_h:
        cut |= boxes[:, 3] >= h - margin
    return cut


def roi_of(
    points: np.ndarray | None, frame_w: int, frame_h: int, margin: float = 0.08
) -> tuple[int, int, int, int] | None:
    """Bounding box of polygon points (N×2 px) grown by `margin` × frame size."""
    if points is None or len(points) == 0:
        return None
    p = np.asarray(points, np.float32).reshape(-1, 2)
    mx, my = margin * frame_w, margin * frame_h
    return (
        int(max(0, p[:, 0].min() - mx)),
        int(max(0, p[:, 1].min() - my)),
        int(min(frame_w, p[:, 0].max() + mx)),
        int(min(frame_h, p[:, 1].max() + my)),
    )


def weighted_box_fusion(
    boxes: np.ndarray, scores: np.ndarray, water: np.ndarray, iou: float = 0.55
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [v3] Phase 8 (plan 7.2): weighted box fusion over the full-frame + tile detections of ONE frame.

    Boxes are visited best score first; a box joins the first cluster whose current fused box it overlaps
    by IoU > `iou`, else it starts a new cluster. A cluster's box is the score-weighted mean of its members'
    coordinates (so a person seen whole by the full frame and by a tile gets one box between the two views,
    instead of whichever happened to score highest); its score is the BEST member score (tiles exist to
    find people the full-frame pass scores low or misses - averaging them down would undo that); its water
    vote is the score-weighted mean of the members that have one (NaN when none do).
    Fragments a tile cut in half are removed before (cut_by_tile) and after (merge_boxes containment).
    """
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    scores = np.asarray(scores, np.float32).reshape(-1)
    water = np.asarray(water, np.float32).reshape(-1)
    if len(boxes) == 0:
        return boxes, scores, water
    order = np.argsort(-scores, kind="stable")
    fused: list[np.ndarray] = []
    members: list[list[int]] = []
    for i in order:
        best, best_iou = -1, iou
        for c, fb in enumerate(fused):
            ix = max(0.0, min(fb[2], boxes[i, 2]) - max(fb[0], boxes[i, 0]))
            iy = max(0.0, min(fb[3], boxes[i, 3]) - max(fb[1], boxes[i, 1]))
            inter = ix * iy
            union = (
                (fb[2] - fb[0]) * (fb[3] - fb[1]) + (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) - inter
            )
            ov = inter / union if union > 0 else 0.0
            if ov > best_iou:
                best, best_iou = c, ov
        if best < 0:
            fused.append(boxes[i].copy())
            members.append([int(i)])
        else:
            members[best].append(int(i))
            m = np.asarray(members[best])
            w = scores[m]
            fused[best] = (boxes[m] * w[:, None]).sum(0) / max(float(w.sum()), 1e-6)
    out_b = np.asarray(fused, np.float32).reshape(-1, 4)
    out_s = np.asarray([scores[m].max() for m in members], np.float32)
    out_w = np.full(len(members), np.nan, np.float32)
    for c, m in enumerate(members):
        mm = np.asarray(m)
        ok = ~np.isnan(water[mm])
        if ok.any():
            out_w[c] = float((water[mm][ok] * scores[mm][ok]).sum() / max(float(scores[mm][ok].sum()), 1e-6))
    return out_b, out_s, out_w


# ── [v7 F4] resolution mosaic ─────────────────────────────────────────────────────────────────────────────────────
# The detector sees the whole frame at 640 px: a 2560 x 1440 camera is shrunk 4x (mid-pool swimmers become 15-40 px)
# and a 626 x 360 clip leaves a struggling child at ~30 px. The mosaic re-scans only the water rows where people are
# small, at DET_MOSAIC_GAIN x the full-frame resolution (0.5x native at 2560, a 2x upscale at 626), cut into strips
# packed into at most `max_canvases` canvases of size x size - one detector call each, batched like any other.


def mosaic_region(
    roi: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    in_size: int,
    gain: float,
    far_frac: float,
    near_frac: float,
) -> tuple[tuple[int, int, int, int], float] | None:
    """(region in frame px, scale) to re-scan, or None. scale = gain x the full-frame pass's scale, at most 1.0 unless
    the frame is smaller than the detector input (then it is an upscale). The top far_frac of the water rows (the far
    end) is never re-scanned; for scale <= 1 (a large frame) the rows below near_frac are not either - people there
    are already big in the full-frame pass. For an upscale (a small frame) everything below the far end is."""
    x1, y1, x2, y2 = (int(v) for v in roi)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    base = in_size / float(max(frame_w, frame_h))
    scale = gain * base
    if base < 1.0:
        scale = min(scale, 1.0)
    if scale <= base * 1.05:
        return None  # nothing to gain over the full-frame pass
    h = y2 - y1
    top = y1 + int(round(far_frac * h))
    bottom = y2 if scale > 1.0 else y1 + int(round(near_frac * h))
    if bottom - top < 16:
        return None
    return (x1, top, x2, bottom), scale


def mosaic_plan(
    region: tuple[int, int, int, int], scale: float, size: int, overlap: float, max_canvases: int
) -> list[list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]]:
    """Canvases, each a list of (src rect in frame px, dst rect in canvas px). The scaled region is cut into
    size-wide strips (overlap between neighbours, horizontally and vertically) stacked top to bottom; strips that do
    not fit in max_canvases are dropped from the BOTTOM of the region (the nearer rows, best seen by the full frame)."""
    x1, y1, x2, y2 = region
    sw = max(1, int(round((x2 - x1) * scale)))
    sh = max(1, int(round((y2 - y1) * scale)))
    strip_h = min(sh, size)
    step_x = max(1, int(size * (1.0 - overlap)))
    step_y = max(1, int(strip_h * (1.0 - overlap)))
    xs = [0] if sw <= size else list(range(0, sw - size + 1, step_x))
    if sw > size and xs[-1] + size < sw:
        xs.append(sw - size)
    ys = [0] if sh <= strip_h else list(range(0, sh - strip_h + 1, step_y))
    if sh > strip_h and ys[-1] + strip_h < sh:
        ys.append(sh - strip_h)
    strips = [(sx, sy, min(sx + size, sw), min(sy + strip_h, sh)) for sy in ys for sx in xs]
    per_canvas = max(1, size // strip_h)
    canvases: list[list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]] = []
    for k, (sx1, sy1, sx2, sy2) in enumerate(strips):
        c, slot = divmod(k, per_canvas)
        if c >= max_canvases:
            break
        if slot == 0:
            canvases.append([])
        src = (
            x1 + int(sx1 / scale),
            y1 + int(sy1 / scale),
            x1 + int(np.ceil(sx2 / scale)),
            y1 + int(np.ceil(sy2 / scale)),
        )
        dst = (0, slot * strip_h, sx2 - sx1, slot * strip_h + (sy2 - sy1))
        canvases[-1].append((src, dst))
    return canvases


def mosaic_canvas(
    frame: np.ndarray, placements: Sequence[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]], size: int
) -> np.ndarray:
    canvas = np.zeros((size, size, 3), np.uint8)
    for (sx1, sy1, sx2, sy2), (dx1, dy1, dx2, dy2) in placements:
        crop = frame[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            continue
        interp = cv2.INTER_CUBIC if (dx2 - dx1) > (sx2 - sx1) else cv2.INTER_AREA
        canvas[dy1:dy2, dx1:dx2] = cv2.resize(crop, (dx2 - dx1, dy2 - dy1), interpolation=interp)
    return canvas


def mosaic_unmap(
    boxes: np.ndarray,
    placements: Sequence[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]],
    region: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    margin: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """(frame-px boxes, indices of the canvas boxes kept). A box belongs to the strip holding its centre; a box that
    crosses into another strip, or touches a strip edge that is not a frame edge (a person cut by the strip - the
    overlapping neighbour or the full-frame pass sees them whole), is dropped."""
    out, keep = [], []
    for i, b in enumerate(np.asarray(boxes, np.float32).reshape(-1, 4)):
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        for (sx1, sy1, sx2, sy2), (dx1, dy1, dx2, dy2) in placements:
            if not (dx1 <= cx < dx2 and dy1 <= cy < dy2):
                continue
            if b[0] < dx1 - margin or b[1] < dy1 - margin or b[2] > dx2 + margin or b[3] > dy2 + margin:
                break  # crosses into another strip
            if (sx1 > 0 and b[0] <= dx1 + margin) or (sy1 > 0 and b[1] <= dy1 + margin):
                break
            if (sx2 < frame_w and b[2] >= dx2 - margin) or (sy2 < frame_h and b[3] >= dy2 - margin):
                break
            fx, fy = (sx2 - sx1) / float(max(dx2 - dx1, 1)), (sy2 - sy1) / float(max(dy2 - dy1, 1))
            out.append(
                [sx1 + (b[0] - dx1) * fx, sy1 + (b[1] - dy1) * fy, sx1 + (b[2] - dx1) * fx, sy1 + (b[3] - dy1) * fy]
            )
            keep.append(i)
            break
    return np.asarray(out, np.float32).reshape(-1, 4), np.asarray(keep, int)
