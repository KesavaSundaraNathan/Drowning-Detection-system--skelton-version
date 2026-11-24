"""Detector output decoding (three ONNX layouts) and person-class selection.

[v3] From the v2 baseline detection/rtm_backend.py: class-name tables, person_class_ids,
agnostic_nms, _person_view, run_detector_people and decode_detections verbatim; run_detector
without the rtmlib-object branch; rtmlib's multiclass_nms vendored verbatim below.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from lifeguard.vision.constants import DET_DETR_MIN_SCORE, DET_DETR_NESTED_IOA, DET_DETR_NMS_IOU, DET_DETR_TRACK_FLOOR
from lifeguard.vision.inference.models import FORMAT_XYXY_SCORES, DetModel

logger = logging.getLogger(__name__)


# Class names a detector may use for a person, and the two zone-aware classes
# the client-trained detector predicts (documentation/ML_CORE_V2.md).
WATER_CLASS_NAMES = ("person_in_water", "in_water", "swimmer")
LAND_CLASS_NAMES = ("person_out_of_water", "out_of_water", "person_on_land")
PERSON_CLASS_NAMES = ("person",) + WATER_CLASS_NAMES + LAND_CLASS_NAMES
COCO80 = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


# ── vendored from rtmlib 0.0.16 (Apache License 2.0, Copyright (c) Tau-J), file
# rtmlib/tools/object_detection/post_processings.py - verbatim, so the default detector's
# class-aware NMS is bit-identical to the v2 baseline (rtmlib is not a server dependency). ──
def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr, classes=None):
    """Multiclass NMS implemented in Numpy.

    Class-aware version.

    Args:
        boxes (np.ndarray): Boxes in shape (N, 4), xyxy format.
        scores (np.ndarray): Per-class scores in shape (N, num_classes).
        nms_thr (float): IoU threshold for NMS.
        score_thr (float): Score threshold to filter out low-confidence
            boxes before NMS.
        classes (Sequence[int] | None): [v6] only these class columns are
            suppressed and returned (NMS is per class, so each returned
            class is identical to the all-class result). None: all classes.

    Returns:
        tuple:
        - dets (np.ndarray | None): Kept detections in shape (M, 6),
            formatted as (x1, y1, x2, y2, score, cls_ind). ``None`` if no
            detection survives.
        - keep (np.ndarray | None): Indices into the *original* ``boxes``/
            ``scores`` arrays (axis 0) for each row of ``dets``, i.e.
            ``dets[i, :4] == boxes[keep[i]]``. ``None`` if no detection
            survives. Note that the same original index may appear more
            than once if the corresponding box survives NMS under more
            than one class.
    """
    final_dets = []
    final_keep = []
    num_classes = scores.shape[1]
    all_indices = np.arange(boxes.shape[0])
    wanted = range(num_classes) if classes is None else sorted({int(c) for c in classes if 0 <= int(c) < num_classes})
    for cls_ind in wanted:
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            valid_indices = all_indices[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                final_dets.append(dets)
                # Map the NMS-local `keep` indices (relative to
                # `valid_boxes`) back to indices in the original,
                # unfiltered `boxes`/`scores` arrays. Callers (e.g.
                # RTMO) rely on this to index other per-box arrays
                # (such as keypoints) that share the original ordering.
                final_keep.append(valid_indices[keep])
    if len(final_dets) == 0:
        return None, None
    return np.concatenate(final_dets, 0), np.concatenate(final_keep, 0)


def person_class_ids(names: Sequence[str] | None) -> tuple[list[int], list[int], list[int]]:
    """
    (person ids, in-water ids, out-of-water ids) for a detector's class list.
    No names / COCO-80: class 0 is the only person class (the historical
    `labels == 0` rule). A model with no class named like a person keeps
    class 0 as person, so an unnamed single-class export still works.
    """
    if not names:
        return [0], [], []
    low = [str(n).strip().lower() for n in names]
    water = [i for i, n in enumerate(low) if n in WATER_CLASS_NAMES]
    land = [i for i, n in enumerate(low) if n in LAND_CLASS_NAMES]
    person = [i for i, n in enumerate(low) if n in PERSON_CLASS_NAMES]
    return (person or [0]), water, land


def agnostic_nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Indices kept by class-agnostic NMS, best score first."""
    if len(boxes) == 0:
        return np.zeros(0, int)
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    return np.asarray(cv2.dnn.NMSBoxes(xywh, scores.astype(float).tolist(), 0.0, iou), int).reshape(-1)


def detr_floor() -> float:
    """[2.22.0] Lowest DETR score kept: DET_DETR_MIN_SCORE, or DET_DETR_TRACK_FLOOR when set below it. The boxes in
    between only reach ByteTrack's low-score stages (extend / re-find a track): a birth needs
    TRACK_HIGH_SCORE_THRESHOLD + TRACK_BIRTH_SCORE_MARGIN and an untracked box shows from TENTATIVE_DISPLAY_MIN_SCORE,
    both >= DET_DETR_MIN_SCORE (tests/vision/test_detr_track_floor.py holds that)."""
    if 0.0 < DET_DETR_TRACK_FLOOR < DET_DETR_MIN_SCORE:
        return float(DET_DETR_TRACK_FLOOR)
    return float(DET_DETR_MIN_SCORE)


def _person_view(
    boxes: np.ndarray,
    cls_scores: np.ndarray,
    person: Sequence[int],
    water: Sequence[int],
    land: Sequence[int],
    score_thr: float,
    nms_thr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse per-class person scores into ONE box per person: score = best
    person-class score, water = P(in water | person) from the in/out class
    pair (NaN when the model has no such pair). Class-agnostic NMS — per-class
    NMS would keep an in-water AND an out-of-water box for the same swimmer.
    """
    ps = cls_scores[:, list(person)]
    score = ps.max(1)
    if water and land:
        pw = cls_scores[:, list(water)].max(1)
        pl = cls_scores[:, list(land)].max(1)
        wprob = pw / np.maximum(pw + pl, 1e-6)
    else:
        wprob = np.full(len(score), np.nan, np.float32)
    keep = score >= score_thr
    boxes, score, wprob = boxes[keep], score[keep], wprob[keep]
    k = agnostic_nms(boxes, score, nms_thr)
    return boxes[k].astype(np.float32), score[k].astype(np.float32), wprob[k].astype(np.float32)


def _drop_nested(boxes: np.ndarray, scores: np.ndarray, min_ioa: float) -> np.ndarray:
    """[v7] DETR only: indices kept after dropping every box lying >= min_ioa (of its own area) inside a
    higher-scoring kept box - a second query on the same person. Best score first, like NMS."""
    order = np.argsort(-scores)
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9)
    kept: list[int] = []
    for i in order.tolist():
        b = boxes[i]
        inside = False
        for k in kept:
            ix = min(b[2], boxes[k][2]) - max(b[0], boxes[k][0])
            iy = min(b[3], boxes[k][3]) - max(b[1], boxes[k][1])
            if ix > 0 and iy > 0 and ix * iy / area[i] >= min_ioa:
                inside = True
                break
        if not inside:
            kept.append(i)
    return np.asarray(sorted(kept), int)


def run_detector_people(
    model: Any, frame: np.ndarray, score_thr: float, nms_thr: float = 0.45, names: Sequence[str] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PERSON detections only: (boxes[N,4] frame px, scores[N], water_prob[N]).
    water_prob is the detector's in-water vote in [0,1], NaN for models that
    do not predict the in/out-of-water classes (every pre-v2 model).
    """
    person, water, land = person_class_ids(names)
    zone_aware = bool(water and land)
    if isinstance(model, DetModel) and model.detr:
        outputs, _ = model.run(frame)
        logits, boxes = np.asarray(outputs[0])[0], np.asarray(outputs[1])[0]
        cls = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
        ih, iw = frame.shape[:2]
        cx, cy, bw, bh = boxes[:, 0] * iw, boxes[:, 1] * ih, boxes[:, 2] * iw, boxes[:, 3] * ih
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1).astype(np.float32)
        # DETR queries are near-duplicate-free by construction; the light NMS only
        # merges the rare same-person pair, so it uses a looser IoU than CNN heads.
        # [v7] The fine-tuned D-FINE is not: ~25 boxes per frame on dev clips, mostly stacked on the same person.
        # DET_DETR_NMS_IOU / DET_DETR_MIN_SCORE / DET_DETR_NESTED_IOA tighten that (defaults: the old 0.7 / none / off).
        b, s, w = _person_view(xyxy, cls, person, water, land, max(score_thr, detr_floor()), DET_DETR_NMS_IOU)
        if DET_DETR_NESTED_IOA > 0 and len(b) > 1:
            keep = _drop_nested(b, s, DET_DETR_NESTED_IOA)
            b, s, w = b[keep], s[keep], w[keep]
        return b, s, w
    if zone_aware and isinstance(model, DetModel) and model.fmt == FORMAT_XYXY_SCORES:
        outputs, ratio = model.run(frame)
        first = np.asarray(outputs[0])
        first = first[None] if first.ndim == 2 else first
        return _person_view(
            first[0, :, :4].astype(np.float32) / float(ratio or 1.0),
            first[0, :, 4:].astype(np.float32),
            person,
            water,
            land,
            score_thr,
            nms_thr,
        )
    boxes, scores, labels = run_detector(model, frame, score_thr, nms_thr, classes=person)
    keep = np.isin(labels, person)
    w = np.full(int(keep.sum()), np.nan, np.float32)
    if zone_aware:  # hard labels from an end2end export
        lab = labels[keep]
        w = np.where(np.isin(lab, water), 1.0, np.where(np.isin(lab, land), 0.0, np.nan)).astype(np.float32)
    return boxes[keep], scores[keep], w


def decode_detections(
    outputs: Sequence[np.ndarray],
    ratio: float,
    input_hw: tuple[int, int],
    score_thr: float,
    nms_thr: float = 0.45,
    fmt: str = "",
    classes: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Three output layouts:
      (a) OpenMMLab onnx_sdk / mmdeploy end2end: `dets (1,N,5)` [+ `labels (1,N)`], NMS baked in;
      (b) detection/trainer.py exports: decoded `(1,N,4+C)` xyxy + per-class scores (metadata
          lifeguard.det_format=xyxy_scores), NMS done here;
      (c) raw YOLOX-style `(1,N,5+C)` grid outputs, decoded + NMS here.
    """
    empty = (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64))
    if not outputs:
        return empty
    first = np.asarray(outputs[0])
    if first.ndim == 2:
        first = first[None]
    if first.ndim != 3 or first.shape[0] < 1:
        raise ValueError(f"unexpected detector output shape {first.shape}")
    last = first.shape[-1]
    ratio = float(ratio) if ratio else 1.0

    if last == 5 and fmt != FORMAT_XYXY_SCORES:
        boxes = first[0, :, :4].astype(np.float32) / ratio
        scores = first[0, :, 4].astype(np.float32)
        labels = np.zeros(len(scores), np.int64)
        if len(outputs) > 1:
            lab = np.asarray(outputs[1]).reshape(-1)
            if lab.shape[0] == scores.shape[0]:
                labels = lab.astype(np.int64)
        keep = scores >= score_thr
        return boxes[keep], scores[keep], labels[keep]

    if fmt == FORMAT_XYXY_SCORES:
        boxes = first[0, :, :4].astype(np.float32) / ratio
        cls_scores = first[0, :, 4:].astype(np.float32)
    elif last > 5:
        pred = first[0].astype(np.float32).copy()
        grids, strides = [], []
        for stride in (8, 16, 32):
            hs, ws = input_hw[0] // stride, input_hw[1] // stride
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((hs * ws, 1), stride))
        grid = np.concatenate(grids, 0)
        stride_arr = np.concatenate(strides, 0)
        if grid.shape[0] != pred.shape[0]:
            raise ValueError(
                f"raw detector output has {pred.shape[0]} anchors, expected {grid.shape[0]} for input {input_hw}"
            )
        pred[:, :2] = (pred[:, :2] + grid) * stride_arr
        pred[:, 2:4] = np.exp(pred[:, 2:4]) * stride_arr
        boxes = np.empty((pred.shape[0], 4), np.float32)
        boxes[:, 0] = pred[:, 0] - pred[:, 2] / 2
        boxes[:, 1] = pred[:, 1] - pred[:, 3] / 2
        boxes[:, 2] = pred[:, 0] + pred[:, 2] / 2
        boxes[:, 3] = pred[:, 1] + pred[:, 3] / 2
        boxes /= ratio
        cls_scores = pred[:, 4:5] * pred[:, 5:]
    else:
        raise ValueError(f"unexpected detector output shape {first.shape}")

    dets, _keep = multiclass_nms(boxes, cls_scores, nms_thr=nms_thr, score_thr=score_thr, classes=classes)
    if dets is None or len(dets) == 0:
        return empty
    return dets[:, :4].astype(np.float32), dets[:, 4].astype(np.float32), dets[:, 5].astype(np.int64)


def run_detector(
    model: Any, frame: np.ndarray, score_thr: float, nms_thr: float = 0.45, classes: Sequence[int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect on a BGR frame. Returns (boxes[N,4] xyxy in frame pixels,
    scores[N], labels[N]) — scores are preserved, unlike RTMDet.__call__.
    """
    outputs, ratio = model.run(frame)
    fmt = model.fmt
    return decode_detections(outputs, ratio, tuple(model.model_input_size), score_thr, nms_thr, fmt, classes)
