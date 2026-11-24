"""Top-down RTMPose on the person boxes RTMDet found and ByteTrack tracked: 17 COCO keypoints in frame pixels.

    x, centers, scales = prepare(model, frame, boxes)   # crop + zoom each box to 192x256 (camera thread)
    simcc_x, simcc_y = infer(model, x)                   # one batched call (live: shared by every camera)
    keypoints, scores = decode(model, simcc_x, simcc_y, centers, scales)

Top-down is deliberate: every box, however small (a swimmer 25 m away), is cropped and enlarged to the pose
model's full input, which a one-stage model working on the whole frame cannot do. The detector stays the
authority on who exists: pose only adds keypoints to boxes it was given, it never adds or drops a person.

[v3] Ported from the v2 baseline detection/rtm_backend.py (_crop_params, run_pose_multi, run_pose) and
detector.py (pose_order); the batched call is split out so live/pose_batcher.py can gather crops of all cameras.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np

from lifeguard.vision.inference.models import PoseModel

SIMCC_SPLIT_RATIO = 2.0
POSE_BBOX_PADDING = 1.25
NUM_KEYPOINTS = 17

# (model, crops (N,3,H,W)) -> (simcc_x (N,K,Wx), simcc_y (N,K,Hy))
InferFn = Callable[[PoseModel, np.ndarray], tuple[np.ndarray, np.ndarray]]


def crop_params(boxes: np.ndarray, input_wh: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """rtmlib's bbox_xyxy2cs(padding=1.25) + aspect fix -> (centers[N,2], scales[N,2])."""
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    center = np.stack([(b[:, 0] + b[:, 2]) * 0.5, (b[:, 1] + b[:, 3]) * 0.5], 1)
    scale = np.stack([(b[:, 2] - b[:, 0]), (b[:, 3] - b[:, 1])], 1) * POSE_BBOX_PADDING
    w, h = input_wh
    aspect = w / h
    bw, bh = scale[:, 0:1], scale[:, 1:2]
    scale = np.where(bw > bh * aspect, np.hstack([bw, bw / aspect]), np.hstack([bh * aspect, bh]))
    return center, np.maximum(scale, 1e-3)


def pose_order(boxes: np.ndarray, max_people: int) -> np.ndarray:
    """Indices of the boxes that get a pose crop (largest first, capped), in box order."""
    n = len(boxes)
    if n <= max_people:
        return np.arange(n)
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return np.sort(np.argsort(-area, kind="stable")[:max_people])


def _empty(k: int = NUM_KEYPOINTS) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((0, k, 2), np.float32), np.zeros((0, k), np.float32)


def prepare(model: PoseModel, frame: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Affine crop of every box to the model input, normalised -> (crops (N,3,H,W), centers, scales)."""
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    w, h = model.model_input_size
    x = np.empty((len(b), 3, h, w), np.float32)
    if len(b) == 0:
        return x, np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    centers, scales = crop_params(b, model.model_input_size)
    _x, crop = model.bufs()
    order = (2, 1, 0) if model.rgb else (0, 1, 2)
    for r in range(len(b)):
        c, s = centers[r], scales[r]
        sc = w / s[0]
        m = np.array([[sc, 0.0, w * 0.5 - c[0] * sc], [0.0, sc, h * 0.5 - c[1] * sc]], np.float32)
        cv2.warpAffine(
            frame, m, (w, h), dst=crop, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        for ch in range(3):
            np.subtract(crop[:, :, order[ch]], model.mean[ch], out=x[r, ch], dtype=np.float32)
            x[r, ch] *= model._inv_std[ch]
    return x, centers, scales


def infer(model: PoseModel, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SimCC outputs for every crop: chunks of at most `max_batch`, each zero-padded to a batch bucket so the
    engine only ever sees a handful of shapes (TensorRT / cuDNN never re-tune mid-stream)."""
    n_all = len(x)
    if n_all == 0:
        return np.zeros((0, NUM_KEYPOINTS, 0), np.float32), np.zeros((0, NUM_KEYPOINTS, 0), np.float32)
    w, h = model.model_input_size
    outs_x: list[np.ndarray] = []
    outs_y: list[np.ndarray] = []
    for start in range(0, n_all, model.max_batch):
        chunk = x[start : start + model.max_batch]
        n = len(chunk)
        bsz = model.bucket_for(n)
        if bsz > n:
            chunk = np.concatenate([chunk, np.zeros((bsz - n, 3, h, w), np.float32)])
        model.calls += 1
        sx, sy = model.session.run(model.output_names, {model.input_name: np.ascontiguousarray(chunk)})[:2]
        outs_x.append(np.asarray(sx[:n]))
        outs_y.append(np.asarray(sy[:n]))
    return np.concatenate(outs_x), np.concatenate(outs_y)


def infer_parts(
    model: PoseModel, parts: list[np.ndarray], buffers: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """[v7] infer() over several requests' crops without concatenating them first: each chunk is copied ONCE into a
    reusable bucket-sized buffer (`buffers`, owned by the single caller thread), pad rows zeroed in place - the same
    input tensor infer() builds with two full copies (concatenate + pad). Outputs are identical."""
    n_all = sum(len(p) for p in parts)
    if n_all == 0:
        return np.zeros((0, NUM_KEYPOINTS, 0), np.float32), np.zeros((0, NUM_KEYPOINTS, 0), np.float32)
    w, h = model.model_input_size
    outs_x: list[np.ndarray] = []
    outs_y: list[np.ndarray] = []
    queue = [(p, 0) for p in parts if len(p)]
    while queue:
        bsz_cap = model.max_batch
        take: list[tuple[np.ndarray, int, int]] = []  # (part, start, stop)
        n = 0
        while queue and n < bsz_cap:
            p, start = queue[0]
            k = min(len(p) - start, bsz_cap - n)
            take.append((p, start, start + k))
            n += k
            if start + k >= len(p):
                queue.pop(0)
            else:
                queue[0] = (p, start + k)
        bsz = model.bucket_for(n)
        buf = buffers.get(bsz)
        if buf is None:
            buf = buffers[bsz] = np.zeros((bsz, 3, h, w), np.float32)
        pos = 0
        for p, a, b in take:
            buf[pos : pos + (b - a)] = p[a:b]
            pos += b - a
        if pos < bsz:
            buf[pos:] = 0.0
        model.calls += 1
        sx, sy = model.session.run(model.output_names, {model.input_name: buf})[:2]
        outs_x.append(np.asarray(sx[:n]))
        outs_y.append(np.asarray(sy[:n]))
    return np.concatenate(outs_x), np.concatenate(outs_y)


def decode(
    model: PoseModel, sx: np.ndarray, sy: np.ndarray, centers: np.ndarray, scales: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """SimCC argmax -> (keypoints (N,K,2) in frame pixels, scores (N,K) in [0, 1])."""
    if len(sx) == 0:
        return _empty()
    w, h = model.model_input_size
    xl = np.argmax(sx, axis=2).astype(np.float32)
    yl = np.argmax(sy, axis=2).astype(np.float32)
    vals = 0.5 * (np.max(sx, axis=2) + np.max(sy, axis=2))
    locs = np.stack([xl, yl], -1)
    locs[vals <= 0] = -1
    cs = np.asarray(centers, np.float32)[:, None, :]
    ss = np.asarray(scales, np.float32)[:, None, :]
    k = locs / SIMCC_SPLIT_RATIO / np.array([w, h], np.float32) * ss + cs - ss / 2
    return k.astype(np.float32), np.clip(vals, 0.0, 1.0).astype(np.float32)


def run_pose(
    model: PoseModel, frame: np.ndarray, boxes: np.ndarray, infer_fn: InferFn | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """RTMPose on each box of one frame. An empty box list returns empty arrays (rtmlib would pose the WHOLE
    image and invent a skeleton for an empty pool)."""
    if boxes is None or len(boxes) == 0:
        return _empty()
    x, centers, scales = prepare(model, frame, boxes)
    sx, sy = (infer_fn or infer)(model, x)
    return decode(model, sx, sy, centers, scales)


def run_pose_multi(
    model: PoseModel, items: Sequence[tuple[np.ndarray, np.ndarray]]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """RTMPose on every box of every (frame, boxes) item in ONE batched inference."""
    prepared = [prepare(model, frame, boxes) for frame, boxes in items]
    if not any(len(p[0]) for p in prepared):
        return [_empty() for _ in items]
    sx, sy = infer(model, np.concatenate([p[0] for p in prepared]))
    out: list[tuple[np.ndarray, np.ndarray]] = []
    pos = 0
    for x, centers, scales in prepared:
        n = len(x)
        out.append(decode(model, sx[pos : pos + n], sy[pos : pos + n], centers, scales) if n else _empty())
        pos += n
    return out
