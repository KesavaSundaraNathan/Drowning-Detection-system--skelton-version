"""[rtmpose clone] Skeleton overlay: drawn from the keypoints the pipeline already has, display only."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from lifeguard.live import skeleton as skel
from lifeguard.live.view import annotate, boxes_json, render, render_client


def _kp(x0: float, y0: float, conf: float = 0.9) -> list[list[float]]:
    return [[x0 + 2 * j, y0 + 5 * j, conf] for j in range(17)]


def _track(tid: int, bbox: tuple[int, int, int, int], kp: list[list[float]] | None) -> SimpleNamespace:
    return SimpleNamespace(track_id=tid, bbox=bbox, keypoints=kp, tentative=tid < 0)


def _dets(*tracks: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(tracks=list(tracks), coasting=[])


def test_skeleton_pixels_drawn_only_when_enabled() -> None:
    img = np.zeros((200, 200, 3), np.uint8)
    d = _dets(_track(1, (10, 10, 60, 120), _kp(20, 20)))
    plain = annotate(img, d, 200)
    poses = skel.MEMORY.poses("cam-a", d)
    drawn = annotate(img, d, 200, poses=poses)
    assert len(poses) == 1
    assert (drawn != plain).any()
    assert render((img, d), 200, "cam-a", True) != render((img, d), 200, "cam-a", False)


def test_low_confidence_joints_are_not_drawn() -> None:
    img = np.zeros((200, 200, 3), np.uint8)
    d = _dets(_track(2, (10, 10, 60, 120), _kp(20, 20, conf=0.1)))
    plain = annotate(img, d, 200)
    assert (annotate(img, d, 200, poses=skel.MEMORY.poses("cam-b", d)) == plain).all()


def test_pose_skipped_frame_reuses_last_pose_moved_with_box() -> None:
    first = _dets(_track(3, (0, 0, 50, 100), _kp(10, 10)))
    skel.MEMORY.poses("cam-c", first)
    skipped = _dets(_track(3, (100, 0, 150, 100), None))  # POSE_STRIDE: keypoints None, box moved +100 px
    ((_, moved),) = skel.MEMORY.poses("cam-c", skipped)
    assert moved[0, 0] == 110 and moved[0, 1] == 10


def test_memory_is_per_camera_and_expires(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    skel.MEMORY.poses("cam-d", _dets(_track(4, (0, 0, 50, 100), _kp(10, 10))))
    assert skel.MEMORY.poses("cam-e", _dets(_track(4, (0, 0, 50, 100), None))) == []
    t0 = skel.time.monotonic()
    monkeypatch.setattr(skel.time, "monotonic", lambda: t0 + skel.POSE_MEMORY_S + 0.5)
    assert skel.MEMORY.poses("cam-d", _dets(_track(4, (0, 0, 50, 100), None))) == []


def test_boxes_json_carries_scaled_keypoints_for_the_browser() -> None:
    img = np.zeros((100, 400, 3), np.uint8)
    d = _dets(_track(5, (0, 0, 100, 100), _kp(40, 20)))
    out = render_client((img, d), 200, "cam-f", True)
    assert out is not None
    doc = json.loads(out[1])
    (row,) = doc["k"]
    assert row[0] == 5 and len(row) == 1 + 17 * 3
    assert row[1:4] == [20, 10, 90]  # x, y halved (400 -> 200 px), conf 0..100
    off = render_client((img, d), 200, "cam-f", False)
    assert off is not None and "k" not in json.loads(off[1])
    assert "k" not in json.loads(boxes_json(d, frozenset(), 1.0, (400, 100)))
