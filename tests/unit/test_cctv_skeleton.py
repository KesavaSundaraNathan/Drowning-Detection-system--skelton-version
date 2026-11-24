"""[rtmpose clone] Standard CCTV cameras run pose for the skeleton overlay only: shown, never scored or decisive."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lifeguard.vision.inference import detector as D


def _stub(surveillance: bool) -> SimpleNamespace:
    boxes_posed = np.array([0])
    return SimpleNamespace(
        _pose_model=object(),
        surveillance=surveillance,
        pose_infer=None,
        pose_plan=lambda boxes, ids: (boxes_posed, np.zeros(len(boxes), bool)),
    )


def _run(stub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> tuple[np.ndarray, ...]:
    monkeypatch.setattr(
        D.pose_stage,
        "run_pose",
        lambda model, frame, boxes, infer: (np.ones((len(boxes), 17, 2)), np.ones((len(boxes), 17))),
    )
    boxes = np.array([[10.0, 10.0, 60.0, 120.0]])
    return D.MultiPersonDetector.stage_pose(stub, np.zeros((200, 200, 3), np.uint8), boxes, np.array([1]))  # type: ignore[arg-type]


def test_standard_cctv_gets_keypoints_for_the_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D, "CCTV_SKELETON", True)
    idx, kpts, conf, _ = _run(_stub(surveillance=True), monkeypatch)
    assert idx.tolist() == [0] and kpts.shape == (1, 17, 2) and conf.shape == (1, 17)


def test_the_original_behaviour_is_one_switch_away(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D, "CCTV_SKELETON", False)  # LIFEGUARD_CCTV_SKELETON=0
    idx, kpts, _, _ = _run(_stub(surveillance=True), monkeypatch)
    assert len(idx) == 0 and len(kpts) == 0


def test_pool_cameras_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    for flag in (True, False):
        monkeypatch.setattr(D, "CCTV_SKELETON", flag)
        idx, kpts, _, _ = _run(_stub(surveillance=False), monkeypatch)
        assert idx.tolist() == [0] and kpts.shape == (1, 17, 2)
