"""
[v3] Ported verbatim from the v2 baseline detection/bytetrack/kalman.py (imports only).

Constant-velocity Kalman filter in (cx, cy, aspect, height) space — the
standard SORT/DeepSORT/ByteTrack formulation (noise scaled by box height).
"""

from __future__ import annotations

import numpy as np

_STD_POS = 1.0 / 20
_STD_VEL = 1.0 / 160


class KalmanFilterXYAH:
    ndim = 4

    def __init__(self, std_position: float = _STD_POS, std_velocity: float = _STD_VEL) -> None:
        # [v6] noise weights (relative to box height) are configurable; the defaults are the ByteTrack values.
        # std_position is both the process noise of position and the measurement noise, as in ByteTrack.
        self.std_position = float(std_position)
        self.std_velocity = float(std_velocity)
        self._motion = np.eye(8)
        for i in range(4):
            self._motion[i, 4 + i] = 1.0
        self._update = np.eye(4, 8)

    def initiate(self, xyah: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[xyah, np.zeros(4)]
        h = xyah[3]
        std = [
            2 * self.std_position * h,
            2 * self.std_position * h,
            1e-2,
            2 * self.std_position * h,
            10 * self.std_velocity * h,
            10 * self.std_velocity * h,
            1e-5,
            10 * self.std_velocity * h,
        ]
        return mean, np.diag(np.square(std))

    def multi_predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised predict for N tracks: mean (N,8), cov (N,8,8)."""
        h = mean[:, 3]
        sp, sv = self.std_position, self.std_velocity
        std_pos = np.stack([sp * h, sp * h, np.full_like(h, 1e-2), sp * h], 1)
        std_vel = np.stack([sv * h, sv * h, np.full_like(h, 1e-5), sv * h], 1)
        sqr = np.square(np.concatenate([std_pos, std_vel], 1))
        motion_cov = np.zeros((len(mean), 8, 8))
        idx = np.arange(8)
        motion_cov[:, idx, idx] = sqr
        mean = mean @ self._motion.T
        cov = self._motion @ cov @ self._motion.T + motion_cov
        return mean, cov

    def update(self, mean: np.ndarray, cov: np.ndarray, xyah: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self.std_position * h, self.std_position * h, 1e-1, self.std_position * h]
        proj_mean = self._update @ mean
        proj_cov = self._update @ cov @ self._update.T + np.diag(np.square(std))
        # K = P Hᵀ S⁻¹ via a linear solve (S is symmetric positive definite)
        gain = np.linalg.solve(proj_cov, (cov @ self._update.T).T).T
        mean = mean + gain @ (xyah - proj_mean)
        cov = cov - gain @ proj_cov @ gain.T
        return mean, cov
