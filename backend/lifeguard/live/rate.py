"""Dynamic live analysis rate (owner request 2026-11-18): every RTSP camera is analysed between a floor and a ceiling.

``settings.target_analysis_fps`` (15) stays the FLOOR: admission, the capacity model and CAMERA_LAGGING all count it,
so nothing about what the server promises changes. ``settings.max_analysis_fps`` (20) is the CEILING. The monitor
(live/runtime.py, every 0.5 s) feeds this controller the worst camera's queue lag and the GPU busy fraction:

* step DOWN by STEP_DOWN_FPS at once when any camera's oldest queued frame is older than LAG_DOWN_S, or the GPU is
  busier than GPU_HIGH. LAG_DOWN_S (0.4 s) is well below lag_threshold_s (1 s, the LAGGING chip) and far below the
  10 s of lag that raises CAMERA_LAGGING, so the rate backs off long before either fires;
* step UP by STEP_UP_FPS only after UP_CALM_S of calm (every camera's lag <= LAG_UP_S and the GPU below GPU_LOW), and
  never within HOLD_AFTER_DOWN_S of a step down (no oscillation);
* admitting a camera resets the rate to the floor (the capacity model was computed at the floor).

One rate for all cameras: the GPU is shared, and a camera never gets fewer frames than another one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STEP_UP_FPS = 1.0
STEP_DOWN_FPS = 2.0
LAG_DOWN_S = 0.4  # oldest queued frame older than this on ANY camera: step down
LAG_UP_S = 0.2  # every camera at or under this for UP_CALM_S: may step up
GPU_HIGH = 0.70  # detector + pose session time per wall second above this: step down
GPU_LOW = 0.50  # ... below this (and calm): may step up - the ceiling is only used while the GPU is lightly loaded
UP_CALM_S = 5.0
HOLD_AFTER_DOWN_S = 15.0


@dataclass
class RateController:
    floor: float
    ceiling: float
    rate: float = field(init=False)
    steps_up: int = field(default=0, init=False)
    steps_down: int = field(default=0, init=False)
    _calm_since: float | None = field(default=None, init=False)
    _last_down: float = field(default=float("-inf"), init=False)

    def __post_init__(self) -> None:
        self.ceiling = max(float(self.floor), float(self.ceiling))
        self.rate = float(self.floor)

    @property
    def enabled(self) -> bool:
        return self.ceiling > self.floor

    def reset(self, now: float) -> float:
        """Back to the floor (a camera was admitted), held there for HOLD_AFTER_DOWN_S."""
        self.rate = float(self.floor)
        self._calm_since = None
        self._last_down = now
        return self.rate

    def update(self, now: float, worst_lag_s: float, gpu_busy: float | None) -> float | None:
        """One monitor pass. Returns the new rate when it changed, else None."""
        if not self.enabled:
            return None
        hot = worst_lag_s > LAG_DOWN_S or (gpu_busy is not None and gpu_busy > GPU_HIGH)
        if hot:
            self._calm_since = None
            if self.rate > self.floor:
                self.rate = max(self.floor, self.rate - STEP_DOWN_FPS)
                self._last_down = now
                self.steps_down += 1
                return self.rate
            return None
        calm = worst_lag_s <= LAG_UP_S and (gpu_busy is None or gpu_busy < GPU_LOW)
        if not calm:
            self._calm_since = None
            return None
        if self._calm_since is None:
            self._calm_since = now
        if (
            self.rate < self.ceiling
            and now - self._calm_since >= UP_CALM_S
            and now - self._last_down >= HOLD_AFTER_DOWN_S
        ):
            self.rate = min(self.ceiling, self.rate + STEP_UP_FPS)
            self._calm_since = now
            self.steps_up += 1
            return self.rate
        return None

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "enabled": self.enabled,
            "rate_fps": self.rate,
            "floor_fps": self.floor,
            "ceiling_fps": self.ceiling,
            "steps_up": self.steps_up,
            "steps_down": self.steps_down,
        }
