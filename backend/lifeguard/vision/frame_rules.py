"""[v7] Frame-count rules at any analysed frame rate (plan v6 Phase 2).

Two rules count CAMERA frames: POSE_STABLE_MIN_FRAMES (a track may skip pose only after that many frames) and
FALL_IN_LAST_FRAMES (the no-zone fall-in check looks at the last N land observations, one per frame). Both were tuned at
FRAME_RULES_REFERENCE_FPS (15). With FRAME_RULES_SCALE_TO_FPS on, `frames_at` rescales them to the analysed fps so they
keep the same duration in seconds at 15, 18 or 30 fps; at the reference fps the result is the value unchanged.

Not frame rules: BEHAVIOR_BUFFER_FRAMES / VELOCITY_WINDOW_FRAMES count behaviour samples taken at a fixed
BEHAVIOR_SAMPLE_HZ, whatever the camera fps.
"""

from __future__ import annotations

from lifeguard.vision import constants as C


def frames_at(frames_at_reference: int, fps: float | None) -> int:
    """`frames_at_reference` (tuned at FRAME_RULES_REFERENCE_FPS) as a frame count at `fps`; never below 1."""
    if not C.FRAME_RULES_SCALE_TO_FPS or not fps or fps <= 0 or C.FRAME_RULES_REFERENCE_FPS <= 0:
        return int(frames_at_reference)
    return max(1, round(frames_at_reference * fps / C.FRAME_RULES_REFERENCE_FPS))
