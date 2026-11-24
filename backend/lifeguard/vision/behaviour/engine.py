"""detection/risk_engine.py — Swimmer kinematics and scenario evidence (Phase 4).

REWRITE
───────
The previous engine added "risk points" for a tall bounding box, erratic motion
and inactivity, which is why lap swimmers were flagged as Active Drowning. This
module replaces it with pose kinematics computed over a rolling per-track
buffer, and turns them into *evidence* (0–1) for each scenario:

    N  Normal Swimming      travel speed high, torso horizontal (< 45°),
                            periodic (harmonic) stroke oscillation
    A  Active Drowning      torso vertical (> 60°), frantic vertical arm
                            motion, negligible forward displacement, head
                            bobbing around the surface
    P  Passive Drowning     negligible velocity for > passive_seconds (8 s)
                            on the surface, or face-down / inverted and still
    C  Floatable Failure    decoupling from a detected floatable, sudden
                            displacement or abrupt posture failure (weighted
                            by child likelihood) followed by distress
    D  Blackout/Submersion  pose confidence collapses inside the water, away
                            from the pool edge, for > submersion_seconds (5 s)
                            (the "track vanished" variant lives in
                            classifier.SubmersionMonitor)

Units
─────
Distances are divided by a robust per-track body scale (≈ torso length, from
shoulders / hips / ears, median over the window), so every speed is in
"scale units per second" (su/s) and thresholds work at any resolution or
distance from the camera. Time comes from sample timestamps (camera clock for
live feeds, frame_index / fps for uploaded video) — never from sample counts.

Reliability (Phase 2)
─────────────────────
Every sample carries `pose_rel` (0–1: upper-body keypoint confidence × box
size). Keypoint-derived quantities — anchor displacement, limb and wrist
motion, torso angle, face / shoulder visibility, head bob — only use samples
whose pose is reliable; otherwise the box / track kinematics stand in
(box-centre travel, box-top bob, box aspect, box height vs the track's own
baseline). `RiskEngine.evidence` then requires several independent cues to
agree before any drowning evidence reaches a threshold, and recognises
normal activity (swimming, treading, standing in the water, resting at the
edge) positively.

`TrackBuffer` is the rolling deque of the last BEHAVIOR_BUFFER_FRAMES samples
for ONE track_id plus the long-running timers (stillness, face-down, pose-loss,
floatable coupling) that must outlive the 60-sample window.

[v3] Split into behaviour/numeric.py, samples.py, features.py and evidence.py; this module
assembles RiskEngine from them and re-exports the public names.
"""

from __future__ import annotations

import logging

from lifeguard.vision.behaviour.evidence import EvidenceScorer, ScenarioEvidence, step_evidence
from lifeguard.vision.behaviour.features import FeatureExtractor, KinematicFeatures
from lifeguard.vision.behaviour.numeric import clamp01, finite, inv_ramp, ramp
from lifeguard.vision.behaviour.samples import FrameSample, TrackBuffer
from lifeguard.vision.constants import STEP_PRE_LIMIT

logger = logging.getLogger(__name__)


class RiskEngine(FeatureExtractor, EvidenceScorer):
    """Stateless: turns a TrackBuffer into KinematicFeatures and ScenarioEvidence."""


__all__ = [
    "STEP_PRE_LIMIT",
    "EvidenceScorer",
    "FeatureExtractor",
    "FrameSample",
    "KinematicFeatures",
    "RiskEngine",
    "ScenarioEvidence",
    "TrackBuffer",
    "clamp01",
    "finite",
    "inv_ramp",
    "ramp",
    "step_evidence",
]
