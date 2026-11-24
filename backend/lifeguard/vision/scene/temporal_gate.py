"""[2.20.0] Per-track temporal risk gate: an alert needs sustained evidence in the track's own recent history.

Decoupled from detection, tracking and the behaviour rules: the scene tracker feeds it what it saw (which track
ids were visible on each analysed frame, where, and whether each scoring round was positive) and asks it, at the
moment an alert would rise, whether the track's history over the last `window_s` seconds supports it.

    gate = TemporalRiskGate(TemporalGateConfig.from_constants())
    gate.observe_frame(t, {track_id: bbox, ...})          # every analysed frame
    gate.observe_score(track_id, t, positive)              # every scoring round of a visible track
    ok, feats = gate.allow(track_id, t, ghost=False)       # at a rising edge

Safety: the gate DEFERS a live track's alert, it never cancels it. A track whose evidence keeps coming is asked
again on every scoring round and fires as soon as its window qualifies. For a vanished track (submersion ghost)
the history is frozen at loss, so its answer is final. Nothing here looks at box size or pose, so it cannot
single out small or pose-less victims; what it measures is how long and how steadily the track was seen and how
consistently it scored positive. Off by default (TEMPORAL_GATE false): the features are always computed and
attached to the alert (`AlertFire.gate`) so a candidate setting can be judged on recorded runs first.
"""

# Phase 4 Step 1: the code moved to behaviour/temporal.py, verbatim; this module re-exports it.
from lifeguard.vision.behaviour.temporal import (
    Box,
    GateFeatures,
    TemporalGateConfig,
    TemporalRiskGate,
)

__all__ = ["Box", "GateFeatures", "TemporalGateConfig", "TemporalRiskGate"]
