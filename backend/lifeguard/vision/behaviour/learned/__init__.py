"""
[v3] Ported verbatim from the v2 baseline detection/behavior_model/__init__.py (imports only).

detection/behavior_model — the LEARNED behaviour classifier (ML core v2).

Trained on the client's labelled videos (training/behavior, torch — never
imported here), shipped as ONNX + a JSON sidecar, run with onnxruntime in the
live path. It classifies each in-water track's last few seconds as

    NORMAL | DISTRESS | ACTIVE_DROWNING | PASSIVE_DROWNING

and is fused with the rule engine in detection/classifier.py (the rules stay
as a safety net; SUBMERSION — a vanished swimmer — stays rule-based).

    features.py  window features from a TrackBuffer (shared by training + live)
    infer.py     BehaviorModel: load ONNX + sidecar, predict class probabilities
"""

from lifeguard.vision.behaviour.learned.features import CLASSES, SEQ_CHANNELS, SEQ_LEN, SUMMARY_FIELDS, window_features
from lifeguard.vision.behaviour.learned.infer import BehaviorModel, get_behavior_model

__all__ = [
    "CLASSES",
    "SEQ_CHANNELS",
    "SEQ_LEN",
    "SUMMARY_FIELDS",
    "BehaviorModel",
    "get_behavior_model",
    "window_features",
]
