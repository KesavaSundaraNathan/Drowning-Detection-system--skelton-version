"""
[v3] Ported verbatim from the v2 baseline detection/bytetrack/__init__.py (imports only).

detection/bytetrack — in-repo ByteTrack multi-object tracker.

Algorithm: ByteTrack (Zhang et al., ECCV 2022; reference implementation
github.com/ifzhang/ByteTrack, MIT licence). Re-implemented here so the live
path no longer depends on `supervision`'s ByteTrack, which upstream is
deprecating (it was pinned <0.31). Behaviour intentionally matches the
supervision version the project used (same thresholds, activation and
lost-buffer semantics) so existing tuning and tests carry over.

MIT licence notice for the algorithm/reference design:
Copyright (c) 2021 Yifu Zhang — see THIRD_PARTY_LICENSES.md.
"""

from lifeguard.vision.tracking.bytetrack.tracker import ByteTracker, STrack, TrackState

__all__ = ["ByteTracker", "STrack", "TrackState"]
