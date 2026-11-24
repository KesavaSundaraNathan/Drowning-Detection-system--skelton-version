# mypy: disable-error-code="index"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""Stage 2: window kinematics (KinematicFeatures) extracted from a TrackBuffer.

[v3] Split out of the v2 baseline detection/risk_engine.py: KinematicFeatures verbatim, and
RiskEngine.extract verbatim as FeatureExtractor.extract (RiskEngine inherits it).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from lifeguard.vision.behaviour.numeric import (
    HEAD,
    L_SH,
    NAN,
    R_EAR,
    R_SH,
    _dist,
    _median,
    _moving_average,
    _nanmedian,
    _periodicity,
    _resample,
    clamp01,
    finite,
    ramp,
)
from lifeguard.vision.behaviour.samples import TrackBuffer
from lifeguard.vision.constants import (
    CHILD_RATIO_HI,
    CHILD_RATIO_LO,
    KEYPOINT_CONF_THRESHOLD,
    KINEMATICS,
    PASSIVE_NEVER_TRAVELLED_SU,
    PROGRESS_SECONDS,
    SHALLOW_CHILD_PRIOR,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Window features
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class KinematicFeatures:
    n: int = 0
    duration: float = 0.0
    rate_hz: float = 0.0
    scale_px: float = 1.0
    pose_quality: float = 0.0  # median upper-body keypoint confidence (0 for HOG)
    travel_speed: float = 0.0  # su/s, bobbing-free (median of 1-s-lag speeds)
    progress_speed: float = 0.0  # su/s, net displacement over the last PROGRESS_SECONDS
    travel_max: float = 0.0
    axis_alignment: float = NAN  # |cos| between body axis and travel direction
    uprightness: float = NAN  # median, degrees
    torso_angle: float = NAN  # |uprightness|
    # ── spec-named aliases (Phase 1.3) ───────────────────────────────────
    torso_inclination: float = NAN  # θ_torso vs the water plane, degrees (0 flat, 90 upright)
    v_horizontal: float = NAN  # |v⃗_horizontal| over the rolling 30-frame window (su/s)
    stroke_frequency: float = NAN  # f_stroke from vertical wrist/shoulder oscillation (Hz)
    zero_velocity_seconds: float = 0.0  # continuous zero-keypoint-velocity time (s)
    angle_valid: float = 0.0  # fraction of samples with a body axis
    arm_amp: float = NAN  # vertical wrist-vs-shoulder oscillation (su std)
    arm_periodicity: float = NAN
    arm_freq: float = NAN
    head_bob: float = NAN  # detrended head height std (su)
    bob_periodicity: float = NAN  # how metronomic the head bob is (breaststroke ≈ high)
    bob_freq: float = NAN
    head_flicker: float = 0.0  # head visibility transitions per second
    bbox_bob: float = NAN  # HOG fallback bob proxy
    aspect_ratio: float = 1.0
    still_seconds: float = 0.0
    moved_before_still: float = NAN  # [Phase 4 Step 3] max `moving` in rounds before the current still stretch
    still_upright_face_frac: float = 0.0
    still_ultra_frac: float = 0.0
    face_down_seconds: float = 0.0
    inverted_seconds: float = 0.0
    pose_lost_seconds: float = 0.0
    pose_motion: float = NAN
    child_likelihood: float = 0.0
    decoupled_ago: float = NAN
    spike_ago: float = NAN
    spike_ratio: float = 0.0
    flip_ago: float = NAN
    flip_deg: float = 0.0
    flip_to: float = NAN
    floatable_seen: bool = False
    floatable_iou: float = 0.0  # best person↔floatable bbox IoU seen on this track
    detach_ago: float = NAN  # seconds since an abrupt >2.5 body-width separation
    detach_widths: float = 0.0  # how many body widths that separation spanned
    detach_vertical: bool = False  # posture turned vertical within the detachment window
    upper_conf_recent: float = NAN
    shoulder_rel: float = NAN  # median shoulder / head keypoint confidence (recent)
    # ── Phase 2: reliability + box-level fallbacks ───────────────────────
    pose_reliability: float = 0.0  # median pose_rel over the last 2 s
    box_aspect: float = NAN  # median box h/w over the last 2 s
    box_height_ratio: float = NAN  # recent box height / the track's own baseline height
    conf_recent: float = NAN  # median detector confidence over the last 1 s
    face_conf_recent: float = NAN  # median face keypoint confidence (reliable samples)
    since_normal: float = NAN  # seconds since a normal activity was last recognised
    last_activity: str = ""
    edge_su: float = NAN  # distance to the water edge in body-scale units (+ inside)

    @property
    def moving(self) -> float:
        """Speed that counts as real travel: capped by net progress (su/s)."""
        return float(min(self.travel_speed, 1.25 * self.progress_speed))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, str):
                out[k] = v
            elif isinstance(v, bool):
                out[k] = v
            elif isinstance(v, (int, float, np.floating)):
                out[k] = round(float(v), 3) if math.isfinite(float(v)) else None
        return out


class FeatureExtractor:
    """Stage 2 of RiskEngine."""

    # ── Stage 2: kinematics ───────────────────────────────────────────────────
    def extract(self, buf: TrackBuffer, zone_type: str | None = None) -> KinematicFeatures:
        f = KinematicFeatures()
        n = len(buf.samples)
        f.n = n
        if n == 0:
            return f
        samples = list(buf.samples)
        now = samples[-1].t
        t = np.array([s.t for s in samples], dtype=float)
        f.duration = float(t[-1] - t[0])
        f.rate_hz = (n - 1) / f.duration if f.duration > 0 else 0.0
        scale = buf.scale()
        f.scale_px = scale
        f.pose_quality = _median([s.upper_conf if finite(s.upper_conf) else 0.0 for s in samples])
        f.aspect_ratio = _median([s.aspect_ratio for s in samples])

        # ── travel (bobbing-free) ─────────────────────────────────────────────
        rec = buf.recent(3.0)
        spd = [buf.speed[i] for i in rec if finite(buf.speed[i])]
        f.travel_speed = _median(spd) if spd else 0.0
        f.travel_max = float(max(spd)) if spd else 0.0
        prog = buf.recent(PROGRESS_SECONDS)
        if len(prog) >= 3:
            span = samples[-1].t - samples[prog[0]].t
            if span >= 0.8 * PROGRESS_SECONDS:
                f.progress_speed = _dist(buf.smooth[prog[0]], buf.smooth[-1]) / span / scale
            else:
                f.progress_speed = f.travel_speed
        else:
            f.progress_speed = f.travel_speed
        if len(buf.smooth) >= 3 and rec:
            a, b = buf.smooth[rec[0]], buf.smooth[-1]
            dvec = (b[0] - a[0], b[1] - a[1])
            dn = math.hypot(*dvec)
            if dn > 0.3 * scale:
                ux, uy = dvec[0] / dn, dvec[1] / dn
                al = [
                    abs(samples[i].axis[0] * ux + samples[i].axis[1] * uy) for i in rec if samples[i].axis is not None
                ]
                if al:
                    f.axis_alignment = _median(al)

        # ── reliability + box-level features (Phase 2.1) ──────────────────────
        r2 = buf.recent(2.0)
        f.pose_reliability = _median([samples[i].pose_rel for i in r2]) if r2 else 0.0
        f.box_aspect = _median([samples[i].aspect_ratio for i in r2]) if r2 else NAN
        r1 = buf.recent(1.0)
        base_h = buf.height_baseline()
        if r1 and finite(base_h) and base_h > 0:
            f.box_height_ratio = _median([samples[i].bbox[3] - samples[i].bbox[1] for i in r1]) / base_h
        f.conf_recent = _median([samples[i].conf for i in r1]) if r1 else NAN
        f.face_conf_recent = _nanmedian([samples[i].face_conf for i in buf.recent(2.5) if samples[i].reliable])
        f.since_normal = now - buf.last_normal_t if buf.last_normal_t is not None else NAN
        f.last_activity = buf.last_activity
        edge = samples[-1].edge_dist_px
        f.edge_su = edge / scale if math.isfinite(edge) else NAN

        # ── posture (reliable samples only) ───────────────────────────────────
        f.uprightness = _nanmedian([samples[i].uprightness for i in buf.recent(2.5) if samples[i].reliable])
        f.torso_angle = abs(f.uprightness) if finite(f.uprightness) else NAN
        f.angle_valid = sum(1 for s in samples if s.reliable and finite(s.uprightness)) / n
        # θ_torso relative to the WATER PLANE. `uprightness` is signed
        # (+90 upright … 0 flat … −90 inverted); the inclination the spec's
        # rules use is its magnitude, so a face-down and a face-up swimmer are
        # both "horizontal".
        f.torso_inclination = f.torso_angle

        # |v⃗_horizontal| over the rolling 30-frame window, capped by net
        # progress so bobbing in place can never look like travel.
        vh = [buf.v_horizontal[i] for i in buf.recent(3.0) if finite(buf.v_horizontal[i])]
        f.v_horizontal = _median(vh) if vh else NAN

        # ── oscillations on a uniform grid over the last ≤ 4 s ───────────────
        rec_osc = buf.recent(4.0)
        if len(rec_osc) >= 8 and f.rate_hz > 0:
            ts = t[rec_osc]
            span = float(ts[-1] - ts[0])
            if span > 0:
                m = max(8, int(round(span * f.rate_hz)) + 1)
                grid = np.linspace(ts[0], ts[-1], m)
                rate = (m - 1) / span

                # arms: wrist height relative to the same-side shoulder
                best_amp, best_per, best_freq = NAN, NAN, NAN
                for side in (0, 1):
                    ys = np.array(
                        [
                            samples[i].wrist_rel[side][1] / scale
                            if samples[i].wrist_rel[side] is not None and samples[i].reliable
                            else NAN
                            for i in rec_osc
                        ]
                    )
                    r = _resample(ts, ys, grid)
                    if r is None:
                        continue
                    detr = r - _moving_average(r, max(3, int(round(2.5 * rate)) | 1))
                    amp = float(np.std(detr))
                    per, freq = _periodicity(detr, rate) if amp > 0.03 else (0.0, NAN)
                    if not finite(best_amp) or amp > best_amp:
                        best_amp, best_per, best_freq = amp, per, freq
                f.arm_amp, f.arm_periodicity, f.arm_freq = best_amp, best_per, best_freq
                f.stroke_frequency = best_freq  # f_stroke (spec alias)

                # head bobbing around the surface. Each head keypoint is detrended
                # on its own series and the residuals are averaged, so eyes / ears
                # flickering around the confidence threshold never look like a bob.
                win_b = max(3, int(round(1.5 * rate)) | 1)
                resid = []
                for idx in HEAD + (L_SH, R_SH):
                    ys = np.array(
                        [
                            samples[i].kp[idx, 1] / scale
                            if samples[i].reliable and samples[i].kp[idx, 2] >= KEYPOINT_CONF_THRESHOLD
                            else NAN
                            for i in rec_osc
                        ]
                    )
                    r = _resample(ts, ys, grid, min_valid=0.6)
                    if r is not None:
                        resid.append(r - _moving_average(r, win_b))
                    if idx == R_EAR and resid:
                        break  # shoulders only as a fallback
                if resid:
                    bob_sig = np.mean(np.vstack(resid), axis=0)
                    f.head_bob = float(np.std(bob_sig))
                    if f.head_bob > 0.03:
                        f.bob_periodicity, f.bob_freq = _periodicity(bob_sig, rate)

                # bbox-top oscillation (the bob cue when keypoints are unreliable),
                # in units of the BOX scale so it does not depend on keypoints at all
                bscale = max(buf.box_scale_baseline(), KINEMATICS["min_scale_px"])
                by = np.array([samples[i].bbox[1] / bscale for i in rec_osc])
                r = _resample(ts, by, grid)
                if r is not None:
                    detr = r - _moving_average(r, max(3, int(round(1.5 * rate)) | 1))
                    f.bbox_bob = float(np.std(detr))

                vis = [samples[i].head_visible for i in rec_osc if samples[i].reliable]
                f.head_flicker = sum(1 for a, b in zip(vis, vis[1:]) if a != b) / span

        # ── long-running timers ──────────────────────────────────────────────
        f.still_seconds = buf.still.duration(now)
        if PASSIVE_NEVER_TRAVELLED_SU > 0:  # [Phase 4 Step 3] off = no state kept, NaN
            f.moved_before_still = buf.travel_before_still(now, f.still_seconds, f.moving)
        if buf.streak_n > 0:
            f.still_upright_face_frac = buf.streak_upright_face / buf.streak_n
            f.still_ultra_frac = buf.streak_ultra_still / buf.streak_n
        f.face_down_seconds = buf.face_down.duration(now)
        f.inverted_seconds = buf.inverted.duration(now)
        f.pose_lost_seconds = buf.pose_lost.duration(now)
        f.zero_velocity_seconds = buf.zero_velocity.duration(now)
        f.pose_motion = _nanmedian([buf.pose_motion[i] for i in buf.recent(1.5)])
        f.upper_conf_recent = _nanmedian([samples[i].upper_conf for i in buf.recent(1.5)])
        f.shoulder_rel = _nanmedian([samples[i].shoulder_rel for i in buf.recent(2.5) if samples[i].reliable])

        # ── child likelihood: head-to-shoulder ratio + SHALLOW prior ─────────
        # head-to-shoulder proportions only mean something on a reliable pose
        child = ramp(_nanmedian([s.child_ratio for s in samples if s.reliable]), CHILD_RATIO_LO, CHILD_RATIO_HI)
        if (zone_type or "").upper() == "SHALLOW":
            child = clamp01(child + SHALLOW_CHILD_PRIOR)
        f.child_likelihood = child

        # ── abrupt events ────────────────────────────────────────────────────
        f.floatable_seen = buf.floatable_seen
        f.floatable_iou = buf.floatable_iou_max
        f.detach_ago = now - buf.detach_t if buf.detach_t is not None else NAN
        f.detach_widths = buf.detach_widths
        f.detach_vertical = buf.detach_upright_at is not None
        f.decoupled_ago = now - buf.decoupled_at if buf.decoupled_at is not None else NAN
        f.spike_ago = now - buf.spike_t if buf.spike_t is not None else NAN
        f.spike_ratio = buf.spike_ratio
        f.flip_ago = now - buf.flip_t if buf.flip_t is not None else NAN
        f.flip_deg = buf.flip_deg
        f.flip_to = buf.flip_to
        return f
