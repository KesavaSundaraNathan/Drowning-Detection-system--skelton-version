# mypy: disable-error-code="index"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""One observation of one person (FrameSample) and the per-track rolling buffer plus
long-running timers (TrackBuffer).

[v3] Split out of the v2 baseline detection/risk_engine.py; definitions verbatim.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lifeguard.vision.behaviour.numeric import (
    CORE,
    FACE,
    HEAD,
    L_EAR,
    L_HIP,
    L_SH,
    L_WR,
    NAN,
    NOSE,
    R_EAR,
    R_HIP,
    R_SH,
    R_WR,
    UPPER,
    Point,
    _dist,
    _median,
    _nanmedian,
    _percentile,
    finite,
    ramp,
)
from lifeguard.vision.constants import (
    BEHAVIOR_BUFFER_FRAMES,
    BOX_SHRINK_RATIO,
    CHILD_RATIO_MIN_SHOULDER_PX,
    FLOATABLE_ASSOCIATION_IOU,
    FLOATABLE_DETACH_BODY_WIDTHS,
    FLOATABLE_DETACH_POSTURE_SECONDS,
    KEYPOINT_CONF_THRESHOLD,
    KINEMATICS,
    POSE_FACE_DOWN_MIN_REL,
    POSE_KP_CONF_HI,
    POSE_KP_CONF_LO,
    POSE_RELIABLE_MIN,
    POSE_SIZE_HI_PX,
    POSE_SIZE_LO_PX,
    SUBMERSION_CONF_DROP,
    TORSO_VERTICAL_DEG,
    VELOCITY_WINDOW_FRAMES,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# One observation of one person
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class FrameSample:
    """A single detection of a tracked person plus everything derivable from it alone."""

    t: float  # seconds (monotonic per source)
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (px)
    conf: float = 1.0
    keypoints: Any | None = None  # 17 × [x, y, conf] (list or ndarray) or None
    in_water: bool = True
    edge_dist_px: float = math.inf  # signed distance to the water edge (+ inside)
    floatable_dist_px: float | None = None  # None = no floatable detector ran
    floatable_iou: float | None = None  # bbox IoU with the nearest floatable (Phase 2 module)

    # ── derived (filled by __post_init__) ─────────────────────────────────────
    kp: np.ndarray | None = field(init=False, default=None, repr=False)
    center: Point = field(init=False, default=(0.0, 0.0))
    anchor: Point | None = field(init=False, default=None)
    anchor_kind: str = field(init=False, default="bbox")
    scale_px: float = field(init=False, default=1.0)
    uprightness: float = field(init=False, default=NAN)  # +90 upright … 0 horizontal … −90 inverted
    axis: Point | None = field(init=False, default=None)
    head: Point | None = field(init=False, default=None)
    head_visible: bool = field(init=False, default=False)
    face_conf: float = field(init=False, default=NAN)
    upper_conf: float = field(init=False, default=NAN)
    body_visible: bool = field(init=False, default=False)
    wrist_rel: tuple[Point | None, Point | None] = field(init=False, default=(None, None))
    child_ratio: float = field(init=False, default=NAN)
    aspect_ratio: float = field(init=False, default=1.0)
    body_width_px: float = field(init=False, default=1.0)  # shoulder span (fallback: bbox width)
    axis_ratio: float = field(init=False, default=NAN)  # image torso length / body scale (foreshortening)
    shoulder_rel: float = field(init=False, default=NAN)  # shoulder conf / head conf (low = sitting low in the water)
    size_px: float = field(init=False, default=0.0)  # sqrt(box area)
    pose_rel: float = field(init=False, default=0.0)  # 0–1: can keypoint cues be trusted on this sample?

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = (float(v) for v in self.bbox)
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        self.center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self.aspect_ratio = h / w
        self.body_width_px = w
        self.size_px = math.sqrt(w * h)
        bbox_scale = 0.55 * math.sqrt(w * h)

        kp = None
        if self.keypoints is not None:
            arr = np.asarray(self.keypoints, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 13 and arr.shape[1] >= 3:
                kp = arr
        self.kp = kp
        if kp is None:
            self.scale_px = max(bbox_scale, KINEMATICS["min_scale_px"])
            return

        thr = KEYPOINT_CONF_THRESHOLD

        def ok(i: int) -> bool:
            return kp[i, 2] >= thr

        def pt(i: int) -> Point:
            return (float(kp[i, 0]), float(kp[i, 1]))

        def mid(i: int, j: int) -> Point | None:
            a, b = ok(i), ok(j)
            if a and b:
                return ((kp[i, 0] + kp[j, 0]) / 2.0, (kp[i, 1] + kp[j, 1]) / 2.0)
            if a:
                return pt(i)
            if b:
                return pt(j)
            return None

        sh = mid(L_SH, R_SH)
        hip = mid(L_HIP, R_HIP)
        head_pts = [pt(i) for i in HEAD if ok(i)]
        if head_pts:
            self.head = (sum(p[0] for p in head_pts) / len(head_pts), sum(p[1] for p in head_pts) / len(head_pts))
            self.head_visible = True
        self.face_conf = float(np.mean(kp[list(FACE), 2]))
        head_max = float(np.max(kp[list(HEAD), 2]))
        if head_max >= thr:
            self.shoulder_rel = float(np.mean(kp[[L_SH, R_SH], 2])) / head_max
        self.upper_conf = float(np.mean(kp[list(UPPER), 2]))
        self.body_visible = any(ok(i) for i in (L_SH, R_SH, L_HIP, R_HIP))

        # The anchor kind records WHICH points formed it, so a shoulder dropping
        # out (anchor jumps half a shoulder width) is never read as motion.
        if sh is not None:
            kind = "shoulders" if (ok(L_SH) and ok(R_SH)) else ("shoulder_l" if ok(L_SH) else "shoulder_r")
            self.anchor, self.anchor_kind = sh, kind
        elif ok(NOSE):
            self.anchor, self.anchor_kind = pt(NOSE), "nose"
        elif ok(L_EAR) and ok(R_EAR):
            self.anchor, self.anchor_kind = mid(L_EAR, R_EAR), "ears"

        # Body scale: every estimate is a (possibly foreshortened) projection of
        # the same physical length, so the largest one is the least shrunk.
        ests: list[float] = []
        sw = _dist(pt(L_SH), pt(R_SH)) if ok(L_SH) and ok(R_SH) else 0.0
        if sw > 0:
            ests.append(1.25 * sw)
            # Shoulder span is the honest body width; the bbox width inflates
            # whenever the arms are out (which is exactly the drowning case).
            self.body_width_px = max(sw, 1.0)
        if sh is not None and hip is not None:
            ests.append(_dist(sh, hip))
        if ok(L_EAR) and ok(R_EAR):
            ests.append(3.0 * _dist(pt(L_EAR), pt(R_EAR)))
        ests.append(0.4 * bbox_scale)
        self.scale_px = max(max(ests), KINEMATICS["min_scale_px"])

        # Uprightness of the body axis (hips → shoulders, else shoulders → head).
        if sh is not None and hip is not None and _dist(sh, hip) >= 0.2 * self.scale_px:
            vx, vy = sh[0] - hip[0], sh[1] - hip[1]
            self.axis_ratio = _dist(sh, hip) / self.scale_px
        elif sh is not None and self.head is not None and _dist(sh, self.head) >= 0.12 * self.scale_px:
            vx, vy = self.head[0] - sh[0], self.head[1] - sh[1]
        else:
            vx = vy = 0.0
        norm = math.hypot(vx, vy)
        if norm > 0:
            self.uprightness = math.degrees(math.atan2(-vy, abs(vx)))
            self.axis = (vx / norm, vy / norm)

        rel: list[Point | None] = []
        for s_i, w_i in ((L_SH, L_WR), (R_SH, R_WR)):
            if ok(s_i) and ok(w_i):
                rel.append((float(kp[w_i, 0] - kp[s_i, 0]), float(kp[w_i, 1] - kp[s_i, 1])))
            else:
                rel.append(None)
        self.wrist_rel = (rel[0], rel[1])

        # [2.30.0] CHILD_RATIO_MIN_SHOULDER_PX: a small pose's ear / shoulder geometry is noise (off at 0)
        if sw > max(1.0, CHILD_RATIO_MIN_SHOULDER_PX) and ok(L_EAR) and ok(R_EAR):
            self.child_ratio = _dist(pt(L_EAR), pt(R_EAR)) / sw

        # Reliability = are ENOUGH body keypoints confident to trust the
        # geometry (the 6 best of head, arms, shoulders, hips)? Not the mean of
        # all of them: a face-down or low-in-the-water person legitimately has
        # an invisible face, and that absence IS the cue.
        best6 = float(np.mean(np.sort(kp[:13, 2])[-6:]))
        self.pose_rel = ramp(best6, POSE_KP_CONF_LO, POSE_KP_CONF_HI) * ramp(
            self.size_px, POSE_SIZE_LO_PX, POSE_SIZE_HI_PX
        )

    @property
    def reliable(self) -> bool:
        return self.kp is not None and self.pose_rel >= POSE_RELIABLE_MIN

    @property
    def has_pose(self) -> bool:
        return self.kp is not None

    def face_hidden(self) -> bool:
        return self.kp is not None and self.face_conf < KINEMATICS["face_hidden_conf"]


# ═════════════════════════════════════════════════════════════════════════════
# Rolling buffer + long-running timers for ONE track
# ═════════════════════════════════════════════════════════════════════════════
class _Streak:
    """A duration timer that tolerates short interruptions ("blips")."""

    __slots__ = ("since", "_break_since", "tolerance")

    def __init__(self, tolerance: float) -> None:
        self.since: float | None = None
        self._break_since: float | None = None
        self.tolerance = tolerance

    def update(self, active: bool, t: float) -> bool:
        """Returns True when a new streak started on this update."""
        if active:
            self._break_since = None
            if self.since is None:
                self.since = t
                return True
            return False
        if self.since is not None:
            if self._break_since is None:
                self._break_since = t
            elif t - self._break_since >= self.tolerance:
                self.since = None
                self._break_since = None
        return False

    def duration(self, now: float) -> float:
        return max(0.0, now - self.since) if self.since is not None else 0.0

    def reset(self) -> None:
        self.since = None
        self._break_since = None


class TrackBuffer:
    """
    Rolling deque (BEHAVIOR_BUFFER_FRAMES samples) for one track_id plus the
    timers Phase 4 needs beyond that window (8 s passive, 5 s submersion …).
    """

    def __init__(self, track_id: int, maxlen: int = BEHAVIOR_BUFFER_FRAMES) -> None:
        self.track_id = track_id
        self.samples: deque[FrameSample] = deque(maxlen=maxlen)
        self.traj: deque[Point] = deque(maxlen=maxlen)  # integrated anchor path
        self.smooth: deque[Point] = deque(maxlen=maxlen)  # trailing-average path
        self.speed: deque[float] = deque(maxlen=maxlen)  # smoothed travel speed su/s
        self.short_speed: deque[float] = deque(maxlen=maxlen)  # 0.3 s lag speed su/s
        self.v_horizontal: deque[float] = deque(maxlen=maxlen)  # |v| over the 30-sample spec window (su/s)
        self.pose_motion: deque[float] = deque(maxlen=maxlen)  # limb motion su/s
        self.wrist_motion: deque[float] = deque(maxlen=maxlen)  # wrist-vs-shoulder motion su/s (strokes / sculling)
        self.scales: deque[float] = deque(maxlen=maxlen)
        self.box_scales: deque[float] = deque(maxlen=maxlen)  # sqrt(box area), for box-only kinematics
        self.first_t: float | None = None
        self.last_t: float | None = None
        self._scale = KINEMATICS["min_scale_px"]
        self.total_samples = 0
        self.good_pose_samples = 0
        # timers
        self.still = _Streak(0.6)
        self.face_down = _Streak(0.6)
        self.inverted = _Streak(0.6)
        self.pose_lost = _Streak(0.4)
        # Spec: PASSIVE_DROWNING needs ZERO KEYPOINT VELOCITY for > 8 s. That is
        # stricter than `still` (which tolerates slow drift), so it gets its own
        # timer measured purely on median keypoint displacement.
        self.zero_velocity = _Streak(0.5)
        self.streak_n = 0
        self.streak_upright_face = 0
        self.streak_ultra_still = 0
        # floatable coupling
        self.coupled_time = 0.0
        self.last_coupled_t: float | None = None
        self.decoupled_at: float | None = None
        self.floatable_seen = False
        # [Phase 4 Step 3] fastest travel in scoring rounds before the current still stretch (travel_before_still)
        self._tb_last_t = -math.inf
        self._tb_before_max = -math.inf  # rounds before the current still stretch
        self._tb_before_n = 0
        self._tb_stretch_start: float | None = None
        self._tb_stretch_max = -math.inf  # rounds inside the current still stretch
        self._tb_stretch_n = 0
        # Phase 2 module: abrupt separation measured in BODY WIDTHS (spec 2.5×)
        # plus the IoU history that decides which float was "theirs".
        self.floatable_iou_max = 0.0
        self.floatable_iou_seen = False
        self.detach_t: float | None = None
        self.detach_widths = 0.0
        self.detach_upright_at: float | None = None
        # abrupt events
        self.spike_t: float | None = None
        self.spike_ratio = 0.0
        self.flip_t: float | None = None
        self.flip_deg = 0.0
        self.flip_to = NAN
        # Phase 2: last time a NORMAL activity was recognised (set by the classifier)
        self.last_normal_t: float | None = None
        self.last_activity: str = ""

    # ── basic properties ──────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        if self.first_t is None or self.last_t is None:
            return 0.0
        return self.last_t - self.first_t

    @property
    def had_pose(self) -> bool:
        return self.good_pose_samples >= 3

    def travel_before_still(self, now: float, still_seconds: float, moving: float) -> float:
        """[Phase 4 Step 3] Record one scoring round's travel speed and return the fastest travel among the rounds
        BEFORE the current still stretch (NaN when there are none: still since the track appeared, or not still).
        A round inside the current stretch counts only once that stretch has ended. One update per timestamp."""
        if now > self._tb_last_t:
            self._tb_last_t = now
            start = now - still_seconds if still_seconds > 0 else None
            new_stretch = start is None or self._tb_stretch_start is None or abs(start - self._tb_stretch_start) > 0.05
            if new_stretch and self._tb_stretch_n:
                self._tb_before_max = max(self._tb_before_max, self._tb_stretch_max)
                self._tb_before_n += self._tb_stretch_n
                self._tb_stretch_max, self._tb_stretch_n = -math.inf, 0
            m = float(moving) if math.isfinite(moving) else 0.0
            if start is None:
                self._tb_stretch_start = None
                self._tb_before_max = max(self._tb_before_max, m)
                self._tb_before_n += 1
            else:
                self._tb_stretch_start = start if new_stretch else self._tb_stretch_start
                self._tb_stretch_max = max(self._tb_stretch_max, m)
                self._tb_stretch_n += 1
        if still_seconds <= 0 or self._tb_before_n == 0:
            return math.nan
        return self._tb_before_max

    def scale(self) -> float:
        # 75th percentile: per-sample estimates only ever SHRINK when points
        # are hidden (foreshortening / submerged shoulders), never grow.
        # Cached — recomputed once per add().
        return self._scale

    def height_baseline(self, seconds: float = 6.0) -> float:
        """75th percentile of this track's box height over the last `seconds` (its 'normal' height)."""
        if not self.samples:
            return NAN
        t_end = self.samples[-1].t
        hs = [x.bbox[3] - x.bbox[1] for x in self.samples if t_end - x.t <= seconds]
        return _percentile(hs, 75) if len(hs) >= 3 else NAN

    def box_scale_baseline(self) -> float:
        return _percentile(self.box_scales, 75) if self.box_scales else 0.0

    def _index_before(self, t: float) -> int:
        """Index of the newest sample with timestamp <= t (or 0)."""
        for i in range(len(self.samples) - 1, -1, -1):
            if self.samples[i].t <= t:
                return i
        return 0

    def recent(self, seconds: float) -> list[int]:
        if not self.samples:
            return []
        t_end = self.samples[-1].t
        return [i for i, s in enumerate(self.samples) if t_end - s.t <= seconds]

    # ── ingestion ─────────────────────────────────────────────────────────────
    def add(self, s: FrameSample) -> None:
        if self.samples and s.t <= self.samples[-1].t:
            return  # out-of-order / duplicate timestamp
        prev = self.samples[-1] if self.samples else None
        dt = (s.t - prev.t) if prev else 0.0

        # integrated path. The step is the median displacement of core keypoints
        # visible in BOTH frames — immune to keypoints flickering in/out (which
        # moves the bbox centre and any single anchor). Fallbacks: same-kind
        # anchor, then bbox centre (HOG / no pose).
        if prev is None:
            pos = s.anchor if s.anchor is not None else s.center
        else:
            # Travel is a track-level cue: the median displacement of the core
            # keypoints both samples agree on is steadier than the box centre
            # (which jumps with every arm movement), so it is used whenever it
            # exists; only pose-DERIVED cues are gated by reliability.
            d = None
            if s.kp is not None and prev.kp is not None:
                idx = [
                    i
                    for i in CORE
                    if s.kp[i, 2] >= KEYPOINT_CONF_THRESHOLD and prev.kp[i, 2] >= KEYPOINT_CONF_THRESHOLD
                ]
                if len(idx) >= 2:
                    d = (
                        float(np.median(s.kp[idx, 0] - prev.kp[idx, 0])),
                        float(np.median(s.kp[idx, 1] - prev.kp[idx, 1])),
                    )
            if d is None and s.anchor is not None and prev.anchor is not None and s.anchor_kind == prev.anchor_kind:
                d = (s.anchor[0] - prev.anchor[0], s.anchor[1] - prev.anchor[1])
            if d is None:
                d = (s.center[0] - prev.center[0], s.center[1] - prev.center[1])
            pos = (self.traj[-1][0] + d[0], self.traj[-1][1] + d[1])

        self.samples.append(s)
        self.traj.append(pos)
        self.scales.append(s.scale_px)
        self.box_scales.append(s.size_px)
        self._scale = max(_percentile(self.scales, 75), KINEMATICS["min_scale_px"])
        self.total_samples += 1
        if self.first_t is None:
            self.first_t = s.t
        self.last_t = s.t
        if s.reliable:
            self.good_pose_samples += 1

        scale = self.scale()
        k = KINEMATICS

        # trailing-average path (removes 1–2 Hz bobbing from travel speed)
        i0 = self._index_before(s.t - k["smooth_seconds"])
        seg = list(self.traj)[i0:]
        self.smooth.append((sum(p[0] for p in seg) / len(seg), sum(p[1] for p in seg) / len(seg)))

        # smoothed travel speed over `travel_lag_seconds`
        j = self._index_before(s.t - k["travel_lag_seconds"])
        lag = s.t - self.samples[j].t
        self.speed.append(_dist(self.smooth[j], self.smooth[-1]) / lag / scale if lag > 0.25 else NAN)

        # short-lag raw speed (for sudden displacement)
        j2 = self._index_before(s.t - 0.3)
        lag2 = s.t - self.samples[j2].t
        self.short_speed.append(_dist(self.traj[j2], self.traj[-1]) / lag2 / scale if lag2 > 0.1 else NAN)

        # Spec: forward velocity vector magnitude |v_horizontal| over a rolling
        # 30-FRAME window. This is net displacement / elapsed time, so bobbing
        # in place (no net travel) reads as ~0 however violent it is.
        n_win = min(VELOCITY_WINDOW_FRAMES, len(self.traj))
        if n_win >= 3:
            i_win = len(self.samples) - n_win
            span = s.t - self.samples[i_win].t
            self.v_horizontal.append(_dist(self.smooth[i_win], self.smooth[-1]) / span / scale if span > 0.3 else NAN)
        else:
            self.v_horizontal.append(NAN)

        # limb motion: median keypoint displacement over ≈1 s
        pm = NAN
        j3 = self._index_before(s.t - 1.0)
        lag3 = s.t - self.samples[j3].t
        ref = self.samples[j3]
        if s.reliable and ref.reliable and lag3 > 0.4:
            both = (s.kp[:, 2] >= KEYPOINT_CONF_THRESHOLD) & (ref.kp[:, 2] >= KEYPOINT_CONF_THRESHOLD)
            if both.sum() >= 3:
                disp = np.hypot(s.kp[both, 0] - ref.kp[both, 0], s.kp[both, 1] - ref.kp[both, 1])
                pm = float(np.median(disp)) / lag3 / scale
        self.pose_motion.append(pm)

        # wrist motion relative to the shoulder over ≈0.5 s (arms stroking / sculling)
        wm = NAN
        j4 = self._index_before(s.t - 0.5)
        lag4 = s.t - self.samples[j4].t
        if lag4 > 0.2 and s.reliable and self.samples[j4].reliable:
            ref4 = self.samples[j4]
            vals = [_dist(a, b) for a, b in zip(s.wrist_rel, ref4.wrist_rel) if a is not None and b is not None]
            if vals:
                wm = max(vals) / lag4 / scale
        self.wrist_motion.append(wm)

        self._update_timers(s, dt, scale)
        self._update_events(s)

    def _update_timers(self, s: FrameSample, dt: float, scale: float) -> None:
        k = KINEMATICS
        spd = self.speed[-1]
        pm = self.pose_motion[-1]
        spd_ok = finite(spd)
        # arms working (strokes, sculling, pressing) — median over ≈1 s
        wms = [self.wrist_motion[i] for i in self.recent(1.0) if finite(self.wrist_motion[i])]
        arms_working = bool(wms) and _median(wms) >= k["arms_working_su"]

        # ── stillness (translation + limbs) ─────────────────────────────────
        is_still = (
            spd_ok and spd < k["still_speed"] and (not finite(pm) or pm < k["still_pose_energy"]) and not arms_working
        )
        # Spec PASSIVE rule: *zero* keypoint velocity. Median limb displacement
        # under `ultra_still_pose` with no translation at all — a stricter test
        # than `is_still`, which still allows a slow drift.
        zero_vel = spd_ok and spd < k["still_speed"] and finite(pm) and pm < k["ultra_still_pose"] and not arms_working
        self.zero_velocity.update(zero_vel, s.t)

        if self.still.update(is_still, s.t):
            self.streak_n = self.streak_upright_face = self.streak_ultra_still = 0
        if self.still.since is None:
            self.streak_n = self.streak_upright_face = self.streak_ultra_still = 0
        elif is_still:
            self.streak_n += 1
            if finite(s.uprightness) and s.uprightness > 50 and finite(s.face_conf) and s.face_conf >= 0.4:
                self.streak_upright_face += 1
            if finite(pm) and pm < k["ultra_still_pose"]:
                self.streak_ultra_still += 1

        # ── face-down (prone, face in the water, not swimming) ──────────────
        # A freestyle swimmer's face is in the water too — the arms tell them apart.
        fd = (
            s.pose_rel >= POSE_FACE_DOWN_MIN_REL
            and s.face_hidden()
            and s.body_visible
            and (not finite(s.uprightness) or s.uprightness < 45)
            and spd_ok
            and spd < 0.20
            and (not finite(pm) or pm < 0.45)
            and not arms_working
        )
        self.face_down.update(fd, s.t)

        # ── inverted (head below hips) ──────────────────────────────────────
        # A swimmer coming straight TOWARD an elevated camera has the head below
        # the hips in the image too — require a limp body and an un-foreshortened
        # torso so lap swimmers at the near wall are never "inverted".
        inv = (
            s.pose_rel >= POSE_FACE_DOWN_MIN_REL
            and finite(s.uprightness)
            and s.uprightness < -45
            and spd_ok
            and spd < 0.3
            and (not finite(pm) or pm < 0.45)
            and finite(s.axis_ratio)
            and s.axis_ratio >= 0.6
        )
        self.inverted.update(inv, s.t)

        # ── pose collapse inside the water, away from the edge ─────────────
        away_from_edge = s.edge_dist_px >= k["edge_margin_su"] * scale
        lost = False
        if self.had_pose and s.in_water and away_from_edge:
            if not s.has_pose:
                lost = True
            elif finite(s.upper_conf) and s.upper_conf < k["pose_collapse_conf"]:
                lost = True
            # Phase 2: a collapsed pose alone is what far / blurry / turned-away
            # swimmers look like too. It only counts when the box agrees that
            # the person went down: the box shrank against the track's own
            # recent height, or the detector itself lost confidence — and the
            # box is big enough that a pose would normally be found.
            if lost:
                base_h = self.height_baseline()
                h_now = s.bbox[3] - s.bbox[1]
                shrunk = finite(base_h) and h_now <= BOX_SHRINK_RATIO * base_h
                faint = s.conf < SUBMERSION_CONF_DROP
                big_enough = self.box_scale_baseline() >= POSE_SIZE_HI_PX * 0.75
                lost = big_enough and (shrunk or faint)
        self.pose_lost.update(lost, s.t)

        # ── floatable detachment measured in BODY WIDTHS (Phase 2 spec) ────
        # "distance spikes abruptly (> 2.5 × body width) while child posture
        #  shifts to vertical" — the two halves are timed independently so a
        #  slow drift away from a ring never fires.
        if s.floatable_iou is not None:
            self.floatable_iou_seen = True
            self.floatable_iou_max = max(self.floatable_iou_max, float(s.floatable_iou))
        if s.floatable_dist_px is not None and math.isfinite(s.floatable_dist_px):
            widths = s.floatable_dist_px / max(s.body_width_px, 1.0)
            # "Theirs" means they actually held it: continuous coupling for at
            # least floatable_hold_seconds (or an already-registered decouple).
            # A stranger's ring drifting past is never a detachment.
            was_associated = self.coupled_time >= KINEMATICS["floatable_hold_seconds"] or self.decoupled_at is not None
            if was_associated and self.floatable_iou_seen:
                was_associated = self.floatable_iou_max >= FLOATABLE_ASSOCIATION_IOU
            if was_associated and widths >= FLOATABLE_DETACH_BODY_WIDTHS:
                if self.detach_t is None or s.t - self.detach_t > FLOATABLE_DETACH_POSTURE_SECONDS:
                    self.detach_t, self.detach_widths = s.t, widths
                    self.detach_upright_at = None
                else:
                    self.detach_widths = max(self.detach_widths, widths)
            elif widths < 1.0:
                self.detach_t = None  # re-grabbed it
                self.detach_upright_at = None
        if (
            self.detach_t is not None
            and self.detach_upright_at is None
            and finite(s.uprightness)
            and s.uprightness > TORSO_VERTICAL_DEG
            and s.t - self.detach_t <= FLOATABLE_DETACH_POSTURE_SECONDS
        ):
            self.detach_upright_at = s.t

        # ── floatable coupling ──────────────────────────────────────────────
        if s.floatable_dist_px is not None:
            fd_su = s.floatable_dist_px / scale if math.isfinite(s.floatable_dist_px) else math.inf
            if fd_su <= k["floatable_coupled_su"]:
                self.floatable_seen = True
                if self.last_coupled_t is not None and s.t - self.last_coupled_t > 1.0:
                    self.coupled_time = 0.0  # holding must be continuous, not accumulated passes
                self.coupled_time += min(dt, 0.5)
                self.last_coupled_t = s.t
                if self.decoupled_at is not None and s.t - self.decoupled_at > 1.0:
                    self.decoupled_at = None  # re-grabbed the float
            elif fd_su >= k["floatable_decoupled_su"]:
                if (
                    self.coupled_time >= k["floatable_hold_seconds"]
                    and self.last_coupled_t is not None
                    and s.t - self.last_coupled_t <= 3.0
                    and self.decoupled_at is None
                ):
                    self.decoupled_at = s.t
                    self.coupled_time = 0.0

    def _update_events(self, s: FrameSample) -> None:
        # sudden displacement: short-lag speed spike vs the preceding baseline
        v = self.short_speed[-1]
        if finite(v) and len(self.short_speed) > 6:
            cut = self._index_before(s.t - 0.6)
            base = _nanmedian(list(self.short_speed)[: max(cut, 1)])
            base = base if finite(base) else 0.0
            ratio = v / (base + 0.3)
            if 1.8 <= v <= 8.0 and ratio >= 3.0:
                if self.spike_t is None or s.t - self.spike_t > 1.0 or ratio > self.spike_ratio:
                    self.spike_t, self.spike_ratio = s.t, ratio

        # abrupt posture change: median uprightness before vs after
        if s.reliable and finite(s.uprightness) and len(self.samples) > 6:
            before = [
                x.uprightness
                for x in self.samples
                if s.t - 2.2 <= x.t <= s.t - 1.0 and x.reliable and finite(x.uprightness)
            ]
            after = [
                x.uprightness for x in self.samples if s.t - 0.8 <= x.t <= s.t and x.reliable and finite(x.uprightness)
            ]
            if len(before) >= 2 and len(after) >= 2:
                mb, ma = _median(before), _median(after)
                delta = abs(ma - mb)
                # A lap swimmer's turn at the far wall also "flips" the image
                # posture (foreshortening): only a person who was NOT travelling
                # beforehand can have an abrupt posture failure.
                pre = [
                    self.speed[i]
                    for i, x in enumerate(self.samples)
                    if s.t - 2.5 <= x.t <= s.t - 1.0 and finite(self.speed[i])
                ]
                travelling = bool(pre) and _median(pre) > 0.5
                if (
                    delta >= 35.0
                    and not travelling
                    and (self.flip_t is None or s.t - self.flip_t > 1.5 or delta > self.flip_deg)
                ):
                    self.flip_t, self.flip_deg, self.flip_to = s.t, delta, ma
