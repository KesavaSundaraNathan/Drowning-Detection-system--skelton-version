# mypy: disable-error-code="arg-type, assignment, operator"
# (ported verbatim from the v2 baseline - narrowing / NumPy typing gaps of v2 code; see pyproject.toml)
"""
[v3] Ported verbatim from the v2 baseline detection/simulation.py (imports only).

detection/simulation.py — Synthetic pool footage at the keypoint level (Phase 4).

Why
───
There is no public, labelled dataset of real drownings, and the behaviour
classifier works on pose *kinematics*, not pixels. This module synthesises what
a YOLO-pose tracker would output for a pool camera, so the classifier can be
validated and its thresholds calibrated (behavior_calibration.py) on thousands
of randomised sequences — and so the false-positive rate on normal swimming
can be measured instead of guessed.

What is modelled
────────────────
* A pinhole camera 3.5–7 m above the pool edge, pitched down 25–50°, yawed ±15°.
* A 3-D stick figure (COCO-17 joints) per person with heading, body pitch
  (0° horizontal … 90° upright … −90° inverted), roll, stroke / scull / press
  arm patterns, head tilt, leg kicks and a world position in the pool.
* The water surface: joints below it lose keypoint confidence exponentially with
  depth (≈0.3 at 10 cm, ≈0.05 at 30 cm) — the same effect that makes real
  swimmers' legs invisible. Faces pointing away from the camera or into the
  water are hidden. Pixel noise, random dropouts and missed detections.
* Detection disappears when too little of the body is visible (true
  submersion), optional tracker id switches, optional floatable objects.

Scenarios (ground-truth label in brackets)
    freestyle, breaststroke, backstroke, lap_turns          [NORMAL_SWIMMING]
    treading, resting_wall, standing_shallow, playing        [NON-DROWNING]
    short_dive                                               [NON-DROWNING]
    active_drowning                                          [ACTIVE_DROWNING]
    passive_face_down, passive_still                         [PASSIVE_DROWNING]
    floatable_failure, child_posture_failure                 [FLOATABLE_FAILURE]
    submersion_vanish, submersion_visible                    [SUBMERSION]
    deck_walker, deck_edge                                   [BYSTANDER]

This is a validation harness, not a substitute for real footage: calibrate
again with operator-labelled clips from your own cameras (alert feedback).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from lifeguard.vision.inference.detector import FrameDetections, TrackedPerson, foot_point_from
from lifeguard.vision.scene.floatables import FloatableObject
from lifeguard.vision.scene.geometry import ZoneGeometry, ZonePolygon, normalized_to_contour

# ── ground-truth labels ───────────────────────────────────────────────────────
SCENARIO_LABELS: dict[str, str] = {
    "freestyle": "NORMAL_SWIMMING",
    "breaststroke": "NORMAL_SWIMMING",
    "backstroke": "NORMAL_SWIMMING",
    "lap_turns": "NORMAL_SWIMMING",
    "treading": "RESTING",
    "resting_wall": "RESTING",
    "standing_shallow": "RESTING",
    "playing": "NORMAL_SWIMMING",
    "short_dive": "NORMAL_SWIMMING",
    "exit_pool": "NORMAL_SWIMMING",
    "active_drowning": "ACTIVE_DROWNING",
    "passive_face_down": "PASSIVE_DROWNING",
    "passive_still": "PASSIVE_DROWNING",
    "floatable_failure": "FLOATABLE_FAILURE",
    "child_posture_failure": "FLOATABLE_FAILURE",
    "submersion_vanish": "SUBMERSION",
    "submersion_visible": "SUBMERSION",
    "deck_walker": "BYSTANDER",
    "deck_edge": "BYSTANDER",
}
NORMAL_SCENARIOS = tuple(k for k, v in SCENARIO_LABELS.items() if v in ("NORMAL_SWIMMING", "RESTING"))
DROWNING_SCENARIO_NAMES = tuple(
    k
    for k, v in SCENARIO_LABELS.items()
    if v in ("ACTIVE_DROWNING", "PASSIVE_DROWNING", "FLOATABLE_FAILURE", "SUBMERSION")
)
BYSTANDER_SCENARIOS = ("deck_walker", "deck_edge")

POOL_X = (-4.0, 4.0)  # metres (width)
POOL_Z = (6.0, 20.0)  # metres (length, away from the camera)
TWO_PI = 2.0 * math.pi


# ═════════════════════════════════════════════════════════════════════════════
# Camera
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Camera:
    width: int = 1280
    height: int = 720
    focal: float = 950.0
    cam_height: float = 5.0
    pitch_deg: float = 35.0
    yaw_deg: float = 0.0
    x_offset: float = 0.0

    def __post_init__(self) -> None:
        p, y = math.radians(self.pitch_deg), math.radians(self.yaw_deg)
        self.fwd = np.array([math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p)])
        self.right = np.array([math.cos(y), 0.0, -math.sin(y)])
        self.up = np.cross(self.fwd, self.right)
        self.pos = np.array([self.x_offset, self.cam_height, 0.0])

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rel = pts - self.pos
        xc, yc, zc = rel @ self.right, rel @ self.up, rel @ self.fwd
        zc = np.maximum(zc, 1e-3)
        u = self.width / 2 + self.focal * xc / zc
        v = self.height / 2 - self.focal * yc / zc
        return np.stack([u, v], axis=1), zc

    def to_camera(self, p: np.ndarray) -> np.ndarray:
        d = self.pos - p
        return d / (np.linalg.norm(d) + 1e-9)

    @staticmethod
    def random(rng: np.random.Generator) -> Camera:
        h = float(rng.uniform(3.5, 7.0))
        centre = math.degrees(0.5 * (math.atan(h / POOL_Z[0]) + math.atan(h / POOL_Z[1])))
        return Camera(
            cam_height=h,
            pitch_deg=float(np.clip(centre + rng.uniform(-3, 3), 18, 55)),
            yaw_deg=float(rng.uniform(-12, 12)),
            x_offset=float(rng.uniform(-1.5, 1.5)),
        )

    def water_geometry(self, camera_id: str = "sim") -> ZoneGeometry:
        corners = np.array(
            [
                [POOL_X[0], 0, POOL_Z[0]],
                [POOL_X[1], 0, POOL_Z[0]],
                [POOL_X[1], 0, POOL_Z[1]],
                [POOL_X[0], 0, POOL_Z[1]],
            ],
            dtype=float,
        )
        uv, _ = self.project(corners)
        poly = _clip_polygon(uv.tolist(), self.width, self.height)
        norm = [[x / self.width, y / self.height] for x, y in poly]
        zone = ZonePolygon(1, "Main Pool", "WATER", normalized_to_contour(norm, self.width, self.height), 1.0, None)
        return ZoneGeometry(camera_id, self.width, self.height, [zone])


def _clip_polygon(poly: list[list[float]], w: int, h: int) -> list[list[float]]:
    """Sutherland–Hodgman clip of a polygon to the image rectangle."""

    def clip(pts, inside, inter):
        out = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(inter(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(inter(prev, cur))
        return out

    def ix(a, b, x):
        t = (x - a[0]) / (b[0] - a[0] + 1e-12)
        return [x, a[1] + t * (b[1] - a[1])]

    def iy(a, b, y):
        t = (y - a[1]) / (b[1] - a[1] + 1e-12)
        return [a[0] + t * (b[0] - a[0]), y]

    pts = poly
    for inside, inter in (
        (lambda p: p[0] >= 0, lambda a, b: ix(a, b, 0)),
        (lambda p: p[0] <= w, lambda a, b: ix(a, b, w)),
        (lambda p: p[1] >= 0, lambda a, b: iy(a, b, 0)),
        (lambda p: p[1] <= h, lambda a, b: iy(a, b, h)),
    ):
        if not pts:
            break
        pts = clip(pts, inside, inter)
    return pts


# ═════════════════════════════════════════════════════════════════════════════
# Body model
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Pose:
    root: np.ndarray  # hip centre (world, m)
    heading: float  # rad, direction the body faces / travels (0 = +Z away from camera)
    pitch: float  # deg: 0 horizontal (prone), 90 upright, −90 inverted
    roll: float = 0.0  # deg about the body axis (180 = supine)
    scale: float = 1.0  # 1 adult, ~0.6 child
    arm_l: tuple[np.ndarray, np.ndarray] = None  # (upper dir, fore dir) in body coords (a, n, l)
    arm_r: tuple[np.ndarray, np.ndarray] = None
    leg_kick: float = 0.0  # rad
    head_tilt: float = 0.0  # deg, + = tilted back (face up)


def _rot(v: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * np.dot(axis, v) * (1 - math.cos(ang))


def body_frame(heading: float, pitch: float, roll: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit vectors (a = hips→shoulders, n = chest normal, l = person's left)."""
    a0 = np.array([0.0, 1.0, 0.0])
    n0 = np.array([math.sin(heading), 0.0, math.cos(heading)])
    l0 = np.cross(a0, n0)
    beta = math.radians(90.0 - pitch)
    a = a0 * math.cos(beta) + n0 * math.sin(beta)
    n = n0 * math.cos(beta) - a0 * math.sin(beta)
    r = math.radians(roll)
    return a, _rot(n, a, r), _rot(l0, a, r)


def joints(p: Pose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """COCO-17 joints (17, 3) plus the chest normal and left vector."""
    a, n, l = body_frame(p.heading, p.pitch, p.roll)
    s = p.scale
    J = np.zeros((17, 3))
    hip = p.root
    sh = hip + a * 0.5 * s
    tilt = math.radians(p.head_tilt)
    head_axis = a * math.cos(tilt) - n * math.sin(tilt)
    face_n = n * math.cos(tilt) + a * math.sin(tilt)
    head = sh + head_axis * 0.24 * s
    hw = 0.075 * s * min(1.5, 1.0 + 1.2 * max(0.0, 1.0 - s))  # children: proportionally larger heads
    J[0] = head + face_n * 0.10 * s
    J[1] = head + face_n * 0.08 * s + head_axis * 0.03 * s + l * 0.03 * s
    J[2] = head + face_n * 0.08 * s + head_axis * 0.03 * s - l * 0.03 * s
    J[3] = head + l * hw
    J[4] = head - l * hw
    J[5] = sh + l * 0.2 * s
    J[6] = sh - l * 0.2 * s
    J[11] = hip + l * 0.14 * s
    J[12] = hip - l * 0.14 * s

    def arm(sh_pt, dirs, side):
        up, fore = dirs if dirs is not None else (np.array([-1.0, 0.0, 0.25 * side]), np.array([-1.0, 0.0, 0.1 * side]))
        u = up[0] * a + up[1] * n + up[2] * l
        f = fore[0] * a + fore[1] * n + fore[2] * l
        u /= np.linalg.norm(u) + 1e-9
        f /= np.linalg.norm(f) + 1e-9
        el = sh_pt + u * 0.30 * s
        return el, el + f * 0.28 * s

    J[7], J[9] = arm(J[5], p.arm_l, +1)
    J[8], J[10] = arm(J[6], p.arm_r, -1)
    for hi, ki, ai, sgn in ((11, 13, 15, 1), (12, 14, 16, -1)):
        kick = a * math.cos(p.leg_kick * sgn) * -1 + n * math.sin(p.leg_kick * sgn) * 0.4
        kick /= np.linalg.norm(kick) + 1e-9
        J[ki] = J[hi] + kick * 0.42 * s
        J[ai] = J[ki] + kick * 0.40 * s
    return J, face_n, l


def in_pool(x: float, z: float) -> bool:
    return POOL_X[0] <= x <= POOL_X[1] and POOL_Z[0] <= z <= POOL_Z[1]


# ═════════════════════════════════════════════════════════════════════════════
# Actors (one per scenario)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class ActorFrame:
    pose: Pose | None
    force_blob: bool = False  # detector still fires on a submerged body (pose lost)
    floatable: np.ndarray | None = None  # world position of a held float (centre)


@dataclass
class Actor:
    track_id: int
    scenario: str
    label: str
    step: Callable[[float, float], ActorFrame]
    onset: float | None = None  # when the drowning behaviour starts (ground truth)
    id_switch_at: float | None = None


def _pool_point(rng: np.random.Generator, margin: float = 1.5) -> np.ndarray:
    return np.array(
        [rng.uniform(POOL_X[0] + margin, POOL_X[1] - margin), 0.0, rng.uniform(POOL_Z[0] + margin, POOL_Z[1] - margin)]
    )


def _bounce(pos: np.ndarray, heading: float) -> tuple[np.ndarray, float]:
    x, z = pos[0], pos[2]
    if x < POOL_X[0] + 0.8 or x > POOL_X[1] - 0.8:
        heading = -heading
        pos[0] = float(np.clip(x, POOL_X[0] + 0.8, POOL_X[1] - 0.8))
    if z < POOL_Z[0] + 0.8 or z > POOL_Z[1] - 0.8:
        heading = math.pi - heading
        pos[2] = float(np.clip(z, POOL_Z[0] + 0.8, POOL_Z[1] - 0.8))
    return pos, heading


def _v(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=float)


def make_actor(scenario: str, track_id: int, rng: np.random.Generator, duration: float) -> Actor:
    label = SCENARIO_LABELS[scenario]
    pos = _pool_point(rng)
    heading = float(rng.uniform(-math.pi, math.pi))
    state: dict[str, float] = {}

    # ── normal swimming ─────────────────────────────────────────────────────
    if scenario in ("freestyle", "lap_turns", "backstroke"):
        speed = rng.uniform(0.45, 1.2)
        freq = rng.uniform(0.45, 0.8)
        supine = scenario == "backstroke"
        if scenario == "lap_turns":
            heading = float(rng.choice([0.0, math.pi]))
            pos[0] = rng.uniform(-2.5, 2.5)

        def step(t: float, dt: float) -> ActorFrame:
            nonlocal pos, heading
            pos = pos + _v(math.sin(heading), 0, math.cos(heading)) * speed * dt
            pos, heading = _bounce(pos, heading)
            ph = TWO_PI * freq * t
            arms = []
            for side, off in ((1, 0.0), (-1, math.pi)):
                phi = ph + off
                d = _v(math.cos(phi), math.sin(phi) * (-1 if supine else 1), 0.2 * side)
                arms.append((d, d + _v(0, 0, 0.05 * side)))
            root = pos + _v(0, -0.07, 0)
            return ActorFrame(
                Pose(
                    root,
                    heading,
                    pitch=float(rng.normal(8, 2)),
                    roll=180.0 if supine else 20 * math.sin(ph),
                    arm_l=arms[0],
                    arm_r=arms[1],
                    leg_kick=0.35 * math.sin(TWO_PI * 2.2 * t),
                    head_tilt=-10 if not supine else 10,
                )
            )

        return Actor(track_id, scenario, label, step)

    if scenario == "breaststroke":
        speed = rng.uniform(0.35, 0.8)
        freq = rng.uniform(0.45, 0.75)

        def step(t: float, dt: float) -> ActorFrame:
            nonlocal pos, heading
            ph = TWO_PI * freq * t
            surge = 0.6 + 0.8 * max(0.0, math.sin(ph))
            pos = pos + _v(math.sin(heading), 0, math.cos(heading)) * speed * surge * dt
            pos, heading = _bounce(pos, heading)
            sweep = 0.9 * (0.5 + 0.5 * math.sin(ph))
            lift = max(0.0, math.sin(ph))
            arms = (
                (_v(math.cos(sweep), 0.3, math.sin(sweep)), _v(math.cos(sweep), 0.5, 0.3 * math.sin(sweep))),
                (_v(math.cos(sweep), 0.3, -math.sin(sweep)), _v(math.cos(sweep), 0.5, -0.3 * math.sin(sweep))),
            )
            root = pos + _v(0, -0.10 + 0.08 * lift, 0)
            return ActorFrame(
                Pose(
                    root,
                    heading,
                    pitch=10 + 22 * lift,
                    arm_l=arms[0],
                    arm_r=arms[1],
                    leg_kick=0.5 * math.sin(ph + 1.5),
                    head_tilt=15 * lift,
                )
            )

        return Actor(track_id, scenario, label, step)

    # ── stationary non-drowning ─────────────────────────────────────────────
    if scenario in ("treading", "resting_wall", "standing_shallow"):
        facing = math.pi + float(rng.uniform(-0.9, 0.9))  # roughly facing the camera
        scull_f = rng.uniform(0.7, 1.3)
        if scenario == "resting_wall":
            pos[2] = POOL_Z[1] - 1.1
        drift = _v(rng.normal(0, 0.03), 0, rng.normal(0, 0.03))

        def step(t: float, dt: float) -> ActorFrame:
            nonlocal pos
            pos = pos + drift * dt
            look = facing + 0.5 * math.sin(TWO_PI * 0.08 * t + track_id)
            if scenario == "treading":
                sc = math.sin(TWO_PI * scull_f * t)
                arms = (
                    (_v(-0.05, 0.6 + 0.35 * sc, 0.8), _v(0.0, 0.7 + 0.3 * sc, 0.6)),
                    (_v(-0.05, 0.6 - 0.35 * sc, -0.8), _v(0.0, 0.7 - 0.3 * sc, -0.6)),
                )
                root = pos + _v(0, -0.55 + 0.015 * math.sin(TWO_PI * 0.5 * t), 0)
                return ActorFrame(
                    Pose(
                        root,
                        look,
                        pitch=82,
                        arm_l=arms[0],
                        arm_r=arms[1],
                        leg_kick=0.4 * math.sin(TWO_PI * 1.1 * t),
                        head_tilt=3,
                    )
                )
            if scenario == "resting_wall":
                arms = ((_v(0.2, 1.0, 0.35), _v(0.1, 1.0, 0.1)), (_v(0.2, 1.0, -0.35), _v(0.1, 1.0, -0.1)))
                root = pos + _v(0, -0.5, 0)
                return ActorFrame(Pose(root, look, pitch=78, arm_l=arms[0], arm_r=arms[1], head_tilt=5))
            gest = 0.3 * math.sin(TWO_PI * 0.3 * t) if int(t / 6) % 2 == 0 else 0.0
            arms = ((_v(-1.0, 0.2 + gest, 0.3), _v(-0.6, 0.6 + gest, 0.2)), (_v(-1.0, 0.2, -0.3), _v(-0.8, 0.4, -0.2)))
            root = pos + _v(0, -0.25, 0)
            return ActorFrame(Pose(root, look, pitch=88, arm_l=arms[0], arm_r=arms[1], head_tilt=0))

        return Actor(track_id, scenario, label, step)

    if scenario == "playing":
        speed = rng.uniform(0.3, 0.6)
        scale = rng.uniform(0.6, 0.85)

        def step(t: float, dt: float) -> ActorFrame:
            nonlocal pos, heading
            heading += rng.normal(0, 0.6) * dt
            pos = pos + _v(math.sin(heading), 0, math.cos(heading)) * speed * dt
            pos, heading = _bounce(pos, heading)
            splash = math.sin(TWO_PI * 1.6 * t + 0.6 * math.sin(t))
            arms = (
                (_v(0.3 + 0.4 * splash, 0.6, 0.7), _v(0.2 + 0.5 * splash, 0.6, 0.6)),
                (_v(0.3 - 0.4 * splash, 0.6, -0.7), _v(0.2 - 0.5 * splash, 0.6, -0.6)),
            )
            jump = 0.12 * max(0.0, math.sin(TWO_PI * 0.5 * t))
            root = pos + _v(0, -0.45 * scale + jump, 0)
            return ActorFrame(Pose(root, heading, pitch=70, scale=scale, arm_l=arms[0], arm_r=arms[1], head_tilt=5))

        return Actor(track_id, scenario, label, step)

    if scenario == "short_dive":
        speed = rng.uniform(0.5, 1.0)
        t_dive = rng.uniform(4.0, 8.0)
        under = rng.uniform(1.5, 3.0)

        def step(t: float, dt: float) -> ActorFrame:
            nonlocal pos, heading
            pos = pos + _v(math.sin(heading), 0, math.cos(heading)) * speed * dt
            pos, heading = _bounce(pos, heading)
            depth = 0.0
            if t_dive <= t <= t_dive + under:
                depth = 0.9 * math.sin(math.pi * (t - t_dive) / under)
            ph = TWO_PI * 0.6 * t
            arms = (
                (_v(math.cos(ph), math.sin(ph), 0.2), _v(math.cos(ph), math.sin(ph), 0.25)),
                (
                    _v(math.cos(ph + math.pi), math.sin(ph + math.pi), -0.2),
                    _v(math.cos(ph + math.pi), math.sin(ph + math.pi), -0.25),
                ),
            )
            root = pos + _v(0, -0.07 - depth, 0)
            return ActorFrame(
                Pose(
                    root,
                    heading,
                    pitch=8 - 25 * (depth > 0.1),
                    arm_l=arms[0],
                    arm_r=arms[1],
                    leg_kick=0.35 * math.sin(TWO_PI * 2.0 * t),
                    head_tilt=-10,
                )
            )

        return Actor(track_id, scenario, label, step, id_switch_at=t_dive + under + 0.3 if rng.random() < 0.5 else None)

    if scenario == "exit_pool":
        # swim to the near wall, climb out onto the deck, walk along it
        pos[2] = rng.uniform(POOL_Z[0] + 3.0, POOL_Z[0] + 6.0)
        speed = rng.uniform(0.5, 0.9)
        st = {"phase": "swim", "t0": 0.0, "x": 0.0}

        def step(t: float, dt: float) -> ActorFrame:
            nonlocal pos
            ph = TWO_PI * 0.6 * t
            if st["phase"] == "swim":
                pos = pos + _v(0, 0, -speed * dt)
                arms = (
                    (_v(math.cos(ph), math.sin(ph), 0.2), _v(math.cos(ph), math.sin(ph), 0.2)),
                    (_v(-math.cos(ph), -math.sin(ph), -0.2), _v(-math.cos(ph), -math.sin(ph), -0.2)),
                )
                if pos[2] <= POOL_Z[0] + 0.5:
                    st["phase"], st["t0"] = "climb", t
                return ActorFrame(Pose(pos + _v(0, -0.07, 0), math.pi, pitch=8, arm_l=arms[0], arm_r=arms[1]))
            k = min(1.0, (t - st["t0"]) / 2.0)
            if st["phase"] == "climb":
                root = _v(pos[0], -0.3 + 1.2 * k, POOL_Z[0] + 0.3 - 1.0 * k)
                if k >= 1.0:
                    st["phase"], st["x"] = "walk", pos[0]
                push = ((_v(-0.6, 0.8, 0.3), _v(-0.9, 0.3, 0.2)), (_v(-0.6, 0.8, -0.3), _v(-0.9, 0.3, -0.2)))
                return ActorFrame(Pose(root, math.pi, pitch=30 + 60 * k, arm_l=push[0], arm_r=push[1]))
            st["x"] += 1.1 * dt
            swing = math.sin(TWO_PI * 0.9 * t)
            arms = (
                (_v(-1.0, 0.25 * swing, 0.2), _v(-1.0, 0.3 * swing, 0.1)),
                (_v(-1.0, -0.25 * swing, -0.2), _v(-1.0, -0.3 * swing, -0.1)),
            )
            return ActorFrame(
                Pose(
                    _v(st["x"], 0.9, POOL_Z[0] - 0.7),
                    math.pi / 2,
                    pitch=90,
                    arm_l=arms[0],
                    arm_r=arms[1],
                    leg_kick=0.35 * swing,
                )
            )

        return Actor(track_id, scenario, label, step)

    # ── Scenario A: active drowning (instinctive drowning response) ─────────
    if scenario == "active_drowning":
        onset = rng.uniform(3.0, 6.0)
        pre_speed = rng.uniform(0.3, 0.7)
        facing = math.pi + float(rng.uniform(-1.2, 1.2))
        f_press = rng.uniform(1.2, 2.2)
        f_bob = rng.uniform(0.8, 1.6)
        phase = {"p": 0.0, "b": 0.0}

        def step(t: float, dt: float) -> ActorFrame:
            nonlocal pos, heading
            if t < onset:  # swimming before the struggle
                pos = pos + _v(math.sin(heading), 0, math.cos(heading)) * pre_speed * dt
                pos, heading = _bounce(pos, heading)
                ph = TWO_PI * 0.6 * t
                arms = (
                    (_v(math.cos(ph), math.sin(ph), 0.2), _v(math.cos(ph), math.sin(ph), 0.2)),
                    (_v(-math.cos(ph), -math.sin(ph), -0.2), _v(-math.cos(ph), -math.sin(ph), -0.2)),
                )
                return ActorFrame(Pose(pos + _v(0, -0.08, 0), heading, pitch=10, arm_l=arms[0], arm_r=arms[1]))
            phase["p"] += TWO_PI * f_press * (1 + 0.25 * rng.normal()) * dt
            phase["b"] += TWO_PI * f_bob * (1 + 0.25 * rng.normal()) * dt
            press = math.sin(phase["p"]) * (0.8 + 0.3 * rng.random())
            bob = 0.13 * math.sin(phase["b"])
            pos = pos + _v(rng.normal(0, 0.05), 0, rng.normal(0, 0.05)) * dt
            arms = (
                (_v(0.2 + 0.55 * press, 0.25, 0.9), _v(0.1 + 0.7 * press, 0.2, 0.8)),
                (_v(0.2 + 0.55 * press, 0.25, -0.9), _v(0.1 + 0.7 * press, 0.2, -0.8)),
            )
            root = pos + _v(0, -0.62 + bob, 0)
            return ActorFrame(
                Pose(
                    root,
                    facing,
                    pitch=float(rng.normal(84, 3)),
                    arm_l=arms[0],
                    arm_r=arms[1],
                    leg_kick=0.3 * math.sin(phase["b"]),
                    head_tilt=28,
                )
            )

        return Actor(track_id, scenario, label, step, onset=onset)

    # ── Scenario B: passive drowning ────────────────────────────────────────
    if scenario in ("passive_face_down", "passive_still"):
        onset = rng.uniform(3.0, 6.0)
        pre_speed = rng.uniform(0.3, 0.8)

        def step(t: float, dt: float) -> ActorFrame:
            nonlocal pos, heading
            ph = TWO_PI * 0.6 * t
            if t < onset:
                pos = pos + _v(math.sin(heading), 0, math.cos(heading)) * pre_speed * dt
                pos, heading = _bounce(pos, heading)
                arms = (
                    (_v(math.cos(ph), math.sin(ph), 0.2), _v(math.cos(ph), math.sin(ph), 0.2)),
                    (_v(-math.cos(ph), -math.sin(ph), -0.2), _v(-math.cos(ph), -math.sin(ph), -0.2)),
                )
                return ActorFrame(Pose(pos + _v(0, -0.08, 0), heading, pitch=10, arm_l=arms[0], arm_r=arms[1]))
            pos = pos + _v(0.02, 0, 0.01) * dt
            if scenario == "passive_face_down":
                limp = ((_v(0.2, 1.0, 0.3), _v(0.1, 1.0, 0.2)), (_v(0.2, 1.0, -0.3), _v(0.1, 1.0, -0.2)))
                return ActorFrame(
                    Pose(pos + _v(0, -0.06, 0), heading, pitch=3, arm_l=limp[0], arm_r=limp[1], head_tilt=-25)
                )
            slump = ((_v(-0.7, 0.6, 0.3), _v(-0.8, 0.5, 0.2)), (_v(-0.7, 0.6, -0.3), _v(-0.8, 0.5, -0.2)))
            return ActorFrame(
                Pose(pos + _v(0, -0.42, 0), math.pi + 0.4, pitch=62, arm_l=slump[0], arm_r=slump[1], head_tilt=-55)
            )

        return Actor(track_id, scenario, label, step, onset=onset)

    # ── Scenario C: floatable failure ───────────────────────────────────────
    if scenario in ("floatable_failure", "child_posture_failure"):
        onset = rng.uniform(4.0, 7.0)
        scale = rng.uniform(0.55, 0.7)
        drift = rng.uniform(0.05, 0.2)
        facing = math.pi + float(rng.uniform(-1.0, 1.0))
        float_pos = {"p": None}
        tip_to = float(rng.choice([90.0, -35.0]))
        with_float = scenario == "floatable_failure"
        phase = {"p": 0.0}

        def step(t: float, dt: float) -> ActorFrame:
            nonlocal pos
            if t < onset:
                pos = pos + _v(math.sin(facing), 0, math.cos(facing)) * drift * dt
                paddle = 0.2 * math.sin(TWO_PI * 1.0 * t)
                arms = (
                    (_v(0.3, 0.8, 0.5 + paddle), _v(0.2, 0.9, 0.3)),
                    (_v(0.3, 0.8, -0.5 - paddle), _v(0.2, 0.9, -0.3)),
                )
                root = pos + _v(0, -0.30 * scale, 0)
                fl = pos + _v(0, 0.0, 0) if with_float else None
                float_pos["p"] = fl
                return ActorFrame(
                    Pose(root, facing, pitch=60, scale=scale, arm_l=arms[0], arm_r=arms[1], head_tilt=5), floatable=fl
                )
            k = min(1.0, (t - onset) / 0.5)
            pitch = 60 + (tip_to - 60) * k
            sink = min(0.45, 0.06 * (t - onset))
            phase["p"] += TWO_PI * 1.6 * (1 + 0.3 * rng.normal()) * dt
            press = math.sin(phase["p"])
            arms = (
                (_v(0.2 + 0.6 * press, 0.3, 0.9), _v(0.1 + 0.7 * press, 0.2, 0.8)),
                (_v(0.2 + 0.6 * press, 0.3, -0.9), _v(0.1 + 0.7 * press, 0.2, -0.8)),
            )
            pos = pos + _v(rng.normal(0, 0.05), 0, rng.normal(0, 0.05)) * dt
            root = pos + _v(0, -0.45 * scale - sink + 0.06 * math.sin(TWO_PI * 1.2 * t), 0)
            fl = None
            if with_float and float_pos["p"] is not None:
                float_pos["p"] = float_pos["p"] + _v(math.sin(facing + 1.2), 0, math.cos(facing + 1.2)) * 0.8 * dt
                fl = float_pos["p"]
            return ActorFrame(
                Pose(
                    root,
                    facing,
                    pitch=pitch,
                    scale=scale,
                    arm_l=arms[0],
                    arm_r=arms[1],
                    head_tilt=25 if tip_to > 0 else -20,
                ),
                floatable=fl,
            )

        return Actor(track_id, scenario, label, step, onset=onset)

    # ── Scenario D: sudden blackout / submersion ────────────────────────────
    if scenario in ("submersion_vanish", "submersion_visible"):
        onset = rng.uniform(4.0, 7.0)
        speed = rng.uniform(0.3, 0.9)
        visible = scenario == "submersion_visible"

        def step(t: float, dt: float) -> ActorFrame:
            nonlocal pos, heading
            ph = TWO_PI * 0.6 * t
            arms = (
                (_v(math.cos(ph), math.sin(ph), 0.2), _v(math.cos(ph), math.sin(ph), 0.2)),
                (_v(-math.cos(ph), -math.sin(ph), -0.2), _v(-math.cos(ph), -math.sin(ph), -0.2)),
            )
            if t < onset:
                pos = pos + _v(math.sin(heading), 0, math.cos(heading)) * speed * dt
                pos, heading = _bounce(pos, heading)
                return ActorFrame(Pose(pos + _v(0, -0.07, 0), heading, pitch=8, arm_l=arms[0], arm_r=arms[1]))
            depth = min(1.4, 0.9 * (t - onset))
            limp = ((_v(0.3, 0.9, 0.4), _v(0.2, 1.0, 0.3)), (_v(0.3, 0.9, -0.4), _v(0.2, 1.0, -0.3)))
            return ActorFrame(
                Pose(pos + _v(0, -0.07 - depth, 0), heading, pitch=-20, arm_l=limp[0], arm_r=limp[1]),
                force_blob=visible and depth > 0.3,
            )

        return Actor(track_id, scenario, label, step, onset=onset + 0.8)

    # ── bystanders ──────────────────────────────────────────────────────────
    if scenario in ("deck_walker", "deck_edge"):
        z = rng.uniform(4.0, 5.2) if scenario == "deck_walker" else POOL_Z[0] - rng.uniform(0.25, 0.5)
        x = rng.uniform(-3.0, 3.0)
        speed = 1.2 if scenario == "deck_walker" else 0.0
        head = math.pi / 2 if rng.random() < 0.5 else -math.pi / 2
        d = {"x": x, "h": head}

        def step(t: float, dt: float) -> ActorFrame:
            d["x"] += math.sin(d["h"]) * speed * dt
            if abs(d["x"]) > 3.5:
                d["h"] = -d["h"]
            swing = math.sin(TWO_PI * 0.9 * t) if speed > 0 else 0.0
            arms = (
                (_v(-1.0, 0.25 * swing, 0.2), _v(-1.0, 0.3 * swing, 0.1)),
                (_v(-1.0, -0.25 * swing, -0.2), _v(-1.0, -0.3 * swing, -0.1)),
            )
            root = _v(d["x"], 0.9, z)
            return ActorFrame(
                Pose(
                    root,
                    d["h"] if speed > 0 else math.pi,
                    pitch=90,
                    arm_l=arms[0],
                    arm_r=arms[1],
                    leg_kick=0.35 * swing,
                )
            )

        return Actor(track_id, scenario, label, step)

    raise ValueError(f"unknown scenario {scenario!r}")


# ═════════════════════════════════════════════════════════════════════════════
# Rendering an actor → TrackedPerson
# ═════════════════════════════════════════════════════════════════════════════
def observe(
    cam: Camera,
    actor: Actor,
    af: ActorFrame,
    rng: np.random.Generator,
    track_id: int,
    noise: float = 1.0,
    dropout: float = 0.04,
) -> TrackedPerson | None:
    if af.pose is None:
        return None
    J, face_n, left = joints(af.pose)
    # Refraction: submerged points appear ≈1/1.33 as deep. A point whose line of
    # sight crosses the surface outside the pool is hidden by the pool wall.
    Jv = J.copy()
    hidden = np.zeros(17, dtype=bool)
    for i in range(17):
        x, y, z = J[i]
        if in_pool(x, z) and y < 0.0:
            Jv[i, 1] = y / 1.33
            ray = Jv[i] - cam.pos
            s_ = cam.pos[1] / max(cam.pos[1] - Jv[i, 1], 1e-6)
            sx, sz = cam.pos[0] + s_ * ray[0], cam.pos[2] + s_ * ray[2]
            hidden[i] = not in_pool(sx, sz)
    uv, depth = cam.project(Jv)
    conf = np.empty(17)
    for i in range(17):
        x, y, z = J[i]
        submerged = in_pool(x, z) and y < 0.0
        c = rng.uniform(0.75, 0.95)
        if submerged:
            c = 0.85 * math.exp(y / 0.10) * rng.uniform(0.8, 1.1)
        conf[i] = 0.0 if hidden[i] else c
    to_cam = cam.to_camera(J[0])
    facing = float(np.dot(face_n, to_cam))
    for i in (0, 1, 2):
        if facing < -0.1:
            conf[i] *= 0.15
    for i in (3, 4):
        if facing < -0.6:
            conf[i] *= 0.5
    drop = rng.random(17) < dropout
    conf[drop] = rng.uniform(0.0, 0.1, drop.sum())
    conf = np.clip(conf, 0.0, 0.99)

    vis = conf >= 0.3
    frac = vis.mean()
    if frac < 0.12 and not af.force_blob:
        return None
    seen = ~hidden
    pts = uv[conf >= 0.2] if (conf >= 0.2).sum() >= 2 else (uv[seen] if seen.sum() >= 2 else uv)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    pad = 0.08 * max(x2 - x1, y2 - y1) + 4
    bw, bh = cam.width, cam.height
    if x2 < 0 or y2 < 0 or x1 > bw or y1 > bh:
        return None
    sigma = noise * (1.2 + 0.012 * (y2 - y1))
    noisy = uv + rng.normal(0, 1, uv.shape) * sigma * np.where(vis, 1.0, 2.5)[:, None]
    if af.force_blob:
        conf = rng.uniform(0.0, 0.08, 17)
    bbox = (int(max(0, x1 - pad)), int(max(0, y1 - pad)), int(min(bw, x2 + pad)), int(min(bh, y2 + pad)))
    if bbox[2] - bbox[0] < 4 or bbox[3] - bbox[1] < 4:
        return None
    kps = [[float(noisy[i, 0]), float(noisy[i, 1]), float(conf[i])] for i in range(17)]
    det_conf = float(0.35 + 0.6 * frac) if not af.force_blob else 0.32
    return TrackedPerson(track_id, bbox, det_conf, kps, foot_point_from(bbox, kps))


# ═════════════════════════════════════════════════════════════════════════════
# Scenes
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class SimScene:
    camera: Camera
    geometry: ZoneGeometry
    frames: list[tuple[FrameDetections, list[FloatableObject] | None]]
    truth: dict[int, tuple[str, str, float | None]]  # track_id → (scenario, label, onset)
    rate_hz: float
    duration: float
    actor_of: dict[int, int] = field(default_factory=dict)  # track_id → actor (stable across id switches)


def simulate_scene(
    scenarios: Sequence[str],
    *,
    seed: int = 0,
    duration: float = 30.0,
    rate_hz: float = 8.0,
    floatable_detector: bool = True,
    noise: float = 1.0,
    miss_rate: float = 0.02,
    camera: Camera | None = None,
) -> SimScene:
    """
    Simulate several people at once (one per scenario). Returns per-frame
    FrameDetections (+ floatables when `floatable_detector`) and ground truth.
    Timestamps start at 1000.0 s so nothing depends on t == 0.
    """
    rng = np.random.default_rng(seed)
    cam = camera or Camera.random(rng)
    geom = cam.water_geometry()
    actors = [make_actor(sc, i + 1, rng, duration) for i, sc in enumerate(scenarios)]
    truth = {a.track_id: (a.scenario, a.label, a.onset) for a in actors}
    actor_of = {a.track_id: a.track_id for a in actors}
    id_map = {a.track_id: a.track_id for a in actors}
    next_id = 100
    dt = 1.0 / rate_hz
    frames: list[tuple[FrameDetections, list[FloatableObject] | None]] = []
    n = int(round(duration * rate_hz))
    t0 = 1000.0
    for k in range(n):
        t = k * dt
        people: list[TrackedPerson] = []
        floats: list[FloatableObject] = []
        for a in actors:
            af = a.step(t, dt)
            if a.id_switch_at is not None and t >= a.id_switch_at and id_map[a.track_id] == a.track_id:
                id_map[a.track_id] = next_id
                truth[next_id] = truth[a.track_id]
                actor_of[next_id] = a.track_id
                next_id += 1
            if af.floatable is not None and floatable_detector:
                c = af.floatable
                ring = np.array([[c[0] - 0.35, 0.05, c[2] - 0.35], [c[0] + 0.35, 0.05, c[2] + 0.35]])
                uv, _ = cam.project(ring)
                floats.append(
                    FloatableObject(
                        (int(min(uv[:, 0])), int(min(uv[:, 1])) - 4, int(max(uv[:, 0])), int(max(uv[:, 1])) + 4),
                        0.7,
                        "swim ring",
                    )
                )
            if rng.random() < miss_rate:
                continue
            p = observe(cam, a, af, rng, id_map[a.track_id], noise=noise)
            if p is not None:
                people.append(p)
        det = FrameDetections("sim", t0 + t, k, (cam.height, cam.width, 3), people, 1.0, "sim")
        frames.append((det, floats if floatable_detector else None))
    return SimScene(cam, geom, frames, truth, rate_hz, duration, actor_of)
