"""Behaviour suite: synthetic scenes and operator-labelled recordings replayed through the exact
live logic (TrackBuffer -> BehaviorClassifier -> AlertHysteresis / SceneBehaviorTracker).

[v3] The evaluation half of the v2 baseline detection/behavior_calibration.py, verbatim except
load_recordings (an explicit directory; unreadable files are logged). The calibration search
(calibrate / apply_calibration / async / CLI) is NOT ported: v2 wrote its result to an override
file the classifier merged at run time; in v3 a calibrated change is a new, reviewed
thresholds.json version (Phase 7 accuracy workstream). save_recording / label_recording belong
to alert persistence (Phase 5).
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lifeguard.eval.simulation import (
    BYSTANDER_SCENARIOS,
    DROWNING_SCENARIO_NAMES,
    NORMAL_SCENARIOS,
    SCENARIO_LABELS,
    SimScene,
    simulate_scene,
)
from lifeguard.vision.behaviour.classifier import SWIMMER, BehaviorClassifier, thresholds_source
from lifeguard.vision.behaviour.engine import FrameSample, TrackBuffer
from lifeguard.vision.constants import BEHAVIOR_SCORE_INTERVAL_S
from lifeguard.vision.scene.scene_tracker import AlertHysteresis, SceneBehaviorTracker

logger = logging.getLogger(__name__)


# Synthetic scenes are simulated at a CCTV frame rate and ingested frame by
# frame, exactly like the live pipeline (which pools every frame into
# BEHAVIOR_SAMPLE_HZ behaviour samples and scores every BEHAVIOR_SCORE_INTERVAL_S).
SIM_FRAME_RATE_HZ = 25.0


def default_rate_hz(mode: str = "Medium") -> float:
    """Frame rate synthetic scenes are simulated at (every frame is analysed; modes only change thresholds)."""
    return SIM_FRAME_RATE_HZ


NEGATIVE_LABELS = {"NORMAL_SWIMMING", "RESTING", "BYSTANDER"}
ParamsFn = Callable[[str], dict[str, Any]]


# ═════════════════════════════════════════════════════════════════════════════
# Running a scene
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class SceneRun:
    scene: SimScene
    fires: list[tuple[float, int, str, bool]]  # (t since start, track_id, scenario, ghost)
    scored_bystander: bool = False  # a deck bystander was ever risk-scored (must never happen)


def run_scene(
    scene: SimScene, mode: str = "Medium", params_fn: ParamsFn | None = None, score_every: int | None = None
) -> SceneRun:
    """
    Drive a scene through the live logic. Scoring runs on the live cadence
    (every BEHAVIOR_SCORE_INTERVAL_S of scene time) unless `score_every`
    (frames) is given.
    """
    tracker = SceneBehaviorTracker("calibration", mode, BehaviorClassifier(params_fn=params_fn))
    tracker.set_geometry(scene.geometry)
    t0 = scene.frames[0][0].timestamp if scene.frames else 0.0
    fires: list[tuple[float, int, str, bool]] = []
    scored_bystander = False
    last_score = -1e18
    for k, (det, floats) in enumerate(scene.frames):
        tracker.ingest(det, floats)
        due = (
            (k % score_every == 0) if score_every else (det.timestamp - last_score >= BEHAVIOR_SCORE_INTERVAL_S - 1e-6)
        )
        if due:
            last_score = det.timestamp
            for f in tracker.score(det.timestamp):
                fires.append((det.timestamp - t0, f.track_id, f.result.scenario, f.ghost))
            for ts in tracker.tracks.values():
                truth = scene.truth.get(ts.track_id)
                if truth and truth[0] in BYSTANDER_SCENARIOS and ts.result is not None and ts.result.is_drowning:
                    scored_bystander = True
        tracker.expire(det.timestamp)
    return SceneRun(scene, fires, scored_bystander)


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation on synthetic scenes
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class ScenarioStats:
    scenario: str
    label: str
    runs: int = 0
    alerts: int = 0  # normal → false alerts; drowning → detections after onset
    correct_type: int = 0
    early: int = 0  # drowning scenario alerted BEFORE the behaviour began (false)
    latencies: list[float] = field(default_factory=list)
    types: dict[str, int] = field(default_factory=dict)

    @property
    def is_negative(self) -> bool:
        return self.label in NEGATIVE_LABELS

    def to_dict(self) -> dict[str, Any]:
        lat = [round(x, 2) for x in self.latencies]
        d = {
            "scenario": self.scenario,
            "label": self.label,
            "runs": self.runs,
            "negative": self.is_negative,
            "types": self.types,
        }
        if self.is_negative:
            d.update(false_alerts=self.alerts, false_alert_rate=round(self.alerts / max(self.runs, 1), 4))
        else:
            d.update(
                detected=self.alerts,
                recall=round(self.alerts / max(self.runs, 1), 4),
                correct_type=self.correct_type,
                early_alerts=self.early,
                median_latency_s=round(float(np.median(lat)), 2) if lat else None,
                max_latency_s=round(max(lat), 2) if lat else None,
            )
        return d


def _score_run(run: SceneRun, stats: dict[str, ScenarioStats]) -> None:
    by_track: dict[int, list[tuple[float, int, str, bool]]] = {}
    for f in run.fires:
        by_track.setdefault(f[1], []).append(f)
    # ground truth is keyed by every id the actor ever had (tracker id switches)
    actors: dict[int, list[int]] = {}
    for tid in run.scene.truth:
        actors.setdefault(run.scene.actor_of.get(tid, tid), []).append(tid)
    for actor, ids in actors.items():
        scen, _label, onset = run.scene.truth[actor]
        st = stats.setdefault(scen, ScenarioStats(scen, SCENARIO_LABELS[scen]))
        st.runs += 1
        fired = sorted((f for tid in ids for f in by_track.get(tid, [])), key=lambda f: f[0])
        if st.is_negative:
            if fired:
                st.alerts += 1
                st.types[fired[0][2]] = st.types.get(fired[0][2], 0) + 1
            continue
        early = [f for f in fired if onset is not None and f[0] < onset - 0.5]
        good = [f for f in fired if onset is None or f[0] >= onset - 0.5]
        if early:
            st.early += 1
        if good:
            st.alerts += 1
            st.latencies.append(good[0][0] - (onset or 0.0))
            st.types[good[0][2]] = st.types.get(good[0][2], 0) + 1
            if good[0][2] == SCENARIO_LABELS[scen]:
                st.correct_type += 1


def _summary(stats: dict[str, ScenarioStats]) -> dict[str, Any]:
    neg = [s for s in stats.values() if s.is_negative]
    pos = [s for s in stats.values() if not s.is_negative]
    lat = [x for s in pos for x in s.latencies]
    n_neg, n_pos = sum(s.runs for s in neg), sum(s.runs for s in pos)
    fa, det = sum(s.alerts for s in neg), sum(s.alerts for s in pos)
    return {
        "normal_runs": n_neg,
        "false_alerts": fa,
        "false_alert_rate": round(fa / max(n_neg, 1), 4),
        "drowning_runs": n_pos,
        "detected": det,
        "recall": round(det / max(n_pos, 1), 4),
        "correct_type_rate": round(sum(s.correct_type for s in pos) / max(det, 1), 4),
        "early_alerts": sum(s.early for s in pos),
        "median_latency_s": round(float(np.median(lat)), 2) if lat else None,
        "p90_latency_s": round(float(np.percentile(lat, 90)), 2) if lat else None,
    }


class SceneCache:
    """Simulated scenes are deterministic per (scenarios, seed) — simulate once, replay many."""

    def __init__(self) -> None:
        self._scenes: dict[tuple[tuple[str, ...], int, float, float], SimScene] = {}

    def get(self, scenarios: Sequence[str], seed: int, duration: float, rate_hz: float) -> SimScene:
        key = (tuple(scenarios), seed, duration, rate_hz)
        if key not in self._scenes:
            self._scenes[key] = simulate_scene(scenarios, seed=seed, duration=duration, rate_hz=rate_hz)
        return self._scenes[key]


def evaluate(
    mode: str = "Medium",
    seeds: Iterable[int] = range(10),
    scenarios: Sequence[str] | None = None,
    crowd_scenes: int = 0,
    duration: float = 30.0,
    rate_hz: float | None = None,
    params_fn: ParamsFn | None = None,
    cache: SceneCache | None = None,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """
    Evaluate on synthetic scenes: one person per scene for every scenario and
    seed, plus `crowd_scenes` crowded pools (6–9 people, mixed behaviour,
    occlusions). Returns per-scenario and summary metrics.
    """
    rate = rate_hz or default_rate_hz(mode)
    names = list(scenarios or SCENARIO_LABELS.keys())
    seeds = list(seeds)
    cache = cache or SceneCache()
    stats: dict[str, ScenarioStats] = {}
    bystander_scored = 0
    total = len(names) * len(seeds) + crowd_scenes
    done = 0
    t_start = time.time()
    for name in names:
        for seed in seeds:
            run = run_scene(cache.get([name], seed, duration, rate), mode, params_fn)
            _score_run(run, stats)
            bystander_scored += int(run.scored_bystander)
            done += 1
            if progress:
                progress(done / max(total, 1))
    crowd_stats: dict[str, ScenarioStats] = {}
    for c in range(crowd_scenes):
        rng = np.random.default_rng(90_000 + c)
        n = int(rng.integers(6, 10))
        pool = list(NORMAL_SCENARIOS) + list(BYSTANDER_SCENARIOS)
        mix = [str(rng.choice(pool)) for _ in range(n)]
        if rng.random() < 0.7:
            mix[int(rng.integers(0, n))] = str(rng.choice(DROWNING_SCENARIO_NAMES))
        run = run_scene(cache.get(mix, 90_000 + c, duration, rate), mode, params_fn)
        _score_run(run, crowd_stats)
        bystander_scored += int(run.scored_bystander)
        done += 1
        if progress:
            progress(done / max(total, 1))
    return {
        "mode": mode,
        "source": "simulation",
        "seeds": len(seeds),
        "duration_s": duration,
        "rate_hz": rate,
        "scenarios": {k: v.to_dict() for k, v in stats.items()},
        "summary": _summary(stats),
        "crowd": {
            "scenes": crowd_scenes,
            "summary": _summary(crowd_stats),
            "scenarios": {k: v.to_dict() for k, v in crowd_stats.items()},
        }
        if crowd_scenes
        else None,
        "bystanders_scored": bystander_scored,
        "thresholds": thresholds_source(),
        "elapsed_s": round(time.time() - t_start, 1),
    }


def load_recordings(directory: Path | None = None, labelled_only: bool = True) -> list[dict[str, Any]]:
    d = directory
    out: list[dict[str, Any]] = []
    if d is None or not d.exists():
        return out
    for p in sorted(d.glob("alert_*.json")):
        try:
            data = json.loads(p.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("[suite] unreadable recording %s: %s", p, exc)
            continue
        if labelled_only and data.get("label") not in ("drowning", "false_alarm"):
            continue
        out.append(data)
    return out


def replay_recording(
    rec: dict[str, Any], params_fn: ParamsFn | None = None, score_every: int | None = None
) -> str | None:
    """Replay a recording through the live logic; returns the scenario that fired (or None)."""
    clf = BehaviorClassifier(params_fn=params_fn)
    mode = rec.get("mode", "Medium")
    params = clf.params(mode)
    buf = TrackBuffer(int(rec.get("track_id", 0) or 0))
    hyst = AlertHysteresis()
    last_score = -1e18
    for i, s in enumerate(rec.get("samples", [])):
        buf.add(
            FrameSample(
                t=float(s["t"]),
                bbox=tuple(s["bbox"]),
                conf=float(s.get("conf", 1.0)),
                keypoints=s.get("kp"),
                in_water=bool(s.get("in_water", True)),
                edge_dist_px=float(s["edge"]) if s.get("edge") is not None else math.inf,
                floatable_dist_px=float(s["float"]) if s.get("float") is not None else None,
            )
        )
        t_s = float(s["t"])
        if score_every:
            if i % score_every:
                continue
        elif t_s - last_score < BEHAVIOR_SCORE_INTERVAL_S - 1e-6:
            continue
        last_score = t_s
        res = clf.classify(
            buf,
            category=SWIMMER,
            mode=mode,
            sensitivity_multiplier=float(rec.get("sensitivity_multiplier", 1.0)),
            zone_type=rec.get("zone_type"),
        )
        if hyst.update(res, params, buf.last_t or 0.0):
            return res.scenario
    return None


def evaluate_recordings(recs: Sequence[dict[str, Any]], params_fn: ParamsFn | None = None) -> dict[str, Any]:
    tp = fn = fp = tn = 0
    for rec in recs:
        fired = replay_recording(rec, params_fn) is not None
        if rec.get("label") == "drowning":
            tp += fired
            fn += not fired
        else:
            fp += fired
            tn += not fired
    return {
        "recordings": len(recs),
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
    }
