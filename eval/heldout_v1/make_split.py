"""Builds eval/heldout_v1: the Phase 7 clip-level dev / test split of the 16 `dev-train` clips.

FROZEN on 2026-10-08, before any Phase 7 measurement (committed on its own). Re-running must reproduce
the committed files byte for byte (tests/unit/test_heldout_split.py checks that and the rules below).

Source: the event-level ground truth of the v2 baseline (build/v2_baseline/eval/dataset_v1, written by
eval/dataset_v1/make_gt.py on 2026-10-05) and Datasets/RIGHTS.csv (incident groups, allowed use).
Videos stay in the read-only baseline copy (build/v2_baseline/Datasets, gitignored); sha256 pinned here.

Rules (in this order):
  1. Only `allowed_use = dev-train` clips (16). The 3 Shutterstock clips are `excluded` and unused.
  2. The split is by INCIDENT GROUP: every copy / cut of one incident is on the same side. Groups come from
     RIGHTS.csv, made stricter where the footage shows the same camera + incident: `active drowning (9)`
     (G08) is a close-up of the G02 lane pool ("Camera 01", same ladder) and is merged into G02.
  3. A group anything was tuned on is `dev`: G02 (= web_lane_pool_1080p), G04 (= web_garden_pool_portrait),
     G11 (= web_pool_ring_lowres), G05 (= web_slide_pool_submersion, where ID stitching and faint births
     were developed; docs: TRACKING_V3.md in the baseline).
  4. `test` is stratified by event type over the remaining groups: 2 active, 2 passive, 1 floatable and the
     only normal-swimming clip (the release gate needs 0 false alerts on normal swimmers).
  5. `test` is never trained on (D-FINE uses dev clips only) and never tuned on; it is scored for decisions
     only by the pre-registered rules in split.json["decision_rules"].

Disclosure: before the rebuild, v2-era thresholds were looked at against aggregate results over all 19
clips (v2 report dataset_v1). The test split is held out from Phase 7 on; it is not pristine before that.
Ground truth is EVENT-LEVEL (incident window + approximate victim box at keyframes, no per-frame person
boxes): event recall, time-to-alert and false alerts are measurable on test; box recall / IDF1 are not.

    python eval/heldout_v1/make_split.py            # writes annotations/, manifest.json, split.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "build" / "v2_baseline"
SRC = BASE / "eval" / "dataset_v1" / "annotations"
DATASETS = BASE / "Datasets"
VIDEO_REL = Path("..", "..", "build", "v2_baseline", "Datasets")  # relative to this eval dir

MERGED_GROUPS = {"G08": "G02"}  # rule 2
TUNED_GROUPS = {"G02", "G04", "G05", "G11"}  # rule 3
TEST_CLIPS = {  # rule 4: clip_id -> stratum
    "ds_active_drowning": "active",
    "ds_active_drowning_v7": "active",
    "ds_passive_drowning": "passive",
    "ds_passive_drowning_v3": "passive",
    "ds_floatable_drowning_v2": "floatable",
    "ds_normal_swimming": "normal",
}

DECISION_RULES = {
    "applies_to": ["ID_STITCHING", "TRACK_FAINT_BIRTHS"],
    "arms": {
        "both_on": {"ID_STITCHING": True, "TRACK_FAINT_BIRTHS": True},
        "stitching_off": {"ID_STITCHING": False, "TRACK_FAINT_BIRTHS": True},
        "faint_births_off": {"ID_STITCHING": True, "TRACK_FAINT_BIRTHS": False},
        "both_off": {"ID_STITCHING": False, "TRACK_FAINT_BIRTHS": False},
    },
    "comparison": (
        "A flag stays ON only if the arm WITH it beats the arm WITHOUT it (other flag at its decided value; "
        "evaluated both ways) on the test split by the first criterion that differs, in this order:"
    ),
    "criteria": [
        "1. test event recall (events detected / events) - higher wins",
        "2. false alerts on the normal-swimming clip - fewer wins (any > 0 also fails the release gate)",
        "3. all false alerts on test (unmatched alerts) - fewer wins",
        "4. median time-to-alert over events detected by BOTH arms - lower by >= 0.5 s wins",
        "5. victim ID changes at the labelled victim keyframes inside event windows - fewer wins",
    ],
    "tie": "A tie on every criterion is NOT a win: the flag goes OFF (it must earn its complexity).",
    "dev_split": "Dev numbers are reported for diagnosis only and never decide.",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path: Path, text: str) -> None:
    path.write_text(text + "\n", encoding="utf-8", newline="\n")  # LF on Windows too


def rights() -> dict[str, dict[str, str]]:
    with (DATASETS / "RIGHTS.csv").open(encoding="utf-8") as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def build() -> dict[str, object]:
    rows = rights()
    clips: list[dict[str, object]] = []
    out_dir = HERE / "annotations"
    out_dir.mkdir(exist_ok=True)
    for src in sorted(SRC.glob("*.json")):
        d = json.loads(src.read_text(encoding="utf-8"))
        video_name = Path(d["video"]).name
        r = rows[video_name]
        if r["allowed_use"] != "dev-train":
            continue  # rule 1
        group = MERGED_GROUPS.get(r["incident_group"], r["incident_group"])
        cid = d["clip_id"]
        split = "test" if cid in TEST_CLIPS else "dev"
        if split == "test" and group in TUNED_GROUPS:
            raise SystemExit(f"{cid}: group {group} was tuned on and cannot be test")
        d["split"] = split
        d["video"] = (VIDEO_REL / video_name).as_posix()
        _write(out_dir / src.name, json.dumps(d, indent=2, ensure_ascii=False))
        clips.append(
            {
                "clip_id": cid,
                "video": video_name,
                "video_sha256": sha256(DATASETS / video_name),
                "incident_group": group,
                "split": split,
                "stratum": TEST_CLIPS.get(cid),
                "tuned_on": group in TUNED_GROUPS,
                "events": [e["type"] for e in d.get("events", [])],
            }
        )
    groups: dict[str, set[str]] = {}
    for c in clips:
        groups.setdefault(str(c["incident_group"]), set()).add(str(c["split"]))
    leaks = {g: s for g, s in groups.items() if len(s) > 1}
    if leaks:
        raise SystemExit(f"incident groups on both sides: {leaks}")
    if len(clips) != 16 or {c["clip_id"] for c in clips if c["split"] == "test"} != set(TEST_CLIPS):
        raise SystemExit("expected 16 dev-train clips and every TEST_CLIPS entry present")
    manifest = {
        "format": "aquaguard-eval-manifest/1",
        "clips": [
            {
                "clip_id": c["clip_id"],
                "annotation": f"annotations/{c['clip_id']}.json",
                "split": c["split"],
                "labelled": True,
            }
            for c in clips
        ],
    }
    _write(HERE / "manifest.json", json.dumps(manifest, indent=2))
    split = {
        "format": "aquaguard-split/1",
        "name": "heldout_v1",
        "frozen": "2026-10-08",
        "source": "build/v2_baseline/eval/dataset_v1 (event-level GT, make_gt.py 2026-10-05) + Datasets/RIGHTS.csv",
        "counts": {
            "dev": sum(c["split"] == "dev" for c in clips),
            "test": sum(c["split"] == "test" for c in clips),
            "test_events": sum(len(c["events"]) for c in clips if c["split"] == "test"),  # type: ignore[arg-type]
        },
        "clips": clips,
        "decision_rules": DECISION_RULES,
    }
    _write(HERE / "split.json", json.dumps(split, indent=2))
    return split


if __name__ == "__main__":
    s = build()
    sys.stdout.write(json.dumps(s["counts"]) + "\n")
