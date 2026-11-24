# Phase 4, round 1: consensus of the COCO voters — 2026-10-28

Inputs: `training\labels\raw_voters\pool.jsonl` (4,664 train/val pool frames; D-FINE-L and RTMDet-L with 2×2 tiles,
OWLv2 full frame; 56 min). Consensus: `training\tools\consensus.py` → `training\labels\auto\pool_consensus.jsonl`.

## Provisional voter thresholds (no labels yet)
Set where each voter yields ~9–10 boxes per frame (the scene has ~7–8 people per frame):
D-FINE ≥ 0.35 (9.8 / frame), RTMDet ≥ 0.18, OWLv2 ≥ 0.22. Score medians: D-FINE 0.13, RTMDet 0.08, OWLv2 0.08.

## Result
| | |
|---|---|
| Instances accepted (≥ 2 of 3 voters, IoU ≥ 0.6) | **27,923** (~6.0 per frame) |
| SAM 2 refinement kept (IoU ≥ 0.7 with the consensus box) | 26,742 (96 %); 1,181 kept the consensus box |
| Lone-voter boxes above the voting threshold | D-FINE 4.4 / frame (median score **0.41**), RTMDet 4.6 (0.22), OWLv2 5.2 (0.27) |
| Frames with no flag (auto-accepted) | **13 of 4,664** with the prompt's rules as written |

Review-queue size for stricter "lone voter is confident" bars (crowded-overlap flag on / off):
| Lone-voter bar (D-FINE / RTMDet / OWLv2) | Auto-accepted frames | Frames to review |
|---|---|---|
| 0.50 / 0.35 / 0.35 | 512 / 731 | 4,152 / 3,933 |
| 0.60 / 0.40 / 0.40 | 850 / 1,439 | 3,814 / 3,225 |
| 0.70 / 0.50 / 0.45 | 1,062 / 2,037 | 3,602 / 2,627 |

## What this means
- The three COCO voters disagree because two of them are weak on swimmers (RTMDet and OWLv2 score them low), not
  because the frames are ambiguous. Many lone D-FINE boxes are probably real swimmers.
- Auto-accepting frames with confident lone boxes would turn **missed swimmers into "background"** in training: the
  worst error for a drowning system. Raising the bar until frames pass hides exactly those.
- Reviewing every flagged frame is 2,600–3,800 frames: ~40–60 h of owner time, versus the prompt's 2–3 h.

## Options (owner decision)
**A. Seed, then self-train (recommended).** Review a compact, diverse **seed of ~250 frames** (hardest + DINOv2-diverse,
~4–5 h in 30-frame jobs, SAM click-to-box helps), fully corrected. Train a v0 detector on the seed (+ only frames whose
every lone box is below a low bar). v0 is pool-specific, so it replaces the weak COCO voters in round 2 (the prompt's
Phase 7 mechanism, started earlier); consensus then needs far less review. The 5 % audit measures the error rate each
round, as the prompt requires.
**B. Prompt as written.** Review queue of ~2,600+ frames, hardest first; train when 80 % is done. Weeks of review.
**C. Lower the bar.** Accept D-FINE alone above a score (e.g. 0.5) as a label. Fast, but it trusts one COCO model's
errors and breaks the 2-of-3 rule; not recommended for a safety system.
