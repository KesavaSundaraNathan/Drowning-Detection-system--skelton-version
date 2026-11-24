# Phase 2: frame sampling — 2026-10-28

Scripts: `training\tools\extract_frames.py` (queue job 002, 10 min) → `training\data\frames\<clip>\` (2 fps JPEG,
pHash, DINOv2-base embeddings; 4.2 GB, not in git); `training\tools\sample_frames.py` → `training\data\sampling.json`.
Split: `training\data\splits.json` (approved, GATE 1). No frame was viewed.

## Train / val labelling pool (Option B input)
| | Extracted (2 fps) | After near-duplicate removal | Selected | 0 persons* | 1–3 | 4–8 | 9+ |
|---|---|---|---|---|---|---|---|
| train (16 clips) | 13,232 | 8,445 | **3,975** | 150 | 1,275 | 1,275 | 1,275 |
| val (3 clips) | 1,805 | 1,149 | **689** | 14 | 165 | 255 | 255 |
| **total** | | | **4,664** | 164 (3.5 %) | | | |

\* person counts from COCO D-FINE-M (score > 0.4), used **only** to stratify; this untuned model undercounts swimmers, so
"0 persons" frames are likely hard cases (missed swimmers) as much as true negatives.

Method: within each (split, stratum), DINOv2 k-means; frames picked round-robin over clusters so every look of the
scene is represented. Seed 20261028 (reproducible).

**Deviation — near-duplicates:** whole-frame pHash cannot separate frames of this fixed camera (median Hamming 2 at
0.5 s, 4 at 60 s; the prompt's pHash-only rule removed 97 % of frames). A frame is dropped only if it is < 2 s after
the last kept frame **and** DINOv2 cosine ≥ 0.975 **and** pHash ≤ 2 (36 % removed).

**Shortfall — hard negatives:** target ~10 %, got 3.5 %: the pool is busy all evening (only 164 frames with no
detected person). If more true negatives are needed, the empty-pool moments of another recording are the fix.

## Dense 10-s segments (for tracking / temporal consistency)
51 segments: 3 per train clip, 1 per val clip, placed at different crowd levels of each clip (count quantiles), never
within 20 s of each other. All 25 fps frames of a segment are used (≈ 250 frames each) via propagation in Phase 4.

## Test set (hand labelling only; no detector or automatic labeller was run on test frames)
- **300 frames** from the 4 test clips: 2,416 extracted → 1,939 after near-duplicate removal → 300 by DINOv2 k-means.
- **One dense 10-s segment:** `2026-10-22-18-22-30.mp4`, 137.5–147.5 s (frames 3436–3686), the test clip with the
  widest embedding spread.
- Labelling method: SAM 2 click-to-box only (GATE after Phase 3), keypoints on ~100 of the 300.

## Timing
Extraction 10 min (est. 8), counts + sampling 5 min (est. 15). Logged in `timings.md`.
