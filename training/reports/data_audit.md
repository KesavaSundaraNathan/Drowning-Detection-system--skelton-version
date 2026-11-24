# Phase 1: data audit and split — 2026-10-28

Script: `training\tools\audit_clips.py` → `training\data\audit.json` (numbers only; no frame viewed). Crowd = median /
max person count from COCO D-FINE-M (score > 0.4) on one frame every 10 s; this untuned model undercounts swimmers, so
it is used only to balance the split. Foreground = MOG2 moving fraction at 1 fps. Brightness = mean V (0–255).

## Clips (23 unique; all HEVC 960×1072, 25 fps — no format flags)
| Clip (2026-10-22-…) | Min | Persons median / max | Foreground | Brightness | Split |
|---|---|---|---|---|---|
| 18-14-59 | 5.0 | 7 / 13 | 0.093 | 125.7 | train |
| 18-17-57 | 5.0 | 11 / 17 | 0.069 | 100.7 | train |
| 18-22-30 | 5.0 | 7 / 13 | 0.107 | 120.8 | **test** |
| 18-27-02 | 5.0 | 12 / 20 | 0.082 | 107.2 | **test** |
| 18-31-55 | 5.0 | 7 / 12 | 0.093 | 121.8 | train |
| 18-33-19 | 5.0 | 12 / 19 | 0.071 | 115.2 | train |
| 18-34-34 | 5.0 | 7 / 19 | 0.098 | 117.7 | *val* |
| 18-35-10 | 5.0 | 11 / 15 | 0.086 | 117.3 | *val* |
| 18-36-19 | 5.0 | 6 / 12 | 0.082 | 128.2 | train |
| 18-36-59 | 5.0 | 7 / 15 | 0.076 | 102.2 | **test** |
| 18-38-23 | 5.0 | 7 / 16 | 0.084 | 127.1 | train |
| 18-39-02 (+ byte-identical "(1)" copy, excluded) | 5.0 | 9 / 20 | 0.059 | 116.9 | train |
| 18-42-47 | 10.0 | 8 / 18 | 0.096 | 108.7 | train |
| 18-44-07 | 10.0 | 4 / 10 | 0.046 | 108.4 | train |
| 18-45-30 | 10.0 | 3 / 9 | 0.058 | 113.2 | train |
| 18-46-32 | 10.0 | 10 / 17 | 0.039 | 105.2 | train |
| 18-49-23 | 10.0 | 4 / 9 | 0.064 | 134.2 | train |
| 18-50-32 | 10.0 | 11 / 19 | 0.067 | 125.1 | train |
| 19-07-45 | 5.0 | 9 / 14 | 0.062 | 126.0 | train |
| 19-08-35 | 5.0 | 1 / 9 | 0.050 | 130.3 | train (mostly empty: hard negatives) |
| 19-11-55 | 5.0 | 10 / 16 | 0.040 | 123.4 | train |
| 19-12-41 | 5.0 | 5 / 10 | 0.055 | 130.1 | **test** |
| 19-13-23 | 5.0 | 4 / 10 | 0.044 | 130.5 | *val* |

The prompt expected 20 clips × 5 min; there are 23 unique (7 × 10 min). Note: file-name times overlap (e.g. 18:42:47
+ 10 min and 18:44:07), but frame content at aligned times differs, so these are separate recordings or the names are
export times. All are one session of one evening: **the same swimmers appear in train, val and test**, so the test set
measures generalisation to new moments, not to new people or new days. Recordings from another day would be a stronger
test.

## Proposed split (`training\data\splits.json`, clip-level)
| Split | Clips | Minutes | Mean crowd (range) | Mean brightness (range) |
|---|---|---|---|---|
| train | 16 | 110.2 | 7.4 (1–12) | 119.4 (100.7–134.2) |
| val | 3 | 15.0 | 7.3 (4–11) | 121.8 (117.3–130.5) |
| test | 4 | 20.1 | 7.8 (5–12) | 115.1 (102.2–130.1) |

Rules: no clip in two splits; the 10-minute clips stay in train; val and test each have a low, a middle and a high crowd
clip; test includes the darkest-but-one clip (18-36-59). 16 / 3 / 4 instead of the prompt's 14 / 3 / 3 because there
are 23 clips (owner-agreed).

## Pool mask
CVAT job **http://localhost:8080/tasks/1/jobs/1** (project "Life Guard pool mask"): 3 frames (early, middle, late). Draw
one `pool_water` polygon around the water surface on the first frame (lane ropes inside it), check it fits frames 2
and 3, optionally an `exclude` polygon for areas never to label (screens, posters), then mark the job **Completed**.
Used for `swimmer` vs `person`: any body part in the water → `swimmer`.
