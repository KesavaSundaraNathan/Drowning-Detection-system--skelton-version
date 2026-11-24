# Temporal check, dense segments (Phase 4)

Input: `labels\auto\dense_consensus.jsonl` (12,803 frames, 19 clips, 25 fps; consensus without SAM).
Boxes linked: 89,480 accepted + 59,869 confident lone-voter boxes.

| Check | Count (accepted boxes) |
|---|---|
| singleton (1 frame only) | 1,638 |
| jump (linked only by centre distance) | 1,739 |
| size change > 40 % between linked frames | 4,133 |
| gap frames bridged (1–2 missing frames inside a track; interpolation candidates) | 16,647 |
| frames with at least one non-gap flag | 6,520 of 12,803 |

Tracks with accepted boxes: 7,728; median length 5 frames
(p90 41).
Lone-voter tracks lasting >= 1 s: **174** (9,740 boxes); median lone
track length 1 frames. A box that one model keeps for a second or more is more
likely a real person the other voters miss than noise (not checked by eye), so these are review candidates.
Short accepted tracks (median above) and the many bridged gaps show that the COCO voters' agreement flickers from frame
to frame: per-frame consensus alone under-labels the dense segments.

Use: these flags go into the dense review / self-training round (Phase 7) once v0 replaces the COCO voters; no
dense labels are used for v0 (option A trains on the reviewed seed).
