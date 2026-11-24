# Hard-negative fine-tune (v0 → v0.1): decision rule, written before training (2026-10-30)

Why: on the owner's upload (`2026-10-22-18-46-32` cut, 67 s, High, Whole frame) v0 raised 11 alerts and all were false:
7 on fixtures, flags, floor markings or glare, and 4 on people standing or wading. The old RTMDet-m raised 1 alert, but it
found only 7 tracks because it misses most swimmers. The owner asked to keep v0's recall and stop the glare and marker
boxes.

Change (data only, same D-FINE-L config):
- the v0 training labels (`seed_r1/train.json`) lose every box the owner calls "not a person" on `hardneg_r1.html`;
- 106 new train-split frames, fully decided (auto rules spot-checked by Claude: 30/30 auto negatives and 15/15 auto
  persons correct, plus the owner's answers), are added. These frames carry about 690 distractors with no box, so the model sees them
  as background;
- fine-tune from `work_dirs/dfine_l_v0/best_stg1.pth`, 20 epochs (augmentation off for the last 3), same batch and size.

Measured on the same frames as v0 (val = 50 owner-reviewed frames, never the test split). v0.1 **passes** only if all hold:
1. **Safety (person recall):** val recall@0.5 fppi ≥ v0 − 0.02 (v0 0.373) and val small_recall@0.5 fppi ≥ v0 − 0.03
   (v0 0.202).
2. **Distractors:** unmatched val boxes at score ≥ 0.5 (false positives at the app's working score) drop by ≥ 30 % vs v0.
3. **AP50 val** ≥ 0.52 (v0 0.540).
4. **Upload replay** of the owner's clip (High, Whole frame): false alerts on fixtures, flags, markings or glare go from 7 to ≤ 2, and
   tracks in the water stay ≥ 50 % of v0's. This clip is a train clip, so it is a check, not evidence.

If 1 fails, v0.1 is rejected however good 2–4 are (a person in the water must never be hidden). If 1 holds and 2 fails,
v0 stays and the next step is the label clean-up of the remaining unaudited answers. The wading-person alerts (4 of 11)
are a behaviour-rule matter (vertical torso + no progress) and are not expected to change with this detector fine-tune.

## Addendum (2026-10-30, written BEFORE re-scoring): cleaned val

The owner cleaned val (`data/coco/seed_r1/val_clean.json`: 73 non-person labels dropped, 104 unlabelled swimmers added). The
verdict on v0.1 above (FAIL on the old val, and FAIL rule 4) stands. For the next round, the same rule 1-3 is applied on
`val_clean.json` with these baselines taken from v0 on the SAME file: recall@0.5fppi >= v0 - 0.02; small_recall@0.5fppi >=
v0 - 0.03; fp/frame at score 0.5 down >= 30 %; AP50 >= v0 - 0.02. Rule 4 (owner clip replay): alerts on non-people <= 2.
The cleaned val is used only to judge; no training frame comes from it.
