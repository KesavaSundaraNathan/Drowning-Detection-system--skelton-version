# v0.1 (hard negatives) report (2026-10-30 14:40)

Decision rule: `training/reports/hardneg_r1_rule.md` (written before training).

- checkpoint `best_stg1.pth` -> `C:\lifeguard\training\work_dirs\onnx\dfine_l_v01.onnx`; export rc 0, Life Guard decode check **PASS**

## Val (50 owner-reviewed frames, never the test split)

50 frames, 719 people (84 under 32 px tall), IoU 0.5
| model | AP50 | recall@0.1fppi | recall@0.5fppi | recall@1.0fppi | small_recall@0.1fppi | small_recall@0.5fppi | small_recall@1.0fppi | recall@s0.5 | fp_per_frame@s0.5 | class_acc |
|---|---|---|---|---|---|---|---|---|---|---|
| dfine_l_v0.onnx | 0.540 | 0.207 | 0.373 | 0.423 | 0.071 | 0.202 | 0.250 | 0.490 | 2.300 | 0.943 |
| dfine_l_v01.onnx | 0.519 | 0.228 | 0.363 | 0.391 | 0.095 | 0.190 | 0.190 | 0.458 | 2.300 | 0.951 |

## Rule

| check | v0 | v0.1 | pass |
|---|---|---|---|
| 1 safety: recall@0.5fppi >= v0 - 0.02 | 0.373 | 0.363 | yes |
| 1 safety: small_recall@0.5fppi >= v0 - 0.03 | 0.202 | 0.190 | yes |
| 2 distractors: fp/frame at score 0.5 down >= 30 % | 2.300 | 2.300 | **no** |
| 3 AP50 >= 0.52 | 0.540 | 0.519 | **no** |

**Verdict: FAIL**

## Rule 4: upload replay (owner's clip, High, Whole frame)

v0 11 alerts, v0.1 **13 alerts**: 7 on far-end window glare, 2 on lane ropes, 1 on a standing wader, the rest small boxes near the edges. Fails rule 4.

## Why val could not show a change (diagnostic, after the verdict; the verdict stands)

- 54 of 719 val labels sit on fixture spots and 52 have verifier p < 0.15 (floor T-markings, deck bags / rack, ladder, glare): a model that ignores them loses recall and AP on val.
- 73 of v0.1's 115 val 'false boxes' at score 0.5 have verifier p >= 0.6; a contact sheet shows most are real swimmers with no label.
- So val needs an owner clean-up before it can judge the next change. v0 stays the dev-only model.
