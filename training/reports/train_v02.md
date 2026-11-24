# v0.2 report (2026-10-30 17:50)

Rule: `training/reports/hardneg_r1_rule.md` addendum (written before the cleaned-val scoring). Val = `val_clean.json`.

- checkpoint `best_stg2.pth` -> `C:\lifeguard\training\work_dirs\onnx\dfine_l_v02.onnx`; export rc 0, decode check **PASS**

## Cleaned val (50 frames, owner-cleaned)

50 frames, 750 people (73 under 32 px tall), IoU 0.5
| model | AP50 | recall@0.1fppi | recall@0.5fppi | recall@1.0fppi | small_recall@0.1fppi | small_recall@0.5fppi | small_recall@1.0fppi | recall@s0.5 | fp_per_frame@s0.5 | class_acc |
|---|---|---|---|---|---|---|---|---|---|---|
| dfine_l_v0.onnx | 0.586 | 0.219 | 0.412 | 0.488 | 0.096 | 0.288 | 0.356 | 0.524 | 1.480 | 0.974 |
| dfine_l_v01.onnx | 0.593 | 0.268 | 0.457 | 0.543 | 0.137 | 0.288 | 0.397 | 0.533 | 0.880 | 0.973 |
| dfine_l_v02.onnx | 0.574 | 0.303 | 0.440 | 0.481 | 0.178 | 0.288 | 0.356 | 0.499 | 1.280 | 0.975 |

## Rule (v0.2 vs v0)

| check | v0 | v0.1 | v0.2 | pass |
|---|---|---|---|---|
| 1 recall@0.5fppi >= v0 - 0.02 | 0.412 | 0.457 | 0.440 | yes |
| 1 small_recall@0.5fppi >= v0 - 0.03 | 0.288 | 0.288 | 0.288 | yes |
| 2 fp/frame at score 0.5 down >= 30 % vs v0 | 1.480 | 0.880 | 1.280 | **no** |
| 3 AP50 >= v0 - 0.02 | 0.586 | 0.593 | 0.574 | yes |

**Verdict: FAIL**
