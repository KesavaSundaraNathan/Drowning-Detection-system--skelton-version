# Final test evaluation on AUTO-LABELLED test frames (2026-10-30)

300 frames (TEST A + B), 1,271 auto boxes (56 under 32 px tall). Labels are NOT hand labels: COCO D-FINE-L + COCO RTMDet-L + OWLv2 voters plus the distractor verifier (`tools/test_autolabel.py`, owner decision 2026-10-30). Run once, rule written before the run (STATUS.md): a v0 model beats the app's RTMDet-m if recall@0.5 false positives per frame is >= 5 points higher AND AP50 is not lower.

| model | AP50 | recall@0.1fppi | recall@0.5fppi | recall@1.0fppi | small_recall@0.5fppi | class_acc |
|---|---|---|---|---|---|---|
| RTMDet-m COCO (app default) | 0.406 | 0.156 | 0.249 | 0.317 | 0.071 | n/a |
| D-FINE-M dev-only | 0.429 | 0.163 | 0.328 | 0.398 | 0.054 | 0.606 |
| **D-FINE-L v0** | **0.646** | 0.183 | **0.509** | **0.646** | 0.071 | 0.988 |
| **RTMDet-L v0** | 0.631 | 0.081 | 0.371 | 0.566 | 0.125 | 0.986 |

Rule result: D-FINE-L v0 (+26 points recall@0.5, AP50 +0.24) and RTMDet-L v0 (+12 points, AP50 +0.23) both pass.

Caveats (read before trusting it):
- Auto labels come from COCO detectors. D-FINE-L v0 is fine-tuned from the COCO D-FINE-L that is one of the voters, so the labels favour it; the RTMDet-m baseline is favoured less. A hand-labelled test is still needed for a claim you can rely on.
- The auto labels miss small and submerged swimmers (only 56 small boxes), so the small-person numbers mean little.
- RTMDet-L v0 is worse than RTMDet-m at a very low false-alarm rate (recall@0.1fppi 0.081 vs 0.156).
- v0 was trained on partly unaudited labels (see STATUS).
- Class accuracy is measured against a pool-mask rule, not hand classes.
Nothing is registered in MODELS.json.
