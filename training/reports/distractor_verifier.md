# Distractor verifier (owner-audited boxes only)

1064 boxes: 503 person, 561 distractor, 19 clips; leave-clips-out 5-fold CV.

Best C=0.003: **AUC 0.874**.

| threshold | distractors removed | persons kept |
|---|---|---|
| 0.3 | 63.1% | 92.2% |
| 0.4 | 72.2% | 87.7% |
| 0.5 | 79.0% | 80.5% |
| 0.6 | 86.6% | 72.4% |
| 0.7 | 90.0% | 58.1% |

AUC by C: 0.001: 0.872, 0.003: 0.874, 0.01: 0.869, 0.03: 0.861, 0.1: 0.853
