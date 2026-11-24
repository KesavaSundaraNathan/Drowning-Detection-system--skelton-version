# Labelling progress

| CVAT job | Type | Frames | Status | Owner time (actual) | Hours remaining |
|---|---|---|---|---|---|
| [pool mask, job 1](http://localhost:8080/tasks/1/jobs/1) | pool_water polygon (+ optional exclude) | 3 | **completed** | (not recorded) | 0 |
| TEST A jobs 2–3 (http://localhost:8080/tasks/2) | test: boxes + class + keypoints (hand + SAM click-to-box); jobs 4–5 no longer required (owner, 2026-10-28) | 50 | new | — | ~1.5–2 h |
| TEST B jobs 6–9 (http://localhost:8080/tasks/3) | test: boxes + class (hand + SAM click-to-box); jobs 10–13 no longer required | 100 | new | — | ~1–1.3 h |
| TEST C jobs 14–19 (http://localhost:8080/tasks/4) | test: dense 10 s; owner boxes people once, SAM 2 follows (`tools\follow_test_c.cmd`), owner fixes slips | 251 | new | — | ~1 h |
| ~~SEED R1 train jobs 20–26 (task 5)~~ | superseded by CHECK R1 (renamed OLD, not used) | 200 | — | — | 0 |
| ~~SEED R1 val jobs 27–28 (task 6)~~ | superseded by CHECK R1 (renamed OLD, not used) | 50 | — | — | 0 |
| CHECK R1 train jobs 29–35 (http://localhost:8080/tasks/7) | one key per box (Attribute annotation mode) + add missed people; 4,223 boxes (1,551 preset) | 200 | new | — | ~1.6–2.4 h |
| CHECK R1 val jobs 36–37 (http://localhost:8080/tasks/8) | same, val | 50 | new | — | ~0.4–0.6 h |
