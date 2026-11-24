# Training status

Updated: 2026-10-29 01:40

| | |
|---|---|
| Phase | 4 (auto-label method, owner 2026-10-28): seed CHECK in CVAT tasks 7–8 (one key per box, ≈ 2–3 h of owner time); v0 pipeline starts automatically at 8 of 9 jobs finished |
| Running job | none (dense voters done 18:45; benchmark done 19:05). Next, queued automatically at 80 % of the seed Completed: export (1 min) → D-FINE-L (≈ 4 h) → RTMDet-L (≈ 2–2.5 h) → ONNX export + Life Guard check + val eval (≈ 15 min) → `reports\train_v0.md` + desktop alert. ≈ 6.5 h in total, no pause |
| Latest metrics | Phase 0 smoke tests (below) |
| Open interventions | seed CHECK (tasks 7–8), then the reduced test set (TEST C 14–19 with SAM 2 follow, TEST B 6–9, TEST A 2–3) |
| Live system | stopped by the owner (2026-10-28) until testing and training finish; training runs at full GPU speed |

## Phase 0 progress
| Item | State |
|---|---|
| Folders, MASTER_PROMPT.md copy, status files, `.gitignore` entries | done |
| `tools\notify.ps1` | done, tested (balloon + beep + `logs\notify.log`) |
| Claude Code Notification hook + pre-approved commands | **owner to add** (Claude may not edit its own settings) |
| Queue (`tools\queue\run_queue.ps1`, `jobqueue.py`) + watchdog (`tools\watchdog.ps1`) | done; tested: crash of runner + job → resumed from the checkpoint; always-failing job → failed after 2 restarts + alert; overheat pause → resumed from the checkpoint |
| Watchdog task `LifeGuard-Training-Watchdog` (log-on + every 5 min) | registered |
| Power plan High performance, no sleep / hibernate; Windows Update active hours 06–24; firewall TCP 8080 Private | done (elevated script, `logs\phase0_admin.log`) |
| CUDA 12.8 toolkit (was already installed), VS 2022 Build Tools (C++), WSL2, Docker Desktop | installed; restarted 2026-10-28 |
| `training\.venv`: torch 2.7.1+cu128, MMCV 2.1.0 (compiled, sm_120), MMEngine 0.10.7, MMDetection 3.3.0, MMPose 1.3.2, SAM 2, transformers 4.57.1 | done; `requirements.lock` |
| Weights (COCO-only detectors, SAM 2.1 b+, OWLv2, DINOv2) | downloaded; `docs\weights_manifest.txt`, `docs\LICENSES.md`; RTMPose held (licence) |
| CVAT | v2.80.0 at http://localhost:8080 (this PC only), project 1 schema done |

## Smoke tests (one 960×1072 frame of `2026-10-22-18-36-19.mp4`, RTX 5060, numbers only)
| Model | ms / frame | Peak VRAM | Output |
|---|---|---|---|
| D-FINE-L COCO @ 640 / 1024 | 25.8 / 42.1 | 0.4 GB | 9 / 6 persons > 0.4 |
| D-FINE-M COCO @ 640 / 1024 | 16.1 / 28.5 | 0.3 GB | 13 / 7 persons > 0.4 |
| RTMDet-L COCO @ 640 | 57.2 (incl. pre/post-processing) | 0.3 GB | 0 persons > 0.4; best person 0.27 (low swimmer scores, as in the live system) |
| SAM 2.1 hiera-b+ | 48.4 embed + 5.6 per box | 0.8 GB | mask 1072×960 |
| OWLv2 base ensemble | 294 | 1.9 GB | 22 boxes > 0.2 for 3 prompts |
| DINOv2 base | 12.8 | 1.0 GB | 768-d embedding |

Notes: D-FINE needs one model per input size (its decoder anchors are size-specific). The RTMDet checkpoint was
converted to a plain state dict with a narrow weights-only allowlist (`rtmdet_l_coco_statedict.pth`).

## Dataset
- Folder: `training_dataset\` (read-only originals). 24 files, **23 unique**: `2026-10-22-18-39-02 (1).mp4` is
  byte-identical to `2026-10-22-18-39-02.mp4` and is excluded.
- 16 clips × ~5 min, 7 clips × ~10 min (≈ 145 min). HEVC, 960×1072, 25 fps. One camera (owner, 2026-10-28).

## Resume commands for running jobs
- 005_voters_dense_r2: automatic (queue + watchdog; finished frames are skipped). By hand: `powershell -ExecutionPolicy Bypass -File training\tools\queue\run_queue.ps1`

## Phase 6 (v0) pieces, built 2026-10-28
| Piece | File | Checked |
|---|---|---|
| CVAT seed → COCO export + QA (blocking: `candidate` left, zero-area box; reported: tiny, duplicate IoU > 0.9, class vs pool mask) | `tools\export_seed.py` | run on all 9 unreviewed jobs (`--all-jobs`): 400 train entries (200 frames x2), 50 val frames |
| D-FINE-L v0 config: 2 classes, 1024, AMP + EMA, COCO weights (`-t`), 60 epochs, augmentation + multi-scale off for the last 9 (15 %), batch 2, last.pth every epoch | `configs\dfine_l_v0.yml` | — |
| Wrapper: pool photometric augmentation (HSV / brightness, glare, motion blur, noise, JPEG; no geometric warps), seed 0, `--resume` from last.pth | `tools\train_dfine.py` | CPU smoke: 1 epoch train + val on 8 / 4 frames, then resume into epoch 2 |
| Trigger (watchdog, every 5 min): >= 80 % of the seed jobs Completed and >= 1 val job → queue export + training once | `tools\seed_trigger.py` | dry run: 0/9 Completed |
| GPU speed / VRAM at 1024 | job 006 + 40-frame runs | batch 4: peak 10.4 GB > 8 GB, spilled to shared memory, 11.6 s/step (an epoch 19 min, 60 epochs ~20 h): rejected. **Batch 2: peak 5.4 GB, ~1.2 s/step → ~4 min per epoch, 60 epochs ≈ 4 h** (config set to 2). D-FINE-M not needed |
| RTMDet-L v0 config: 2 classes, 1024, AMP + EMA, COCO weights, 60 epochs, backbone frozen for 6 (10 %) then 0.1x LR, mosaic off for the last 9 (15 %), empty frames kept, same photometric augmentation, batch 2. **No copy-paste**: it needs instance masks and the seed has boxes only | `configs\rtmdet_l_v0.py`, `tools\train_rtmdet.py` | GPU, 40 frames: frozen batch 4 3.1 GB; unfrozen batch 4 7.7 GB at ~5 s/step (spills, rejected); **unfrozen batch 2 4.3 GB**; resume from the epoch checkpoint works (checkpoints keep no logging history, so they load weights-only) |
| Trigger queues: export → D-FINE-L → RTMDet-L → post_v0 | `tools\seed_trigger.py` | dry run |
| Dense temporal check (Phase 4) | `tools\temporal_check.py` → `reports\temporal_check_dense.md` | run: 12,803 frames; 1,638 one-frame boxes, 1,739 jumps, 4,133 size jumps, 16,647 gap frames; 174 single-model tracks >= 1 s |
| ONNX export in the app's two layouts (D-FINE `detr_logits`, RTMDet `xyxy_scores`), classes `swimmer` / `person_out_of_water` so the app reads the in-water vote | `tools\export_v0.py`, `tools\verify_v0_onnx.py` | stand-in checkpoints: ORT = PyTorch (D-FINE worst query 3.9e-4; RTMDet 5.7e-2 px), batch 2 = batch 1, Life Guard loader + decoder PASS for both |
| Val evaluation through Life Guard's decoder: AP50, recall at 0.1 / 0.5 / 1 FP per frame, small-person recall, class accuracy; test split refused without `--final` | `tools\eval_v0.py` | run on unreviewed val pre-labels (plumbing only, not a result) |
| Post-training job: best checkpoints → ONNX → check → val eval vs the app's RTMDet-m and D-FINE-M → `reports\train_v0.md` | `tools\post_v0.py` | full run on stand-ins: both PASS, report written |
| `onnx==1.23.2` (+ ml-dtypes 0.6.0) added to the training venv for the export | `requirements.lock` | — |

## Auto-label method (owner-approved 2026-10-28), built 2026-10-29
| Piece | File | Checked |
|---|---|---|
| Grounding DINO base (Apache-2.0) as a 4th voter: 50-frame trial on val seed frames | `tools\gdino_voter.py` | at 0.15: confirms 59 % of 2-of-3 boxes, 32 % of single-detector boxes, finds no person the others all missed → **dropped** (not run on all frames) |
| FiftyOne crop review | — | not used: bundles MongoDB (SSPL), outside the MIT / Apache rule. CVAT Attribute Annotation Mode does the same box-by-box check |
| Seed CHECK tasks: every proposal a `proposal` box with `verdict` (? / swimmer / person / not_a_person); 2-of-3 boxes preset from the pool mask; single-detector boxes (D-FINE >= 0.30, RTMDet >= 0.20, OWLv2 >= 0.20, de-cluttered at IoU 0.4) on "?" | `tools\check_r1.py` → tasks 7 (jobs 29–35, 4,223 boxes, 1,551 preset) and 8 (jobs 36–37, 1,069 boxes, 394 preset); old seed tasks 5–6 renamed OLD, nothing deleted | created |
| Export reads verdicts; a box left on "?" blocks its job; the trigger watches CHECK R1 | `tools\export_seed.py`, `tools\seed_trigger.py` | smoke export + dry run |
| TEST C: SAM 2.1 follows the owner's hand-drawn boxes (only `manual` shapes are accepted as prompts) and writes CVAT tracks | `tools\sam2_follow.py`, `tools\follow_test_c.cmd` | scratch task on 60 train frames: 2 tracks over 60 frames, max 31 px/frame; scratch deleted |
| Test set reduced (owner): TEST A jobs 2–3, TEST B jobs 6–9, TEST C 14–19 | `LABELING_PROGRESS.md` | — |
| Attribute annotation mode, checked 2026-10-29 in the built-in browser (owner logged in) on a synthetic grey-image scratch task (no footage; deleted after): keys **0–3** (0 ?, 1 swimmer, 2 person, 3 not_a_person), Tab cycles the boxes of one frame, F next frame, Ctrl+S saved the verdict to the server, Menu → Finish the job → Continue sets the state to completed | `docs\OWNER_STEPS.md` corrected (it said 1–4) | verified |

## Claude checks the seed "?" boxes (owner-approved 2026-10-29: crops of the 250 seed frames may go to Anthropic; never the test set; 10 % owner audit)
| Piece | File | Result |
|---|---|---|
| Whole-frame images with the "?" boxes numbered (tiles were tried first: too little context to judge) | `tools\claude_check_sheets.py make` | 3,309 "?" boxes on 249 frames |
| **Val (tasks 8, 49 frames) judged by Claude** | `labels\auto\claude_check_r1.jsonl` | 637 boxes: **373 not a person, 21 person (18 swimmer / 3 person by the pool mask), 243 left on "?" (unsure)**. Written to CVAT: job 36 "?" 410 → 169, job 37 98 → 74 |
| What the "?" boxes mostly are | — | group / area boxes, lane ropes and the floor T-markings, pennants, glare, splash, deck equipment, second boxes on a swimmer already boxed, and two static spots that recur in many frames (a dark patch at the left wall, small items by the far ladder) |
| Problems in the PRESET (2-of-3) boxes seen while judging | `labels\auto\claude_preset_issues_r1.jsonl` | the floor T-marking preset as a person (3 frames), one box over a group of people, one deck person boxed twice |
| Swimmers with no box at all, noted | `labels\auto\claude_missed_r1.jsonl` | 6 frames, mostly far-lane swimmers under the pennants |
| Owner audit page (offline, embedded crops): all 94 "person" answers + 260 "not a person" (val + train, 354 crops) | `tools\claude_audit.py` → `data\audit\audit_r1.html` | waiting for the owner; `claude_audit.py score` reads the downloaded result |
| **Train (task 7, 200 frames, 2,672 "?") judged by Claude** | `labels\auto\claude_check_r1.jsonl` | 1,736 verdicts written for frames 99-248 (1,465 not a person, 66 person, 205 left on "?"); frames 49-98 earlier. 2,690 decided answers in total. The owner still answers the "?" boxes and adds missed people |

### Audit of Claude's seed answers (2026-10-29): FAILED
Owner audited 354/354 crops: **89 wrong = 25.1 %** (rule: <= 3 %). not_a_person: 74 of 260 wrong; swimmer: 14 of 83 wrong; person: 1 of 11 wrong. Report: `reports/claude_check_audit_r1.md`.
Claude's answers must NOT train v0. v0 stays blocked (no job is finished by the owner, so the trigger has not fired). Likely cause of most "not_a_person" errors: Claude dismissed loose / duplicate / group boxes that still contain a person, which the owner's rule counts as a person.

### Round 2 of Claude's seed check (2026-10-29)
Rule after the failed audit: any visible part of a person (dark head, partly submerged body, loose / duplicate / group box holding a person) = person. Tool `tools/claude_redo.py`. The 354 audited boxes were set from the owner's audit (ground truth); 1,758 more boxes (mostly on water, "not a person" or "?") were re-judged on contact sheets: 1,263 person, 495 not a person. 2,112 verdicts written to CVAT (`labels/auto/claude_check_r2.jsonl`). Audit page for round 2: `data/audit/audit_r2.html` (300 crops: 200 person + 100 not a person). Stopping rule unchanged (<= 3 %).

### Round 2 audit (2026-10-29): FAILED again
Owner audited 300/300: 59 wrong = **19.7 %** (swimmer 45 of 200 wrong, not_a_person 14 of 100 wrong). Round 1: 25.1 %. Rule is <= 3 %, so Claude's labels cannot train v0. The 654 owner-audited boxes are now ground truth in CVAT (logged in `labels/auto/claude_check_r2.jsonl`, src owner_audit_r2). All other Claude answers stay unverified. v0 not started.

### Distractor verifier (2026-10-29)
Request: learn reflections, underwater markings and other obstacles. `tools/distractor_verifier.py`: frozen DINOv2-base crop features + geometry -> logistic regression, trained only on the 654 owner-audited boxes (322 person, 332 distractor, 19 clips). Leave-clips-out CV AUC **0.886**; at threshold 0.3 it removes 61 % of distractors and keeps 92.5 % of persons. Model `work_dirs/verifier/verifier.joblib`, report `reports/distractor_verifier.md`. NOT wired into the app. Safety rule (CLAUDE.md): a person in the water must never be hidden, so at 7.5 % person loss it may only re-rank / flag boxes for label cleaning, not suppress live detections. More owner-verified boxes will raise it.
Verifier flags (2026-10-29): `tools/verifier_flag.py --apply` set 306 of 2,553 unverified Claude answers back to "?" (Claude person but p<0.30, or not_a_person but p>=0.60); list in `labels/auto/verifier_flags.jsonl`. The other ~2,250 Claude answers remain unaudited and are NOT trusted for training.

### v0 TRIAL started (2026-10-29, owner request "just try to start v0")
Export: `export_seed.py --all-jobs` (every job, "?" boxes dropped): train 411 entries / 211 frames / 5,119 boxes, val 50 frames / 669 boxes. Labels are the owner-audited 654 boxes + Claude's UNAUDITED answers (rounds 1-2 failed the audit: 25 % and 19.7 % wrong) + preset boxes. Dropped "?" boxes become false negatives. Treat results as a pipeline/trial run, NOT as a valid v0: rename work_dirs/*_v0 to *_trial before the real v0, and do not add to MODELS.json. Trigger marker written so the watchdog does not queue a second run.

### Static-background map (2026-10-29) - part 1 of the "separate background" request
`tools/static_background.py` finds boxes that fire at the same place in many clips (fixed camera). Against the owner's audit (332 distractors / 322 persons): 8 clips+150 frames: 58 % / 23 % persons lost; 12+400: 42 % / 10 %; 15+800: 30 % / 5 %; **17+1500 (saved, 14 fixtures): 20 % / 1 %**. Position alone cannot separate fixtures from swimmers safely; the DINOv2 verifier (AUC 0.886, 61 % / 7.5 %) is stronger. Neither is wired into the app (a person in the water must never be hidden). Part 2 (live monitoring) is blocked by the live-APIs-off rule; part 3 (strokes) has no stroke labels yet.

### Environment monitor, offline replay (2026-10-29)
`tools/env_monitor.py` (CPU; live stays off): per clip, 120 sampled detector boxes scored by the verifier. Distractor rate 21-54 %, median 35 %; worst clips 18:38:23 (54 %), 18:39:02 (50 %), 18:31:55 / 19:11:55 (48 %); best 19:07:45 (21 %). Fixture-map hits 2-17 %. No clear lighting trend across 18:14 -> 19:13: it varies by clip content, so fixtures are an every-clip problem, not a drift one. Report `reports/env_monitor.md`. Stroke first-guess (RTMPose rules) waits for the GPU after the v0 trial.

### GPU freed and pose probe (2026-10-29 14:42)
Owner asked to free the GPU: v0 trial jobs 007 (stopped at about epoch 20 of 60, partial `work_dirs/dfine_l_v0`), 008, 009 cancelled; GPU back to idle. The trial output is not valid anyway (unaudited labels); the real v0 needs the owner's CVAT answers first, then `seed_trigger.py` marker removed to re-arm.
Pose probe (`tools/pose_probe.py`, `reports/pose_probe.md`): RTMPose-L on the 322 owner-verified swimmer boxes: wrists confident in 0-1 %, shoulders 3 %, nose 6 % (60 px and up: 1 %, 6 %, 11 %). **Pose rules cannot give a stroke first guess on this footage** (overhead, small, blurred, under water). Stroke classification needs owner-tagged track clips and a motion/appearance classifier, not keypoints.

### Stroke tagging tasks (2026-10-29)
TEST C is the final test set, so strokes are NOT tagged there (a classifier trained on it would contaminate the test). `tools/cvat_stroke_tasks.py` made CVAT tasks 12-15 "STROKE 1-4" (train clips 18-42-47, 18-49-23, 18-50-32, 18-31-55; 10 s from 120 s; 251 frames each; one job per task). SAM 2 follow: `tools/follow_stroke.cmd`. Owner draws boxes on the first frame, runs the cmd, sets track attribute `activity`.
SAM 2 follow on STROKE 1-4 done (2026-10-29): tracks written, owner reviews boxes and sets activity per track.

### v0 trial b started (2026-10-29, owner: "then proceed with v0")
Jobs `train_dfine_l_v0b`, `train_rtmdet_l_v0b`, `post_v0b_export_eval`. Labels: owner-audited 654 boxes + Claude answers with the verifier's 306 flags back on "?" (dropped at export) + preset boxes. Still open in CVAT: 131 "?" in val, 286 "?" in train; 0 of 9 jobs finished by the owner. Old partial run moved to `work_dirs/dfine_l_v0_trial1_partial`. Not a valid v0 until the owner finishes the jobs; do not add to MODELS.json.

### Owner answer sheet applied (2026-10-29)
`tools/answer_page.py`: owner answered all 410 remaining "?" boxes on one page (113 = 28 % changed from the verifier's suggestion; so the verifier is ~72 % right). 0 "?" left in the tasks. Owner-verified boxes now ~1,064 (654 audited + 410 answered). About 2,000 Claude answers stay unaudited (~20 % error). v0 trial c (`*_v0c` jobs) restarted on these labels; trial b stopped (`work_dirs/dfine_l_v0_trialb_partial`). Still no job marked finished and nobody has added missing swimmers.
Verifier retrained on 1,064 owner-verified boxes (2026-10-29): CV AUC 0.874 (was 0.886 on 654; the added boxes were the hard "?" cases). More data barely moves it: frozen DINOv2 features are the limit. Next gain would come from fine-tuning the detector itself with the hard negatives (v0 does this through the not_a_person boxes being left unlabelled).

### Owner HTML tools (2026-10-29) - `tools/annotator_pages.py`
`data/audit/missing_r1.html` (250 seed frames, draw boxes on people without one; then `apply missing`), `data/test_upload/label_test.html` (300 TEST A/B frames, boxes only; then `export test` -> `data/coco/test_hand.json`; keypoints and TEST C stay in CVAT), `data/audit/stroke_r1.html` (26 SAM 2 tracks as looping clips; then `apply stroke`). The pages autosave in the browser.

### 2026-10-30 results applied
- v0c finished (`reports/train_v0.md`): val AP50 D-FINE-L v0 0.540, RTMDet-L v0 0.615 vs app RTMDet-m 0.342 / dev D-FINE-M 0.281; recall@0.5fppi 0.373 / 0.293 vs 0.136 / 0.147; both ONNX pass the Life Guard decode check. Labels: owner-verified + unaudited Claude answers, so still a trial, not registered in MODELS.json.
- Owner's `missing_r1` result applied to CVAT: 283 false detections removed (task 8: 54, task 7: 229), 10 boxes added.
- Owner's stroke tags applied: 26 tracks (activity stored on the track shapes). Counts: freestyle 13, standing_wading 11, backstroke 1, butterfly 1, underwater 1, floating 1 (note 27 tracks, 1 too short to show). **Not enough for a stroke classifier**: breaststroke 0, backstroke 1, butterfly 1.
- TEST A+B labels: AUTO (owner decision), `tools/test_autolabel.py`, from independent COCO voters + verifier; report any result on them as "auto-labelled test (not hand)".
- Decision rule for the one final test run, written BEFORE it: a v0 model counts as better than the app's RTMDet-m on the auto test if recall@0.5 false positives per frame is at least 5 points higher AND AP50 is not lower. The auto labels come from COCO detectors, which favours the baselines, so a v0 win is conservative.
- FINAL test run done on the auto-labelled test (2026-10-30), once: `reports/test_final_auto.md`. D-FINE-L v0 AP50 0.646 / recall@0.5fppi 0.509; RTMDet-L v0 0.631 / 0.371; app RTMDet-m 0.406 / 0.249. Pre-registered rule passed by both, with the caveat that auto labels favour COCO-derived models. Not registered in MODELS.json; owner decides.
- D-FINE-L v0 registered (2026-10-30, owner request) in `backend/models/MODELS.json` as `lifeguard_dfine_l_pool_v0`, `dev_only: true`, classes swimmer / person_out_of_water; file `backend/models/lifeguard_dfine_l_pool_v0.onnx` (sha256 e93b5e02..., 124,956,086 bytes). `verify_models.py` OK, `check_licenses.py` OK. Not the default, not committed; use per deployment via `LIFEGUARD_UPLOAD_DETECTOR_MODEL`. Live use stays off until the owner re-enables the live APIs.

### 2026-10-30 v0 upload check and hard-negative fine-tune (v0.1)
- Owner upload (`2026-10-22-18-46-32` cut, High, Whole frame): v0 raised 11 alerts, all false. 7 are on fixtures, flags, floor markings or glare, and 4 on people standing or wading. App RTMDet-m on the same clip: 1 alert, but only 7 tracks (it misses most swimmers). Native full screen is turned off on purpose (`controlsList="nofullscreen"`); the stage button / double-click works in Chrome but not in the in-app browser pane.
- `tools/hardneg.py`: v0 on 535 new train-clip frames (12,331 boxes) -> verifier + fixture map -> owner sheet `data/audit/hardneg_r1.html` (624 questions, all answered). **274 old train labels (545 boxes incl. 2x copies) were not people**, so v0 had been taught that markers and flags are swimmers. Built `data/coco/hardneg_r1/train.json`: these removed, plus 106 new fully decided frames (320 person boxes, about 690 distractors with no box).
- Rule written before training: `reports/hardneg_r1_rule.md`. Baseline v0 on val (CPU): recall@0.5fppi 0.373, small 0.190, fp/frame at score 0.5 2.30, AP50 0.538.
- Queue: `016_train_dfine_l_v01` (from v0 best_stg1, 20 epochs, ~2 h), then `017_post_v01_export_eval` -> `reports/train_v01.md` with the rule verdict. API, upload worker and UI are stopped for training.
- v0.1 result (2026-10-30 14:40): **FAIL** (eports/train_v01.md): val recall held (0.363 vs 0.373) but fp/frame at score 0.5 unchanged (2.30) and AP50 0.519 < 0.52; upload replay 13 alerts vs v0 11 (mostly far-end window glare). v0 stays. Diagnostic: val labels contain ~50+ non-person boxes and miss many real swimmers, so val must be cleaned before the next decision.
- Val clean-up (2026-10-30): `tools/val_clean.py make` -> `data/audit/val_clean_r1.html`, 240 questions (83 val labels that look like non-people or sit on fixture spots, 157 model detections with no label, merged across v0 and v0.1). After the owner saves `val_clean_r1_result.json`: `val_clean.py apply` writes `data/coco/seed_r1/val_clean.json` (val.json is kept). Then re-score v0 and v0.1 on the cleaned val, with the rule fixed before looking.

### 2026-10-30 v0.1 re-scored on the cleaned val; v0.2 queued
- Owner's val clean-up applied: `data/coco/seed_r1/val_clean.json` (750 labels: 73 dropped, 3 class changes, 104 added). Re-score (`work_dirs/onnx/eval_valclean.json`): v0 AP50 0.586, recall@0.5fppi 0.412, fp/frame at score 0.5 1.48; **v0.1 0.593 / 0.457 / 0.88 (-41 % false boxes, recall up)**; RTMDet-L v0 0.621 / 0.287 / 1.50. So v0.1 passes rules 1-3 on the honest val; it failed rule 4 (replay 13 alerts) because the behaviour rules turn a few persistent glare boxes into alerts.
- Round 2 (no owner questions, owner asked to proceed): `tools/hardneg2.py` mined 1,250 more train-clip frames with v0.1, decided boxes automatically (verifier + fixture map), blurred undecided regions in copies (`data/frames_r2`), 393 frames (1,903 person boxes, 1,371 distractors with no box) added to hardneg_r1 -> `data/coco/hardneg_r2/train.json`. Contact-sheet spot check of 40 + 40 auto decisions: fine.
- Queue: `018_train_dfine_l_v02` (from v0.1, 12 epochs), `019_post_v02_export_eval` -> `reports/train_v02.md`. API / worker / UI are stopped while it trains.
- Behaviour rule for wading people (owner request 2026-10-30): thresholds 2.16.0 adds `BOX_ONLY_ACTIVE_LOW_REQUIRED` (false) and `WADING_STILL_CAP` (1.0), both off by default; code in `vision/behaviour/evidence.py`, 8 unit tests in `tests/unit/test_wading_rule.py` (old rules do alert on a wader with 3.1 cues; with the keys on they do not, and sinking / flat / frozen bodies still do). Full non-GPU suite: 805 passed, 1 known env failure. Pre-registered A/B in `docs/evidence/wading/rule.md`; runs after the v0.2 training frees the GPU. Nothing committed.
- v0.2 result (2026-10-30 17:50, `reports/train_v02.md`): **FAIL rule 2** on the cleaned val: fp/frame at score 0.5 1.28 (v0 1.48, v0.1 0.88), recall 0.440, AP50 0.574. Auto pseudo-labels from v0.1 did not help; v0.1 remains the best candidate (rules 1-3 pass on the cleaned val, rule 4 replay failed). Queue now runs `020_wading_ab` (docs/evidence/wading/ab.md).
