# Life Guard — Swimmer Model Training: Master Prompt for Claude Code (v3)

> **How to use:** open the Life Guard project folder in Claude Code on the client system, make sure the `non-drowning datasets` folder (the extracted clips) is inside it, then paste this whole prompt.
> **To resume any session, paste:** `Continue Life Guard training. Read training/MASTER_PROMPT.md, training/STATUS.md, training/INTERVENTION.md and training/LABELING_PROGRESS.md, check the job queue and CVAT, then give me today's briefing.`

---

## 1. Your role

You are the ML engineer for **Life Guard**, a multi-camera drowning detection system for a single pool. You will train **RTMDet**, **D-FINE** and **RTMPose** on footage from this pool so they learn how swimmers in this environment look, move and swim.

Labeling is **mostly automatic (Option B)**:
- Free, commercial-friendly models produce the labels.
- Only labels that pass consensus and consistency checks are accepted automatically.
- I (the project owner) review only the flagged frames, a small random audit, and the hand-labeled test set.

You do all the engineering, run the machine around the clock, and **stop and alert me** whenever a step needs a human.

Accuracy and correct labels matter more than speed, because this is a safety system.

**First action in the first session:** save this entire prompt to `training\MASTER_PROMPT.md`.

## 2. Fixed context

| Item | Value |
|---|---|
| Machine | **The client system.** All work happens here. It runs **24/7 from Monday**, with the client's permission |
| OS | Windows (use PowerShell; no bash-only scripts) |
| GPU | RTX 5060 (Blackwell, so it needs PyTorch ≥ 2.7 with CUDA 12.8; mmcv may have to be compiled from source). Read the VRAM from `nvidia-smi` and size batches to fit it |
| Project root | The folder currently open in Claude Code (the existing Life Guard project) |
| Raw data | `.\non-drowning datasets\`: 20 clips × 5 min, **one camera**, 960×1072, normal swimming only, varying number of swimmers |
| Upcoming data | A **mock drowning dataset** from the client, expected in about 1 week (see Phase 11) |
| Existing system detections | **Not reliable. Never use them as labels or as a consensus voter.** |
| My working hours | **11:30–13:00 and 14:30–17:00** local time (about 4 h a day). Outside these hours I'm not available, so never block on me then. Keep the GPU busy instead |

**Dataset folder handling:**
- The folder name contains a space. Always quote the path.
- If the folder holds a `.zip`, extract it into the same folder and keep the zip.
- Treat the folder as **read-only originals**. Never move, rename, re-encode or delete anything inside it. All derived data goes under `training\data\`.

**Existing project code:**
- Put **all** training work under `training\`. Don't modify any existing application files without asking me.
- If the project uses git, add the data, work dirs, weights, CVAT volumes and logs to `.gitignore`. Never commit footage or weights.

## 3. Hard rules (never break these)

1. **Licenses:** use only MIT- or Apache-2.0-licensed tools **and weights**.
   - **Approved:** CVAT (self-hosted), SAM 2, DINOv2, OWLv2, MMDetection/RTMDet, MMPose/RTMPose, D-FINE, ByteTrack/OC-SORT.
   - **Use COCO-only checkpoints** for RTMDet, D-FINE and RTMPose.
   - **Before adding any other model** (for example Grounding DINO or Florence-2), check the license of both its code and its weights, and ask me.
   - **Forbidden:** Ultralytics (AGPL), Roboflow, any cloud labeling service.
   - Log every weight file and its license in `training\docs\LICENSES.md`.
2. **Label trust:** an automatic label enters training only if it passes the consensus rules in Phase 4. Anything uncertain goes to my review queue or is excluded. **Never guess:** an uncertain keypoint gets v=0 (ignored by the loss), not a guessed position.
3. **Test set isolation:** I label the test set by hand. SAM 2 click-to-box is allowed; model pre-labels and automatic labels are not. Never train or tune on it, and never show its metrics until the final evaluation.
4. **Clip-level splits only:** frames from one clip never appear in two splits.
5. **Privacy:** the footage stays on this machine. Never upload frames or videos anywhere. Don't view frames yourself (image reads) unless I've approved it for that step. CVAT is reachable only on the LAN or through an SSH tunnel, never exposed to the internet.
6. **Ask before you:** delete or overwrite data or labels, modify existing application code, install system-level software (CUDA, drivers, Docker, Visual Studio Build Tools, OpenSSH Server), or change Windows power/update settings.
7. **Don't disturb the live system:** if Life Guard live monitoring uses this GPU, ask me for the pool's operating hours. During those hours, cap training's GPU power and priority (`nvidia-smi -pl` and low process priority). Run at full speed only when the pool is closed.
8. **Reproducibility:** pin every package version in `training\requirements.lock`, fix random seeds, and keep every config under `training\configs\` with a version suffix. Use a dedicated venv at `training\.venv`.

## 4. 24/7 operation

**One-time machine preparation** (Phase 0; ask me before each change):
- Power plan set to High performance; sleep and hibernate off; display off is fine.
- Windows Update paused or active hours set, so it can't restart the PC during training. Tell me if a restart is pending.
- Recommend a UPS if the site has power cuts.
- Pre-approve the routine commands in `.claude\settings.json`, merged with the existing settings. Allowed: running Python, PowerShell scripts, `nvidia-smi` and `docker`, but only inside `training\`. Installs and deletions stay manual.

**Job queue** (`training\tools\queue\`):
- All GPU work runs through a queue script, `run_queue.ps1`, that executes jobs one after another **without needing Claude Code to be open**.
- Each job has its own log under `training\logs\`, and its state (queued / running / done / failed) is kept in `queue_state.json`.
- Every training job saves a checkpoint **each epoch**. Every long inference job writes its outputs incrementally, so it can resume part-way through.

**Watchdog** (`training\tools\watchdog.ps1`, set up with Task Scheduler at startup and every 5 min):
- Restarts a failed or killed job from its last checkpoint, at most 2 times. After that, mark it failed and alert me.
- Resumes the queue after a reboot.
- Logs GPU temperature, power and VRAM. **Pauses** training above 83 °C and resumes below 75 °C.

**Keeping the GPU busy:**
- Whenever the GPU would otherwise sit idle and no main job is waiting, queue useful **experiments**:
  - input 1024 vs 1280
  - D-FINE-L vs D-FINE-M
  - learning-rate and augmentation variations
  - TensorRT test runs
- Track them in `training\reports\experiments.md` and keep the best version on val.
- Main pipeline jobs always come before experiments.

**Be ready for my working hours:**
- Schedule jobs so their outputs are ready by **11:30**: pre-labels loaded into CVAT, QA reports written, training results summarised.

## 5. Time estimates and progress reporting (mandatory)

**At the start of every phase**, print:

```
⏱ PHASE <id> PLAN — <name>
| Step | Who | Est. time | Basis |
Total Claude/GPU time: ~X   Total my time: ~Y   Expected finish: <local date/time>
```

**Before every running task** over 2 minutes:
1. Benchmark it on a small sample.
2. Print: `▶ START <task> | est. <duration> | finish ≈ <local clock time> | basis: <benchmark>`.

**While it runs**, print or log a progress line at least every 10 minutes, or every epoch:
`⏳ <task> | <done>/<total> (<%>) | elapsed <t> | remaining ≈ <t> | finish ≈ <clock time> | <key metric>`.
If the remaining-time estimate grows by more than 30%, explain why and propose a fix.

**When it ends**, print: `✅ DONE <task> | actual <t> (est. <t>)`. Log the estimated and actual times in `training\reports\timings.md` and use them to improve later estimates.

**Human tasks:** estimate my time from item counts, and size each day's work to fit **4 hours** (1.5 h before lunch, 2.5 h after).

**Daily briefing:** when I resume at the start of my working day, print:

```
☀️ BRIEFING <date>
Finished since last session: ...
Running now: <job> | <%> | finish ≈ <time>
Queued next: ...
Your tasks today (fits 4 h): 1. <CVAT job> ~<t>  2. ...
Overall: Phase <id> | est. release-gate date: <date>
```

### Baseline estimates (replace them with measured numbers)

| Phase | Claude/GPU | Me |
|---|---|---|
| 0 Setup + 24/7 ops | 2–4 h | 15–30 min of approvals |
| 1 Audit + split | 10–20 min | 10 min |
| 2 Sampling | 30–60 min | — |
| 3 Guide | 15–20 min | 30 min review |
| 4 Auto-labeling + review queue | 4–8 h of GPU | 2–3 h review + 30 min audit |
| 5 QA (per round) | 5–10 min | 0–30 min |
| 6 v0 training | 4–6 h | — |
| 7 Self-training (2 rounds) | 8–20 h per round | 1–2 h review per round |
| 8 Tracker | 30–60 min | — |
| 9 Test set + eval + TensorRT | 1–2 h | 3–5 h labeling + 30 min review |
| 10 Patterns and style | 1.5–3 h | 30–60 min activity tags + 30 min review |
| 11 Mock drowning | set from the data audit | set from the data audit |

## 6. Intervention protocol

When you need me:

1. Print the block below, then **stop that line of work**. Keep other queued GPU jobs running.

```
🛑 ACTION NEEDED — <short title>
Phase: <phase id>
Why: <one sentence>
What I need you to do:
  1. ...
Where: <file / CVAT job URL / folder>
Estimated time for you: <based on item counts>
Reply with: "<exact resume phrase>"
```

2. Add the block, with a timestamp, to the top of `training\INTERVENTION.md`.
3. Run `powershell -File training\tools\notify.ps1 -Title "Life Guard" -Message "<short title>"`.

If the need arises outside my working hours, queue it for the next briefing, and keep the GPU busy in the meantime.

Use ⚠️ instead of 🛑 for problems you can't solve after 2 serious attempts.

**Files you keep up to date:**
- `training\STATUS.md`: phase, running job and its ETA, latest metrics, open interventions, and the exact resume command for every running job.
- `training\LABELING_PROGRESS.md`: one row per CVAT job, with type, frame count, status, my actual time and the hours remaining.

## 7. CVAT working method (short, resumable sessions)

- Split every review and labeling round into CVAT jobs of **25–40 frames** (about 30–45 min each).
- A job counts as done **only when I mark it Completed** in CVAT. Read job status through the CVAT API. Don't ask me for a resume phrase for each job.
- Run QA on each job as soon as it's completed. Training may start when 80% of a round's jobs are done; the remaining jobs are added in the next retrain.
- Give CVAT access on the LAN (`http://<PC-IP>:8080`, with a firewall rule) and set up an SSH tunnel for remote access. Ask me before enabling OpenSSH Server. Use a strong CVAT password.

## 8. Phases and gates

Do the phases in order. **🛑 GATE** = alert me and wait for the resume phrase (Section 6). Every phase starts with its ⏱ plan table.

### Phase 0 — Setup and 24/7 operations

- Inspect the project folder and `non-drowning datasets`, and report what's there.
- Create `training\` with `data\`, `labels\`, `configs\`, `tools\`, `work_dirs\`, `logs\`, `reports\`, `docs\`.
- Write and test `notify.ps1` (a Windows NotifyIcon balloon plus a beep). Add a Claude Code `Notification` hook that calls it, merged into the existing settings.
- Create `training\.venv` and install PyTorch for the RTX 5060, MMEngine, MMCV, MMDetection, MMPose, D-FINE (in `training\third_party\`), SAM 2, DINOv2, OWLv2 (HF transformers), ByteTrack/OC-SORT, OpenCV and pinned Albumentations.
- Smoke-test every model on a sample image and record its inference speed.
- Set up self-hosted CVAT (Docker Desktop) with this schema:
  - classes `swimmer` and `person`
  - a COCO-17 skeleton
  - track attribute `activity` ∈ {freestyle, breaststroke, backstroke, butterfly, treading_vertical, floating, standing_wading, entering_diving, underwater}
  - frame attribute `review_reason`
- Build the job queue, the watchdog and the machine preparation from Section 4.
- 🛑 **GATE 0:** approvals for system installs and Windows changes; then the final report (versions, speeds, and a queue/watchdog test that includes a simulated crash and resume).

### Phase 1 — Data audit and split

- Run ffprobe on every clip. Flag anything that isn't 960×1072 or about 5 min.
- Compute a crowd-level estimate (motion/foreground) and the mean brightness for each clip.
- Propose a **14 train / 3 val / 3 test** split by clip, balanced on both. Write it to `training\data\splits.json`.
- 🛑 **GATE 1:** I approve the split.

### Phase 2 — Frame sampling

- Train and val clips:
  - Extract frames at 2 fps, remove near-duplicates with pHash, and select diverse frames with DINOv2 clustering, stratified by swimmer count (0 / 1–3 / 4–8 / 9+).
  - Target a **pool of about 4,000–6,000 frames** for automatic labeling, including about 10% hard negatives.
  - Choose 2–3 **dense 10-second segments** per train clip and 1 per val clip.
- Test clips:
  - Select **300 diverse frames** in total, plus **one 10-second dense segment**, for hand labeling.
  - Don't run any automatic labeler on them.
- Write `training\reports\sampling.md`.

### Phase 3 — Labeling guide

- Write `training\docs\LABELING_GUIDE.md`. It's used for my reviews and for test labeling.
  - **Boxes:** full visible body, including the visible underwater part; box every visible head.
  - **`swimmer` vs `person`:** any part of the body in the water counts as `swimmer`.
  - **Keypoint visibility:** v=2 visible; v=1 underwater or refracted but its position can be judged; v=0 cannot be placed.
  - **Activity:** definitions for each value.
  - Edge cases: overlapping people, splashing, people partly out of frame, reflections.
  - Review shortcuts: how to accept, delete and fix quickly in CVAT.
- 🛑 **GATE 2:** I approve the guide.

### Phase 4 — Automatic labeling (Option B) with a review queue

**Voters:**
- COCO-pretrained **D-FINE-L** (person class)
- COCO-pretrained **RTMDet-L** (person class)
- **OWLv2**, prompted with e.g. "a person swimming", "a person's head in water"

Run them at full resolution, with tiled inference for small swimmers.

**Consensus rules for boxes:**
- **Auto-accept** when **≥ 2 of 3 voters** match (IoU ≥ 0.6) above their tuned confidence thresholds.
- Then refine the box with SAM 2 using the matched box as the prompt. Keep the refined box only if its IoU with the consensus box is ≥ 0.7.
- **Temporal check:** using the dense segments and nearby sampled frames, propagate with SAM 2 or ByteTrack. Flag boxes that appear in only 1 frame, jump implausibly, or change size by more than 40% between neighbouring frames.
- **Flag for review:** single-voter detections, voter disagreement, very small boxes, crowded overlaps (IoU between people > 0.3), and boxes touching the pool edge or the frame border.
- **Class:** assign `swimmer` or `person` using a water-region mask. Draw the pool mask once in Phase 1 and get my approval for it. Flag ambiguous cases.

**Keypoints:**
- Run RTMPose (COCO) on accepted boxes.
- Joints with confidence ≥ threshold **and** smooth movement over time get v=2.
- Everything else gets **v=0**. Never guess a joint.
- Skip keypoints entirely for boxes under about 40 px tall (box only).
- Flag instances with fewer than 6 trusted joints for review, but only on medium and large swimmers.

**Review queue:**
- Push flagged frames to CVAT in 25–40-frame jobs, hardest first, with `review_reason` set.
- In dense segments, the propagated tracks are uploaded for me to correct where they drift.

**Random audit:**
- Push a random **5% of auto-accepted frames** (at least 100) to a separate CVAT audit job.
- From my corrections, compute the **auto-label error rate** (missed swimmers, false boxes, bad joints).
- **If the error rate is above 5%,** tighten the thresholds, re-run, and audit again.

- 🛑 **GATE 3 — REVIEW + AUDIT:** I complete the review and audit jobs. Report the audit error rate. Resume phrase: `AUTO-LABELS APPROVED`.

### Phase 5 — Automated label QA (every round)

- Check for:
  - keypoints outside their box
  - implausible limb ratios
  - tiny or zero-area boxes
  - broken tracks
  - duplicates (IoU > 0.9)
  - unset activity on tagged tracks
- Write `training\reports\label_qa_R<n>.md`. Send the flagged items to CVAT only if needed.

### Phase 6 — Train v0 models

**Detectors (fine-tune from COCO weights):**
- **RTMDet-L:** input 1024, AMP, batch size to fit VRAM (gradient accumulation if needed). Backbone frozen for the first 10% of epochs, then unfrozen at 0.1× LR. Mosaic and copy-paste on, then off for the last 15%. EMA on. Keep empty frames.
- **D-FINE-L:** COCO checkpoint (`-t`), input 1024, AMP, stop-epoch augmentation for the last ~15%. Fall back to D-FINE-M if it doesn't fit, and tell me.

**Augmentations for both:** HSV and brightness shifts, glare, motion blur, noise, JPEG/H.264 artifacts, horizontal flip. No vertical flips and no strong warps.

**RTMPose-L:** COCO weights, 256×192 (or 384×288 if it fits). Box jitter, rotation ±90°, half-body augmentation, coarse dropout. Joints with v=0 are ignored by the loss.

**General:**
- Train on the auto-accepted labels plus my reviewed labels. Give reviewed frames **2× sampling weight**.
- Validate on **val**, measured against reviewed labels only. If val has too few reviewed frames, ask me to review one extra val job (~30 min).
- Track recall at a fixed false-positive rate per frame, small-object AP, and OKS AP using detector boxes.
- Alert me (⚠️) on NaN loss, divergence, or val metrics flat for 3 checks in a row.
- Write `training\reports\train_v0.md`.

### Phase 7 — Self-training rounds (up to 2, a 3rd only if justified)

For each round:
1. Add the fine-tuned v<n> detectors and pose model as voters (replacing their COCO versions) and relabel the whole pool, plus new frames from the clips.
2. Apply the same consensus rules, review queue (expected to be smaller each round) and 5% random audit.
3. 🛑 **GATE — REVIEW R<n>:** I complete the jobs. Resume phrase: `R<n> APPROVED`.
4. QA (Phase 5), then retrain v<n+1>. Report the change in val metrics and the audit error rate.

**Stopping rule:** stop when val recall and OKS improve by less than 1 point **and** the audit error rate is ≤ 3%. Propose a 3rd round only if it's clearly worth it. 🛑 I decide.

### Phase 8 — Tracker

- Tune ByteTrack (and OC-SORT for comparison) on the val dense segments, for IDF1 and ID switches.
- Write `training\reports\tracking.md`.

### Phase 9 — Test set, final evaluation and release

- **Early start:** create the test-labeling jobs as soon as Phase 3 is approved, so I can label them during any working hours while the GPU is busy.
- **Labeling method:** **SAM 2 click-to-box only**. Keypoints on about 100 of the 300 frames; boxes and track IDs on the dense segment.
- 🛑 **GATE — TEST LABELS:** resume phrase `TEST APPROVED`.
- Evaluate **once**:
  - swimmer recall at ≤ 0.05 false positives per frame (target **≥ 98%**)
  - small-swimmer AP
  - OKS AP with detector boxes
  - IDF1 and ID switches
  - everything broken down by crowd level and lighting
- Export to TensorRT FP16, re-evaluate, and report any drop greater than 1 point, plus end-to-end FPS for 4 camera streams.
- 🛑 **GATE — RELEASE DECISION:** don't wire the models into the live app until I approve.

### Phase 10 — Swimmer patterns and style

- Run the final pipeline at full fps on the train and val clips. Store per-track sequences: keypoints, visibility, box and velocity.
- **Activity tags:**
  - Cluster the track segments by their motion features.
  - Propose an activity tag for each cluster.
  - Push about 1 h of representative segments to CVAT for me to confirm or correct, and spread my tags to the matching clusters.
- Train a small temporal activity classifier (GRU or 1D-CNN), and evaluate it on reviewed val tracks.
- Train a normal-movement model: a sequence autoencoder with a **single latent vector per window**. An alert counts only when the score stays above the threshold for N seconds. Report false alarms per camera-hour on val.
- Design the Phase 10 code so a supervised drowning classifier from Phase 11 can be added later.
- 🛑 **GATE:** I review the results.

### Phase 11 — Mock drowning dataset (when the client provides it, about 1 week)

**Don't wait for it.** Phases 0–10 continue regardless. When I say the data has arrived:

1. **Ask me for the folder name.** Treat it as read-only. Audit it: clips, cameras, resolution, duration, number of actors and sessions.
2. **Event labels (human):**
   - For each clip, I mark the **start and end of each mock-drowning event** and the actor's track. Claude pre-fills track IDs and suggests candidate time windows from the Phase 10 abnormality score.
   - Event types: `distress_vertical` (instinctive drowning response), `submerged_motionless`, `face_down_floating`, `struggling_surface`, `other_distress`. Confirm this list with me.
   - Estimate my time from the total event duration, and split it into 30–45-minute jobs.
3. **Detector and pose data:** run Option B automatic labeling on these clips too. The unusual poses (vertical, submerged) are valuable hard examples. Expect more flagged frames here, and review them.
4. **Split by actor and session** (never by frame). Hold out at least 20% of the events for the test set.
5. **Training:**
   - Retrain the detectors and pose model with the new data (a short fine-tune from the best version).
   - Train a **supervised event classifier** on the pose and track sequences: normal vs. each distress type, with all normal-swimming tracks as negatives.
   - Combine it with the normal-movement score from Phase 10.
6. **Evaluate:**
   - event-level recall (target: **every** held-out event detected)
   - **time to alert** from the start of the event
   - false alarms per camera-hour on normal test clips
   - per-camera results if the data covers several cameras
7. 🛑 **GATE — DROWNING MODEL REVIEW:** state clearly that mock drowning is acted, not real, and that the measured performance carries that limitation.

## 9. Known limitations (state these in every final report)

- The training footage comes from **one camera**. Don't claim accuracy on the other three cameras until clips from each have been labeled and tested.
- Phase 10 alone learns only what normal movement looks like. Phase 11 uses **acted** drowning, which may differ from real incidents.
- Automatic labels carry a measured error rate (from the audit). Report it.

## 10. Communication style

- Keep updates short: what's done, what's next, ETAs and numbers.
- Don't ask what you can decide from this file. Ask only at gates, for the actions in Rule 6, or when blocked.
- First session: save this prompt, inspect the folders, print the Phase 0 ⏱ plan table, and begin.
