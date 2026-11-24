# Interventions (newest first)

## 2026-10-29 01:40 — 🛑 ACTION NEEDED — Audit Claude's answers (≈ 10 min), then finish jobs 36–37
Phase: 4 (owner approved Claude judging the seed "?" boxes, with a 10 % audit)
Why: Claude answered the val "?" boxes (373 not a person, 21 person, 243 left unsure). Before these answers train
  anything, you check a sample, so we know Claude's error rate.
What I need you to do:
  1. Open C:\lifeguard\training\data\audit\audit_r1.html (double-click; opens offline). For each of the 58 crops
     press Right or Wrong, then "Save results" (downloads audit_r1_result.json).
  2. Finish jobs 36 and 37 in CVAT as before: only the boxes still on "?" need a key (job 36: 169, job 37: 74),
     then add anyone missing, Finish the job.
Where: training\data\audit\audit_r1.html; http://localhost:8080/tasks/8/jobs/36
Estimated time for you: ≈ 10 min audit; ≈ 30–45 min for both jobs.
Reply with: "audit done" (I score it from your Downloads folder).

## 2026-10-29 00:30 — 🛑 ACTION NEEDED — Seed CHECK (one key per box) + reduced test set
Phase: 4 / 9 (auto-label method approved by the owner, 2026-10-28)
Why: replaces the edit-every-frame seed review. Grounding DINO was tried as a 4th auto-labeller and dropped (50-frame
  trial: found 59 % of the people the other detectors agree on, and nobody new). The boxes come from the three
  detectors; you only say what each one is.
What I need you to do (step by step: training\docs\OWNER_STEPS.md, Parts A and B):
  1. CVAT task 7 "CHECK R1 train" (jobs 29–35) and task 8 "CHECK R1 val" (jobs 36–37): Attribute annotation mode,
     one key per box (Tab where the preset class is right), then add anyone without a box; Finish each job.
     Do one val job early. v0 starts by itself at 8 of 9 jobs finished.
  2. Test set: TEST C jobs 14–19 (box everyone once on the first frame, run tools\follow_test_c.cmd, fix slips),
     TEST B jobs 6–9, TEST A jobs 2–3.
Where: http://localhost:8080/projects/1
Estimated time for you: CHECK ≈ 2–3 h; test set ≈ 3–4 h (estimates; the first job's real time will be logged).
Reply with: nothing per job (Completed states are read automatically); tell me when the test jobs are done.

## 2026-10-28 16:50 — ⏭️ SUPERSEDED (seed CHECK, 2026-10-29) — Review the SEED (option A): do this before the test set
Phase: 4 (option A, owner 2026-10-28)
Why: v0 training starts from these fully corrected frames (it may start when 80 % of the seed jobs are Completed).
What I need you to do:
  1. CVAT task 5 "SEED R1 train" (jobs 20–26, 30 frames each, hardest first) and task 6 "SEED R1 val" (jobs 27–28).
  2. Per frame: fix / delete `swimmer` / `person` boxes, turn every orange `candidate` into `swimmer` / `person` or
     delete it, add anyone missed (SAM click-to-box helps), mark the job Completed. Guide: section 6.
Where: http://localhost:8080/projects/1
Estimated time for you: ~8 h in total (13.7 boxes per frame to check, ~2 min per frame), ~30–60 min per job.
Reply with: nothing per job (I read Completed states). v0 training starts by itself at 80 % Completed (8 of 9 jobs,
  at least one val job): the watchdog queues the export (QA: a job with a `candidate` left is excluded and listed in
  training\reports\label_qa_R1.md) and D-FINE-L v0 training. Jobs finished later join the next retrain.

## 2026-10-28 15:45 — ✅ RESOLVED (owner chose A) — Phase 4 consensus: choose how to label (A / B / C)
Phase: 4
Why: the prompt's 2-of-3 rule auto-accepts 13 of 4,664 frames; the COCO voters miss swimmers, so reviewing every
  flagged frame would take ~40–60 h, and loosening the rule would teach the models to ignore swimmers.
What I need you to do: read training\reports\phase4_consensus_r1.md (5 min) and choose A, B or C.
Where: training\reports\phase4_consensus_r1.md
Estimated time for you: 5 min now; then A ≈ 4–5 h of review (seed), B ≈ 40–60 h.
Reply with: "LABELING: A" (or B / C)

## 2026-10-28 15:00 — ✅ RESOLVED (block rule added, owner-approved) — SAM is ready in CVAT; close a network exposure
Phase: 9 (test labelling)
Why: the SAM 1 ViT-H function (float16, OOM → CPU fallback) is deployed; Nuclio publishes it on host port 57538 on
  ALL interfaces, and Docker Desktop's firewall rule allows every port on Public networks (your Wi-Fi is Public).
What I need you to do: pick one.
  A. Mark the Wi-Fi as **Private** (Settings > Network & internet > Wi-Fi > your network > Private). Docker's allow
     rule is Public-only, so this closes it for all Docker ports. (Recommended; it also allows CVAT on the LAN later.)
  B. Let me add a Windows Firewall **block** rule for TCP 57538 on Public (UAC prompt).
  C. Disable the "Docker Desktop Backend" inbound rules yourself (Windows Defender Firewall > Inbound rules).
How to use SAM in CVAT: in a job, left toolbar **AI Tools > Interactors > Segment Anything**, click on the person
  (positive point; Shift/right-click for a negative point), Enter to finish. Check the result and make it the
  `swimmer` / `person` box. (I have checked the function and CVAT's listing, not the in-browser clicking itself.)
Reply with: "WIFI PRIVATE DONE", "BLOCK 57538", or "DOCKER RULE DISABLED"

## 2026-10-28 14:40 — ⏭️ SUPERSEDED (reduced test set + SAM 2 follow, 2026-10-29) — Hand-label the test set
Phase: 9 (early start, allowed after GATE 2)
Why: the test set must be labelled by you, by hand, before the final evaluation; it can be done in pieces.
What I need you to do (follow training\docs\LABELING_GUIDE.md section 7):
  1. TEST B jobs 6–13 (boxes + class, 25 frames each, ~20 min per job) — start here.
  2. TEST C jobs 14–19 (dense segment, boxes + track ids; use CVAT track mode + interpolation).
  3. TEST A jobs 2–5 (boxes + class + keypoints for people >= 40 px tall).
  Mark each job Completed in CVAT; I read the state myself.
Where: http://localhost:8080/projects/1 (tasks 2, 3, 4)
Estimated time for you: ~11–14 h in total (300 frames x ~7 people) — higher than the prompt's 3–5 h baseline.
  Option to cut it: I set up CVAT's serverless add-on (Nuclio) with SAM 2 for click-to-box (allowed by rule 3).
  That is a system change in Docker (~1 h of my time, GPU in WSL) — say "SETUP SAM IN CVAT" to approve.
Reply with: nothing needed per job; "SETUP SAM IN CVAT" if you want the click-to-box tool.

## 2026-10-28 15:00 — 🛑 GATE 2 — Approve the labelling guide (+ draw the pool mask)
Phase: 3
Why: every review, the audit and the hand-labelled test set follow this guide; the pool mask decides swimmer vs person.
What I need you to do:
  1. Read training\docs\LABELING_GUIDE.md (~20 min) and send changes, especially the swimmer / person rule and
     the activity definitions.
  2. Still open from GATE 1: CVAT job 1 (http://localhost:8080/tasks/1/jobs/1) has no `pool_water` polygon yet:
     draw it, press Save (Ctrl+S), mark the job Completed.
Where: training\docs\LABELING_GUIDE.md, CVAT job 1; Phase 2 result: training\reports\sampling.md
Estimated time for you: ~30 min
Reply with: "GUIDE APPROVED" (or your changes)

## 2026-10-28 14:40 — 🛑 GATE 1 — Approve the split and draw the pool mask
Phase: 1
Why: the clip-level split and the water mask decide every later label and metric.
What I need you to do:
  1. Read training\reports\data_audit.md (split table + the note that all clips are one evening, same swimmers).
  2. In CVAT (http://localhost:8080, user lifeguard_admin, password in training\cvat\credentials.txt), open
     http://localhost:8080/tasks/1/jobs/1, draw one `pool_water` polygon, check it on all 3 frames, mark Completed.
Where: training\data\splits.json, CVAT job 1
Estimated time for you: ~15 min (5 reading + 10 drawing)
Reply with: "SPLIT APPROVED" (or the changes you want); I detect the CVAT job's Completed state myself.

## 2026-10-28 14:10 — 🛑 GATE 0 — Phase 0 complete: review and approve
Phase: 0
Why: the prompt's GATE 0: setup, installs, queue/watchdog crash test and CVAT are done; Phase 1 needs your go.
What I need you to do:
  1. Read training\reports\phase0_report.md (5 min).
  2. Answer the RTMPose question below (A / B / C).
  3. Optional: make the Wi-Fi Private + review Docker's Public firewall rule if you want CVAT on the LAN.
Where: training\reports\phase0_report.md; CVAT at http://localhost:8080 (credentials in training\cvat\credentials.txt)
Estimated time for you: 10 min.
Reply with: "GATE 0 APPROVED" (plus "RTMPOSE: A/B/C")

## 2026-10-28 12:00 — 🛑 ACTION NEEDED — RTMPose starting weights (licence)
Phase: 0
Why: the prompt allows only COCO-only checkpoints, and every published RTMPose COCO checkpoint was pre-trained on AI
Challenger (AIC) + COCO (`*_pt-aic-coco_*`); AIC's data terms are research-oriented, so this is your licence call.
What I need you to do: choose one.
  A. Accept the `pt-aic-coco` RTMPose-L checkpoint (common practice; MMPose code is Apache-2.0). Check AIC's terms
     with the client's counsel if it ships.
  B. COCO-only: start RTMPose-L from its ImageNet-pretrained CSPNeXt backbone and train on COCO first (adds about
     1–2 days of GPU), then fine-tune on the pool.
  C. Another pose model whose weights are COCO-only and Apache/MIT (I'd propose candidates first).
Where: training\docs\LICENSES.md
Estimated time for you: 5 min.
Reply with: "RTMPOSE: A" (or B / C)

## 2026-10-28 11:50 — 🛑 ACTION NEEDED — Approve the Phase 0 system changes (one batch)
Phase: 0
Why: these change the machine outside `training\`, so rule 6 of the master prompt needs your approval first.
What I need you to do: answer yes / no for each item.
  1. **WSL2 + Docker Desktop** (for self-hosted CVAT). `wsl --install`, then Docker Desktop. A reboot is likely.
     About 2 GB, plus ~5 GB of CVAT images. Docker Desktop's licence is free for small businesses (< 250 staff and
     < USD 10 M revenue); please confirm the client qualifies, or we use Podman instead.
  2. **NVIDIA CUDA Toolkit 12.8 + Visual Studio 2022 Build Tools (C++ workload)**: needed to compile MMCV's CUDA ops
     for the RTX 5060 (no prebuilt Windows wheel exists for Blackwell). About 3 GB + 6 GB. The driver is not changed.
  3. **Power plan**: switch Balanced → High performance; sleep and hibernate off (the display may still turn off).
  4. **Windows Update**: set active hours or pause updates (Windows 11 Home allows a pause of up to 5 weeks), so no
     automatic restart during training. I'll tell you if a restart is already pending.
  5. **Task Scheduler**: register `LifeGuard-Training-Watchdog` (at log-on and every 5 min), running as your user.
  6. **Firewall rule** for CVAT on TCP 8080, LAN profile only (Private network), not Public.
  7. **OpenSSH Server** for remote CVAT access via a tunnel: optional. I suggest **no** for now (LAN only).
  8. **Claude Code hook + pre-approved commands**: I'm not allowed to edit my own settings. To get the desktop alert,
     add a `Notification` hook that runs
     `powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\lifeguard\training\tools\notify.ps1 -Title "Life Guard (Claude Code)" -Message "Claude Code needs your attention"`
     through `/hooks` in a `claude` terminal (or ask me to show the exact JSON for `.claude\settings.local.json`).
  9. A **UPS** is recommended if the site has power cuts (hardware, your call).
Where: this file; I install / change only the items you approve.
Estimated time for you: 10 min to answer, plus ~15 min at the PC for the installers' prompts (UAC) and one reboot.
Reply with: "PHASE 0 APPROVED: <item numbers>" (for example "PHASE 0 APPROVED: 1 2 3 4 5 6").

## 2026-10-29 09:45 - Claude's seed check is finished: please audit
Claude judged all 249 seed frames (val and train). 2,690 boxes have an answer in CVAT; about 600 stay on "?" for you.
1. Open `C:\lifeguard\training\data\audit\audit_r1.html` in your browser (offline, nothing uploaded), press Right / Wrong on each of the 354 crops, then **Save results**.
2. Tell Claude "audit done"; the error rate must be 3 % or less before v0 trains on these answers.
3. Then in CVAT answer the boxes still on "?" (key 1/2/3), add anyone missing (N), and Finish jobs 36-37 and 29-35.

## 2026-10-29 12:25 - Round 2 audit please
Claude redid the judging with your rule. Open `C:\lifeguard\training\data\audit\audit_r2.html`, press Right/Wrong on 300 crops, **Save results** (downloads `audit_r2_result.json`), then tell Claude "audit 2 done".
