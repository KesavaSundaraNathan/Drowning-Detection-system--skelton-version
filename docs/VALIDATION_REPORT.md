# Life Guard v7: validation report

Date 2026-10-21. Branch `v7-fixes`. Machine: RTX 5060 (Blackwell, sm_120), driver 617.42, i7-14700, Windows 11;
onnxruntime-gpu 1.31 + CUDA 13 + TensorRT 10.16 (FP16); thresholds 2.12.1. Every number comes from a command run on
this date or from the evidence note it cites. Anything that could not be measured is **UNVERIFIED**, with the reason.

**Verdict: 51 / 100. Assist / pilot only.** Not a safety system. No labelled footage from the site exists yet, so the
score is capped at 60 (section 6). The cap is not binding today: the measured score is below it.

## 1. Scope: what was validated and what could not be

| Area | How | Status |
|---|---|---|
| Offline detection and alerting | `eval/heldout_v1`: dev (10 clips, 11 events) for tuning; test (6 clips, 5 events, 249 s) once per change, both detectors | measured |
| Live stability and load | 65 min on the 4 configured RTSP cameras through the real API, metrics every 10 s (`build/eval_v7/live_sampler.py`, `live_run.jsonl`) | measured |
| Live drowning alerts | — | **UNVERIFIED.** The 4 cameras are **office cameras** (Engineer Room, Entrance, Meeting room, Reception) on one NVR. An admin set all 4 to `standard_cctv`: detected and tracked, but never pose-scored or behaviour-scored. No drowning alert can occur, and the pose GPU load is absent. No pool camera is configured. |
| Staged drills (SITE_SESSION D1–D7) | — | **UNVERIFIED.** They need a pool, a drill run by a human and the legal gate. None was in progress; nothing was simulated. |
| Upload path in the UI | Upload worker restarted with D-FINE (its log: `upload detector: lifeguard_dfine_m_devtrain_v1_dynbatch.onnx`). The same pipeline was run through the eval harness | measured programmatically; **uploading through the UI is UNVERIFIED** (needs the admin login) |
| v5 interface with live data | API access log during the run: the UI (the admin's own session) fetched 4,651 camera previews and 28 MJPEG main views, all HTTP 200; alerts were listed and acknowledged | measured from the server side; the picture itself was seen only by the operator |

## 2. Documented requirements vs measured result

Sources: eval targets in `lifeguard.eval` (report `targets`), `docs/v7_fix_plan.md`, `docs/PLAN_v6.md`,
`docs/SITE_REQUIREMENTS.md` / `docs/CAMERA_SETUP.md`, `docs/SITE_SESSION.md`, `CLAUDE.md` safety rules.

| # | Requirement (source) | Measured | Result |
|---|---|---|---|
| 1 | In-water person recall ≥ 0.95 (eval targets) | Dev: RTMDet **0.40** (63 boxes; 40–80 px: 0.03; underwater: 0.00); D-FINE 0.54* | **FAIL** |
| 2 | 0 alerts on normal swimming / standing / out of water (eval targets) | Test: 0 / 0 / 0 for both detectors (the normal-swimming clip is only 12.6 s). Dev RTMDet: 1 on a normal swimmer | PASS on test, FAIL on dev (RTMDet) |
| 3 | p99 ≤ 2 × p50 per-frame latency (eval targets) | Test: RTMDet 1.79–1.81; D-FINE 1.96–2.08 across runs | PASS (RTMDet); borderline (D-FINE) |
| 4 | Every frame analysed offline (eval targets) | 1.0 on every run | PASS |
| 5 | Every frame analysed live, on every stream (eval targets) | Analysed fraction 1.0 / 1.0 / 1.0 / 0.9999; 0 dropped over 65 min | PASS |
| 6 | Live analysis at 18 fps on all 4 cameras (PLAN_v6) | **15.00 fps**, which is the measured camera maximum: each stream delivers 15.2 fps (4 × 60 s probe) | **FAIL against 18; PASS against the measured maximum.** Needs the NVR set to ≥ 18 fps |
| 7 | No CAMERA_LAGGING (SITE_SESSION, PLAN_v6) | 0 lagging samples, max queue lag 0.12 s | PASS |
| 8 | GPU ≤ 80 % (PLAN_v6) | Clean window (38.8 min): p95 29 %, max 34 %. Whole run: max 83 % while eval jobs shared the GPU | PASS for this load, but **UNVERIFIED for a pool load** (no pose runs on standard CCTV) |
| 9 | CPU ≤ 70 % (PLAN_v6) | Clean window: p95 17.8 %, max 20.0 %; whole run max 36.6 % | PASS (same caveat) |
| 10 | 9 people per camera at 18 fps (PLAN_v6) | Live: UNVERIFIED (office scenes, no pose). Bench 2026-10-20: 4 × 1440p crowded clip at 20 fps analysed 0.98, 0 drops (`docs/evidence/v7/crowded_fps_profile.md`) | **UNVERIFIED** live |
| 11 | Admission holds at the target fps (CLAUDE.md, capacity) | All 4 cameras admitted and monitored at the 18 fps target, 12.4 s after start | PASS |
| 12 | Camera stream: H.265, 2560 × 1440, 15 fps, keyframe every 15 frames (CAMERA_SETUP / SITE_REQUIREMENTS) | **H.264, 2688 × 1520**, 15.2 fps, **keyframe every 30 frames (2 s)** (ffprobe on 120 s per camera) | **FAIL** on codec, size and keyframe interval; PASS on fps |
| 13 | Heartbeat "Access is denied" must not recur over 30 min (this task) | 0 errors over 67 min (before: 98 in the log for 10-19/20, up to 24 an hour) | PASS |
| 14 | No alerts on the deck person, lane clip, no zones (v7 plan) | D-FINE (uploads): **0**; RTMDet (live default): **7** | PASS for uploads; FAIL for RTMDet |
| 15 | Deck / water accuracy ≥ 0.95 with either detector (v7 plan) | RTMDet 0.89; D-FINE 1.00* | FAIL (RTMDet); PASS* (D-FINE) |
| 16 | Uploads (D-FINE): test ≥ 4/5 events, median time to alert ≤ 15 s (v7 plan) | 4 / 5, median 9.6 s | PASS |
| 17 | Test false alerts ≤ 8, 0 on normal swimming (v7 plan) | RTMDet 7 / 0; D-FINE 4 / 0 | PASS |
| 18 | Upload processing ≥ 45 fps (≥ 1.5 × real time) (v7 plan) | Test, D-FINE: 67–76 fps (1.7–1.9 ×). One dev run reached 40 fps while the full test suite ran alongside | PASS |
| 19 | Event recall not lower than v6 (dev 7/11, test 3/5) (v7 plan) | RTMDet dev 7/11, test 3/5; D-FINE dev 9/11*, test 4/5 | PASS |
| 20 | Photo / monitor reported as a person on live CCTV: 0 (v7 plan) | Not counted on this run's footage | **UNVERIFIED** (needs a human look at `build/site_capture/normal`) |
| 21 | Stream stalls detected and recovered (CLAUDE.md safety rules) | 3 CAMERA_STALLED faults, all real: Meeting room at start-up (first frame after 8.6 s); twice on a 5th camera whose source never opened (added and deleted by the operator at 12:46) | PASS (faults raised and cleared) |
| 22 | PASSIVE_DROWNING never gated by size or pose (CLAUDE.md) | `small_person_gate` excludes PASSIVE and SUBMERSION; unit tests | PASS |
| 23 | Staged drills D1–D7 (SITE_SESSION) | — | **UNVERIFIED** |
| 24 | In-water recall / alert review on labelled site footage (SITE_SESSION "FINAL") | No labelled site footage exists | **UNVERIFIED** |

\* D-FINE was fine-tuned on pseudo-labels of the **same 10 dev clips**, so its dev numbers are optimistic. Its test numbers
are the fair ones.

## 3. Offline evaluation (heldout_v1, test split, current code)

Thresholds 2.12.1 (small-person gate on at 50 px, fps-scaled frame rules on). High mode, TensorRT. Test footage 249 s
(6 clips, 5 events). Runs: `build/eval_v7/fps_test_rtm_on`, `fps_test_dfine_on` (identical to `f5r2_test_*_G50`).

| | RTMDet (live default) | D-FINE (uploads only, `dev_only`) |
|---|---|---|
| Event recall | **3 / 5 (0.60)** | **4 / 5 (0.80)** |
| Time to alert per hit | 49, 5, 32 s (median 32 s) | 7, 9, 21, 11 s (median 9.6 s) |
| False alerts | 7 | 4 |
| False alerts per camera-hour | **101.3** | **57.9** |
| False alerts on normal swimming | 0 | 0 |
| False alerts on unlabelled people ("no_person") | 7 (all of them) | 4 (all of them) |
| Missed | floatable_drowning_v2, passive_drowning_v3 | active_drowning_v7 |
| Per-frame latency p50 / p99 | 16.2 / 29.2 ms | 14.1 / 27.6 ms |

Ground truth is event-level: only the victim is labelled, so every alert on another swimmer counts as a false alert,
even when that swimmer may also be in trouble. The false-alert rates are on 4 minutes of footage and have wide
uncertainty. Dev, for reference: RTMDet 7/11 events, 49 false alerts (287 per camera-hour); D-FINE 9/11*, 41*.

Changes judged on test today:

| Change | Rule (written before the run) | Test result | Decision |
|---|---|---|---|
| F5 small-person gate, 50 px (`docs/evidence/v7/f5_small_person_gate.md`) | recall not lower, false alerts per camera-hour not higher, both detectors | RTMDet 3/5 = 3/5, FA 8 → 7 (115.7 → 101.3/h); D-FINE 4/5 = 4/5, 4 = 4 | **ON** (2.11.1) |
| Frame rules scaled to fps (`docs/evidence/v7/fps18_frame_rules.md`) | events ≥, FA ≤, normal swimming ≤, both detectors | identical on every clip | **ON** (2.12.1); a correctness change, no measured gain |
| F1, F4 | — | measured earlier, failed | stay **OFF** |

## 4. Upload path (D-FINE)

- The worker loads D-FINE (log line above). The analysis is the same `SourcePipeline` the eval harness runs.
- **Deck person not flagged:** lane clip (`ds_active_drowning_v2`), track 1 = the man on the deck: D-FINE **0** alerts,
  RTMDet 7 (`build/eval_v7/upload_dfine_dev` vs `fps_dev_off`). Dev-split caveat above.
- **Normal swimmers not called drowning:** 0 false alerts on normal swimming on test for both detectors, 0 on dev for
  D-FINE (RTMDet: 1 on dev).
- **Needs a human eye:** uploading the clips through the UI with a camera or water area selected, and checking the
  Model View overlay. This needs the admin login.

## 5. Live results (4 RTSP cameras, 65 min)

Run 11:50:47–12:55:47, API on the current code with `LIFEGUARD_TARGET_ANALYSIS_FPS=18` and the loopback metrics
listener on. **Clean window** 12:17–12:55 (38.8 min): no other GPU jobs. Before 12:17, dev evals and a lock compile
shared the machine.

| Camera | Analysed fps | Frames analysed / read | Proc. ms p50 / p95 / max | Max lag | LAGGING | Reconnects | Dropped |
|---|---|---|---|---|---|---|---|
| Engineer Room | 15.00 | 58,437 / 58,437 | 35.5 / 43.6 / 46.1 | 0.05 s | 0 | 0 | 0 |
| Entrance are | 15.00 | 58,439 / 58,438 | 32.5 / 41.3 / 49.2 | 0.06 s | 0 | 0 | 0 |
| Meeting room | 15.00 (except a 40 s operator stop at 12:46) | 57,697 / 57,698 | 32.5 / 40.0 / 92.3 | 0.12 s | 0 | 0 (1 at start-up) | 0 |
| Reception area | 15.00 | 58,438 / 58,437 | 35.8 / 43.1 / 46.9 | 0.03 s | 0 | 0 | 0 |

| Resource | Clean window p50 / p95 / max | Whole run max |
|---|---|---|
| GPU | 23 / 29 / 34 % | 83 % (with eval jobs) |
| NVDEC | 4 / 7 / 10 % | 10 % |
| CPU (system) | 13.1 / 17.8 / 20.0 % | 36.6 % |
| API process | 2.9 / 3.8 / 4.1 cores | 4.1 cores |
| GPU memory | 3.1 / 3.4 / 3.5 GB | 4.2 GB |

- **Alerts raised:** 3, all CAMERA_STALLED faults (section 2, row 21). No behaviour alerts (standard CCTV).
  "False-alert candidates": none; nothing to review.
- **Footage for labelling** (outside git): `build/site_capture/normal/20261021_115113_*.mp4`, 4 × 120 s, stream copy.
  These are office scenes, so they are useful only for checking photo / monitor / duplicate boxes, not drowning.
  **Needs human review.** There are no alert clips because no behaviour alert was possible.
- **Findings:**
  1. **Start-up stall on the 4th camera:** the NVR opens streams one after another (first frames at 2.3, 4.2, 6.3 and
     8.6 s). The 5 s stall timer fired on the 4th before the 12 s first-frame timeout, a spurious CAMERA_STALLED at
     every start with 4 cameras. The 2 s keyframe interval (row 12) makes it worse.
  2. **WebSocket handler error:** at 12:46:53, `RuntimeError('cannot reuse already awaited coroutine')`, "Task exception
     was never retrieved", when two UI WebSockets reconnected together. No visible effect; not investigated.
  3. The 5th camera added at 12:46 never opened; the faults were raised correctly and the camera was deleted by the
     operator.

## 6. Known gaps, stated plainly

- **Victims are often not detected or tracked.** Ladder victim scored 0.02–0.09 by RTMDet; `active drowning (5)` is
  never tracked; in the 7 labelled ACTIVE events the victim was untracked in most scorings (F5 note, diagnosis 2). Dev
  in-water recall is 0.40, with 0.03 for 40–80 px people and 0.00 under water.
- **"Active drowning" is rarely the label.** When the victim is tracked, pose is unreliable in ~85 % of scorings and the
  vertical cue falls back to the box shape, which fails for a thrashing swimmer.
- **The test clips are phone and social-media video**, mostly 1080p or less, close and low. They are not 2560 × 1440
  CCTV from a 5 m mount. Results may not transfer either way.
- **No labelled site footage**, and no pool camera is connected; the live run tested office cameras in CCTV mode.
- **False alerts are high:** 101 per camera-hour (RTMDet) on test, 287 on dev. That is several alarms an hour per
  camera on a busy pool if it transfers.
- **D-FINE is dev-only** (training data not rights-cleared) and is the better detector on test (4/5 vs 3/5, 58 vs 101
  false alerts per hour). The live default stays RTMDet.
- The camera configuration does not match the documented setup (row 12), and live analysis is capped at 15 fps by the NVR.

## 7. Score

| Criterion (weight) | Score | Why |
|---|---|---|
| Detection reliability (35) | **13** | Test event recall 0.60 (live default) / 0.80 (uploads); in-water recall 0.40; victims often untracked; live drowning detection untested |
| False-alarm rate (25) | **7** | 101 / 58 false alerts per camera-hour on test; 0 on the (short) normal-swimming clip; dev 287 / h |
| Live stability and performance (20) | **14** | 65 min, 4 cameras, every frame analysed, 0 lag, 0 drops, GPU ≤ 34 %, CPU ≤ 20 %. Minus: no pose load (CCTV mode), 15 not 18 fps, spurious start-up stall, camera setup off-spec |
| Consistency across runs (10) | **9** | Re-runs on 10-20 and 10-21 give identical events and false alerts (RTMDet test 0.60 / 115.72 per hour both days); A/B arms reproduce exactly; latency varies with load |
| Documentation and test coverage (10) | **8** | 1,223 tests pass (4 known environment failures); every change has a pre-registered A/B note; CLAUDE.md is kept untracked by the repo's own hygiene rule; the hand UI checks are not done |
| **Total** | **51 / 100** | |

**Cap at 60 while there is no labelled site footage.** The weights reward detection and false-alarm rates measured on
4 minutes of web video that does not look like the site. Until the system has been measured on labelled footage from
its own cameras, no score can claim more than "it works on someone else's clips". The cap would only bind if the
measured score rose above 60. **Label: assist / pilot only.** A lifeguard stays responsible; Life Guard is an extra
pair of eyes whose misses and false alarms are not yet characterised on site.

## 8. Next steps, in priority order

1. **Connect the pool cameras and fix their stream settings:** H.265, 2560 × 1440, keyframe every 15 frames, and
   ≥ 18 fps if 18 fps analysis is wanted (NVR setting). Set them to `pool`, draw zones, calibrate.
2. **Collect and label site footage** (SITE_SESSION, legal gate first): ≥ 4 camera-hours of normal sessions plus
   staged drills D1–D7 with adults. Label victims and every person at 1 fps. Re-run the eval on it; that replaces this
   report's offline numbers.
3. **Retrain D-FINE on rights-cleared footage** (the site's own, after the agreements), so the better detector can
   ship live, and re-measure in-water recall for small and underwater people.
4. **Retry F4 (resolution mosaic) on site footage:** it failed on web clips, but the gap it targets (40–80 px people,
   recall 0.03) is exactly what elevated CCTV shows.
5. Fix the start-up stall race (stall timer vs. sequential NVR opens) and the WebSocket "already awaited coroutine"
   error.
6. A vertical / distress cue that works without full pose (ACTIVE_DROWNING), tuned only on labelled site data.
7. Human items: the admin uploads the test clips through the UI and checks the Model View; a person checks the 4
   normal-footage clips for photo / monitor boxes; sign-off on shipping the TensorRT runtime (licence, `tools/check_licenses.py`);
   start Life Guard from the other Windows account once.
