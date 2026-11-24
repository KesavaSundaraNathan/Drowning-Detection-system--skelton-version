# Site session (Phase 11 WP6)

One visit to the client pool measures the use-case criteria on the real cameras, with the real lighting, on the client's
server. The kit is `tools/site_session.ps1` (it runs `tools/site_session.py`). Every rule it applies was fixed **before**
any recording: `eval/site/levers_phase11.json` → `gates_wp6`. The pass levels there are **proposed**; the client
confirms or changes them, recorded as an amendment of that file, before the session and never after a result is seen.

## 1. Before the visit

| Item | Who | Why |
|---|---|---|
| DPIA, data-sharing agreement, counsel approval, retention period (`DATA_PROTOCOL.md` §5) | Client + RVV + counsel | The kit refuses to record without their **references** (the documents themselves are never stored). |
| Performer consent forms, one per adult performer, with a pseudonymous id (P01, P02, …) | Client | Only listed ids can take part in a drill. |
| Recording signage up; for a normal session, the public notice of the recording window | Client | Part of the legal gate. |
| **Adults only** | Everyone | No person under 18 in the water or in view during ANY recording (`gates_wp6` decision 1). The operator attests it before every run. D6 uses adult learners. |
| Mounting, lenses, lights, network and UPS as in `SITE_REQUIREMENTS.md`; cameras configured as in `CAMERA_SETUP.md`; all 4 calibrated to the same pool plan, the pool zones drawn (the teaching platform at least) | Client + installer | The `check` step verifies the measurable parts. |
| `deploy/site/site_profile.json`: the 4 camera names as registered on the *Cameras* page | Installer | The configuration check compares against it. |
| The pass levels confirmed by the client (amendment of `gates_wp6.criteria`) | Client + RVV | Pre-registration. |
| A lifeguard in the water and a deck supervisor for the whole session | Client | Safety (§4). |

**Dry run (recommended, without swimmers, about 15 min, a day or more before):**

```powershell
powershell -ExecutionPolicy Bypass -File tools\site_session.ps1 -Base https://<site> -Step header -SessionId DRY1
powershell -ExecutionPolicy Bypass -File tools\site_session.ps1 -Base https://<site> -Step check -SessionId DRY1
powershell -ExecutionPolicy Bypass -File tools\site_session.ps1 -Base https://<site> -Step capacity -SessionId DRY1
powershell -ExecutionPolicy Bypass -File tools\site_session.ps1 -Base https://<site> -Step live -SessionId DRY1
```

Fix every FAIL before the drill day: GOP, codec, frame rate or B-frames in the camera menu; coverage holes by camera
aim, lens or mounting height; capacity by fewer cameras per GPU or a stronger GPU.

## 2. What the kit does

Run elevated on the server, as an administrator. It reads the camera sources with the server's own keys. The admin
credentials come from `LIFEGUARD_SITE_USER` / `LIFEGUARD_SITE_PASSWORD`, or a prompt; never the command line.

- **Pinned tools.** FFmpeg, ffprobe (and MediaMTX for `-Smoke`) are checked against `deploy\windows\TOOLS.json`
  (sha256). Any mismatch: refused.
- **Session folder.** `C:\ProgramData\LifeGuard\site_session\<SessionId>\`, with inheritance removed and access for
  SYSTEM and Administrators only, verified after creation. The recordings are personal data; no service account
  needs them.
- **No camera URL or password is ever written.** FFmpeg's error output is redacted; the geometry snapshot has the
  camera source emptied.

| Step | What it does | Output |
|---|---|---|
| `header` | DATA_PROTOCOL 9.3 header + the legal references (prompted) | `session.json` |
| `check` | Configuration vs `site_profile.json`. Per camera: a 20 s ffprobe window over RTSP/TCP (codec, size, **measured** fps, keyframe interval over ≥ 2 GOPs, B-frame reordering, frame-rate timing) + the live capture's own open (first frame, hardware decoding). Coverage: every 0.5 m cell must have a responsible camera, all cameras on one pool | `checks\config_check.json`, `camera_check.json`, `coverage.json` + `coverage.png` |
| `capacity` | Tiling off: the running API's measured capacity. Tiling on: **computed** from the same table (monitoring is never stopped) | `checks\capacity.json` |
| `live` | Every enabled camera for 5 min: MONITORED throughout, analysed fraction ≥ 0.99, 0 dropped | `checks\live_run.json` |
| `record` | One FFmpeg `-c copy` recorder per camera (wall-clock timestamps), then the drill runner (§3). Afterwards: the geometry snapshot, the SYNC offsets, `RIGHTS.csv` rows | `recordings\drills\*.mkv`, `recordings.json`, `drill_log.json/.csv`, `geometry.sqlite`, `sync.json` |
| `normal` | An adults-only normal session for `-Hours` (same legal gate, plus the public-notice reference) | `recordings\normal\` |
| `replay` | OFF / GATING / MULTIVIEW arms through the real runtime (lockstep, one session clock); the OFF arm runs the detector, the others replay its output exactly | `replay\<block>\<arm>\` |
| `report` | The PROVISIONAL report (`-Final` after labels and review, §5) | `report\site_session_report.md/.json`, `review_normal.csv`, `clips\` |
| `all` | header (if missing) → check → capacity → live → record → replay → report | |

`-Smoke` (laptop / staged install, no elevation needed; start the staged stack first with `deploy\windows\stage_run.ps1`): one simulated camera (MediaMTX on 127.0.0.1 with MoQ off, plus FFmpeg `-c copy`
of a 1440p H.265 clip), registered for the run and removed afterwards, with scripted drills. No person is recorded.

## 3. The drills (`lifeguard/site/drills.py`)

Every run: performer ids, position, the **adults-only attestation** (type `YES`), then one Enter per mark. Marks are
taken on the server clock. Outcome `aborted` is allowed at any time; the performer can stop with the agreed signal.

| Code | Drill | Marks | Runs |
|---|---|---|---|
| SYNC | One light flash (or clap) every camera sees | flash | 1 (at the start) |
| D1 | 6-15 adults spread over all lanes, 3 min | start, end | 1+ |
| D2 | ONE adult swims 4 × 25 m through every camera handoff, alone in the water | start, end | 1+ |
| D3 | Entries one at a time by adults: step in, ladder, jump | entry (each) | 1+ |
| D4 | Lane changes, on and off the teaching platform | start, end | 1+ |
| D5 | Controlled submersion: a trained adult sits on the bottom 10-15 s, rescuer in the water; far end of every camera and the overlap areas | under, surfaced | **10** |
| D5b | As D5; 2-5 s after 'under' a second adult swims past in the ADJACENT lane within ~2 m, **never over the performer** | under, pass, surfaced | **5** |
| D6 | A standing adult instructor + ADULT learners with floats, 5 min (false-alarm check) | start, end | 1+ |
| D7 | Glare passes (afternoon), 3 min | start, end | 1+ |
| D7s | Optional, lifeguard's decision: one controlled submersion inside the glare | under, surfaced | 0-1 |

D5b measures the WP3 safety finding on site: someone surfacing near a vanished person can cancel their watch. A miss is
reported and becomes a pre-registered lever proposal; nothing changes automatically.

## 4. Safety (non-negotiable)

Adults only. A qualified lifeguard in the water for every submersion. Nobody swims over a submerged performer. The
lifeguard can stop any drill. DATA_PROTOCOL §4.1 applies in full.

## 5. After the session

1. **Provisional report** (on the day): rows computable from the drill log, the live run and the replay. Rows that
   need labels or a review are **PENDING**, never PASS.
2. **Normal-session review**: `report\review_normal.csv` lists every audible alert. The reviewer fills `verdict`
   (`TRUE` / `FALSE`) and saves it as `report\review_normal_completed.csv`.
3. **Box labels** (CVAT, 1 fps sampled frames of D1 and D6) for in-water recall, id switches inside alerts and
   time-to-first-detection.
4. **Final report**: `-Step report -Final`. Lever decisions follow the registered rules only. A tie keeps the shipped
   default; a lever whose inputs are missing stays PENDING and is not enabled.

**Data:** the session folder stays on the server, or on RVV-controlled encrypted storage as the data-sharing
agreement says. `RIGHTS.csv` holds one row per recording (`rights_status` `pending` until the agreement is
countersigned). Retention and deletion follow the agreement, and deletion is logged (DATA_PROTOCOL §8).

## 6. Troubleshooting

| Symptom | Meaning / action |
|---|---|
| `legal gate: ... missing` | A reference is empty: re-run `-Step header`. |
| `REFUSED: adults-only attestation refused` | Nothing was recorded for that run. Clear the pool of minors, then repeat the run. |
| Camera check `gop` FAIL "irregular keyframes" | H.265+ / smart codec / dynamic GOP is on: switch it off (`CAMERA_SETUP.md`). |
| `b_frames` FAIL | B-frames / SVC on in the camera menu. |
| `fps_measured` FAIL | The camera is not at 15 fps. |
| `opencv_fps` WARN | The stream carries no frame-rate timing; Life Guard uses `CAMERA_FPS_FALLBACK`. |
| Coverage holes | Aim, lens or height; recalibrate after any change. |
| `sync.json` verdict NO_FLASH | No clear flash seen: the replay uses no offsets (reported). Repeat SYNC at the next block if possible. |
| `WARNING: recorder stopped for ...` | A camera dropped its recording session: check the camera's RTSP session limit (§ CAMERA_SETUP). |
