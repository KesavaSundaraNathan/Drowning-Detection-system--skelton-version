<!-- Frozen copy of sections 1.0-1.7 of the 2026-10-07 production-readiness audit of the v2 tree, taken from the audit session transcript on 2026-10-08 (Phase 10). The row IDs used by docs/AUDIT_TRACEABILITY.md are derived from these tables by tools/check_traceability.py, so this file must not be edited. Redaction: the probe-returned camera credential is replaced by <user>:<password>. -->

# AquaGuard Production Readiness Audit

**What this is based on.** I read these line by line: the entry point, config, auth and security code, the DB layer, every route file (with auth mapped per endpoint), the upload, alert, webhook and watchdog services, the shared pipeline, the GPU runtime, the live worker, the frame buffer, and the scene-tracker alert logic. For the detector, risk engine, classifier, tracker, camera capture, training, evaluation and frontend I mapped their functions and used the eval reports, tests and docs rather than reading every line.

I also ran four checks:
- **Test suite:** 278 passed, 1 failed (the known flaky `test_worker_reconnects_with_backoff`), 1 skipped.
- **Licence gate:** passed (99 packages, 10 model files).
- **Frontend build:** passed.
- **Live API probe:** run against a throwaway database. Your real DB was not touched.

## 1. Performance of Project

### 1.0 Critical findings across the codebase (ranked)

| # | Severity | Finding | Evidence |
|---|---|---|---|
| 1 | **BLOCKER** | Detection accuracy is far below what a life-safety product needs | In-water person recall is **0.41** against a 0.95 target. Underwater swimmers: 0.07. Swimmers 40–80 px tall: **0.03**. Drowning-event recall is **6/20 (0.30)** across 19 hand-labelled clips. Median time to alert is **23.4 s** (p90 is 46.2 s). There are 32 false alerts per camera-hour (`eval/reports/dataset_v1_before`). Box-level ground truth totals only 0.009 camera-hours, and the code was tuned on that same clip. |
| 2 | **BLOCKER** | Anyone on the network can read everything without logging in | Confirmed by the probe: `GET /api/cctv/cameras` with no token returned `rtsp://<user>:<password>@10.0.0.5:554/ch1`. The live MJPEG video of swimmers, alert snapshots, CSV export, `/api/metrics` and the WebSocket are also open. `require_auth` is applied only to routes that change data. |
| 3 | **BLOCKER** | Nothing raises an alarm when the system itself fails | There is no heartbeat or dead-man's switch, so if the process crashes the pool goes silently unwatched. The watchdog only restarts threads that have died, not ones that are stalled. The siren plays in a browser tab and is blocked by autoplay rules until someone clicks. The webhook is fire-and-forget on a daemon thread: unsigned, not saved, and lost if the process crashes. |
| 4 | **BLOCKER** | A clean install quietly falls back to the CPU | `detection/cuda_dlls.py` finds the CUDA/cuDNN DLLs in the dev venv's `torch\lib` folder. A clean `requirements.txt` install has neither torch nor the `nvidia-*-cu12` wheels, so ONNX Runtime silently runs on the CPU, far below real time. |
| 5 | HIGH | Sessions and login are weak | Tokens never expire. They are stored in plain text in the DB and in `localStorage`, and are accepted as `?token=` (which leaks into logs and browser history). A default `admin/admin123` account is created automatically. There is no rate limit: the probe made 30 failed logins and the next correct one still succeeded. `SECRET_KEY` is never used, `role` is never checked, and `DEBUG` defaults to True. |
| 6 | HIGH | Throughput does not scale with cameras | One GPU thread runs the detector for each camera in turn (~19–30 ms per frame). With just **2 streams**, only **81%** of live frames were analysed. `MAX_CAMERAS=16` cannot run at full frame rate on one GPU. Upload analysis and training share that GPU with the live safety cameras. |
| 7 | HIGH | Remote-code-execution and SSRF openings (require login) | `/api/training/load/{name}` calls `torch.load(weights_only=False)`, which unpickles arbitrary code. Camera `rtsp_url` and `video_path` values go straight to FFmpeg, which accepts any protocol or local file. |
| 8 | HIGH | The release isn't under version control or reproducible | 37 paths are uncommitted or untracked, including all of tracking v3 and `motion.py`. There is no CI, Docker setup or service definition. There is no lockfile: requirements use `>=` ranges only. The dev venv has both `opencv-python` and `opencv-contrib-python` installed, plus torch. |
| 9 | HIGH | Almost no observability | Logging is never configured, so every `logger.info` in the detection code is dropped. There are 46 bare `print()` calls and 95 broad `except` blocks. There is no Prometheus endpoint and no log rotation. |
| 10 | MEDIUM | Legal and compliance exposure | `Datasets/` contains stock, YouTube and WhatsApp footage with unclear rights. Pool video and snapshots (often of children) are unauthenticated and have no retention policy. Nothing has been tested against ISO 20380 (the standard for computer-vision drowning detection). |

### 1.1 Backend: entry point, config, core

| File | Score | Verdict | Required fixes |
|---|---|---|---|
| [main.py](main.py) | 4 | Clean wiring, but startup carries on after failures | Abort startup if `init_database()` fails (today it logs the error and keeps serving). Move to `lifespan`. Configure logging (JSON, rotated). Add a global exception handler and security headers. Refuse to start with default secrets when `DEBUG=false`. |
| [config/settings.py](config/settings.py) | 3 | Every default is a development default; importing it creates folders | Move to `pydantic-settings` with validation. Default `DEBUG` to False. Require `DEFAULT_ADMIN_PASSWORD` on first boot. Use `SECRET_KEY` or delete it. Remove the `YOLO_*` aliases. Take `mkdir` out of import. |
| [config/constants.py](config/constants.py) | 5 | 203 documented tunables, tuned on about one clip | Version the file and record which eval report justifies each value. Validate ranges. Keep per-site overrides separate from code defaults. |
| [core/security.py](core/security.py) | 2 | Auth exists but only guards writes | Add `dependencies=[Depends(require_auth)]` at router level for everything except login and health. Remove `?token=`. Add absolute and idle expiry. Hash tokens at rest. Enforce `role`. |
| [core/passwords.py](core/passwords.py) | 6 | bcrypt is used correctly | Remove the unsalted SHA-256 legacy path. Add a password policy. Run a dummy hash for unknown usernames so timing doesn't reveal which exist. |
| [core/events.py](core/events.py) | 4 | Simple and safe to call from any thread | Broadcast awaits each client in turn, so one slow socket delays every alert. Use per-client queues and `gather` with timeouts. Authenticate the socket. Buffer events sent before the loop is bound. |
| [database/db.py](database/db.py) | 3 | SQLite with several writer threads | Turn on WAL mode, `busy_timeout` and `pool_pre_ping`. Move to PostgreSQL beyond about 4 cameras or more than one node. |
| [database/models.py](database/models.py) | 4 | No foreign keys, few indexes, naive datetimes | Add foreign keys for sessions/alerts/videos. Index `alerts(created_at, camera_id, status)`. Use timezone-aware timestamps. Add `expires_at` to sessions. |
| [database/init_db.py](database/init_db.py) | 3 | Hand-written migrations and automatic default admin | Switch to Alembic. Create the admin only through an explicit one-time CLI command. |
| `database/drowning_demo.db` | — | Development data sitting in the deliverable | Exclude it. |

### 1.2 Backend: routes and schemas

| File | Score | Verdict | Required fixes |
|---|---|---|---|
| [routes/cctv_routes.py](routes/cctv_routes.py) | 2 | Leaks camera credentials (confirmed); MJPEG is open | Require auth. Mask `source` (`rtsp://***@host`). Use short-lived signed stream URLs, because `<img>` tags can't send headers. Accept only `rtsp://` URLs or device indexes. |
| [routes/training_routes.py](routes/training_routes.py) | 2 | Accepts any `dataset_path`, unpickles files, and trains on the live GPU | Remove from the production build. Train on a separate machine. If it stays, use `weights_only=True` and a path allowlist. |
| [routes/auth_routes.py](routes/auth_routes.py) | 3 | No brute-force protection | Add a rate limiter and lockout. Accept tokens from the header only. |
| [routes/alert_routes.py](routes/alert_routes.py) | 3 | List, get, snapshot and CSV are open (snapshot path containment is done correctly) | Require auth. Paginate the list. Escape CSV cells that start with `=`, `+`, `-` or `@`. |
| [routes/ws_routes.py](routes/ws_routes.py) | 3 | Any client can subscribe | Require a token on connect. Check `Origin`. Time out idle clients server-side. |
| [routes/detection_routes.py](routes/detection_routes.py) | 4 | `background=false` processes a whole video inside the request | Remove sync mode or cap it. Use a job queue with concurrency 1 and lower priority than live cameras. |
| [routes/metrics_routes.py](routes/metrics_routes.py), [routes/system_routes.py](routes/system_routes.py) | 4 | Environment and system details are exposed without auth | Require auth. Expose Prometheus `/metrics` on an internal port only. |
| [routes/video_routes.py](routes/video_routes.py) | 5 | Upload streams to disk with a size cap (good); checks the file extension only | Probe the container before accepting. Add quotas. Require auth on reads. |
| [routes/zone_routes.py](routes/zone_routes.py), [routes/schedule_routes.py](routes/schedule_routes.py) | 5 | CRUD is fine; a schedule change can mute sirens | Require auth on reads. Keep an audit log of who changed schedules and zones. |
| `schemas/*` (7 files) | 6 | Adequate Pydantic v2 models | Set maximum string lengths and allowed URL schemes. Mask secrets in responses. |

### 1.3 Backend: services

| File | Score | Verdict | Required fixes |
|---|---|---|---|
| [services/alert_writer.py](services/alert_writer.py) | 7 | The best-built service: never raises, snapshot paths stay inside their folder | Write JPEGs off the behaviour thread. Add snapshot retention and purging. |
| [services/alert_dispatch.py](services/alert_dispatch.py) | 4 | Right idea, but delivery isn't dependable | Add a durable outbox table and retry worker. Sign requests with HMAC plus a timestamp. Use a bounded thread pool. Escalate if not acknowledged within N seconds. Health-check the second channel. |
| [services/camera_watchdog.py](services/camera_watchdog.py) | 5 | Only notices dead threads | Detect stalls (last frame or last analysis older than N seconds) and raise a fault alarm. Escalate after K restarts. Send an external heartbeat. |
| [services/alert_service.py](services/alert_service.py) | 6 | The workflow is sound (`test_clear_alerts.py`) | Use timezone-aware datetimes instead of `utcnow`. Keep an audit trail for acknowledge and false-alarm actions. |
| [services/detection_service.py](services/detection_service.py) | 5 | Progress and telemetry are handled well | Use a bounded job queue isolated from live inference. On restart, recover videos stuck in "Processing". |
| [services/cctv_service.py](services/cctv_service.py) | 4 | Returns the raw source string; worker registry is a module global | Mask the source. Encrypt camera credentials at rest. Move state into a service object. |
| [services/auth_service.py](services/auth_service.py) | 4 | No session lifecycle | Add expiry, rotation, lockout and a last-login audit. |
| [services/zone_service.py](services/zone_service.py), [services/schedule_service.py](services/schedule_service.py), [services/video_service.py](services/video_service.py) | 6 | Sound | Add an audit trail. A muted state must be visible in the UI. |

### 1.4 Backend: detection engine

| File | Score | Verdict | Required fixes |
|---|---|---|---|
| [detection/pipeline.py](detection/pipeline.py) | 7 | One clean entry point shared by upload, live and the harness, so they can't drift apart | None critical. |
| [detection/live_runtime.py](detection/live_runtime.py) | 4 | Runs detection one camera at a time; one pose failure fails the whole round | Export a static-batch detector (4/8) and batch frames across cameras. Isolate failures per camera. Support more than one GPU worker. Lock `batch_sizes`. |
| [detection/live_pipeline.py](detection/live_pipeline.py) | 6 | Lag and dropped frames are measured honestly and surfaced | Raise `processing_lag` as an alarm-class event once it exceeds the alert budget. |
| [detection/camera_stream.py](detection/camera_stream.py) | 6 | Bounded queue, counted drops, reconnect backoff | Allow only certain protocols. Use NVDEC/GStreamer for more than four 1080p streams. Detect stalls separately from the read timeout. |
| [detection/frame_buffer.py](detection/frame_buffer.py) | 6 | Fine | Skip JPEG encoding when nobody is watching. |
| [detection/detector.py](detection/detector.py) | 5 | A 1,000-line class; the HOG fallback doesn't work on pools | Treat HOG as "unavailable" in production: fail loudly and alarm. Split out the zoom, faint-birth and ROI logic. |
| [detection/rtm_backend.py](detection/rtm_backend.py) | 6 | Provider chain and warm-up are good; `RTM_IMGSZ` is silently ignored | Warn or fail when the requested size doesn't match the model. Export with dynamic shapes if a larger input is needed. |
| `detection/bytetrack/*` | 7 | Matches supervision on 3 clips; unit-tested | None critical. |
| [detection/tracking.py](detection/tracking.py) | 5 | Four layered heuristics tuned on one dev clip, so likely overfitted | Validate on held-out labelled footage before trusting it. |
| [detection/motion.py](detection/motion.py) | 6 | Reasonable MOG2 gating | Commit it to git. |
| [detection/scene_tracker.py](detection/scene_tracker.py) | 5 | Careful ghost/submersion logic; 1,000 lines; schedule windows can silence sirens | Split it up. Never mute SUBMERSION. Show a persistent "MUTED" state. |
| [detection/risk_engine.py](detection/risk_engine.py) | 4 | 1,347 lines of hand-tuned kinematics running on boxes only (pose is off) | Freeze it. Replace it gradually with the learned model once reviewed client data exists. Add property tests. |
| [detection/classifier.py](detection/classifier.py) | 4 | Global thresholds can be changed at runtime through the API, with no audit | Version and audit the thresholds file. |
| [detection/behavior_calibration.py](detection/behavior_calibration.py), [detection/simulation.py](detection/simulation.py) | 4 | Calibrating on synthetic scenes is circular: synthetic recall is 1.0 while real event recall is 0.30 | Use for regression testing only. Never cite it as accuracy. |
| [detection/zone_state.py](detection/zone_state.py), [detection/pool_geometry.py](detection/pool_geometry.py), [detection/model_view.py](detection/model_view.py) | 6 | Sensible; leans toward the monitored side when unsure | None critical. |
| [detection/water_region.py](detection/water_region.py) | 4 | Finds the water by colour, which breaks indoors, at night, on blue tiles and in glare | Require calibration or drawn zones in production; keep the colour estimate as a warned fallback. |
| [detection/global_tracker.py](detection/global_tracker.py) | 3 | Never tested on a real multi-camera rig, and `claim_alert` can suppress a real alert | Disable cross-camera alert de-duplication until it has been tested on a real 4-camera, time-synced rig. |
| [detection/enhance.py](detection/enhance.py) | 4 | Tiling has never been measured end to end | Keep it off until measured. |
| [detection/floatables.py](detection/floatables.py) | 3 | Off by default; no trained model | Remove it or train a model. |
| [detection/cuda_dlls.py](detection/cuda_dlls.py), [detection/environment.py](detection/environment.py) | 4 | GPU support depends on torch being present in the venv | Add `requirements-gpu.txt` with the `nvidia-*-cu12` wheels. Refuse to start if `RTM_PROVIDER=cuda` but the CUDA provider isn't active. |
| `detection/behavior_model/*` | 3 | Inference code is fine; the only model is the `dev_only` bootstrap (18 normal, 0 distress windows) | Needs reviewed client data. |
| [detection/trainer.py](detection/trainer.py), [detection/train_worker.py](detection/train_worker.py) | 2 | Legacy OpenMMLab stack, launched from the server, unpickles files | Remove from the production image. |

### 1.5 Training, evaluation, scripts, tests

| Component | Score | Verdict | Required fixes |
|---|---|---|---|
| `evaluation/*`, [scripts/evaluate_pipeline.py](scripts/evaluate_pipeline.py) | 8 | The strongest part of the project: honest metrics, targets and A/B comparison | Run it in CI on a frozen held-out labelled set. |
| [scripts/check_licenses.py](scripts/check_licenses.py), `THIRD_PARTY_LICENSES.md` | 7 | Works | Check `faster-coco-eval` by hand (it has no licence metadata). |
| `training/dfine/*`, `training/behavior/*` | 4 | One real 2-epoch bootstrap run; history of a silent config-key bug | Validate generated configs against D-FINE's schema. Keep training on a separate host. |
| [scripts/setup_models.py](scripts/setup_models.py) | 6 | Works | None critical. |
| [scripts/export_coco_only_detector.py](scripts/export_coco_only_detector.py) | 4 | Needs mmcv compiled with MSVC in a separate venv; not reproducible | Archive the built ONNX with its sha256 as a release asset. |
| `scripts/build_*dataset*.py` | 4 | Produce auto-labels, not ground truth | Feed them only into CVAT review. |
| Other scripts, `utils/*` | 5–6 | Adequate | None critical. |
| `tests/` (18 files, 280 tests) | 7 | Hermetic and logic-heavy | Add an auth-matrix test, a 24-hour soak test, a real-weights smoke test in CI, and fix the flaky reconnect test. |

### 1.6 Frontend

| Component | Score | Verdict | Required fixes |
|---|---|---|---|
| [../frontend/package.json](../frontend/package.json) and build | 3 | A Figma Make export (`@figma/my-make-file`): no `tsconfig`, so TypeScript is never type-checked; no lint; no tests; React is only an optional peer | Add a strict `tsconfig` and run `tsc --noEmit` in CI. Add ESLint and Vitest. Make React a real dependency. |
| Bundle | 4 | One 655 KB JS chunk (199 KB gzipped), shipping MUI + emotion as well as Radix | Drop one UI library. Split code by route. |
| `lib/store.tsx` | 4 | A 1,447-line store that does everything | Split it by domain and test the alert reducer. |
| `lib/api.ts` | 5 | Token kept in `localStorage` | Use HttpOnly-cookie sessions and signed stream URLs. |
| `lib/wsClient.ts` | 6 | Reconnects with backoff | Show a full-screen "NOT MONITORING" banner and play a fault tone after N seconds disconnected. |
| `lib/alarm.ts` | 5 | Web Audio siren that autoplay blocks until the user clicks | Require an "arm audio" click at login. Run periodic self-tests. |
| `screens/LiveFeed`, `ZoneMapping`, `UploadDetection` (720–1,040 lines each) | 4 | Monolithic | Split into components and add component tests. |
| `components/camera/*`, `primitives/*` | 6 | Fine | None critical. |
| `components/ui/*` (~45 shadcn files) | 7 | Vendored, mostly unused | Prune. |
| `frontend-demo/` | 1 | A mock-data fork that has drifted from the real frontend and could be mistaken for live monitoring | Remove from the deliverable. |

### 1.7 Repository and documentation

| Item | Score | Verdict | Required fixes |
|---|---|---|---|
| `documentation/*` | 6 | Unusually honest, but 12 overlapping, mostly historical documents | Write one operator manual and one architecture document for the client. |
| `CLAUDE.md`, `claude md files/`, `Project description file.md`, `backend/run guide.txt`, `.vercel/`, `.pytest_cache/` | 1 | Internal and AI-agent files; the run guide's markdown is broken | Exclude from the deliverable. |
| `Datasets/` | 0 | Third-party footage with unclear rights | Never ship it. |
