# Life Guard v3

> **ASSIST / PILOT: not a certified life-saving system.** A pool-side aid for lifeguards, never a replacement for them.
> The UI states this until a signed acceptance report from the site's own footage passes the release gate. That gate has
> **not** passed in this version.

## 1. Project purpose

Life Guard v3 is a pool drowning-detection **assist** system for one Windows GPU server. It pulls the pool's RTSP cameras
(designed for four corner cameras, H.265 2560x1440 at 15 fps), detects and tracks every person with an ONNX detector
(RTMDet-m, COCO-only) on the GPU, scores each swimmer's behaviour (active drowning, passive drowning, floatable failure,
submersion), and raises alarms:

- in the operator web UI (alarm cards, alarm and fault tones, a full-screen **NOT MONITORING** overlay whenever any camera,
  the GPU path or the live feed is down);
- out of band on three channels: ntfy push, SMTP e-mail and an HMAC-signed webhook, with retry, tiered escalation and a daily self-test;
- through an independent supervisor service that raises a system fault when the API, a camera or the GPU scheduler stops.

It also keeps one identity per person across calibrated cameras (global ids, one alert card per person), and contains the offline evaluation harness (labelled clips, for measurement only; the product is RTSP-only), the signed release gate
and the site-session kit used to validate the system at the client's pool.

It is a clean rebuild of Life Guard v2 (frozen as tag `v2-baseline`; the vision code is a verbatim port for parity).

**Documents**

| Document | For |
|---|---|
| [`docs/architecture_overview.md`](docs/architecture_overview.md) | system design, exact tech stack, data flow, deployment |
| [`docs/current_modules.md`](docs/current_modules.md) | directory tree, every module, responsibilities, dependencies |
| [`docs/system_logic_specs.md`](docs/system_logic_specs.md) | execution flows, schemas, state management, known limitations |
| [`docs/OPERATOR_MANUAL.md`](docs/OPERATOR_MANUAL.md) | lifeguards, supervisors, admins |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | install, acceptance, operate, recover (incl. a developer run, section 9) |
| [`docs/SITE_REQUIREMENTS.md`](docs/SITE_REQUIREMENTS.md), [`CAMERA_SETUP.md`](docs/CAMERA_SETUP.md), [`SITE_SESSION.md`](docs/SITE_SESSION.md) | what the pool, cameras and network must provide; the validation visit |
| [`docs/SECURITY.md`](docs/SECURITY.md), [`DATA_PROTOCOL.md`](docs/DATA_PROTOCOL.md) | security model; footage and labels |
| [`docs/AUDIT_TRACEABILITY.md`](docs/AUDIT_TRACEABILITY.md) | every 2026-10-07 audit finding -> resolution -> evidence (checked in CI) |

## 2. Prerequisites

| Need | Exact requirement |
|---|---|
| OS | Windows 10/11 or Server 2019+ (the installer, NSSM services, DPAPI keys and ACLs are Windows-only) |
| Python | **3.12, 64-bit, from python.org** (for a server install: "for all users"; the Microsoft Store build is refused). The project requires `>=3.12,<3.13` |
| Node.js | `>= 22.12` (frontend build, tests, dev server) |
| GPU | NVIDIA GPU with driver `>= 527.41` (CUDA 12). CUDA 12.1 / cuDNN 9.1 come from the NVIDIA wheels in `requirements/serving.lock`; no CUDA toolkit or torch is needed. TensorRT FP16 (optional) is used through ONNX Runtime when installed. Without a GPU, `LIFEGUARD_INFERENCE_PROVIDER=auto` runs on OpenVINO / CPU and live admission refuses the cameras the CPU cannot analyse at 15 fps; Linux / WSL: `deploy/linux/install.sh` (RUNBOOK §10) |
| Model weights | `backend/models/*.onnx` are **not in git**; they must match the sha256 in `backend/models/MODELS.json` (`python tools/verify_models.py`). Default detector: `rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c_dynbatch.onnx` |
| Tools (server install, tests) | pinned binaries in `build/tools` listed in `deploy/windows/TOOLS.json`: Caddy 2.11.7, NSSM 2.24, ntfy 2.28.0; FFmpeg and MediaMTX v1.21 for the camera tests, the acceptance run and the site kit |
| Browser | a current Chromium-based browser for the UI (Playwright's Chromium is used by the E2E tests) |

## 3. Setup and execution

All commands run from the repository root. Dependencies are installed **only** from the hash-locked files.

### 3.1 Development environment

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --require-hashes --no-deps -r requirements\dev.lock
cd frontend; npm ci; cd ..
```

### 3.2 Run the system for development (API + UI)

Use a data folder **outside** the repository and set these variables in each of the two windows (PowerShell syntax; `cmd` uses
`set "NAME=value"`). Do **not** set `LIFEGUARD_COOKIE_SECURE=false`: the session cookie is `__Host-` prefixed and needs `Secure`;
`http://localhost` is allowed by browsers.

```powershell
$env:PYTHONPATH = "backend"
$env:LIFEGUARD_ENV = "development"
$env:LIFEGUARD_DATA_DIR = "C:\LifeGuardDev"
$env:LIFEGUARD_SESSION_SECRET = "<at least 32 characters>"
$env:LIFEGUARD_CREDENTIAL_KEY = "<a Fernet key; keep it, it encrypts the camera sources>"
$env:LIFEGUARD_LIVE_ENABLED = "true"            # needs the GPU; leave unset to work on the UI only
$env:LIFEGUARD_SUPERVISOR_EXPECTED = "false"
```

```powershell
.venv\Scripts\python -m lifeguard db upgrade                       # once, and after pulling a new migration
.venv\Scripts\python -m lifeguard admin create --username <name>   # once; the password is typed blind
.venv\Scripts\python -m uvicorn --factory lifeguard.app.main:create_app --host 127.0.0.1 --port 8000   # window 1: API
cd frontend; npm run dev                                                                                # window 2: UI
```

Open `http://localhost:5173`, sign in, click **Arm audio and start monitoring**. Navigation: Dashboard, Cameras, Alerts,
Settings (Zones, Pool calibration, Schedules, System health, Users, Audit log).

### 3.3 Tests and the full pipeline

```powershell
.venv\Scripts\python -m pytest -m "not gpu and not soak and not chaos and not integration"   # unit + API + vision
.venv\Scripts\python -m pytest -m "integration and not gpu"                                    # real processes, MediaMTX
cd frontend; npm run typecheck; npm run lint; npm test; cd ..                                  # frontend
powershell -ExecutionPolicy Bypass -File tools\ci.ps1 -Gpu                                     # everything, incl. build, E2E, GPU tests
```

Run GPU tests and `ci.ps1 -Gpu` with the Life Guard services stopped (they share the GPU).

### 3.4 Evaluate the detector on a labelled set

```powershell
$env:PYTHONPATH = "backend"
.venv\Scripts\python -m lifeguard.eval.evaluate --eval-dir eval\heldout_v1 --split test --provider cuda --name run1 --out <folder outside the repo> --trace <folder>\run1.trace.jsonl.gz
.venv\Scripts\python -m lifeguard.eval.trace <a.trace.jsonl.gz> <b.trace.jsonl.gz> --box-tol 0 --conf-tol 0     # bit-for-bit comparison
```

### 3.5 Build, release and install on the server

```powershell
cd frontend; npm run build; cd ..                                                   # frontend\dist (served directly by Caddy)
python tools\release_zip.py --verify ci-artifacts\release\lifeguard-<version>-<commit>.zip
powershell -ExecutionPolicy Bypass -File deploy\windows\install.ps1 -SiteAddress lifeguard.pool.local -WhatIf     # elevated
powershell -ExecutionPolicy Bypass -File deploy\windows\install.ps1 -SiteAddress lifeguard.pool.local -AdminUser <name>
```

`install.ps1 -Staged` proves steps 1-9 without administrator rights (data in `build\stage\data`, Caddy on `127.0.0.1:8443`);
`deploy\windows\stage_run.ps1` starts that stack in the foreground, and `tools\site_session.ps1 -Smoke` exercises the site-session kit
against it. The installed services are `LifeGuardApi`, `LifeGuardSupervisor`, `LifeGuardProxy`, `LifeGuardNtfy`
(`Get-Service Life Guard*`). After an update that adds a migration: stop the services, `lifeguard db upgrade` (elevated, with the
services' env file), start them (`docs/RUNBOOK.md` section 5).

### 3.6 Dependency and quality rules (enforced in CI)

Install only from locks with `--require-hashes` (`tools/check_locks.py --env` fails if the venv differs from `dev.lock`); re-lock with
`python tools/lock.py`. The serving lock has no torch and no opencv-contrib (headless OpenCV only). No GPL / AGPL / SSPL /
non-commercial dependency or model (`tools/check_licenses.py`); no known vulnerability in any lock (`tools/audit_locks.py`).
ruff (120 columns), mypy (strict on `core`, `app`, `alerts`, `live`), bandit, and a frontend bundle budget of 250 KB gzipped.

## 4. Current version status

**Version:** package `3.0.0.dev0`, frontend `3.0.0-dev.0`, thresholds `2.4.0`, database schema head `0007`.
Latest commit at the time of writing: `03a925d` (branch `main`, no remote).

**History:** the 10-phase clean rebuild (Phases 0-10: baseline and evidence, locked dependencies, detector provider guard,
batched live runtime, out-of-band alert delivery, operator UI, calibration, release gate and Windows installer, account and
health features, audit traceability) was completed on 2026-10-08. **Phase 11** (site upgrades for a 4-camera indoor pool) was
then built in seven work packages, all committed:

| WP | Delivered |
|---|---|
| WP1 | one runtime clock, cross-camera global ids, one incident (one card) per person |
| WP2 | pool-plan zones projected into every calibrated camera, per-camera responsibility (gating off), pool-zone editor |
| WP3 | entry / track-break metrics, multi-view confirmation (off), responsible-region tiling, glare mask + hold (off) |
| WP4 | forced RTSP over TCP, first-frame grace, H.265 1440p defaults, HEVC test clips |
| WP5 | capacity cache, startup timeline, fps fallback (a warm restart is MONITORED ~5-6 s after process start with 2 cameras) |
| WP6 | site-session kit: camera check, coverage, capacity, drills, synchronised recording, lockstep replay, report |
| WP7 | camera pre-open, NSSM 0.5 s restart delay, docs, held-out multi-view A/B (bit-identical), the Uploads page with the Model View over the clip (camera zones, timeline, delete), the Settings page and five-entry navigation |

**Verification at this commit:** unit/API 1069 tests, integration 80 passed + 1 expected failure, vitest 95, Playwright E2E 13, all
129 Phase 11 negative controls caught, `verify_verbatim` 0 unexpected, traceability green (81 audit rows). The last
`tools\ci.ps1 -Gpu` passed 24 of 25 steps; the GPU step failed on one WP5 test's informational comparison tail (borderline
two-camera admission on the dev laptop's GPU; it passed 3/3 alone). It has not been re-run and the test has not been changed:
that is awaiting a decision.

**What is not done / not claimed:**
- The release gate has not passed, so ASSIST / PILOT stays. On the held-out TEST split the shipped detector found 1 of 5 drowning
  events (Medium, no zones) with no false alerts on normal swimming.
- The site validation has not been run: client confirmation of the pass levels, the dry run, the drills (D1-D7, 10 controlled
  submersions, 5 watch-clearing runs), the replay, labels and the final report (`docs/SITE_SESSION.md`).
- Off by their pre-registered rules, awaiting site data: `MULTIVIEW_CONFIRM`, `RESPONSIBILITY_GATING`, `GLARE_HOLD`, `DET_TILING`.
- No pose model runs in v3; the learned behaviour model is off by default; the D-FINE detector is `dev_only`.
- Open items from earlier phases that need the target site: 24 h soak and chaos run, behaviour-model data, ISO 20380.
- Installed services (if any) need `lifeguard db upgrade` (migration 0007), a re-run of `services.ps1` for the 0.5 s restart delay,
  and a restart.
