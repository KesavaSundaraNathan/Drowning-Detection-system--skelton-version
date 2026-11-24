# Life Guard runbook: install, accept, operate, recover

**For:** the RVV engineer or site IT administrator installing and running Life Guard on the pool's Windows GPU server. **Replaces:** the v2 install notes. **Applies to:** v3 (`main`).

> **Life Guard is an ASSIST / PILOT system.** Lifeguards must keep watching the pool. The banner stays until a signed release gate (§6) passes on this site's own footage. As of Phase 8 it **fails**: 1 of 5 held-out events caught.

---

## 1. What runs where

| Service (NSSM) | Account | Listens | What |
|---|---|---|---|
| `LifeGuardProxy` | `NT SERVICE\LifeGuardProxy` | **0.0.0.0:443** (only port open in the firewall) | Caddy: TLS, the operator UI, `/api` → API |
| `LifeGuardApi` | `NT SERVICE\LifeGuardApi` | 127.0.0.1:8000 | API, live runtime, batched GPU scheduler, alerts, WebSocket |
| `LifeGuardSupervisor` | `NT SERVICE\LifeGuardSupervisor` | none | watches the API heartbeat; raises SYSTEM FAULT out of band |
| `LifeGuardNtfy` | `NT SERVICE\LifeGuardNtfy` | 127.0.0.1:2586 | push notifications to phones (`-NtfyHost` publishes it on 443 by host name) |

| Path | Content |
|---|---|
| `<InstallDir>` (the release folder) | code, `.venv`, `frontend\dist`, `backend\models` |
| `C:\ProgramData\LifeGuard` (`DataDir`) | `lifeguard.env`, `keys\` (DPAPI machine scope), `db\`, `logs\`, `snapshots\`, `recordings\`, `backups\`, `caddy\` (config + internal CA), `ntfy\`, `release_gate.json`. ACL: SYSTEM, Administrators and the five service accounts only. |

**Nothing starts on its own (Phase 12).** The services are registered with start type **Manual** and left stopped, so after
a reboot no Life Guard process (API, workers, Caddy, ntfy) runs until someone starts it:

```
powershell -ExecutionPolicy Bypass -File tools\start_lifeguard.ps1     # elevated: ntfy, API, supervisor, Caddy
powershell -ExecutionPolicy Bypass -File tools\stop_lifeguard.ps1      # elevated: the reverse order
powershell -ExecutionPolicy Bypass -File tools\check_autostart.ps1     # read-only: exit 1 if anything of ours starts at boot
```

A live pool site that must survive a power cut unattended opts in explicitly: `install.ps1 -StartMode Automatic -StartServices`.
NSSM's restart-after-crash applies only to a service that has been started.

**Scheduled tasks** (only with `install.ps1 -ScheduleTasks`; run as SYSTEM): `LifeGuard-Backup` daily at 02:30,
`LifeGuard-Retention` daily at 03:15, logs in `DataDir\logs\LifeGuard-*.log`. Without them (the default), run
`lifeguard db backup` and `lifeguard retention run` by hand (elevated, `LIFEGUARD_ENV_FILE` set). While the system runs, the
supervisor still raises **BACKUP_MISSED** when no backup is recent. An install without `-ScheduleTasks` disables tasks that an
earlier install registered (it does not delete them).

---

## 2. Install (elevated PowerShell, on the GPU server)

**Prerequisites:**
- Windows 10/11 or Server 2019+;
- NVIDIA driver ≥ 527.41;
- **Python 3.12 (64-bit) from python.org, installed with "Install Python 3.12 for all users"** (`C:\Program Files\Python312`). **Not** the Microsoft Store Python: it is a per-user app alias that the service accounts cannot start (Phase 8 field report), so `install.ps1` refuses it. The installer prefers `C:\Program Files\Python312\python.exe`; otherwise pass `-Python <path>`;
- the release folder: the CI release zip `lifeguard-<version>-<commit>.zip` (from `ci-artifacts\release`, with its `.sha256`), or a checkout plus `frontend\dist`.
  - **Verify the zip before unpacking:** `python tools\release_zip.py --verify lifeguard-<…>.zip`. It must say `OK`: every file matches the sha256 in its `RELEASE_MANIFEST.json`, and nothing is missing or extra.
  - A name ending in `-dirty` was built from uncommitted changes. Don't install it on a site.
  - Model weights and the `build\tools` binaries are **not** in the zip. Copy them separately; the installer checks them against `MODELS.json` and `TOOLS.json`.
- `build\tools` holding the pinned tools of `deploy\windows\TOOLS.json` (Caddy 2.11.7, NSSM 2.24, ntfy 2.28.0; MediaMTX and FFmpeg are only for acceptance).

```
powershell -ExecutionPolicy Bypass -File deploy\windows\install.ps1 -SiteAddress lifeguard.pool.local -AdminUser pool.admin -WhatIf
powershell -ExecutionPolicy Bypass -File deploy\windows\install.ps1 -SiteAddress lifeguard.pool.local -AdminUser pool.admin
```
- **`-WhatIf` first:** it prints every change without making it.
- **TLS:**
  - **LAN install** (`-TlsMode internal`, the default): Caddy's own CA signs the site certificate. Install `DataDir\caddy\caddy\pki\authorities\local\root.crt` into *Trusted Root Certification Authorities* on each operator PC (Group Policy or `certutil -addstore Root root.crt`).
  - **Public DNS name:** `-TlsMode acme -AcmeEmail ops@…` gets a Let's Encrypt certificate; port 80 must reach the server once for the challenge.
- **Re-running** the same command updates in place. Existing keys, database, env values and the admin account are never overwritten.
- **Admin password:** prompted, never on the command line. It must be ≥ 12 characters and not a common password. A re-run with `-AdminUser` keeps an existing admin and does not prompt.
- **Services get their own venv:** `<InstallDir>\build\service\venv` (override with `-Venv`). No service ever uses the development `.venv`.
- **Changing a password later:** `docs/OPERATOR_MANUAL.md` §7 covers the UI, a reset by another admin, and the console command in §5 below.
- **What it does, step by step:** see the header of `install.ps1` (14 steps: preflight → venv from `serving.lock` with hashes → model sha256 → env file → DPAPI keys → migrations → UI → Caddy → ntfy → ACLs → services (Manual, not started) → firewall → tasks (opt-in) → release-gate key).
- **Start it** when you want it: `tools\start_lifeguard.ps1` (or add `-StartServices` to the install command).

**After install, set the alert channels** in `DataDir\lifeguard.env`, then restart `LifeGuardApi` and `LifeGuardSupervisor`:
- **ntfy:** `LIFEGUARD_NTFY_URL` / `_NTFY_TOPIC` are pre-set to the local server with a random topic. Phones subscribe to that topic, via `-NtfyHost` or a VPN.
- **Email:** `LIFEGUARD_SMTP_HOST`, `_SMTP_PORT`, `_SMTP_FROM`, `_SMTP_TO` (JSON list), `_SMTP_USERNAME` / `_SMTP_PASSWORD`. STARTTLS is mandatory in production.
- **Webhook:** `LIFEGUARD_WEBHOOK_URL` (https) plus `_WEBHOOK_SECRET` (≥ 32 characters).
- **Escalation:** `LIFEGUARD_ESCALATION_*` tiers.
- **Then** run the **self-test**: *System health → Run self-test* in the UI. Every channel must show a receipt.

---

## 3. Acceptance (before the site relies on it)
```
powershell -ExecutionPolicy Bypass -File tools\acceptance.ps1 -Base https://lifeguard.pool.local -Minutes 60
```
It starts simulated cameras: MediaMTX on 127.0.0.1 plus FFmpeg looping a clip with `-c copy`. Then it runs, through the public HTTPS entry point:
- TLS and the security headers;
- `health/ready`;
- **401s and a refused WebSocket without a session**;
- **CUDA provider**;
- capacity;
- **N cameras for 60 min**: every camera MONITORED, nothing dropped, analysed fraction ≥ 0.99;
- the release status;
- **last**, 30 bad logins for a random user name, which must end in lockout. This locks *this machine's IP* out of logins for 15 minutes.

Credentials come from `LIFEGUARD_ACCEPT_USER` / `_PASSWORD` or a prompt. The report is written to `DataDir\acceptance\acceptance-<time>.json`.

**Then, on real cameras:**
1. Configure each camera as in `CAMERA_SETUP.md` (H.265, 2560×1440, 15 fps, I-frame interval 1 s, read-only RTSP user). Register the cameras (*Cameras*) and check each one (`CAMERA_SETUP.md` §5). Draw zones or **calibrate the pool** (*Pool calibration*).
2. The use-case validation visit (camera check, coverage, capacity, drills, replay, report): `SITE_SESSION.md` (`tools\site_session.ps1`).
3. Watch the NOT MONITORING overlay appear: stop `LifeGuardApi` for 30 s, and the screen must go red within ~7 s.
4. Start the API again; the screen must recover by itself.

---

## 4. Daily operation
- **Operators:** log in, **Arm audio**, keep the Dashboard open. **A red full-screen NOT MONITORING means the pool is not watched.** Supervise directly and call IT.
- **Acknowledging an alarm** is audited. Resolve or mark it false-alarm with a note on *Alerts*.
- **MUTED** (yellow banner): a zone outside its schedule window. SUBMERSION and FAULT alerts still sound.
- **System health:**
  - one overall verdict (*All systems OK* / *DEGRADED* / *STATUS UNKNOWN* / *ACTION REQUIRED*) with every check;
  - live runtime and per-camera table, open faults, delivery channels and self-test;
  - the release gate with its criteria.

  The Dashboard shows the same verdict to every role in its health strip.
- **The full operator procedures** are in `docs/OPERATOR_MANUAL.md`: shift start, alarms, calibration, NOT MONITORING, passwords.
- **Restart after a crash** (Phase 11 WP5): NSSM restarts `LifeGuardApi` after 0.5 s (`AppRestartDelay 500`; was 2 s until Phase 11 WP7; re-run `services.ps1` elevated for it to take effect). The API re-uses the GPU capacity it measured last time, stored in `DataDir\capacity_cache.json`, after a 5-run batch-1 spot check. The cached table is used only if the GPU is at most 10 % slower and at most 30 % faster than when it was measured; otherwise the full measurement runs (about 3 s).
  - **Timeline:** each phase is logged (`startup: <phase> at <s> s`, seconds since the process was created), plus one `startup timeline:` line when every camera is MONITORED. The same data is in the live status (`GET /api/v1/live/status` → `startup`, and `capacity_source`: `cache` or `measured` with the reason).
  - **Measured on the dev laptop** (2 cameras, warm restart): cameras admitted 2.7–2.9 s and all MONITORED 6.2–6.8 s after the process was created, against 9.0 s before WP5. Each camera's stream open takes ~1.8 s, and OpenCV opens them **one at a time**, so 4 cameras add ~3.7 s.
  - **Cameras open while the server starts** (Phase 11 WP7, `LIFEGUARD_CAMERA_PREOPEN`, default on): the enabled RTSP cameras are opened right after the database is ready, in parallel with the provider guard and the capacity phase. Their frames are **read and discarded** until admission; nothing is queued, tracked or analysed before a camera is admitted, and admission itself is unchanged. After admission the runtime takes over the already-open stream. A camera that is refused (capacity, invalid source, disabled) or a startup that is refused closes its stream at once. The timeline shows `preopen_started`; the camera's capture status shows `preopen_adopted: true`. Measured on the same rig, alternating the switch in one run (5 restarts per arm): all MONITORED **4.98–5.02 s** with it on against **6.07–6.28 s** with it off (the off arm also had one restart of 9.20 s, when the batch-1 spot check found the GPU slower and re-measured), i.e. 1.28 s earlier by the medians. The rest is the serialised ~1.8 s per camera open, which starts at 1.4 s instead of after admission. Set `LIFEGUARD_CAMERA_PREOPEN=false` to open cameras only at admission.
  - **Forcing a fresh measurement:** stop `LifeGuardApi`, delete `DataDir\capacity_cache.json` (elevated), start it. The cache is ignored by itself after a detector, GPU, driver, ONNX Runtime, app-version, batch, tiling or frame-size change, after 7 days (`LIFEGUARD_CAPACITY_CACHE_MAX_AGE_H`), or when unreadable. `LIFEGUARD_CAPACITY_CACHE=false` turns it off.

---

## 5. Troubleshooting

| Symptom | Check | Fix |
|---|---|---|
| UI unreachable | `Get-Service Life Guard*`; `DataDir\logs\service-LifeGuardProxy.log` | `Restart-Service LifeGuardProxy`. Certificate errors: install the root CA on the PC (§2). |
| NOT MONITORING on every screen | `service-LifeGuardApi.log`; `https://<site>/api/v1/health/ready` | `Restart-Service LifeGuardApi`. If the GPU is not found: check the driver (`nvidia-smi`). There is **no CPU fallback**, by design. |
| One camera NOT MONITORED | *Cameras* page state; `NOT STARTED (capacity)` | Capacity: fewer cameras per server, or a stronger GPU. Source: RTSP URL / network. |
| No phone / email alerts | *System health → delivery*; run the self-test | Fix `lifeguard.env` channel values, restart the API + supervisor. FAULT `CHANNEL_SELF_TEST_FAILED` names the channel. |
| SUPERVISOR_DOWN fault | `Get-Service LifeGuardSupervisor` | Restart it. While it is down, out-of-band fault delivery is not guaranteed. |
| Lost admin password / admin locked out | Can another admin sign in? Then use *Users → Reset a password*. | Otherwise use the server console (elevated): `cd <InstallDir>\backend; $env:LIFEGUARD_ENV_FILE='C:\ProgramData\LifeGuard\lifeguard.env'; ..\build\service\venv\Scripts\python -m lifeguard admin reset-password --username <admin>`. Add `--reactivate` for a deactivated account. It prompts twice, signs out every session and lifts the lock-out (`OPERATOR_MANUAL.md` §7). |
| Services do not start (`Start-Service : Failed to start service`) | `services.ps1` prints the SCM event and the tail of the service log. SCM 7024/7000 *cannot find the file/path specified* means the venv rests on a per-user Python. | Install python.org Python 3.12 *for all users*, delete `build\service\venv`, then re-run `install.ps1 -Python "C:\Program Files\Python312\python.exe" …`. The repeated `Invalid account name!` lines that NSSM 2.24 printed on a re-run were cosmetic; they are now filtered. |

**Service control:** `Get-Service Life Guard*`, `Restart-Service LifeGuardApi`. NSSM restarts a crashed service after 0.5 s, and Windows service recovery is the second line.

**After an update that changes the database** (for example migration `0009`, which drops the removed upload tables): stop the services, run `lifeguard db upgrade` (elevated, with the services' env file, as in the password-recovery row above), start them. The API refuses to start on an older database. The NSSM restart delay (0.5 s) takes effect only after `deploy\windows\services.ps1` is re-run elevated with the install's own arguments.

**Site session kit** (`SITE_SESSION.md`): `tools\site_session.ps1`, elevated, on the server, one visit per site. It does not change the running system: it reads the camera sources with the server's keys, records the drills with FFmpeg `-c copy` and replays them offline.

---

## 6. Release gate (lifting ASSIST / PILOT)
1. Collect rights-clean footage and labels by `docs/DATA_PROTOCOL.md`, with a frozen test split.
2. Measure on this server:
   ```
   python -m lifeguard.eval.evaluate --eval-dir <site eval dir> --split test --out <dir>
   ```
   Use the acceptance report (§3) as the live run.
3. Sign it, on this server, as the admin who ran the install (the key is DPAPI-protected for that account):
   ```
   python -m lifeguard.eval.release_gate sign --report <dir>\report.json --live <acceptance.json> --site "<pool name>"
   ```
   **False-alert limit (site policy, agreed 2026-10-08):** at most **1.0 false alert per camera-hour** on the test split.
   It applies by default. A site may sign a *stricter* limit with `--max-false-alerts-per-hour 0.5`; a looser
   value is refused.
4. Check it: `/system/release` (the banner and *System health*) shows GATE_PASSED **only if**:
   - the signature verifies;
   - the report is bound to the detector and thresholds **this server runs**;
   - the verdict applies the false-alert policy (≤ 1.0 per camera-hour);
   - every criterion is MET.

   Swapping the model or the thresholds re-instates ASSIST / PILOT automatically.

---

## 7. Backup, restore, update, uninstall
- **Backup:** the nightly task writes `DataDir\backups\`. Copy them off-site. They contain personal data; encrypt at rest.
- **Restore:**
  1. Stop the services.
  2. Restore `DataDir` (database, keys **and** env file together). Keys are machine-scoped: restoring to **another** machine needs `keys init` there plus camera sources re-entered.
  3. Start the services.
- **Update:** unpack the new release folder over the old one and re-run `install.ps1` with the same arguments. Migrations run, services are reconfigured, data is kept.
- **Uninstall:**
  1. `deploy\windows\services.ps1 -Nssm <nssm.exe> -Remove`.
  2. Remove the firewall rules: `Get-NetFirewallRule -Group Life Guard | Remove-NetFirewallRule`.
  3. Remove the tasks: `Unregister-ScheduledTask LifeGuard-*`.
  4. Delete `DataDir` only after the retention period agreed with the site.

---

## 8. Staging (developers)
`install.ps1 -Staged` runs steps 1-9 for real without administrator rights:
- `DataDir = build\stage\data`;
- venv `build\stage\venv`;
- Caddy on 127.0.0.1:8443.

`deploy\windows\stage_run.ps1` starts the five processes in the foreground, then:
```
tools\acceptance.ps1 -Base https://localhost:8443 -DataDir build\stage\data -Minutes 5
```
Stop it with `stage_run.ps1 -Stop`. This is how Phase 8 proved the installer on the development laptop.

## 9. Developer run (API + UI, no services)

For working on the UI on a development PC. The data folder is **outside the repository**
(for example `C:\LifeGuardDev`); the installed services use `C:\ProgramData\LifeGuard` and must be stopped first
if they use the same ports and the GPU. Two windows, each in the repository with these variables set
(`cmd` syntax; a `dev_env.cmd` holding them is convenient):

```
set "PYTHONPATH=backend"
set "LIFEGUARD_ENV=development"
set "LIFEGUARD_DATA_DIR=C:\LifeGuardDev"
set "LIFEGUARD_SESSION_SECRET=<at least 32 characters>"
set "LIFEGUARD_CREDENTIAL_KEY=<a Fernet key, kept: it encrypts the camera sources>"
set "LIFEGUARD_LIVE_ENABLED=true"
set "LIFEGUARD_SUPERVISOR_EXPECTED=false"
```

Do **not** set `LIFEGUARD_COOKIE_SECURE=false`: the session cookie is named `__Host-lifeguard_session`, and browsers
refuse a `__Host-` cookie that is not Secure. `http://localhost` counts as a secure context, so the default works.

1. Once: `.venv\Scripts\python.exe -m lifeguard db upgrade`, then `.venv\Scripts\python.exe -m lifeguard admin create --username <name>` (the password is typed blind, no characters are echoed).
2. API: `.venv\Scripts\python.exe -m uvicorn --factory lifeguard.app.main:create_app --host 127.0.0.1 --port 8000` (add `--reload --reload-dir backend` while editing the backend).
3. UI: `cd frontend`, then `npm run dev`, open `http://localhost:5173`. It proxies `/api` to port 8000 (`LIFEGUARD_DEV_API_PORT` changes that).

Run `db upgrade` again after pulling changes that add a migration. Sign in, **Arm audio**, open **Cameras**.

## 10. Pose model, TensorRT and Linux / WSL (thresholds 2.5.1)

The engine is top-down **RTMDet → ByteTrack → RTMPose**.

* **Pose model** - RTMPose-m COCO 256×192, exported once with `tools/export_rtmpose.py --register` (separate venv from
  `requirements/pose_export.in`), then `python tools/verify_models.py`. Until its sha256 is registered the default
  model check fails. `LIFEGUARD_POSE_MODEL=none` runs without pose (shown in health and the startup audit).
* **Provider** - `LIFEGUARD_INFERENCE_PROVIDER=auto` (default) uses TensorRT FP16 → CUDA → OpenVINO → CPU, whichever
  really runs first; an explicit value is strict. After installing or updating the GPU driver / TensorRT, rebuild the
  engines: `python tools/build_trt_engines.py --data-dir <data dir>` (also checks 4 cameras at 15 fps).
* **Learned behaviour model** - on by default from `backend/models/behavior/behavior_v1.onnx`;
  `LIFEGUARD_BEHAVIOR_MODEL=none` = rules only. Its status is in the startup audit (`vision.provider.ok`).
* **Linux / WSL2** - `sudo deploy/linux/install.sh [--admin-user NAME] [--enable]`: service user, hash-locked venv,
  owner-only (0600) key files, migrations, systemd units `lifeguard-api`, `lifeguard-supervisor`
  (installed disabled; start them with `sudo systemctl start ...`). On WSL2 install only the
  Windows NVIDIA driver.
* **Held-out re-evaluation** - `python tools/pose_eval.py --eval-dir eval/heldout_v1` (pose OFF vs ON, DEV + TEST).
