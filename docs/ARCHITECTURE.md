# Life Guard v3 architecture

This document is for engineers and reviewers. It covers what runs, how data flows, where the trust boundaries are, and which invariants keep the pool safe.
- Operators: `docs/OPERATOR_MANUAL.md`.
- Install: `docs/RUNBOOK.md`.
- Security model: `docs/SECURITY.md`.
- How every finding of the 2026-10-07 audit was resolved: `docs/AUDIT_TRACEABILITY.md`.

## 1. Processes (one Windows server, one GPU)

| Service (NSSM, virtual account) | Code | Job |
|---|---|---|
| **LifeGuardApi** | `backend/lifeguard/app` (uvicorn, 127.0.0.1:8000, 1 worker) | REST API + WebSocket feed. It hosts the **live runtime** (camera capture, batched GPU detection, behaviour analysis, alerts), the DB writer, and alert delivery (outbox dispatcher and escalator). |
| **LifeGuardSupervisor** | `backend/lifeguard/supervisor` | Independent watchdog. It reads the heartbeat files and polls `/health/ready`, and raises an out-of-band **SYSTEM FAULT** when the API, a camera or the GPU scheduler stops. It has its own channel clients, so a dead API cannot silence it. |
| **LifeGuardProxy** | Caddy (`deploy/windows/Caddyfile.template`) | TLS on 443, security headers, serves `frontend/dist`, proxies `/api` to the API. |
| **LifeGuardNtfy** | ntfy (loopback) | Push notifications to phones. |

The API and the supervisor watch each other:
- the supervisor reports the API's death off-site;
- the API raises `SUPERVISOR_DOWN` when the supervisor's heartbeat goes stale.

Nightly scheduled tasks run `lifeguard db backup` and `lifeguard retention run`.

## 2. Live data flow

```
camera (RTSP) -> live/capture.py (bounded queue, reconnect backoff, stall detection -> FAULT)
             -> live/scheduler.py (one GPU round per frame set, dynamic batch, per-camera failure isolation)
             -> vision/pipeline.py (detector -> tracking -> scene / zone state -> behaviour engine)
             -> live/policy.py (schedule muting; SUBMERSION and FAULT are never muted)
             -> alerts/writer.py (alert row + outbox rows in ONE transaction) -> dispatcher -> ntfy / SMTP / webhook
             -> WebSocket feed -> operator UI (alarm card + tone; NOT MONITORING overlay when anything is down)
```

- **Admission control** (`live/capacity.py`) starts only as many cameras as the measured GPU budget allows. A camera that does not fit is `NOT STARTED (capacity)`, never silently under-analysed.
- **Fail-safe UI** (`frontend/src/state/monitoring.ts`, `state/health.ts`): a lost feed, a stalled camera or a status that cannot be read shows **NOT MONITORING** or **STATUS UNKNOWN**, never a healthy screen.

## 3. Backend packages (`backend/lifeguard/`)

| Package | Responsibility |
|---|---|
| `core` | settings (pydantic-settings, secure production defaults), JSON logging with redaction, audit log, security (sessions, RBAC, rate limit, passwords, crypto), heartbeats, schedules, CSV safety |
| `db` | SQLAlchemy models, Alembic migrations, **one writer thread** (`writer.py`) for SQLite in WAL mode, online backup |
| `app` | FastAPI app, middleware (security headers, request IDs), routers under `api/v1`, WebSocket hub, startup guards (`vision_startup.py`: the detector and provider must work, or the server refuses to start) |
| `live` | capture, GPU scheduler, capacity, worker per camera, MJPEG view (encoded only while someone watches) |
| `vision` | v2 detection and behaviour logic, ported verbatim for parity; thresholds in a versioned, hashed `thresholds.json` |
| `alerts` | durable outbox, retrying dispatcher, escalation tiers, daily channel self-test, snapshots and recordings |
| `jobs` | retention (nightly purge) |
| `eval` | evaluation harness, A/B trace comparison, and the **signed release gate** (`release_gate.py`) that alone may lift the ASSIST / PILOT banner |
| `supervisor` | the out-of-band watchdog service |

Import boundaries are enforced by `tests/unit/test_import_boundaries.py`. The server never imports torch, training code or the evaluation-only simulation.

## 4. Frontend (`frontend/`)

Vite, React 18 and strict TypeScript, with v2's look (Tailwind v4, glass theme, Phase 9b).
- **Session:** cookie-only (`__Host-` cookies plus a CSRF header); no token in JavaScript.
- **Live data:** the WebSocket feed with a heartbeat watchdog, and polling `/live/status` as a fallback.
- **Safety layer, on every screen:** the arm-audio gate, the alarm and fault tones, the ASSIST / PILOT and MUTED banners, and the NOT MONITORING overlay.
- **Loading:** routes are code-split; the initial load budget is 250 KB gzipped (CI).

## 5. Trust boundaries

1. **Network to Caddy (443).** Only TLS traffic is allowed in. Everything else is loopback or blocked by the firewall.
2. **Caddy to API.** Every route needs a session except login and the two health probes (`tests/api/test_auth_matrix.py`).
3. **API to disk.** `<DataDir>` is readable only by SYSTEM, Administrators and the five service accounts. Keys are DPAPI-protected.
4. **Server to the outside.** Only the alert channels: a signed webhook (HMAC, timestamp and nonce), SMTP with STARTTLS, and ntfy.
5. **Release gate.** Only an Ed25519-signed verdict qualifies. It must be bound to the running detector and thresholds and apply the false-alert policy (at most 1.0 per camera-hour).

## 6. Build, test and release

- `tools/ci.ps1 -Gpu` runs these gates:
  - lock checks; ruff; mypy (strict);
  - pytest (unit, API and integration, plus GPU tests);
  - licence gate; pip-audit; bandit; PowerShell checks;
  - frontend tsc, eslint, vitest, build, size budget and npm audit;
  - the **release zip**; Playwright E2E against the real API on CUDA;
  - the staged install; the model sha256 check;
  - the **audit traceability** gate.
- The release is `tools/release_zip.py`: the tracked tree plus the prebuilt `frontend/dist` plus `RELEASE_MANIFEST.json` (the sha256 of every file, the tools and models manifests). The zip is reproducible and verified in CI. Third-party binaries and model weights are fetched separately and checked against `deploy/windows/TOOLS.json` and `backend/models/MODELS.json`.
