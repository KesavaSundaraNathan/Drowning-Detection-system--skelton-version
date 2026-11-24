# Life Guard security guidelines

This covers the security model of Life Guard v3, the decisions behind it, the trade-offs accepted on purpose, and what the site must do to keep it.
- Operators: `docs/OPERATOR_MANUAL.md`.
- Install and repair: `docs/RUNBOOK.md`.
- Where the code enforces each rule: the module named in each section, with its tests.

Life Guard is life-safety software. **Availability and integrity come before confidentiality.** Monitoring must not silently stop, and an alarm must not be silently lost or forged. The camera footage of swimmers, including children, is still highly sensitive personal data.

---

## 1. Threat model (summary)

| Asset | Threat | Main controls |
|---|---|---|
| Live monitoring | silent stop, stale screen, crash | NOT MONITORING overlay, supervisor service, SYSTEM FAULT off-site, service restart (§8) |
| Alarms and acknowledgements | forged / suppressed acknowledgements; unattributed actions | per-user accounts, RBAC, CSRF, audit log (§3, §7) |
| Footage, snapshots | viewing by unauthorised people; leaks | authentication on every route, retention, ACLs, TLS (§3, §6) |
| Camera credentials (RTSP URLs) | disclosure (v2 audit: unauthenticated reads leaked them) | encrypted at rest, masked on every read (§5) |
| Server secrets | theft, copying to another machine | DPAPI-protected key files, ACLs (§5) |
| Release gate verdict | forging "GATE_PASSED" to remove the ASSIST / PILOT banner | Ed25519 signature, binding to the running model and thresholds (§5.4) |
| Supply chain | malicious package, binary or model | hash-locked dependencies, pinned tool and model sha256, licence and audit gates (§10) |

Out of scope: physical attackers at the server console with administrator rights (they own the box), and compromise of the camera network itself. Isolate cameras on their own VLAN.

---

## 2. Network exposure

- **One port in:** TCP 443 to `caddy.exe` only (`install.ps1` step 12). The API (127.0.0.1:8000) and ntfy (127.0.0.1:2586) listen on loopback, and the firewall also blocks those ports explicitly.
- **TLS everywhere:** Caddy terminates TLS (`tls internal` with the site's own CA on a LAN, or ACME for a public name). Operator PCs must trust the site root CA (`RUNBOOK.md` §2). **Never click through a certificate warning.** Treat one as an incident.
- **HTTP security headers** (`deploy/windows/Caddyfile.template`):
  - HSTS;
  - `X-Frame-Options: DENY`;
  - a strict `Content-Security-Policy` (`default-src 'self'`, no inline scripts, `frame-ancestors 'none'`, `object-src 'none'`).
- **No CORS** in production. The WebSocket accepts only an `Origin` naming the server's own host, which is the WebSocket CSRF defence (`backend/lifeguard/app/ws.py`).
- **Remote access:** do not port-forward 443 to the internet. Use a VPN. Phones reach ntfy through `-NtfyHost` on the LAN or a VPN.

---

## 3. Authentication, sessions and authorisation

**Accounts and roles.**
- Three roles: `lifeguard` < `supervisor` < `admin` (`core/security/rbac.py`).
- **Every route needs an authenticated session** except `POST /auth/login`, `/health/live` and `/health/ready`, which reveal only a status word. `tests/api/test_auth_matrix.py` fails CI if a new route is not explicitly reviewed.
- One person, one account. Shared accounts defeat the audit log.

**Passwords** (`core/security/passwords.py`):
- bcrypt, cost 12 (lower values are refused outside `env=test`);
- at least 12 characters, not the username, not a common password;
- never accepted on a command line: the CLI reads a hidden prompt or `--password-stdin`.

**Sessions** (`core/security/sessions.py`, `app/deps.py`):
- 256-bit random token in a `__Host-` cookie: `Secure`, `HttpOnly`, `SameSite=Strict`, `Path=/`, no `Domain`;
- the database stores only HMAC hashes of tokens;
- idle timeout 30 min, absolute limit 12 h;
- CSRF: a second `__Host-` cookie must be echoed in the `X-CSRF-Token` header on every state-changing request;
- the UI keeps no token in JavaScript or `localStorage`.

**Login throttling** (`core/security/ratelimit.py`):
- 5 failures in 15 min lock the **username** and, separately, the **client IP**;
- attempts are reserved atomically, so parallel guessing cannot exceed the limit;
- attempts made while locked are audited but do not extend the lock.

**Password changes and resets.**

| Path | Who | Effect |
|---|---|---|
| *My account* → `POST /auth/password` | any user, needs the current password | other sessions of the account revoked; audited `auth.password.change` |
| *Users → Reset a password* → `POST /users/{id}/password` | admin | all sessions of the target revoked; audited `user.password.reset` |
| `lifeguard admin reset-password` (server console) | whoever can read the protected env file and DPAPI keys, i.e. a Windows administrator of the server | new password; all sessions revoked; lock-outs cleared for the account and for the IPs that failed on it within the window; optional `--reactivate`; audited `user.password.reset` by `local-console`, with counts and IPs |

The console command is **not a network back door**. It needs local administrator rights on the server, and those already give full control of the box. Keep at least two admin accounts, so the console path is rarely needed (`OPERATOR_MANUAL.md` §7).

---

## 4. Accepted trade-offs (by design)

1. **Per-user lock-out denial of service.** Anyone can lock a known username out for 15 minutes with 5 bad passwords. The plan requires a per-user limit, because per-IP limits alone allow distributed guessing. Mitigations: another admin's reset, the console reset, and the audit trail (`auth.login.locked`).
2. **A successful login does not reset the IP counter.** Otherwise an attacker with one valid account could reset the counter between guesses at other accounts. Only the console reset clears an IP's lock-out, and only for IPs that failed on the account being reset.
3. **No multi-factor authentication** in v3. Life Guard is LAN or VPN only, behind TLS. Reconsider MFA before any internet-facing deployment.
4. **The UI shows camera names and snapshots to every role.** Lifeguards need the evidence to respond. Footage access is controlled by sign-in and retention, not by role.
5. **SQLite on the server's disk.** Confidentiality at rest relies on the ACLs and on BitLocker for the system disk (§6). There is no application-level database encryption.

---

## 5. Secrets and keys

### 5.1 Server keys (`core/security/keys.py`)
- **Session HMAC key** and **camera-credential Fernet key**: DPAPI-protected files in `<DataDir>\keys` (`session.key.dpapi`, `credentials.key.dpapi`).
- They are created once by `lifeguard keys init --machine` (machine scope, so the service accounts can use them). A copied key file is useless on another machine.
- The server refuses to start without them. Secret values never appear in logs or messages.
- `init` refuses to overwrite: a new credential key would orphan every stored camera credential.

### 5.2 Camera credentials
RTSP URLs with user and password are Fernet-encrypted in the database. The API returns only `source_display` (credentials masked), on every read and in the audit log. This fixes v2 audit item #2: unauthenticated reads leaked RTSP credentials.

### 5.3 The env file and channel secrets
`<DataDir>\lifeguard.env` holds the SMTP password, the webhook secret and the ntfy topic. Its ACL allows only SYSTEM, Administrators and the Life Guard service accounts, with read access (`install.ps1` step 10).

The settings refuse:
- a webhook without a secret of at least 32 characters;
- SMTP without STARTTLS in production;
- non-Secure cookies in production.

### 5.4 Release-gate signing key (`eval/release_gate.py`)
- An **Ed25519** private key per site, DPAPI-protected for the **Windows account that ran `keygen`** (the installing admin, `install.ps1` step 14). The API holds only the public key.
- The signed verdict is bound to the detector file sha256, the thresholds version and sha256, and the app version. `/system/release` re-verifies the signature **and** the binding against what is running.
- A missing, tampered, stale or failing verdict keeps ASSIST / PILOT. Swapping the model or the thresholds re-instates the banner automatically.
- **False-alert policy:** at most **1.0 false alert per camera-hour** (agreed 2026-10-08). It is part of every signed verdict. `/system/release` refuses a verdict signed against a looser limit, or with no limit, even when the signature is valid. Signing allows only a stricter limit. The limit exists to prevent lifeguard alarm fatigue: an alarm that is usually wrong gets ignored when it is right.
- Protect that admin account. Whoever signs as it decides when the banner disappears.

### 5.5 Webhook signatures (`alerts/channels/webhook.py`)
Every webhook POST carries these headers:
- `X-LifeGuard-Timestamp`;
- `X-LifeGuard-Nonce` (unique per delivery);
- `X-LifeGuard-Signature: v1=<hex HMAC-SHA256(secret, "<timestamp>.<nonce>." + body)>`.

Receivers **must**:
1. verify the signature with a constant-time compare (`verify_signature` in that module is the reference);
2. reject timestamps outside ±300 s;
3. remember nonces for that window, which stops replays.

---

## 6. Data protection

**Footage.**
- Snapshots and recordings are kept for `RETENTION_DAYS`, 30 by default. The scheduled retention task deletes older files, only inside the managed directories (`jobs/retention.py`).
- Set retention to the shortest period your incident-review policy allows. Keep it consistent with local law (in the EU: a DPIA, signage at the pool, a named data controller).

**At rest.**
- `<DataDir>` (database, footage, keys, env, logs) allows only SYSTEM, Administrators and the five service accounts (`NT SERVICE\Life Guard*`), with inheritance removed.
- Enable **BitLocker** on the server's disks.

**Training and evaluation data:** `docs/DATA_PROTOCOL.md` covers consent, rights-clean sources and the frozen test split.

**Backups:** a nightly online SQLite backup with an integrity check (`db backup`, 02:30). Copy the backup folder **off the server**, encrypted. Backups contain password hashes and the audit log.

---

## 7. Audit log

Security-relevant actions are written to `audit_events`, in the same transaction as the change: logins (success, failure, locked), password changes and resets, user changes, camera, zone, schedule and calibration changes, alarm acknowledgements and closures, live start/stop, and self-tests. Each entry records the actor, the IP, the target and the outcome.

Admins read it under **Audit log**. Review it **weekly** for:
- `auth.login.locked` bursts;
- resets you did not expect (`user.password.reset`, especially by `local-console`);
- accounts created or re-activated outside working hours.

---

## 8. Service hardening and availability

**Services** (`deploy/windows/services.ps1`):
- **Five services**, each with its own **virtual account** `NT SERVICE\<name>`: no password, least privilege, nothing shared with a human login.
- **Restart policy:** NSSM restarts a crashed service after 2 s; Windows service recovery is the second line.
- **Interpreter:** services need a Python they can run. That means python.org Python 3.12 *for all users*, **not** the Microsoft Store build, which is a per-user app alias. `install.ps1` and `services.ps1` refuse a venv built on a per-user interpreter (`deploy/windows/service_python.ps1`).
- **Dedicated venv:** services use `build\service\venv`, installed from `requirements/serving.lock` with `--require-hashes`, never the development venv.

**Watchdogs:**
- The **supervisor** service raises an out-of-band SYSTEM FAULT when the API dies or hangs.
- The API raises SUPERVISOR_DOWN when the supervisor dies.

**Fail-safe UI.** A lost feed, a stalled camera or a status the UI cannot read is shown as **NOT MONITORING** or **STATUS UNKNOWN**, never as healthy (`frontend/src/state/monitoring.ts`, `state/health.ts`).

---

## 9. Incident response

| Event | Do |
|---|---|
| Suspected account compromise | Deactivate the account (*Users*). Reset its password; this revokes its sessions. Review the audit log for its actions. |
| Lost or stolen operator PC | Reset the passwords of every account used on it. Sessions expire within 12 h at the latest; a reset ends them at once. |
| Admin locked out / forgot password | Another admin resets it, or use the console reset (`OPERATOR_MANUAL.md` §7). |
| Certificate warning in a browser | Stop using that PC for Life Guard. Check the root CA installation and the proxy (`RUNBOOK.md` §5). |
| Env file or key files copied off the server | Rotate everything: SMTP password, webhook secret, ntfy topic. Treat camera credentials as disclosed and change them on the cameras. |
| Release-gate key account compromised | Run `release_gate keygen` again under a new admin account, re-sign, and investigate who signed what. |

Report vulnerabilities in Life Guard itself privately to the RVV maintainers, not in a public issue.

---

## 10. Supply chain and change control

**Dependencies.**
- `requirements/serving.lock` is installed with `--require-hashes`.
- CI runs `pip-audit` on it, `npm audit --omit=dev` on the frontend, `bandit` on the backend, and `tools/check_licenses.py` (FOSS-compatible licences only).

**Third-party binaries:** Caddy, NSSM, ntfy, MediaMTX and FFmpeg are pinned by sha256 in `deploy/windows/TOOLS.json`. The installer refuses a modified binary.

**Models:**
- sha256 recorded in `backend/models/MODELS.json`; `tools/verify_models.py` runs at install;
- a different model also voids the release-gate verdict (§5.4).

**Changes:** every change goes through CI (`tools/ci.ps1 -Gpu`), with lint, type checks, unit, API and integration tests, the E2E, and the gates above. Threshold changes are versioned and hashed.

---

## 11. Site checklist (at install and after every change)

- [ ] Only TCP 443 reachable from the LAN. No port-forward from the internet.
- [ ] Root CA installed on every operator PC. No certificate warnings.
- [ ] Two admin accounts, each a named person. No shared accounts.
- [ ] Default and bootstrap passwords changed (`OPERATOR_MANUAL.md` §7).
- [ ] BitLocker on. Backups copied off the server, encrypted.
- [ ] Cameras on their own VLAN, with changed default camera passwords.
- [ ] Webhook receivers verify signatures, timestamps and nonces (§5.5).
- [ ] Retention period agreed and set. Pool signage and DPIA done where required.
- [ ] Weekly audit-log review assigned to a named person.
