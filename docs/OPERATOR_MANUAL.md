# Life Guard operator manual

For lifeguards, duty supervisors and the site admin. It covers what is on the screen, what to do, and in what order.
Installing, upgrading and repairing the server are in `docs/RUNBOOK.md`. The security model is in `docs/SECURITY.md`.

> **Life Guard is an ASSIST / PILOT system until the release gate passes** (yellow banner at the top). It is
> **not a primary safety system.** Lifeguard supervision is always required, whatever the screen shows.

---

## 1. Roles

| Role | Can do |
|---|---|
| **lifeguard** | Dashboard, cameras, alarms (acknowledge, resolve, false alarm), own password |
| **supervisor** | All of the above, plus **Settings** (zones, pool calibration, schedules, *System health*) |
| **admin** | All of the above, plus, in **Settings**, users (create, deactivate, reset passwords) and the audit log, and the alert-channel self-test |

Every account changes its own password under **My account** (click your name at the bottom of the left sidebar). See §7.

---

## 2. Start of shift (every shift, about 2 minutes)

1. Open the Life Guard address in the browser (for example `https://lifeguard.pool.local`) and sign in **with your own account**. Never share accounts: the audit log records who acknowledged each alarm.
2. Click **Arm alarm audio**. Browsers block sound until you click, so monitoring only appears after this click. If the browser refuses, fix the speaker or site-sound setting and click again. **Never run a shift with audio unarmed.**
3. Check the speaker volume. It must be audible from the poolside position.
4. Read the **health strip** under the alarm list on the Dashboard:
   - **All systems OK** (green): carry on.
   - **DEGRADED** (amber): monitoring works, with less margin. Tell the duty supervisor.
   - **STATUS UNKNOWN** (orange): the screen cannot read a status. Treat it as not monitored until it clears.
   - **ACTION REQUIRED** (red): part of the pool is, or may be, unwatched. Supervise directly and call the supervisor (§6).
5. Check that every camera tile is labelled **MONITORED** and its picture is moving.
6. Supervisors: open **Settings → System health**. *Overall* should read **All systems OK**, and *Alert delivery* should list every channel as working (§9).

---

## 3. The Dashboard

| Element | Meaning |
|---|---|
| Yellow **ASSIST / PILOT** banner | The release gate has not passed. It stays until it does. |
| Amber **MUTED** banner | A zone is outside its schedule window, so ordinary alerts there are silent. **SUBMERSION and FAULT alerts always sound.** |
| Alarm cards (top) | Unacknowledged alarms, **one card per person** (§4), oldest first, each with an **Acknowledge** button. |
| Health strip | The overall system verdict and up to four problems, worst first (§9). Supervisors get a *Details* link. |
| Camera tile, **MONITORED** chip | Life Guard is analysing this camera now. |
| Camera tile, **NOT MONITORING** chip + red cover over the video | This camera is not analysed (§6). |
| **LAGGING** chip | Analysis runs more than 1 s behind live. Alarms still fire, but later. Tell the supervisor. |
| **MUTED: \<zone\>** chip | That zone is outside its schedule window. |
| `LIVE` / `CONNECTING` / `DOWN` (top of the sidebar) | The real-time connection to the server. |
| Left sidebar | Four entries: **Dashboard**, **Cameras**, **Alerts** (with the number of unacknowledged alarms) and **Settings** (for supervisors and admins). **Settings** holds Zones, Pool calibration, Schedules, System health and, for admins, Users and the Audit log, as a list of sections on its left. **Collapse** shrinks the sidebar to icons. |
| **Dark theme** switch (bottom of the sidebar) | Light (default) or dark display; remembered by this browser. Alarm and NOT MONITORING colours are the same in both. |

---

## 4. Alarms

**Tones.** There are two sounds, and the alarm takes priority:
- **Alarm** (fast high/low two-tone): there is an unacknowledged drowning alert.
- **Fault** (slow low pulse): **NOT MONITORING** or an open SYSTEM FAULT.

**When an alarm sounds:**
1. **Look at the pool first, the screen second.** The card names the camera, the zone and the reason. The thumbnail opens the snapshot and the recording.
2. Respond as your pool's emergency procedure says. Life Guard does not replace it.
3. Press **Acknowledge**. This stops the alarm tone and the escalation, and records your name and the time.
4. Afterwards, on **Alerts**, close the alarm with a note:
   - **Resolve** if it was real;
   - **False alarm** if it was not. False alarms are counted and feed the release gate. Write down what triggered it (for example "child jumping", "reflection").

**Escalation.** If nobody acknowledges within 15 s (the default), the alarm goes to the next tier on every configured channel (phone push, email, webhook). The last tier repeats every 2 minutes. An unacknowledged alarm is never forgotten.

---

**One card per person (incidents).** On a calibrated pool, Life Guard follows each person across the four cameras
with one **global ID** (**G7**, the same number on every camera). When the same person alerts on more than one camera,
or alerts again, the Dashboard shows **one card for that person**:
- the card reads **Person G7 · 3 alerts, same person · Deep end, Shallow end**, and lists the linked alerts under it;
- **Acknowledge person (3 alerts)** acknowledges all of them at once and stops one alarm tone, not three;
- alerts of the same person within the incident window (`INCIDENT_WINDOW_S`) are linked; a person who is **not**
  matched across cameras (an uncalibrated camera, or a handoff Life Guard could not confirm) gets their own card, and
  you should treat two cards in two cameras as possibly the **same** person until you have looked at the pool.

**Global IDs on the Model View.** The live picture of every calibrated camera labels each person **#3 G7**: `#3` is
that camera's own track number and `G7` the pool-wide ID. If the same person carries **different G numbers** on two
cameras, the handoff failed: tell the supervisor (the site session measures this rate, `SITE_SESSION.md`). Grey boxes
are tentative detections that are not yet confirmed as a person.

**Recording continuity.** The video clip of an alarm (pre- and post-event) is recorded from the camera's own stream. If a
clip shows a gap, or **Evidence** says no recording is available, the camera was reconnecting during the event: the
alarm itself is still valid, and the snapshot is the evidence. During the site session, the recorder is restarted
automatically into a new part file when a camera drops its recording session, and the session report lists every
restart under *recording_continuity*; a camera that could not be recorded for the whole block is a FAIL there.

---

## 5. Pool calibration (supervisor)

Calibration tells Life Guard where the water is and how big the pool is in metres. Each person is then classified as **in water**, **edge** (within the edge band of the pool edge) or **deck**. Every camera of one pool is calibrated against the **same pool plan**, so positions match across cameras.

**When:** after installing or moving a camera, after changing a lens or zoom, and whenever the blue outline no longer matches the pool.

**Steps** (**Settings → Pool calibration**):
1. Choose the camera. The page shows its current frame. Pick a moment when the corners are not hidden by people.
2. Enter the **pool length** and **pool width** in metres. Once one camera is calibrated, the others take the size from it.
3. Click the four corners **in this order**, using the same letters on every camera:
   - **A** = (0, 0), the corner you call the origin;
   - **B** = (length, 0);
   - **C** = (length, width);
   - **D** = (0, width).

   A corner the camera cannot see may be replaced by up to 16 extra points (lane marks, ladders), each with its position in metres.
4. Set the **edge band** in metres (default 0.6). People within this distance of the edge count as **edge**, not in the water.
5. Click **Save calibration**. Then check *This camera*:
   - the **reprojection error** in metres. It should be well below the edge band. If it is close to or above the edge band, re-click the corners more carefully (use **Undo last point** or **Clear points**);
   - the **blue outline** must follow the waterline.
6. If the page says **Stored calibration is unusable**, re-calibrate. Zones drawn by hand still apply in the meantime.

**Zones and calibration together.**
- A calibrated camera with **no monitored zone** drawn watches the whole projected pool ("Pool (calibrated)"). People on the deck are ignored until they enter the water.
- Once you draw a monitored zone on that camera (for example a deep-end zone with its own schedule), **only the drawn zones are watched**. Draw them to cover all of the water this camera sees.
- The calibration still supplies metres and the in-water / edge / deck split.
- **Remove calibration** puts the camera back to zones only.

---

## 6. NOT MONITORING (the "dead feed" overlay)

Life Guard never lets a frozen or stale picture look like a calm pool. **NOT MONITORING means: the pool, or part of it, is not watched by Life Guard right now.**

### When it appears
A **full-screen red NOT MONITORING overlay** with the **fault tone** appears when any of these is true. Each reason is listed on the overlay:

| Reason on the overlay | What happened |
|---|---|
| *No connection to the Life Guard server for N s* | The browser has lost the real-time feed for more than **5 s**: server down, network down, or the API service stopped. **The pictures on screen are no longer live.** |
| *Live monitoring is switched OFF on the server* | The live runtime is disabled in the configuration. |
| *\<camera\>: no video (stream stalled)* | No new frame from that camera for **5 s**. Life Guard keeps reconnecting by itself. |
| *\<camera\>: not analysed* | The camera delivers video, but Life Guard is not analysing it. |
| *Camera \<id\>: not started (capacity or invalid source)* | An enabled camera could not start: the GPU is full, or the camera address is wrong. |

As well as the full-screen overlay, the tile of each affected camera gets its own red **NOT MONITORING** cover with the short reason (*no connection to the server*, *stream stalled* or *not analysed*).

### What to do
1. **Supervise the pool directly, now.** Do not wait for the screen to recover.
2. Tell the duty supervisor. They check *System health* and `docs/RUNBOOK.md` §5.
3. To reach the controls (for example to acknowledge an alarm), press **"I am supervising the pool - show the controls (2 min)"**. The overlay shrinks to a red bar at the top for 2 minutes. If the cause is still there after 2 minutes, it comes back full screen.

   **It cannot be switched off.** It goes away only when its cause is fixed.
4. When every reason has cleared, the overlay disappears by itself and the tiles show **MONITORED** again. Write the outage (start, end, cause) in the pool log.

### What else happens
- **Off-site alerts.** These work even when nobody is looking at the screen:
  - A stalled camera or GPU scheduler raises a **SYSTEM FAULT**. It goes to phones, email and the webhook like any alarm, and escalates if nobody acknowledges it.
  - If the API service itself stops or hangs, the separate **supervisor** service notices and sends the SYSTEM FAULT instead, repeating it every 15 minutes until resolved.
  - If the supervisor stops, the API raises **SUPERVISOR_DOWN**.
- **If the screen PC or its network fails**, the server is unaffected, and so are the off-site alerts. The overlay on that PC tells you that *this screen* is not live.
- **If the whole server loses power**, nothing can send alerts. The overlay on every screen is then the only warning, which is why it is loud and permanent.

### Monthly drill (supervisor, about 5 minutes, outside opening hours)
1. On the server, stop the API service:

   `Stop-Service LifeGuardApi`
2. Within about 5 s, every screen must show **NOT MONITORING**, with the reason *No connection to the Life Guard server*, and play the fault tone.
3. Within a minute, phones and email must receive a SYSTEM FAULT from the supervisor service.
4. Start the service again:

   `Start-Service LifeGuardApi`

   The overlay must clear by itself. Record the drill in the pool log.

---

## 7. Passwords

Rules (enforced by the server): **at least 12 characters**, not the username, and not a common password. Five wrong attempts lock the account **and** the computer's address for **15 minutes**. Use a password manager or a passphrase, for example four unrelated words.

### Change your own password (every role)
1. Click **your name** at the bottom of the left sidebar. This opens **My account**.
2. Enter your **current password**, then the **new password** twice.
3. Click **Change password**. Every *other* session of your account (other PCs, an old browser tab) is signed out. You stay signed in here.

### Reset someone else's password (admin)
1. **Settings → Users**, then **Reset a password**.
2. Choose the user, type a new password and click **Reset password**.
3. All of that user's sessions are signed out. Give them the new password **in person**, and ask them to change it at once under **My account**.

### Admin locked out, or the only admin forgot the password
When no admin can sign in, the person who administers the server recovers the account **at the server console**. This works only on the server itself, from an elevated PowerShell. It needs the protected configuration and keys, so it cannot be done over the network.

```
cd "<InstallDir>\backend"
$env:LIFEGUARD_ENV_FILE = "C:\ProgramData\LifeGuard\lifeguard.env"
..\build\service\venv\Scripts\python.exe -m lifeguard admin reset-password --username <admin name>
```

The command:
- asks for the new password twice (hidden; never type it on the command line);
- signs out every session of that account;
- lifts the lock-out of the account and of the computers that failed to sign in as it;
- records `user.password.reset` by `local-console` in the audit log.

Add `--reactivate` if the account had been deactivated. Then sign in and check **Settings → Audit log** for anything you did not expect.

**Keep two admin accounts** (for example the site manager and the IT contact), so one can always reset the other in the UI.

---

## 8. Shift handover and end of day
- Acknowledged alarms are closed with a note (**Resolve** / **False alarm**).
- At handover, the incoming operator signs in with their own account and arms audio. The outgoing operator logs out.
- Never leave a screen signed in and unattended in a public area. Sessions end after **30 minutes** without activity, or **12 hours** in total.

---

## 9. System health (supervisor)

*Overall* is the **worst** of the checks below. **STATUS UNKNOWN ranks above DEGRADED:** a status the screen cannot read is never shown as healthy.

| Check | ACTION REQUIRED (red) when | DEGRADED (amber) when | First step |
|---|---|---|---|
| Server connection | the real-time feed is lost | - | Network / server; RUNBOOK §5 |
| Live monitoring | runtime off, failed to start, or not running | - | `service-LifeGuardApi.log` |
| GPU scheduler | **STALLED** (frames not analysed for 10 s) | - | `Restart-Service LifeGuardApi`; `nvidia-smi` |
| Inference | - | not running on the GPU | Check the driver; there is no CPU fallback by design |
| Supervisor service | **DOWN**: off-site SYSTEM FAULT alerts not guaranteed | - | `Restart-Service LifeGuardSupervisor` |
| Each camera | stalled, not analysed, or not started | lagging behind live | Camera power / network / RTSP address; capacity |
| Cameras | not every camera is monitored | no camera running | *Cameras* page |
| Alert delivery | no channel configured, or every channel down | one of several channels down | Channel settings, then **Run channel self-test now** (admin) |
| Delivery workers | dispatcher or escalator not running | - | `Restart-Service LifeGuardApi` |
| Delivery queue | - | an alert has waited more than 60 s | Channel reachability |
| System faults | an open SYSTEM FAULT | - | Read the fault reason under *Open faults* |

A daily self-test sends a test message on every channel (time shown on the page). A failed channel raises the fault `CHANNEL_SELF_TEST_FAILED`.

---

## 10. Who to call

| Problem | Who |
|---|---|
| Anything red on the screen | Duty supervisor, then the site IT contact |
| Server, services, certificates, installation | Site IT contact (`docs/RUNBOOK.md`) |
| Suspected misuse of an account, unexplained entries in the audit log | Site admin (`docs/SECURITY.md` §9) |
