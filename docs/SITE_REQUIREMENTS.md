# Site requirements (client side)

What the pool, the cameras, the network and the server room must provide for Life Guard to meet the Phase 11 use
cases: detect several people across 4 cameras (1), keep one clean identity per person across cameras (2), detect
people at entry and in the water, through zone crossings, and when submerged (3). The software was built and
validated **assuming** every item below. An item that is not met does not make the system unsafe, because the
**NOT MONITORING** overlay and the SYSTEM FAULT alarm still work, but it lowers the measured criteria and the
site session (`SITE_SESSION.md`) reports it as a FAIL with the reason.

The site is an indoor 25 m, 5-lane pool with four corner cameras (4 MP, 2560×1440, H.265) and one GPU server (an
RTX 3060 class GPU). Where something below is a number, the site-session kit checks it (the **Checked by** column).

## 1. Mounting

| Requirement | Value | Why | Checked by |
|---|---|---|---|
| Camera height | **≥ 5 m** above the deck. The steel columns and trusses allow it. **Never** at the current 2.3–2.6 m | At 2.3–2.6 m the far water is at a grazing angle: swimmers overlap, the surface glare fills the picture and a person under water at the far end is a few pixels. | Coverage map (`coverage.json`/`.png`): every 0.5 m pool cell must be seen by at least one camera |
| Aim | Tilt each camera so its **near half of the pool (about 12.5 m)** fills the frame; the two cameras of a long side overlap at mid-pool | Each camera is *responsible* for its near half; the overlap is where a person is handed over to the next camera (criterion 2). | Coverage + overlap band reported |
| Stability | Rigid bracket, no pole sway; cable and bracket rated for the pool atmosphere | A swaying camera moves the calibration and the zones. | Reprojection error in *Pool calibration* |
| Calibration | All 4 cameras calibrated to the **same pool plan** (25 m × the measured width), and the pool zones (teaching platform at least) drawn | Cross-camera identity and the zones work in pool metres. Without one shared plan a person has a different position on each camera. See §6. | `check` step: all cameras on ONE pool |

## 2. Lenses

| Camera position | Lens |
|---|---|
| The two cameras looking **down the long axis** (25 m) | **4 mm** |
| Cameras where width matters | **2.8 mm** |
| All | Clean lens and window; **no dome bubble** or condensation. The pool atmosphere clouds a dirty dome within weeks: clean on a schedule. |

A wider lens sees more but gives fewer pixels per person at the far end; the lens table above is the compromise
used for the coverage map. After any lens change recalibrate.

## 3. Light and glare

1. Do **not** mount a camera directly under, or looking straight at, a floodlight.
2. Mount the lights **behind** the cameras, so they light the water and the deck from the camera side.
3. Turn the cameras **away from the open side wall** (daylight on the water surface is the strongest glare).
4. **WDR on** in every camera (`CAMERA_SETUP.md`).
5. A glare pass is part of the site session (drill D7, afternoon). If a glare patch hides people, the measured
   criterion shows it and the lever `GLARE_HOLD` is decided by its pre-registered rule, never by hand.

## 4. Camera configuration

Exactly as in `CAMERA_SETUP.md`: **H.265** (not H.265+), main stream 2560×1440, **15 fps**, **I-frame interval =
15 frames (1 s)**, 3D-DNR low, WDR on, NTP time sync, a **read-only RTSP user**, at least 3 simultaneous RTSP
sessions per camera. The `check` step measures codec, fps and the keyframe interval over the real stream.

## 5. Network and power

| Requirement | Value | Why |
|---|---|---|
| Cabling | **Cat6**, PoE, **≤ 100 m** per run (40 m is fine) | PoE cameras, no separate power in the pool hall. |
| Switch | A **dedicated switch or VLAN** for the cameras; the server on the same segment | Other traffic must not drop frames. 4 cameras × up to 8 Mbps is far below a gigabit link, so the risk is not bandwidth but sharing. |
| Server power | The server (and the camera switch) on a **UPS** | A power cut must not silently stop monitoring: the NOT MONITORING overlay and the fault tones need a running screen and speaker. After a crash of the API process alone, the cameras are admitted again within about 3 s and monitored within about 7 s (measured with 2 cameras, `RUNBOOK.md` §4). |
| Server | RTX 3060 class GPU, with the NVIDIA power mode set to *Prefer maximum performance* | The capacity phase measures what the GPU can do now; a throttling GPU admits fewer cameras (a laptop GPU admits 2 at 15 fps). |
| Time | The server and the cameras on **NTP** | Alert times, recordings and the site session's synchronisation. |
| Alarm audio | A speaker the lifeguard can hear from the deck, tested at the start of every shift | The alarm is audible; the screen alone is not enough. |

## 6. Calibration to one pool plan

For each camera, in **Settings → Pool calibration** (supervisor):
1. Pick the four pool corners A, B, C, D in the image, in the same order and on the same pool plan for all cameras.
2. Enter the measured pool size (25 m × the measured width).
3. Check the reprojection error; recalibrate if it is large.
4. Draw the **pool zones** in the pool-plan zone editor on the same page: the teaching platform at least. A pool
   zone is drawn once on the pool plan and projected into every calibrated camera.

Recalibrate after **any** change in camera aim, zoom or mounting.

## 7. What the site session checks, and when it fails

| Finding | Likely cause | Fix |
|---|---|---|
| `coverage` holes | Aim, lens or height | Re-aim, change lens, recalibrate |
| `gop` FAIL irregular keyframes | H.265+ / smart codec on | Switch it off |
| `fps_measured` FAIL | Camera not at 15 fps | Set 15 fps |
| `live` analysed fraction below 0.99 | GPU too slow for the number of cameras | Fewer cameras per GPU or a stronger GPU |
| Replay: many track breaks per person-minute | Glare, too low a camera, a dirty lens | §1–§3 |

The pass levels for the six criteria are proposed in `eval/site/levers_phase11.json` (`gates_wp6.criteria`); **the
client confirms them before the session**.
