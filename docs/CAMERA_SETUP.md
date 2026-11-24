# Camera setup (site: 4 corner cameras, 4 MP H.265)

How each pool camera must be configured for Life Guard, why, and how to check it. It applies to the site's four
corner 4 MP bullets (2560×1440, ≤ 20 fps, H.265). Mounting, lenses and lighting are in `SITE_REQUIREMENTS.md`
(Phase 11 WP7). The measurements below come from the Phase 11 WP4 tests
(`docs/phase_reports/phase11/WP4.md`): MediaMTX publishing 2560×1440 H.265 test clips made by
`tools/make_hevc_clips.py`.

## 1. Camera settings (set in each camera's web page)

| Setting | Value | Why |
|---|---|---|
| Stream | **Main stream** | The sub-stream is too small to see a child at the far end. |
| Video codec | **H.265** (Main profile). **Not "H.265+"**, "Smart Codec" or "dynamic GOP" | The "+" modes stretch the I-frame interval, often to several seconds and varying with the scene (see §3). |
| Resolution | **2560×1440** | Admission counts capacity at this size (`capacity_frame_size`). |
| Frame rate | **15 fps** | `target_analysis_fps` = 15: every frame of every camera is analysed. |
| I-frame interval (GOP) | **15**, i.e. **1 second** (= the frame rate) | After every (re)connect the first usable picture waits for the next keyframe (§3). |
| B-frames / SVC | Off (if offered) | Camera streams are normally P-frames only. B-frames add latency. |
| Bitrate | CBR or VBR, about **4–8 Mbps** (the tests use 6 Mbps) | 4 cameras × 8 Mbps = 32 Mbps, far below a gigabit link. |
| 3D-DNR (noise reduction) | **Low** | Strong temporal filtering smears moving swimmers. |
| WDR | **On** | Floodlights and the open side wall (glare). |
| Time | **NTP** from the server or the site's time source | Camera overlay times match the alert times. |
| RTSP user | A dedicated **read-only** user (view rights only) | Its password is stored encrypted in Life Guard. Never use the camera's admin account. |
| RTSP port | 554 (default), TCP reachable from the server | Life Guard always uses RTSP over **TCP** (§2). No UDP ports are needed. |
| Simultaneous RTSP sessions | **at least 3** per camera | Live monitoring (1) + the site-session recorder (1) + its camera check (1, a few seconds) during `SITE_SESSION.md`. |

The RTSP URL is entered on the *Cameras* page. Its form depends on the camera maker; check the camera's manual
(typical forms: `rtsp://<user>:<password>@<ip>:554/Streaming/Channels/101` or
`rtsp://<user>:<password>@<ip>:554/cam/realmonitor?channel=1&subtype=0`). The URL holds the password: it is never
logged and never returned by the API.

## 2. RTSP over TCP (forced, nothing to configure)

Life Guard forces `rtsp_transport=tcp` for every camera before the first stream opens (`live/capture.py`
`force_rtsp_tcp`):
- **Why:** over UDP, a 6 Mbps 2560×1440 H.265 stream lost RTP packets even on the server's own loopback (MediaMTX
  "RTP packets lost"; decoder "Could not find ref"), i.e. broken pictures. A real network is worse.
- **Forced, not preferred:** OpenCV's default (`prefer_tcp`) silently falls back to UDP when TCP is refused. The
  forced setting does not: against a UDP-only server the camera does not open at all (tested).
- **A Windows detail, checked at every start:** OpenCV's FFmpeg plugin (`opencv_videoio_ffmpeg*.dll`) reads the
  option from msvcrt.dll's copy of the environment, which Python's `os.environ` does not reach. Life Guard writes it
  there too and reads it back. The live status (`GET /api/v1/live/status`) shows
  `rtsp: {rtsp_transport: tcp, verified: true}`. `verified: false` is also written to the API log as an error.
- If an administrator has set `OPENCV_FFMPEG_CAPTURE_OPTIONS` on the server, its other options are kept. Its
  `rtsp_transport` is replaced by `tcp`, and a warning is logged.

## 3. Keyframes and the first-frame grace

After each (re)connect FFmpeg can show a picture only from the next keyframe:
- **H.264:** the stream *open* itself waits for the keyframe. Measured with an 8 s GOP: the first frame came
  **7.56 s** after the open started (STALL_S is 5 s at the site, 2 s in the test).
- **H.265:** the decoder starts at once, but the pictures before the first keyframe are rebuilt from missing
  reference frames (broken pictures; the log shows "Could not find ref with POC").

So the I-frame interval must be **1 s**: both effects then last at most 1 s after a reconnect.

To make sure a longer keyframe gap can never cause a reconnect loop, each (re)open gets a **first-frame grace** of
`first_frame_timeout_s` = **12 s** (setting `LIFEGUARD_FIRST_FRAME_TIMEOUT_S`):
- the stall monitor does not count a camera as STALLED while it waits for its first frame, for at most 12 s from
  the start of the attempt;
- after 12 s with no frame it is STALLED as usual: FAULT + reconnect with backoff;
- the pause between failed attempts (backoff) is not grace: a camera that cannot be reached at all is still
  STALLED after STALL_S;
- the live status shows per camera `capture.first_frame_s` (open start → first frame of the last (re)connect)
  and `capture.awaiting_first_frame`.

**Accepted cost:** OpenCV cannot shorten a capture's read timeout after the open, and a read that timed out
leaves the capture unusable (both measured). So the read timeout is `max(STALL_S, first_frame_timeout_s)` = 12 s.
When the link to a camera black-holes (no data, no close: a dead switch port), the **CAMERA_STALLED FAULT still
comes at STALL_S** (measured 2.27 s with STALL_S = 2). The reconnect starts only when the blocked read returns:
measured **12.2 s**, against STALL_S before WP4. A camera that closes the connection (reboot, RST) reconnects at
once, as before.

### 3b. Cameras open while the server starts (WP7)

OpenCV opens RTSP streams one at a time (~1.8 s each), so a restart used to open the first camera only after the
provider guard and the capacity phase. Now the enabled RTSP cameras are opened right after the database is ready and
**their frames are discarded** until admission. This changes nothing about what is analysed: no frame is queued,
tracked or analysed for a camera before it is admitted, a camera that is refused or a startup that is refused closes
its stream, and a camera whose address or enabled state changed in the meantime is not reused. It does not add RTSP
sessions: the session opened early is the one the camera keeps. Measured with 2 cameras: all MONITORED 4.98–5.02 s
after the process was created, against 6.07–6.28 s without (`RUNBOOK.md` §4).

## 4. Life Guard settings for the site

All have the right default; set them in `DataDir\lifeguard.env` only to change them.

| Setting (`LIFEGUARD_…`) | Default | Meaning |
|---|---|---|
| `TARGET_ANALYSIS_FPS` | **15** | Admission: every camera must be analysed this fast, or at its real fps if that is higher. A 25 fps camera is still analysed at 25. |
| `CAPACITY_FRAME_SIZE` | **[2560, 1440]** | The cameras' frame size for admission: the per-frame CPU cost is measured at this size, and DET_TILING tiles are counted at it. |
| `FIRST_FRAME_TIMEOUT_S` | **12** | First-frame grace after each (re)connect (§3). Also the open and read timeout. |
| `CAMERA_FPS_FALLBACK` | **15** | The cameras' configured fps. Used when a stream reports an implausible frame rate (outside 1–240, e.g. 90000). Set it to the cameras' real fps if they are not 15. |
| `STALL_S` | 5 | No frame for this long = CAMERA_STALLED FAULT, camera NOT MONITORED, reconnect. |
| `CAMERA_PREOPEN` | **true** | [WP7] Open the RTSP cameras while the server is still starting (§3b). `false` = open them at admission, as before. |
| `PREOPEN_MAX_HOLD_S` | 60 | A pre-opened camera that admission has not taken over by then is closed. |

## 5. Checking a camera

The site-session kit runs every check below automatically for all cameras: `tools\site_session.ps1 -Step check` (`SITE_SESSION.md` §2). By hand:

1. Add the camera and start it. Then:
   - on its camera page (*Cameras* → the camera): **Source fps 15**, **Hardware decoding yes**, **Reconnects 0**
     after a few minutes (*System health* lists fps and reconnects for all cameras);
   - in the live status, `https://<server>/api/v1/live/status` (open it in the browser while signed in; any role):
     for that camera `capture.frame_size` = `[2560, 1440]`, `capture.hw_accel` ≠ 0 (2 = D3D11 for H.265 on the
     dev machine), `capture.first_frame_s` below about 2 s; and at the top level `rtsp.verified` = `true`.
2. **The frame rate must be right.** The tracker's time windows are counted in frames at `capture.fps`, so a wrong
   fps stretches or shrinks them. In the live status, `capture.fps_source` is `stream` when the camera reported a
   plausible fps, and `fallback` when it did not (`capture.reported_fps` shows what it reported). The test H.264
   stream (no timing information) made OpenCV report 90000 on some connects; Life Guard then uses
   `CAMERA_FPS_FALLBACK` (15), not the old fixed 25, and logs a warning. The H.265 test streams always reported 15.
   A camera on `fallback` works, but only as well as the fallback matches its real fps: fix it in the camera
   (H.265, fixed frame rate) and check that `CAMERA_FPS_FALLBACK` equals the cameras' fps.
3. The keyframe interval, measured with the pinned FFmpeg (use the read-only user, then clear the shell history,
   because the URL holds the password):

   ```powershell
   build\tools\ffmpeg\ffmpeg-n8.1-latest-win64-lgpl-8.1\bin\ffprobe.exe -v error -rtsp_transport tcp -read_intervals "%+20" -select_streams v:0 -show_entries packet=pts_time,flags -of csv=p=0 "rtsp://<user>:<password>@<ip>:554/<path>" | Select-String "K"
   ```

   The keyframe (`K`) times must be **1 s** apart. The site session kit (`tools/site_session`, Phase 11 WP6) runs
   these checks for all four cameras and reports codec, fps, measured GOP and hardware decoding.

## 6. Test clips (developers)

`python tools/make_hevc_clips.py` writes 2560×1440 @ 15 fps clips with the pinned FFmpeg (no download) to
`build/test_media/hevc`: H.265 with 1 s and 8 s GOP (libkvazaar, BSD-3, or `--encoder hevc_nvenc`), plus an
H.264 8 s-GOP control (libopenh264). The keyframe interval of each clip is measured, and a clip that does not have
the requested GOP is refused. `tests/integration/test_hevc_rtsp.py` publishes them through MediaMTX.
