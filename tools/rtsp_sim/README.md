# RTSP simulator for the soak and chaos suites (Phase 4)

The 24 h soak (`tests/soak`) and the chaos tests (`tests/chaos`) need real RTSP cameras. They are simulated
with **MediaMTX** (MIT) as the RTSP server and one **FFmpeg** publisher per camera, looping an eval clip.

Neither tool is downloaded by the repository: per the plan every download needs the product owner's approval
(licence + provenance check). After approval, point the suites at the binaries:

```powershell
$env:LIFEGUARD_MEDIAMTX = "C:\tools\mediamtx\mediamtx.exe"   # MediaMTX v1.x, MIT
$env:LIFEGUARD_FFMPEG   = "C:\tools\ffmpeg\bin\ffmpeg.exe"   # an LGPL build is enough (libx264 needed only for publishing test streams)
powershell -ExecutionPolicy Bypass -File tools\soak.ps1            # 24 h (LIFEGUARD_SOAK_HOURS overrides)
powershell -ExecutionPolicy Bypass -File tools\chaos.ps1
```

`mediamtx.yml` binds to 127.0.0.1 only and enables RTSP over TCP only; the harness rewrites the port per run.
