# Life Guard - RTMPose skeleton clone

An independent copy of `C:\aquaguard` (taken 2026-11-24 from branch `lifeguard-rebuild` at 45c65ed, including that
tree's uncommitted changes) with one added feature: each tracked person's RTMPose skeleton is drawn on the live feed.

## What is different from the original

| | Original `C:\aquaguard` | This clone `C:\aquaguard_rtmpose` |
|---|---|---|
| Python venv | `C:\aquaguard\venv` | `C:\aquaguard_rtmpose\venv` (fresh, from `requirements\dev.lock`) |
| Data folder (keys, DB, cameras, logs, recordings) | `C:\LifeGuardDev` | `C:\aquaguard_rtmpose\runtime` (or `LIFEGUARD_RTMPOSE_DATA_DIR`) |
| TensorRT engine cache | `C:\LifeGuardDev\trt_cache` | `C:\aquaguard_rtmpose\runtime\trt_cache` |
| API / UI | :8000 / :5173 | :8100 / :5273 |
| E2E stack (API / control / preview) | :18000 / :8099 / :5173 | :18100 / :8199 / :5373 |
| Skeleton overlay | - | on by default, per-camera "Skeleton" toggle |

Settings keep the `LIFEGUARD_*` names; `Life Guard Start.bat` in this folder sets this clone's values per process,
and always overrides `LIFEGUARD_DATA_DIR` so the clone can never open the original's data folder.

## The skeleton (display only)

The pipeline already runs RTMPose-m top-down on the detector's tracked person crops (GPU, TensorRT FP16, batched
across cameras in `live/pose_batcher.py`, skipped on alternate frames for stable tracks by `POSE_STRIDE`). This clone
adds no inference: it draws the keypoints that are already there.

- `backend/lifeguard/live/skeleton.py` - COCO-17 limbs, drawing (one `cv2.polylines` call per frame), compact JSON,
  and `PoseMemory`: on a pose-skipped frame the track's last pose (at most 1 s old) is moved with its box.
- `backend/lifeguard/live/view.py` - `render` / `render_client` draw or send the skeleton (`"k"` in
  `X-Lifeguard-Boxes`).
- `backend/lifeguard/app/api/v1/live.py` - `skeleton=true|false` on `frame.jpg` and `mjpeg`.
- `frontend/src/components/skeletonOverlay.ts`, `liveFrames.tsx`, `viewMode.ts`, `CameraTile.tsx`,
  `api/endpoints.ts` - the browser draws the skeleton as SVG; "Skeleton" button next to Full / Clean / Alerts only.
  Clean hides it, Alerts only shows only the alerting person's.

Detection, tracking, pose, thresholds, behaviour and alerts are unchanged (`thresholds.json` is untouched).

## Run

First start creates this clone's own keys, database and admin (you type the admin password), then add the cameras
on its Cameras page (http://localhost:5273):

```powershell
& "C:\aquaguard_rtmpose\Life Guard Start.bat"
```

The original keeps running from its own `Life Guard Start.bat` on :8000 / :5173. Both can run at once, but they then
share one GPU: each version measures its capacity as if it had the GPU alone, so run 4 cameras in only one of them
at a time.

For bit-identical detections to the original, seed this clone's engine cache with the original's engines (copy
`C:\LifeGuardDev\trt_cache` to `runtime\trt_cache`): separately built TensorRT FP16 engines differ in the last digits.

## Benchmark (2026-11-24, RTX 5060, TensorRT FP16)

Each tree ran its own `LiveRuntime` (file cameras -> batched RTMDet -> ByteTrack -> RTMPose batcher -> behaviour)
with one simulated operator viewing camera 0 at the HQ rate (1920 px, 10 fps); one tree at a time on the GPU, each with
its own engine cache. Latency = capture-to-result per frame (Windows `time.monotonic`, 15.6 ms resolution).

**2 cameras, `active drowning 3.mp4` (30 fps), browser-drawn view (the UI default), 4 interleaved rounds of 60 s:**

| | analysed fps | processing / frame | latency mean / p95 | viewer render / frame | overlay header / frame |
|---|---|---|---|---|---|
| Original | 59.92 | 30.8 ms | 72.2 / 121 ms | 5.38 ms | 260 B |
| Clone, skeleton off | 59.92 | 31.2 ms | 73.6 / 129 ms | 5.35 ms | 272 B |
| Clone, skeleton on | 59.91 | 31.1 ms | 76.0 / 135 ms* | 5.71 ms | 2.0 KB |

\* mean of 3 rounds; one round (117 ms mean, 344 ms p95) overlapped other disk/CPU work on the machine.

**Saturated loads (both trees over capacity, latency = the full frame queue in both):**

| Load | Original | Clone (skeleton on) |
|---|---|---|
| 4 cameras, same clip, server-drawn view | 45.1-45.9 fps total, 2.07 s | 44.9-45.9 fps total, 2.07 s |
| 3 cameras, Drowning Dataset IMG_5495 / IMG_5498 / IMG_5491 (4K, 30 fps) | 37.5-38.7 fps total, 2.07 s | 38.1-39.1 fps total, 2.06 s |

**Eval harness, one clip at a time (Drowning Dataset, 4K):** per-frame latency mean original 15.3 / 12.8 / 13.9 ms,
clone 14.8 / 12.8 / 13.8 ms (IMG_5495 / IMG_5498 / IMG_5491).

**Parity:** on the same TensorRT engines, `lifeguard.eval.trace --box-tol 0 --conf-tol 0` over those 3 clips: 2,636
frames, 3,895 person observations, 10 alerts, 0 differences.

**Skeleton drawing** (`skeleton.draw`, 10 people, 1920 px frame): 1.06 ms with `LINE_8` (3.3 ms with `LINE_AA`).

**Reading the numbers:** FPS and per-frame processing are unchanged. With the skeleton off, latency matches the
original. With it on, the mean is about 2-4 ms higher while someone watches (below the timer resolution); the
skeleton costs nothing when no one watches. The 4-camera runs use 30 fps (and 4K) files that overload both trees,
unlike the site's 15 fps RTSP cameras. Not measured: real RTSP cameras end to end, the full pytest suite (`tests/` is
absent in this snapshot apart from the skeleton test), Playwright e2e.
