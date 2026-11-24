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
