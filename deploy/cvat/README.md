# CVAT box-labelling pass (Phase 8)

**Goal:** turn the held-out **test** split (`eval/heldout_v1`, 6 clips) from event-level into box-level ground truth, so that in-water recall and ID metrics become measurable and D-FINE can retrain on **reviewed** labels (plan 7.1). The procedure is the one in `docs/DATA_PROTOCOL.md` §6.

| | |
|---|---|
| Tool | CVAT **v2.78.0** (MIT), self-hosted, **loopback only** (`docker-compose.local.yml` binds `127.0.0.1:8080`) |
| Pre-annotations | `build/cvat/heldout_v1_test/<clip>.xml` + `<clip>.events.csv`, made by `tools/cvat_preannotate.py` from the real pipeline's trace |
| Back to Life Guard | `eval/tools/cvat_to_lifeguard.py` → `aquaguard-eval/1`, validated by `lifeguard.eval.annotations` |
| Status (Phase 8) | Pre-annotations generated. **CVAT not yet running:** this laptop has no WSL 2 / Docker. An administrator must install them (below). |

## 1. Prerequisites (administrator, once)
1. Elevated PowerShell: `wsl --install`, then reboot.
2. Install **Docker Desktop**, or **Podman Desktop** with its docker CLI compatibility. Docker Desktop is free for small businesses; check the subscription terms for RVV.
3. Check that `docker compose version` is **≥ 2.24.4**: the override uses `!override`.

## 2. Start CVAT (labelling user)
```
powershell -ExecutionPolicy Bypass -File deploy\cvat\setup.ps1
```
The script does four things:
- clones CVAT at the pinned tag into `build/third_party/cvat`, and refuses any other checkout;
- pulls the images and starts them with the loopback override;
- waits for `http://127.0.0.1:8080`;
- asks once for the CVAT admin account. Choose your own password; it is never stored in this repo.

To stop it: `setup.ps1 -Stop`. Labels stay in CVAT's Docker volumes.

## 3. Create the project and tasks (UI, ~10 min)
1. **Projects → +:** name `lifeguard-heldout_v1`. Under *Constructor → Raw*, paste `deploy/cvat/labels.json`. That gives the `person` rectangle with mutable `visibility` / `state` / `activity`, immutable `age` / `ignore`, plus one tag per condition.
2. **For each test clip, Tasks → +:**
   - **Name:** the clip id, e.g. `ds_active_drowning_v7`. **Project:** the one above.
   - **File:** the video from `build/v2_baseline/Datasets/` (the name is in `build/cvat/heldout_v1_test/README.txt`).
   - **Advanced:** *Image quality 95*, *Use zip chunks* on, **no** cloud storage.
3. Open the task, then **Menu → Upload annotations → format `CVAT 1.1`**, and pick `build/cvat/heldout_v1_test/<clip>.xml`.

## 4. Label (DATA_PROTOCOL §6)
- The pre-filled tracks are **suggestions**. The detector behind them caught 1 of 5 test events. Correct every box at the keyframes (2 Hz), set `visibility` and `state`, add people it missed (tentative people are not pre-filled), and delete false boxes.
- Merge tracks the pipeline split (ID switches), and split tracks it merged.
- Record each event's victim: note the **CVAT track id** of the victim and write it into `track_id` in `<clip>.events.csv`. It is `-1` until then, and the converter refuses `-1`.
- Then a **second person reviews** every clip. Only reviewed labels count.

## 5. Export and convert
1. Task → **Export annotations → `CVAT for video 1.1`**.
2. Convert:
   ```
   python eval/tools/cvat_to_lifeguard.py <export>.xml --video "build/v2_baseline/Datasets/<file>.mp4" ^
       --clip-id <clip> --camera-id <clip> --split test --events build/cvat/heldout_v1_test/<clip>.events.csv ^
       --every 15 --out eval/heldout_v1_boxes/annotations/<clip>.json
   ```
3. Keep the result in a **new** eval dir (`eval/heldout_v1_boxes`), split unchanged. `eval/heldout_v1` stays frozen as committed. Box-level numbers become a new, separately reported baseline; they never retro-fit Phase 7/8 decisions.
