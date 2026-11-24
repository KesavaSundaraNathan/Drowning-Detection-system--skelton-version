# Phase 0 report (GATE 0) — 2026-10-28

## Machine
RTX 5060 (8 GB, Blackwell sm_120, driver 617.42), i7 (28 threads), 32 GB RAM, Windows 11 Home, 1.7 TB free.
Power plan **High performance**, sleep / hibernate off; Windows Update active hours 06:00–24:00; no restart pending.

## Installed (owner-approved, items 1–6)
| Item | Version | Notes |
|---|---|---|
| WSL2 + Docker Desktop | Docker 29.9.2 | engine on WSL2 |
| CUDA Toolkit | 12.8 | was already installed (not on PATH before) |
| VS 2022 Build Tools (C++) | 17.14 | for MMCV |
| Firewall | `Life Guard CVAT 8080 (LAN)`, Private only | |
| Watchdog task | `LifeGuard-Training-Watchdog` | at log-on + every 5 min; ran at log-on after the restart, exit 0 |

## Training environment (`training\.venv`, `training\requirements.lock`, 97 pins)
PyTorch 2.7.1+cu128 · torchvision 0.22.1 · MMCV 2.1.0 (compiled from source for sm_120, CUDA ops verified) ·
MMEngine 0.10.7 · MMDetection 3.3.0 · MMPose 1.3.2 · D-FINE @ 956d170 · SAM 2 @ 2b90b9f (no CUDA extension) ·
transformers 4.57.1 (OWLv2, DINOv2) · OpenCV 4.12 · Albumentations 2.0.8. Licences: `docs\LICENSES.md`; weight hashes:
`docs\weights_manifest.txt`.

## Smoke tests (one 960×1072 frame, numbers only)
| Model | ms / frame | Peak VRAM |
|---|---|---|
| D-FINE-L COCO @ 640 / 1024 | 25.8 / 42.1 | 0.4 GB |
| D-FINE-M COCO @ 640 / 1024 | 16.1 / 28.5 | 0.3 GB |
| RTMDet-L COCO | 57.2 (incl. pre/post) | 0.3 GB |
| SAM 2.1 hiera-b+ | 48.4 embed + 5.6 / box | 0.8 GB |
| OWLv2 base ensemble | 294 | 1.9 GB |
| DINOv2 base | 12.8 | 1.0 GB |
COCO RTMDet scores these swimmers low (best person 0.27 on the test frame; D-FINE finds 6–13 > 0.4): consensus voters
need per-model thresholds, tuned on reviewed frames.

## Job queue + watchdog (`tools\queue\run_queue.ps1`, `tools\watchdog.ps1`)
| Test | Result |
|---|---|
| Two jobs in sequence, per-job logs, state in `queue_state.json` | PASS |
| Runner and job killed at epoch 3 (simulated crash) → watchdog requeues → job resumes at epoch 3 → next job runs | PASS |
| Always-failing job → 1 run + 2 restarts → `failed` + desktop alert | PASS |
| Overheat (threshold forced to 1 °C) → job stopped, PAUSED; cool (forced) → resumed from its checkpoint | PASS |
| Resume after a real reboot | watchdog task ran at log-on after the restart (exit 0); no job was queued, so the resume path is covered by the crash test |
| `notify.ps1` | PASS (balloon + beep + `logs\notify.log`) |

## CVAT (self-hosted, v2.80.0, MIT)
- URL: **http://localhost:8080**, admin `lifeguard_admin`, password in `training\cvat\credentials.txt` (not in git).
- Project 1 "Life Guard swimmers v1": `swimmer` / `person` rectangles with `activity` (freestyle, breaststroke,
  backstroke, butterfly, treading_vertical, floating, standing_wading, entering_diving, underwater, unknown), `pose`
  skeleton (COCO-17, 19 edges), `review` tag with `review_reason` (CVAT has no frame attributes; a tag is per frame).
- **Bound to this PC only (127.0.0.1)**: Docker Desktop created a firewall rule allowing all ports on **Public**
  networks, and this Wi-Fi is Public, so the LAN binding would have exposed CVAT. See "Open items".

## Open items (need you)
1. **RTMPose starting weights**: A / B / C (`INTERVENTION.md`). Blocks pose training only.
2. **LAN access to CVAT** (only if you want to label from another PC): mark this Wi-Fi as **Private**, and review the
   "Docker Desktop Backend" firewall rules (they allow all ports on Public networks). Then I switch CVAT to the LAN
   binding. Until then: label on this PC.
3. **Claude Code Notification hook**: add it yourself (`/hooks` in a `claude` terminal), command in `INTERVENTION.md`.
4. OpenSSH Server: not enabled (your choice: no for now).
5. UPS: your call.

## Deviations from the prompt
- Dataset folder `training_dataset\` (not `non-drowning datasets\`); 23 unique clips (one duplicate), 7 of them 10 min.
- The live Life Guard system is stopped during testing and training (your decision), so no GPU power cap is applied.
- The test split will be 16 / 3 / 4 clips (or balanced by minutes), not 14 / 3 / 3, because there are 23 clips.
