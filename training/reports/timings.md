# Timings (estimate vs actual)

| Date | Task | Estimate | Actual | Basis |
|---|---|---|---|---|
| 2026-10-28 | Phase 0 total (setup, installs, MMCV build, queue/watchdog, CVAT) | 2–4 h | ~3 h 20 min (incl. a restart) | prompt baseline |
| 2026-10-28 | MMCV 2.1.0 source build (sm_120) | 30–60 min | ~25 min | first build |
| 2026-10-28 | CVAT first start (image pull) | 15–30 min | ~10 min | |
| 2026-10-28 | Phase 1 audit (23 clips, 6.3 min GPU) + split | 10–20 min | ~25 min | prompt baseline |
| 2026-10-28 | Phase 2: 2 fps extraction (23 clips) | ~8 min | 10 min | 20 s/clip measured |
| 2026-10-28 | Phase 2: counts + dedup + clustering | ~15 min | 5 min | |
| 2026-10-28 | Voters on pool (benchmark) | — | 1.86 s/frame (20 frames, incl. warm-up) | est. 2 h 20 min for 4,644 (started 14:19) |
| 2026-10-28 | Voters on pool (4,664 frames) | 2 h 20 min | 56 min | 0.73 s/frame steady |
| 2026-10-28 | Consensus + SAM 2 refinement (pool) | — | 13.6 min | |
