"""Site configuration check + capacity (Phase 11 WP6).

* `check_config`: deploy/site/site_profile.json against the installation - cameras (count, names, enabled, RTSP,
  calibrated to ONE pool), required pool zones, and the running API's live status (cameras MONITORED, RTSP over TCP
  verified, target fps, first-frame grace).
* `capacity`: tiling OFF = the running API's own admission profile (GET /live/status "capacity"); tiling ON is
  COMPUTED from the same measured batch table with calls per frame = 1 + the tiles of a 2560x1440 frame (decision 4:
  monitoring is never stopped to benchmark tiling). The same arithmetic admission uses (live/capacity.CapacityProfile).
"""

from __future__ import annotations

from typing import Any

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


def tiles_per_frame(frame_wh: tuple[int, int], input_hw: tuple[int, int] = (640, 640)) -> int:
    """Detector calls per frame with DET_TILING over the whole frame: 1 + tiles, as
    live.runtime.detector_calls_per_frame counts them for admission."""
    from lifeguard.vision.inference import enhance

    w, h = frame_wh
    return 1 + len(enhance.tile_windows((0, 0, w, h), min(input_hw), 0.2, w, h))


def capacity(
    status_capacity: dict[str, Any], *, cameras: int, frame_wh: tuple[int, int], target_fps: float
) -> dict[str, Any]:
    from lifeguard.live.capacity import CapacityProfile

    batch = {int(k): float(v) for k, v in status_capacity["batch_ms"].items()}
    overhead = float(status_capacity["overhead_ms"])
    bmax = int(status_capacity["batch_max"])
    off = CapacityProfile(batch, overhead, bmax, calls_per_frame=int(status_capacity.get("calls_per_frame", 1)))
    calls_on = tiles_per_frame(frame_wh)
    on = CapacityProfile(batch, overhead, bmax, calls_per_frame=calls_on)
    return {
        "target_fps": target_fps,
        "cameras": cameras,
        "tiling_off_limit": off.limit(target_fps),
        "tiling_off_fps_at_cameras": round(off.analysed_fps(cameras), 2),
        "tiling_on_calls_per_frame": calls_on,
        "tiling_on_limit": on.limit(target_fps),
        "tiling_on_fps_at_cameras": round(on.analysed_fps(cameras), 2),
        "tiling_off_admits_all": off.limit(target_fps) >= cameras,
        "tiling_on_admits_all": on.limit(target_fps) >= cameras,
        "source": "computed from the running API's measured batch table (gates_wp6 decision 4)",
    }


def check_config(
    profile: dict[str, Any], cameras: list[dict[str, Any]], zones: list[dict[str, Any]], live: dict[str, Any]
) -> dict[str, Any]:
    """`cameras`: SiteCamera.public() + "pool_key"; `zones`: active zones {name, zone_type, pool_id, camera_id};
    `live`: GET /live/status."""
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, value: Any, expected: Any, warn: bool = False) -> None:
        checks.append(
            {"check": name, "verdict": PASS if ok else (WARN if warn else FAIL), "value": value, "expected": expected}
        )

    enabled = [c for c in cameras if c["enabled"]]
    want = profile["cameras"]
    add("camera_count", len(enabled) == len(want), len(enabled), len(want))
    add(
        "camera_names",
        sorted(c["name"] for c in enabled) == sorted(want),
        sorted(c["name"] for c in enabled),
        sorted(want),
    )
    add("rtsp_sources", all(c["source"].startswith("rtsp") for c in enabled), [c["source"] for c in enabled], "rtsp://")
    uncal = [c["name"] for c in enabled if not c["calibrated"]]
    add("calibrated", not uncal, uncal or "all", "every enabled camera")
    pools = sorted({str(c.get("pool_key")) for c in enabled if c["calibrated"]})
    add("one_pool", len(pools) == 1, pools, "one pool key")
    names = {z["name"].strip().lower() for z in zones if z.get("camera_id") is None}
    missing = [z for z in profile.get("required_pool_zones", []) if z.strip().lower() not in names]
    add("pool_zones", not missing, sorted(names), profile.get("required_pool_zones", []))
    # the running API
    running = {c["camera_id"]: c for c in live.get("cameras", [])}
    not_running = [c["name"] for c in enabled if c["camera_id"] not in running]
    add("cameras_running", not not_running, not_running or "all", "every enabled camera started")
    not_mon = [c["name"] for c in live.get("cameras", []) if c.get("monitoring") != "MONITORED"]
    add("cameras_monitored", not not_mon, not_mon or "all", "MONITORED")
    rtsp = live.get("rtsp") or {}
    add("rtsp_tcp_verified", bool(rtsp.get("verified")) and rtsp.get("rtsp_transport") == "tcp", rtsp, "tcp, verified")
    cap = live.get("capacity") or {}
    add(
        "target_fps",
        float(cap.get("target_fps", 0)) == float(profile["camera"]["fps"]),
        cap.get("target_fps"),
        profile["camera"]["fps"],
    )
    add(
        "first_frame_timeout_s",
        float(live.get("first_frame_timeout_s", 0)) == 12.0,
        live.get("first_frame_timeout_s"),
        12.0,
    )
    fallback = [c["name"] for c in live.get("cameras", []) if (c.get("capture") or {}).get("fps_source") == "fallback"]
    add("stream_fps_reported", not fallback, fallback or "all", "every stream reports its fps", warn=True)
    verdicts = {c["verdict"] for c in checks}
    return {"verdict": FAIL if FAIL in verdicts else WARN if WARN in verdicts else PASS, "checks": checks}


def live_sample(
    cams: list[dict[str, Any]],
    base: dict[str, tuple[int, int, int]],
    counts: dict[str, tuple[int, int, int]],
    t_s: float,
) -> tuple[float, list[dict[str, Any]]]:
    """One /live/status sample against the window's base counters (decoded, analysed, dropped) per camera:
    (worst analysed fraction, problems). One frame may be in flight at a sample (acceptance.py method)."""
    worst = 1.0
    problems: list[dict[str, Any]] = []
    for x in cams:
        cid = x["camera_id"]
        b, n = base.get(cid), counts.get(cid)
        if b is None or n is None:
            continue
        decoded = n[0] - b[0]
        worst = min(worst, (n[1] - b[1]) / decoded + 1.0 / decoded if decoded > 0 else 0.0)
        if x["monitoring"] != "MONITORED" or n[2] > b[2] or x.get("stalled") or decoded <= 0:
            problems.append(
                {
                    "t_s": round(t_s),
                    "camera": x["name"],
                    "monitoring": x["monitoring"],
                    "stalled": bool(x.get("stalled")),
                    "decoded": decoded,
                    "dropped": n[2] - b[2],
                }
            )
    return worst, problems


def live_result(
    cameras: int, seconds: float, samples: int, worst: float, problems: list[dict[str, Any]], dropped: int
) -> dict[str, Any]:
    """The live-run verdict inputs: every camera MONITORED, never stalled, decoding throughout; `dropped` = frames
    dropped in the window (final - base, summed over cameras)."""
    bad = [
        p
        for p in problems
        if p.get("missing_cameras")
        or p.get("stalled")
        or p.get("decoded", 1) <= 0
        or p.get("monitoring", "MONITORED") != "MONITORED"
    ]
    return {
        "cameras": cameras,
        "seconds": round(seconds),
        "samples": samples,
        "min_analysed_fraction": round(min(worst, 1.0), 4) if samples else None,
        "all_monitored": samples > 0 and not bad,
        "dropped": int(dropped),
        "problems": problems[:50],
    }
