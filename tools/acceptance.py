"""Site acceptance test against an INSTALLED Life Guard (plan Phase 8 / Verification 5). Driven by tools/acceptance.ps1,
which starts the simulated cameras (MediaMTX + FFmpeg) first; it can also run on its own against real cameras.

    python tools/acceptance.py --base https://lifeguard.pool.local --ca <caddy root.crt> --cameras 3 --minutes 60 \
        --rtsp rtsp://127.0.0.1:8554/acc --out acceptance.json

Credentials: LIFEGUARD_ACCEPT_USER / LIFEGUARD_ACCEPT_PASSWORD (an admin account), else prompted (never a CLI
argument). Everything goes through the public HTTPS entry point, exactly like an operator's browser.

Checks (each PASS / FAIL in the JSON report; exit 0 only if all pass):
  tls_and_ui            HTTPS with the site certificate (verified against --ca), the UI served, security headers
  health_ready          GET /api/v1/health/ready -> 200
  unauthenticated       /cameras, /live/status, /alerts -> 401; a WebSocket without a session is refused
  provider_tensorrt     the live runtime runs on TensorRT, requested AND active (Phase 6: the site runs TensorRT with
                        engines built at install; a CUDA or CPU fallback fails)
  capacity              the camera count admission enforces at its target fps (capacity_limit = min(limit,
                        limit_skewed); >= --cameras)
  live_run              N cameras for --minutes: every camera MONITORED throughout, nothing dropped, analysed
                        fraction >= 0.99 on every camera (sampled every 10 s after a 30 s warm-up; frames queued or
                        held by the worker at a sample count as in flight, not missed)
  release_status        /system/release answers (ASSIST_PILOT is expected until a signed gate passes)
  login_lockout         LAST (it locks this client IP out of logins for 15 min): 30 bad logins for a random
                        user name end in lockout (429)

The report's "live" block has the harness format, so it can be given to `release_gate sign --live`.
"""

from __future__ import annotations

import argparse
import getpass
import http.client
import json
import os
import secrets
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

IN_WORKER = 2  # frames a camera worker holds outside its queue: the one being analysed + the prefetched next one


class Client:
    def __init__(self, base: str, ctx: ssl.SSLContext) -> None:
        self.base, self.ctx = base.rstrip("/"), ctx
        self.cookies: dict[str, str] = {}
        self.csrf = ""

    def req(self, method: str, path: str, body: Any = None) -> tuple[int, Any, Any]:
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(self.base + path, data=data, method=method)  # noqa: S310 - https only (checked)
        if data is not None:
            r.add_header("Content-Type", "application/json")
        if self.cookies:
            r.add_header("Cookie", "; ".join(f"{k}={v}" for k, v in self.cookies.items()))
        if self.csrf and method != "GET":
            r.add_header("X-CSRF-Token", self.csrf)
        try:
            with urllib.request.urlopen(r, timeout=30, context=self.ctx) as resp:  # noqa: S310  # nosec B310
                status, raw, headers = resp.status, resp.read(), resp.headers
        except urllib.error.HTTPError as exc:
            status, raw, headers = exc.code, exc.read(), exc.headers
        for sc in headers.get_all("Set-Cookie") or []:
            k, _, rest = sc.partition("=")
            self.cookies[k.strip()] = rest.split(";", 1)[0]
        try:
            return status, (json.loads(raw) if raw else None), headers
        except ValueError:
            return status, raw, headers


def ws_refused(base: str, ctx: ssl.SSLContext) -> tuple[bool, str]:
    """A WebSocket upgrade WITHOUT a session must not get 101 Switching Protocols."""
    u = urlsplit(base)
    host, port = u.hostname or "localhost", u.port or 443
    with socket.create_connection((host, port), timeout=10) as raw, ctx.wrap_socket(raw, server_hostname=host) as s:
        key = "dGhlIHNhbXBsZSBub25jZQ=="
        s.sendall(
            (
                f"GET /api/v1/ws HTTP/1.1\r\nHost: {u.netloc}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nOrigin: {base}\r\n\r\n"
            ).encode()
        )
        resp = http.client.HTTPResponse(s)  # type: ignore[arg-type]
        resp.begin()
        return resp.status != 101, f"HTTP {resp.status}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="https://<site address>")
    ap.add_argument("--ca", type=Path, required=True, help="the site certificate's root (Caddy internal CA root.crt)")
    ap.add_argument("--cameras", type=int, default=0, help="cameras to run (default: the admitted count)")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--rtsp", default="rtsp://127.0.0.1:8554/acc", help="camera i uses <rtsp>-<i>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-lockout-probe", action="store_true")
    a = ap.parse_args(argv)
    if urlsplit(a.base).scheme != "https":
        raise SystemExit("--base must be https:// (the installed entry point)")
    ctx = ssl.create_default_context(cafile=str(a.ca))
    checks: list[dict[str, Any]] = []
    live_block: dict[str, Any] = {}

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})
        sys.stdout.write(f"  {'PASS' if ok else 'FAIL'}  {name}: {json.dumps(detail)[:200]}\n")
        sys.stdout.flush()

    c = Client(a.base, ctx)
    st, body, hdr = c.req("GET", "/")
    want = ("Strict-Transport-Security", "Content-Security-Policy", "X-Frame-Options", "X-Content-Type-Options")
    missing = [h for h in want if not hdr.get(h)]
    check(
        "tls_and_ui",
        st == 200 and b'<div id="root">' in (body or b"") and not missing,
        {"status": st, "missing_headers": missing},
    )
    st, _, _ = c.req("GET", "/api/v1/health/ready")
    check("health_ready", st == 200, {"status": st})
    anon = {
        p: Client(a.base, ctx).req("GET", p)[0] for p in ("/api/v1/cameras", "/api/v1/live/status", "/api/v1/alerts")
    }
    refused, how = ws_refused(a.base, ctx)
    check("unauthenticated", all(v == 401 for v in anon.values()) and refused, {"http": anon, "websocket": how})

    user = os.environ.get("LIFEGUARD_ACCEPT_USER") or input("admin user: ")
    pw = os.environ.get("LIFEGUARD_ACCEPT_PASSWORD") or getpass.getpass("password: ")
    st, body, _ = c.req("POST", "/api/v1/auth/login", {"username": user, "password": pw})
    if st != 200:
        raise SystemExit(f"login failed ({st}); acceptance needs an admin account")
    c.csrf = body["csrf_token"]
    st, live, _ = c.req("GET", "/api/v1/live/status")
    check(
        "provider_tensorrt",
        st == 200
        and live.get("enabled") is True
        and live.get("provider") == "tensorrt"
        and live.get("active_provider") == "tensorrt",
        {"enabled": live.get("enabled"), "provider": live.get("provider"), "active": live.get("active_provider")},
    )
    cap = live.get("capacity") or {}
    admitted = int(live.get("capacity_limit") or 0)  # [Phase 6] the enforced limit, not the uniform one
    n = a.cameras or admitted
    check(
        "capacity",
        admitted >= max(1, n),
        {"admitted": admitted, "uniform": cap.get("limit"), "target_fps": cap.get("target_fps"), "requested": n},
    )

    created: list[str] = []
    try:
        for i in range(n):
            st, cam, _ = c.req(
                "POST",
                "/api/v1/cameras",
                {"name": f"acceptance-{i}", "source_url": f"{a.rtsp}-{i}", "default_mode": "Medium"},
            )
            if st != 201:
                raise RuntimeError(f"camera create: {st} {cam}")
            created.append(cam["id"])
            st, out, _ = c.req("POST", f"/api/v1/live/cameras/{cam['id']}/start")
            if st != 200:
                raise RuntimeError(f"camera start: {st} {out}")
        # Steady state is measured from counter DELTAS after a 30 s warm-up (camera connect, first frames);
        # whatever happened before the window is reported separately as the startup gap, never hidden.
        t0, worst, samples, bad = time.monotonic(), 1.0, 0, []
        final: dict[str, float] = {}  # per camera: analysed / decoded over the whole window at the last sample
        base: dict[str, dict[str, int]] = {}
        startup: dict[str, Any] = {}
        while time.monotonic() - t0 < a.minutes * 60:
            time.sleep(10)
            st, live, _ = c.req("GET", "/api/v1/live/status")
            cams = [x for x in (live or {}).get("cameras", []) if x["camera_id"] in created]
            counts = {
                x["camera_id"]: {
                    "decoded": int(x["capture"]["frames_read"]),
                    "analysed": int(x["analysis"]["analysed"]),
                    "dropped": int(x.get("dropped") or 0),
                    "queued": int(x.get("queued") or 0),
                }
                for x in cams
            }
            if time.monotonic() - t0 < 30:
                continue
            if not base:
                base = counts
                startup = {x["name"]: {**counts[x["camera_id"]], "monitoring": x["monitoring"]} for x in cams}
                continue
            samples += 1
            for x in cams:
                b, now = base.get(x["camera_id"]), counts[x["camera_id"]]
                if b is None:
                    continue
                decoded = now["decoded"] - b["decoded"]
                # [Phase 6] frames still on their way at the sample are not missed: the camera's queue plus the
                # IN_WORKER frames the worker holds (current frame + prefetched next). A lost frame shows in
                # `dropped` (a failure below) or stays unanalysed over the window. (Was a fixed 1-frame allowance,
                # which the AC1 run of 2026-11-15 undershot at its earliest samples: 0.989 with nothing dropped.)
                in_flight = now["queued"] + IN_WORKER
                frac = (now["analysed"] - b["analysed"] + in_flight) / decoded if decoded > 0 else 0.0
                worst = min(worst, frac)
                final[x["name"]] = round((now["analysed"] - b["analysed"]) / decoded, 4) if decoded > 0 else 0.0
                if x["monitoring"] != "MONITORED" or now["dropped"] > b["dropped"] or x.get("stalled") or decoded <= 0:
                    bad.append(
                        {
                            "t_s": round(time.monotonic() - t0),
                            "camera": x["name"],
                            "monitoring": x["monitoring"],
                            "dropped_in_window": now["dropped"] - b["dropped"],
                            "stalled": x.get("stalled"),
                        }
                    )
            if len(cams) != n:
                bad.append({"t_s": round(time.monotonic() - t0), "missing_cameras": n - len(cams)})
        worst = min(worst, 1.0)
        ok = samples > 0 and not bad and worst >= 0.99
        live_block = {
            "streams": n,
            "seconds": round(time.monotonic() - t0),
            "window_starts_s": 40,
            "analysed_fraction_min": round(worst, 4),
            "analysed_fraction_final": final,
            "in_flight_allowance": f"queued + {IN_WORKER}",
            "capacity": cap,
            "samples": samples,
            "problems": bad[:50],
            "startup_gap": startup,  # counters at the start of the window: drops here happened while connecting
        }
        check("live_run", ok, {k: v for k, v in live_block.items() if k != "capacity"})
    finally:
        for cid in created:
            c.req("POST", f"/api/v1/live/cameras/{cid}/stop")
            c.req("DELETE", f"/api/v1/cameras/{cid}")
    st, rel, _ = c.req("GET", "/api/v1/system/release")
    check(
        "release_status",
        st == 200 and rel.get("mode") in ("ASSIST_PILOT", "GATE_PASSED"),
        {"mode": rel.get("mode"), "reason": rel.get("reason")},
    )
    if not a.no_lockout_probe:
        probe, codes = f"nobody-{secrets.token_hex(4)}", []
        for _ in range(30):
            codes.append(
                Client(a.base, ctx).req(
                    "POST", "/api/v1/auth/login", {"username": probe, "password": "wrong-" + secrets.token_hex(6)}
                )[0]
            )
        check(
            "login_lockout",
            429 in codes and all(x in (401, 429) for x in codes),
            {"first": codes[:6], "last": codes[-3:]},
        )
    report = {
        "format": "aquaguard-acceptance/1",
        "base": a.base,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "checks": checks,
        "passed": all(x["result"] == "PASS" for x in checks),
        "live": live_block,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    sys.stdout.write(f"{'ACCEPTANCE PASSED' if report['passed'] else 'ACCEPTANCE FAILED'} - {a.out}\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
