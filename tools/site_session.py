"""Site session kit (plan Phase 11 WP6; rules pre-registered in eval/site/levers_phase11.json "gates_wp6").
Driven by tools/site_session.ps1 (elevated, pinned tools, restricted session folder); usable on its own.

    python tools/site_session.py all     --session <DataDir>\\site_session\\<id> --base https://<site> --ca <root.crt>
    python tools/site_session.py check | capacity | live | record | normal | replay | report  (same options)

Steps (each writes <session>/checks/<step>.json or its own files; none ever writes a camera URL):
  header    the DATA_PROTOCOL 9.3 header + legal references (prompted, or --header-file); session.json
  check     configuration vs deploy/site/site_profile.json, camera check per camera (pinned ffprobe + the live
            capture path), calibration coverage + map
  capacity  tiling off (the API's measured table) and on (computed, decision 4)
  live      every enabled camera for --live-minutes: MONITORED, analysed fraction >= 0.99, 0 dropped
  record    the drills: one recorder per camera, the drill runner (adults-only attestation per run), then the
            geometry snapshot, the SYNC offsets and the RIGHTS.csv rows
  normal    an adults-only normal session for --hours (same legal gate + the public-notice reference)
  replay    the OFF / GATING / MULTIVIEW arms (lifeguard.eval.site_replay)
  report    the PROVISIONAL (or --final) report (lifeguard.eval.site_report)

Credentials: LIFEGUARD_SITE_USER / LIFEGUARD_SITE_PASSWORD (an admin account), else prompted - never an argument.
Camera sources are read locally with the server's keys (needs the env file: run elevated on the server).
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import ssl
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from tools.acceptance import Client  # noqa: E402
from tools.make_hevc_clips import find_tool  # noqa: E402

DEFAULT_PROFILE = ROOT / "deploy" / "site" / "site_profile.json"


def out(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


class Kit:
    def __init__(self, a: argparse.Namespace) -> None:
        from lifeguard.site.session import SessionFolder

        self.a = a
        self.folder = SessionFolder(Path(a.session))
        self.profile = json.loads(Path(a.profile).read_text(encoding="utf-8"))
        self.ffmpeg = Path(a.ffmpeg) if a.ffmpeg else find_tool("ffmpeg", "LIFEGUARD_FFMPEG")
        self.ffprobe = Path(a.ffprobe) if a.ffprobe else find_tool("ffprobe", "LIFEGUARD_FFPROBE")
        if self.ffmpeg is None or self.ffprobe is None:
            raise SystemExit("pinned ffmpeg / ffprobe not found (site_session.ps1 passes them)")
        self._client: Client | None = None
        self._cams: list[Any] | None = None
        self.only: set[str] | None = None  # smoke: just the simulated camera

    # ── shared ──
    def client(self) -> Client:
        if self._client is None:
            u = urlsplit(self.a.base)
            if u.scheme == "https":
                ctx = ssl.create_default_context(cafile=self.a.ca) if self.a.ca else ssl.create_default_context()
            elif u.scheme == "http" and self.a.insecure_loopback and u.hostname in ("127.0.0.1", "localhost"):
                ctx = ssl.create_default_context()  # unused for http; loopback test stacks only
            else:
                raise SystemExit("--base must be https:// (http only with --insecure-loopback on 127.0.0.1)")
            c = Client(self.a.base, ctx)
            user = os.environ.get("LIFEGUARD_SITE_USER") or input("admin user: ")
            pw = os.environ.get("LIFEGUARD_SITE_PASSWORD") or getpass.getpass("password: ")
            st, body, _ = c.req("POST", "/api/v1/auth/login", {"username": user, "password": pw})
            if st != 200:
                raise SystemExit(f"login failed ({st})")
            c.csrf = body["csrf_token"]
            self._client = c
        return self._client

    def live_status(self) -> dict[str, Any]:
        st, body, _ = self.client().req("GET", "/api/v1/live/status")
        if st != 200:
            raise SystemExit(f"/live/status: {st}")
        return dict(body)

    def cameras(self) -> list[Any]:
        if self._cams is None:
            from lifeguard.site.sources import load_cameras

            self._cams = load_cameras()
        return self._cams

    def enabled(self) -> list[Any]:
        return [
            c
            for c in self.cameras()
            if c.is_enabled and c.source_type == "rtsp" and (self.only is None or c.camera_id in self.only)
        ]

    def redactor(self) -> Any:
        from lifeguard.site.sources import Redactor

        return Redactor([c.url for c in self.cameras()])

    def header(self) -> Any:
        from lifeguard.site.drills import SessionHeader

        data = self.folder.read_json("session.json", {}).get("header")
        if data is None:
            raise SystemExit("no session header yet: run the 'header' step first")
        return SessionHeader(**data)

    # ── steps ──
    def step_header(self) -> None:
        from dataclasses import asdict

        from lifeguard.site.drills import SessionHeader

        sid = Path(self.a.session).name
        if self.a.header_file:
            h = SessionHeader(**json.loads(Path(self.a.header_file).read_text(encoding="utf-8")))
        else:
            ask = input
            h = SessionHeader(
                session_id=sid,
                site=self.profile["site"],
                date=datetime.now(UTC).date().isoformat(),
                cameras=[c.name for c in self.enabled()],
                lifeguard_in_water=ask("lifeguard in the water (name / id): "),
                deck_supervisor=ask("deck supervisor: "),
                conditions=ask("conditions (light, weather, occupancy): "),
                public_present=ask("members of the public present? [yes/no]: ").strip().lower() == "yes",
                signage_up=ask("recording signage up? [yes/no]: ").strip().lower() == "yes",
                dpia_ref=ask("DPIA reference: "),
                dsa_ref=ask("data-sharing agreement reference: "),
                counsel_ref=ask("counsel approval reference: "),
                retention=ask("retention period (from the agreement): "),
                performer_ids=[
                    p.strip() for p in ask("performer ids with consent on file (P01;P02;...): ").split(";") if p.strip()
                ],
                consent_on_file=ask("performer consent forms on file for every id above? [yes/no]: ").strip().lower()
                == "yes",
                public_notice_ref=ask("public-notice reference (normal session; blank if none): "),
            )
        meta = self.folder.read_json("session.json", {})
        meta.update(
            {
                "session_id": h.session_id,
                "header": asdict(h),
                "cameras": [
                    {**c.public(), "pool_key": (c.calibration or {}).get("pool_id", "main")} for c in self.enabled()
                ],
                "profile": self.profile,
                "kit": {"created_utc": datetime.now(UTC).isoformat(timespec="seconds")},
            }
        )
        self.folder.write_json("session.json", meta)
        out(f"session header written ({h.session_id})")

    def step_check(self) -> dict[str, Any]:
        from lifeguard.site import camera_check, config_check, coverage

        live = self.live_status()
        cams = [
            {**c.public(), "pool_key": (c.calibration or {}).get("pool_id", "main")}
            for c in self.cameras()
            if self.only is None or c.camera_id in self.only
        ]
        zones = self._zones()
        cfg = config_check.check_config(self.profile, cams, zones, live)
        self.folder.write_json("checks/config_check.json", cfg)
        out(f"configuration: {cfg['verdict']}")
        red = self.redactor()
        expect = self.profile["camera"]
        results = {}
        for cam in self.enabled():
            r = camera_check.check_camera(cam.url, self.ffprobe, expect, redact=red, seconds=self.a.probe_seconds)
            results[cam.name] = r
            out(
                f"camera {cam.name}: {r['verdict']} "
                + ", ".join(f"{c['check']}={c['verdict']}" for c in r["checks"] if c["verdict"] != "PASS")
            )
        self.folder.write_json("checks/camera_check.json", results)
        sizes = {
            c.name: (results.get(c.name, {}).get("opencv") or {}).get("frame_size") or expect["size"]
            for c in self.enabled()
        }
        cov = coverage.check_coverage(
            [
                {"camera_id": c.camera_id, "name": c.name, "calibration": c.calibration, "frame_size": sizes[c.name]}
                for c in self.enabled()
            ]
        )
        coverage.draw_map(cov, self.folder.path("checks", "coverage.png"))
        self.folder.write_json("checks/coverage.json", coverage.public(cov))
        out(f"coverage: {cov['verdict']} {cov['problems']}")
        return {
            "config": cfg["verdict"],
            "cameras": {k: v["verdict"] for k, v in results.items()},
            "coverage": cov["verdict"],
        }

    def _zones(self) -> list[dict[str, Any]]:
        from sqlalchemy import select

        from lifeguard.core.config import load_settings
        from lifeguard.db.models import Zone
        from lifeguard.db.session import make_database

        db = make_database(load_settings().effective_database_url)
        try:
            with db.read_sessions() as rs:
                return [
                    {"name": z.name, "zone_type": z.zone_type, "pool_id": z.pool_id, "camera_id": z.camera_id}
                    for z in rs.execute(select(Zone).where(Zone.is_active)).scalars()
                ]
        finally:
            db.dispose()

    def step_capacity(self) -> dict[str, Any]:
        from lifeguard.site.config_check import capacity

        live = self.live_status()
        size = tuple(self.profile["camera"]["size"])
        cap = capacity(
            live["capacity"],
            cameras=len(self.enabled()),
            frame_wh=(int(size[0]), int(size[1])),
            target_fps=float(self.profile["camera"]["fps"]),
        )
        self.folder.write_json("checks/capacity.json", cap)
        out(
            f"capacity: tiling off {cap['tiling_off_limit']} camera(s), tiling on {cap['tiling_on_limit']} "
            f"(computed) at {cap['target_fps']:g} fps"
        )
        return cap

    def step_live(self) -> dict[str, Any]:
        """All enabled cameras for --live-minutes: counter deltas after a warm-up (acceptance.py method)."""
        ids = {c.camera_id for c in self.enabled()}
        from lifeguard.site.config_check import live_result, live_sample

        t0, worst, samples = time.monotonic(), 1.0, 0
        base: dict[str, tuple[int, int, int]] = {}
        last: dict[str, tuple[int, int, int]] = {}
        problems: list[dict[str, Any]] = []
        while time.monotonic() - t0 < self.a.live_minutes * 60:
            time.sleep(self.a.sample_s)
            cams = [x for x in self.live_status().get("cameras", []) if x["camera_id"] in ids]
            counts = {
                x["camera_id"]: (
                    int(x["capture"]["frames_read"]),
                    int(x["analysis"]["analysed"]),
                    int(x.get("dropped") or 0),
                )
                for x in cams
            }
            if time.monotonic() - t0 < self.a.warmup_s:
                continue
            if not base:
                base = counts
                continue
            samples += 1
            w, probs = live_sample(cams, base, counts, time.monotonic() - t0)
            worst = min(worst, w)
            problems += probs
            last = counts
            if len(cams) != len(ids):
                problems.append({"t_s": round(time.monotonic() - t0), "missing_cameras": len(ids) - len(cams)})
        dropped = sum(last[c][2] - base[c][2] for c in last if c in base)
        res = live_result(len(ids), time.monotonic() - t0, samples, worst, problems, dropped)
        self.folder.write_json("checks/live_run.json", res)
        out(f"live run: {res}")
        return res

    def _record(self, block: str, run_body: Any, seconds_hint: float) -> list[dict[str, Any]]:
        from lifeguard.site.recorder import BlockRecorder, check_disk, write_rights_rows

        folder = self.folder.root / "recordings" / block
        disk = check_disk(folder, len(self.enabled()), seconds_hint)
        if not disk["ok"]:
            raise SystemExit(f"not enough disk: {disk}")
        rec = BlockRecorder(self.ffmpeg, self.ffprobe, folder, block, self.redactor())
        rec.start(self.enabled())
        try:
            run_body(rec)
        finally:
            recs = rec.stop()
        self.folder.add_recordings(block, recs)
        h = self.header()
        write_rights_rows(
            self.folder.root / "RIGHTS.csv", recs, site=h.site, session_id=h.session_id, dsa_ref=h.dsa_ref
        )
        bad = [r["camera"] for r in recs if "start_wall" not in r]
        for r in recs:
            if r.get("restarts"):
                parts = [x["file"] for x in r.get("parts", [])]
                out(f"WARNING: recorder of {r['camera']} restarted {r['restarts']} time(s); parts {parts}")
        out(
            f"recorded block {block}: {len(recs) - len(bad)} camera(s)"
            + (f"; NO usable recording from {bad}" if bad else "")
        )
        return recs

    def step_record(self) -> None:
        from lifeguard.core.config import load_settings
        from lifeguard.site import sync
        from lifeguard.site.drills import DRILLS, DrillLog, GateError, legal_problems, run_drill
        from lifeguard.site.session import snapshot_geometry

        h = self.header()
        problems = legal_problems(h, normal=False)
        if problems:
            raise SystemExit("legal gate: " + "; ".join(problems))
        log = (
            DrillLog.load(self.folder.root)
            if (self.folder.root / "drill_log.json").exists()
            else DrillLog(self.folder.root, h)
        )

        def body(rec: Any) -> None:
            time.sleep(2.0)  # the recorders connect before the first mark
            if self.a.script:
                for step in json.loads(Path(self.a.script).read_text(encoding="utf-8")):
                    log.begin(
                        step["code"],
                        performer_ids=step.get("performer_ids", []),
                        position=step.get("position", ""),
                        adults_only=bool(step.get("adults_only")),
                    )
                    for event, delay in step["events"]:
                        time.sleep(float(delay))
                        log.mark(event)
                    log.end(step.get("outcome", "completed"))
                time.sleep(2.0)
                return
            while True:
                code = input(f"\nnext drill {sorted(DRILLS)} or 'done': ").strip()
                if code == "done":
                    return
                if code not in DRILLS or code == "NORMAL":
                    out("unknown drill")
                    continue
                try:
                    run_drill(log, code)
                except GateError as exc:
                    out(f"REFUSED: {exc}")
                alive = rec.alive()
                if not all(alive.values()):
                    out(f"WARNING: recorder stopped for {[k for k, v in alive.items() if not v]}")

        recs = self._record("drills", body, seconds_hint=3600)
        counts = snapshot_geometry(load_settings().effective_database_url, self.folder.path("geometry.sqlite"))
        out(f"geometry snapshot: {counts}")
        flash = [r for r in log.runs if r.code == "SYNC" and r.marks]
        if flash:
            res = sync.offsets(recs, self.folder.root / "recordings" / "drills", flash[-1].marks[0].wall)
            self.folder.write_json("sync.json", res)
            out(f"sync: {res['verdict']} {res.get('offsets_s')}")

    def step_normal(self) -> None:
        from lifeguard.site.drills import DrillLog, legal_problems

        h = self.header()
        problems = legal_problems(h, normal=True)
        if problems:
            raise SystemExit("legal gate (normal session): " + "; ".join(problems))
        adults = (
            self.a.adults_only
            or input("This is an ADULTS-ONLY session; nobody under 18 in view [type YES]: ").strip() == "YES"
        )
        log = (
            DrillLog.load(self.folder.root)
            if (self.folder.root / "drill_log.json").exists()
            else DrillLog(self.folder.root, h)
        )
        log.begin("NORMAL", performer_ids=[], position="whole pool", adults_only=adults, normal=True)
        seconds = self.a.seconds or self.a.hours * 3600

        def body(rec: Any) -> None:
            log.mark("start")
            end = time.monotonic() + seconds
            try:
                while time.monotonic() < end:
                    time.sleep(min(60.0, max(0.1, end - time.monotonic())))
                    out(f"normal session: {max(0, int(end - time.monotonic()))} s left; recorders {rec.alive()}")
            except KeyboardInterrupt:
                out("stopped by the operator")
            log.mark("end")

        try:
            self._record("normal", body, seconds_hint=seconds)
        finally:
            log.end("completed")

    def step_replay(self, block: str) -> None:
        from lifeguard.eval import site_replay

        res = site_replay.run_arms(
            self.folder.root,
            block,
            ["OFF", "GATING", "MULTIVIEW"],
            det_model=self.a.det_model,
            provider=self.a.provider or str(self.live_status().get("provider", "cuda")),
        )
        out(f"replay {block}: " + json.dumps({a: {k: v[k] for k in ("alerts", "seconds")} for a, v in res.items()}))

    def step_smoke(self) -> int:
        return _smoke(self)

    def step_report(self) -> int:
        from lifeguard.eval import site_report

        rep = site_report.build(self.folder.root, stage="FINAL" if self.a.final else "PROVISIONAL", ffmpeg=self.ffmpeg)
        path = site_report.write(self.folder.root, rep)
        for r in rep["rows"]:
            out(f"  {r['verdict']:8s} {r['id']}: {json.dumps(r['value'], default=str)[:100]}")
        out(f"report: {path}")
        return 1 if any(r["verdict"] == "FAIL" for r in rep["rows"]) else 0


SMOKE_NAME = "SMOKE camera"
SMOKE_CAL = {
    "image_points": [[0.05, 0.95], [0.95, 0.95], [0.85, 0.35], [0.15, 0.35]],
    "pool_points": [[0.0, 0.0], [25.0, 0.0], [25.0, 12.5], [0.0, 12.5]],
    "frame_width": 2560,
    "frame_height": 1440,
}
SMOKE_SCRIPT = [
    {"code": "SYNC", "performer_ids": [], "adults_only": True, "events": [["flash", 1.0]]},
    {"code": "D2", "performer_ids": ["P01"], "adults_only": True, "events": [["start", 1.0], ["end", 8.0]]},
    {"code": "D5", "performer_ids": ["P01"], "adults_only": True, "events": [["under", 2.0], ["surfaced", 12.0]]},
]


def _smoke(kit: Kit) -> int:
    """The laptop smoke run (gates_wp6: 1 simulated camera): every step on one camera registered for the run, then
    removed. Legal references are marked SMOKE (no person is recorded: the stream is a stock clip)."""
    if not kit.a.rtsp:
        raise SystemExit("smoke needs --rtsp <the simulated camera stream>")
    c = kit.client()
    st, cam, _ = c.req("POST", "/api/v1/cameras", {"name": SMOKE_NAME, "source_url": kit.a.rtsp})
    if st != 201:
        raise SystemExit(f"smoke camera: {st} {cam}")
    cid = cam["id"]
    try:
        st, body, _ = c.req("PUT", f"/api/v1/cameras/{cid}/calibration", SMOKE_CAL)
        if st != 200:
            raise SystemExit(f"smoke calibration: {st} {body}")
        st, body, _ = c.req("POST", f"/api/v1/live/cameras/{cid}/start")
        if st != 200:
            raise SystemExit(f"smoke start: {st} {body}")
        kit.only = {cid}
        kit.profile = {**kit.profile, "cameras": [SMOKE_NAME], "required_pool_zones": []}
        header = {
            "session_id": Path(kit.a.session).name,
            "site": "SMOKE",
            "date": datetime.now(UTC).date().isoformat(),
            "cameras": [SMOKE_NAME],
            "signage_up": True,
            "dpia_ref": "SMOKE",
            "dsa_ref": "SMOKE",
            "counsel_ref": "SMOKE",
            "retention": "delete after the smoke run",
            "performer_ids": ["P01"],
            "consent_on_file": True,
        }
        kit.a.header_file = str(kit.folder.write_json("smoke/header.json", header))
        kit.a.script = str(kit.folder.write_json("smoke/drills.json", SMOKE_SCRIPT))
        kit.step_header()
        kit.step_check()
        kit.step_capacity()
        kit.step_live()
        kit.step_record()
        if kit.a.provider is None:
            kit.a.provider = kit.live_status().get("provider", "cuda")
        kit.step_replay("drills")
        kit.step_report()
        return 0
    finally:
        c.req("POST", f"/api/v1/live/cameras/{cid}/stop")
        c.req("DELETE", f"/api/v1/cameras/{cid}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "step", choices=("header", "check", "capacity", "live", "record", "normal", "replay", "report", "all", "smoke")
    )
    ap.add_argument("--rtsp", default=None, help="smoke: the simulated camera stream")
    ap.add_argument("--session", required=True)
    ap.add_argument("--profile", default=str(DEFAULT_PROFILE))
    ap.add_argument("--base", default="https://localhost")
    ap.add_argument("--ca", default=None)
    ap.add_argument("--insecure-loopback", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--ffprobe", default=None)
    ap.add_argument("--header-file", default=None, help="the session header as JSON (scripted / smoke)")
    ap.add_argument("--script", default=None, help="scripted drills as JSON (smoke)")
    ap.add_argument("--probe-seconds", type=float, default=20.0)
    ap.add_argument("--live-minutes", type=float, default=5.0)
    ap.add_argument("--warmup-s", type=float, default=30.0)
    ap.add_argument("--sample-s", type=float, default=10.0)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--adults-only", action="store_true", help="attest the adults-only normal session (scripted)")
    ap.add_argument("--block", default="drills", choices=("drills", "normal"))
    ap.add_argument("--provider", default=None, choices=("cuda", "cpu"), help="replay provider (default: the API's)")
    ap.add_argument("--det-model", default=None)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args(argv)
    kit = Kit(a)
    if a.step == "header":
        kit.step_header()
    elif a.step == "check":
        kit.step_check()
    elif a.step == "capacity":
        kit.step_capacity()
    elif a.step == "live":
        kit.step_live()
    elif a.step == "record":
        kit.step_record()
    elif a.step == "normal":
        kit.step_normal()
    elif a.step == "replay":
        kit.step_replay(a.block)
    elif a.step == "report":
        return kit.step_report()
    elif a.step == "smoke":
        return kit.step_smoke()
    else:
        if not (Path(a.session) / "session.json").exists() or "header" not in kit.folder.read_json("session.json", {}):
            kit.step_header()
        kit.step_check()
        kit.step_capacity()
        kit.step_live()
        kit.step_record()
        kit.step_replay("drills")
        return kit.step_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
