"""Release gate (plan Phase 7.4, built Phase 8): a SIGNED verdict the app reads to drop the ASSIST / PILOT banner.

    python -m lifeguard.eval.release_gate keygen                      # once per site (DPAPI-protected key)
    python -m lifeguard.eval.release_gate sign --report <test report.json> --live <live report.json> \
        --site "Pool A"                                                # writes <data_dir>/release_gate.json
        [--max-false-alerts-per-hour 0.5]                              # only STRICTER than the policy (1.0)
    python -m lifeguard.eval.release_gate verify                       # what /system/release will say

Inputs are harness reports (`lifeguard.eval.evaluate`): one on the held-out TEST split, and one with a live
multi-camera benchmark (`--live-streams N`, N = the admitted camera count). The gate PASSES only if every
criterion below is measured and met; anything not measured, or a limit nobody agreed, is a FAIL.

  1 in_water_recall          >= 0.95   box-level recall of people in the water (needs box labels)
  2 event_recall             >= 0.95   drowning events detected
  3 time_to_alert_median_s   <= 10     median seconds from event start to the first alert
  4 normal_standing_false_alerts == 0  false alerts on normal or standing swimmers
  5 false_alerts_per_camera_hour <= 1.0  (FALSE_ALERT_LIMIT_PER_CAMERA_HOUR, agreed 2026-10-08, Phase 10: strict
                                       enough to prevent lifeguard alarm fatigue, practical for complex water;
                                       a site may sign a STRICTER limit, never a looser one - see check_policy)
  6 live_analysed_fraction   >= 0.99   on every stream, with streams >= the admitted camera count
  7 p99_over_p50             <= 2.0    latency tail
  8 evidence_volume          >= 20 test events and >= 1.0 test camera-hour   [v3 addition, not in the plan:
                                       5 events cannot support a 0.95 recall claim either way]

The signed document binds the verdict to WHAT was measured: the detector file + sha256, the thresholds
version + sha256 and the app version. /system/release (app/api/v1/system.py -> status()) re-checks the
signature with the site's public key, the binding against the RUNNING detector and thresholds, and the
false-alert limit against this code's policy (a verdict signed against a looser limit is refused); a
missing, tampered, stale or failed report keeps ASSIST / PILOT. The private key is DPAPI-protected
(Windows account that ran `keygen`); the API only ever holds the public key.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

FORMAT = "aquaguard-release-gate/1"
GATE_FILE = "release_gate.json"
PRIVATE_FILE = "release_gate.key"  # raw 32-byte Ed25519 private key: DPAPI blob (Windows) / 0600 file (POSIX)
PUBLIC_FILE = "release_gate.pub"  # base64 raw 32-byte public key
_ENTROPY = b"aquaguard-release-gate-v1"
MIN_TEST_EVENTS = 20
MIN_TEST_CAMERA_HOURS = 1.0
# Site policy, agreed in writing 2026-10-08 (Phase 10): at most 1.0 false alert per camera-hour. Measured on the
# held-out TEST split; equality passes. Changing it is a policy decision: it is part of every signed verdict and
# status() refuses a verdict whose limit is looser than this value.
FALSE_ALERT_LIMIT_PER_CAMERA_HOUR = 1.0


def check_policy(max_fa_per_hour: float | None) -> float:
    """The false-alert limit to sign with: the policy by default; a stricter site value is allowed, looser is not."""
    if max_fa_per_hour is None:
        return FALSE_ALERT_LIMIT_PER_CAMERA_HOUR
    if not max_fa_per_hour >= 0:  # also rejects NaN
        raise ValueError(f"false-alert limit must be >= 0, got {max_fa_per_hour}")
    if max_fa_per_hour > FALSE_ALERT_LIMIT_PER_CAMERA_HOUR:
        raise ValueError(
            f"false-alert limit {max_fa_per_hour}/camera-hour is looser than the site policy "
            f"({FALSE_ALERT_LIMIT_PER_CAMERA_HOUR}); only a stricter limit may be signed"
        )
    return float(max_fa_per_hour)


ASSIST_MESSAGE = "ASSIST / PILOT - not a primary safety system. Lifeguards must keep watching the pool."


# ── evaluation ───────────────────────────────────────────────────────────────
def _crit(key: str, target: str, value: Any, met: bool | None, note: str = "") -> dict[str, Any]:
    status = "NOT MEASURED" if met is None else ("MET" if met else "NOT MET")
    return {"key": key, "target": target, "value": value, "status": status, **({"note": note} if note else {})}


def _all_zero(values: list[Any]) -> bool | None:
    """None (not measured) if any count is missing, else whether every count is zero."""
    if any(v is None for v in values):
        return None
    return all(int(v) == 0 for v in values)


def evaluate(
    test: dict[str, Any], live: dict[str, Any] | None, max_fa_per_hour: float | None = None
) -> list[dict[str, Any]]:
    limit = check_policy(max_fa_per_hour)
    h = test.get("headline") or {}
    split = (test.get("splits") or {}).get("test") or {}
    events = (split.get("events") or {}).get("total")
    hours = split.get("camera_hours")
    if h.get("labelled_split_used") != "test":
        raise ValueError("the accuracy report must be scored on the held-out TEST split (headline.labelled_split_used)")

    def ge(v: Any, t: float) -> bool | None:
        return None if v is None else float(v) >= t

    def le(v: Any, t: float) -> bool | None:
        return None if v is None else float(v) <= t

    normal = [h.get("false_alerts_normal_swimming"), h.get("false_alerts_standing_in_water")]
    fa_h = h.get("false_alerts_per_camera_hour")
    live_part = (live or {}).get("live") or {}
    live_frac = live_part.get("analysed_fraction_min")
    streams = live_part.get("streams")
    admitted = (live_part.get("capacity") or {}).get("limit")
    live_ok: bool | None = None
    if live_frac is not None and streams and admitted:
        live_ok = float(live_frac) >= 0.99 and int(streams) >= int(admitted) >= 1
    return [
        _crit(
            "in_water_recall",
            ">= 0.95",
            h.get("in_water_recall"),
            ge(h.get("in_water_recall"), 0.95),
            "" if h.get("in_water_recall") is not None else "needs box-level labels on the test split",
        ),
        _crit("event_recall", ">= 0.95", h.get("event_recall"), ge(h.get("event_recall"), 0.95)),
        _crit(
            "time_to_alert_median_s",
            "<= 10",
            h.get("time_to_alert_median_s"),
            le(h.get("time_to_alert_median_s"), 10.0),
        ),
        _crit("normal_standing_false_alerts", "== 0", normal, _all_zero(normal)),
        _crit(
            "false_alerts_per_camera_hour",
            f"<= {limit:g} per camera-hour (site policy {FALSE_ALERT_LIMIT_PER_CAMERA_HOUR:g})",
            fa_h,
            le(fa_h, limit),
        ),
        _crit(
            "live_analysed_fraction",
            ">= 0.99 on every stream, streams >= admitted cameras",
            {"min": live_frac, "streams": streams, "admitted": admitted},
            live_ok,
            "" if live_ok is not None else "no live benchmark at the admitted camera count",
        ),
        _crit("p99_over_p50", "<= 2.0", h.get("p99_over_p50"), le(h.get("p99_over_p50"), 2.0)),
        _crit(
            "evidence_volume",
            f">= {MIN_TEST_EVENTS} test events and >= {MIN_TEST_CAMERA_HOURS} camera-hour",
            {"events": events, "camera_hours": hours},
            None
            if events is None or hours is None
            else int(events) >= MIN_TEST_EVENTS and float(hours) >= MIN_TEST_CAMERA_HOURS,
        ),
    ]


# ── binding: what the verdict is about ──────────────────────────────────────
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_sha_cache: dict[tuple[str, int, int], str] = {}


def _file_sha(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _sha_cache:
        _sha_cache[key] = _sha256(path)
    return _sha_cache[key]


def _running_pose() -> Path | None:
    """The pose model this server runs (None: POSE_ENABLED off or no pose model configured)."""
    from lifeguard.vision import config as vision_config
    from lifeguard.vision.constants import POSE_ENABLED

    path = vision_config.current().pose_model_path
    return Path(path) if (POSE_ENABLED and path) else None


def current_binding(detector: Path, pose: Path | None = None) -> dict[str, Any]:
    """What a verdict is about. Swapping the detector, the pose model or the thresholds voids it."""
    from lifeguard import __version__
    from lifeguard.vision import thresholds

    t = thresholds.active()
    pose = pose if pose is not None else _running_pose()
    return {
        "detector": {"file": detector.name, "sha256": _file_sha(detector) if detector.is_file() else None},
        "pose": {"file": pose.name, "sha256": _file_sha(pose) if pose.is_file() else None} if pose else None,
        "thresholds": {"version": t.version, "sha256": t.sha256},
        "app_version": __version__,
    }


def report_binding(report: dict[str, Any]) -> dict[str, Any]:
    tree = (report.get("meta") or {}).get("tree") or {}
    if tree.get("layout") != "v3":
        raise ValueError("the report must come from the v3 tree (meta.tree.layout == 'v3')")
    det = tree.get("det_model")
    pose = tree.get("pose_model")
    return {
        "detector_file": Path(det).name if det else None,
        "pose_file": Path(pose).name if pose else None,
        "thresholds": tree.get("thresholds") or {},
    }


# ── signing ──────────────────────────────────────────────────────────────────
def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def keygen(keys_dir: Path, machine_scope: bool = False) -> Path:
    from cryptography.hazmat.primitives import serialization

    from lifeguard.core.security.keys import write_protected

    keys_dir.mkdir(parents=True, exist_ok=True)
    if (keys_dir / PRIVATE_FILE).exists():
        raise SystemExit(f"refusing to overwrite {keys_dir / PRIVATE_FILE}: every signed gate would become invalid")
    key = Ed25519PrivateKey.generate()
    raw = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    write_protected(keys_dir / PRIVATE_FILE, raw, _ENTROPY, machine_scope)
    (keys_dir / PUBLIC_FILE).write_text(base64.b64encode(pub).decode("ascii") + "\n", encoding="ascii")
    return keys_dir / PUBLIC_FILE


def _private(keys_dir: Path) -> Ed25519PrivateKey:
    from lifeguard.core.security.keys import read_protected

    return Ed25519PrivateKey.from_private_bytes(read_protected(keys_dir / PRIVATE_FILE, _ENTROPY))


def _public(keys_dir: Path) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(base64.b64decode((keys_dir / PUBLIC_FILE).read_text(encoding="ascii")))


def key_id(pub: Ed25519PublicKey) -> str:
    from cryptography.hazmat.primitives import serialization

    raw = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]


def build(
    test: dict[str, Any], live: dict[str, Any] | None, max_fa: float | None, site: str, detector: Path
) -> dict[str, Any]:
    binding = current_binding(detector)
    measured = report_binding(test)
    if measured["detector_file"] != binding["detector"]["file"]:
        raise ValueError(
            f"the report measured {measured['detector_file']}, this server runs {binding['detector']['file']}"
        )
    running_pose = (binding.get("pose") or {}).get("file")
    if measured.get("pose_file") != running_pose:
        raise ValueError(f"the report measured pose model {measured.get('pose_file')}, this server runs {running_pose}")
    if measured["thresholds"].get("sha256") != binding["thresholds"]["sha256"]:
        raise ValueError("the report was measured with different thresholds than this server runs")
    limit = check_policy(max_fa)
    criteria = evaluate(test, live, limit)
    return {
        "site": site,
        "policy": {"max_false_alerts_per_camera_hour": limit},
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "binding": binding,
        "split": {"name": (test.get("meta") or {}).get("name"), "argv": (test.get("meta") or {}).get("argv")},
        "criteria": criteria,
        "failed": [c["key"] for c in criteria if c["status"] != "MET"],
        "gate_passed": all(c["status"] == "MET" for c in criteria),
    }


def sign(payload: dict[str, Any], key: Ed25519PrivateKey) -> dict[str, Any]:
    sig = key.sign(_canonical(payload))
    return {
        "format": FORMAT,
        "payload": payload,
        "key_id": key_id(key.public_key()),
        "signature": base64.b64encode(sig).decode(),
    }


def verify(doc: dict[str, Any], pub: Ed25519PublicKey) -> bool:
    if doc.get("format") != FORMAT or not isinstance(doc.get("payload"), dict):
        return False
    try:
        pub.verify(base64.b64decode(str(doc.get("signature", ""))), _canonical(doc["payload"]))
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


# ── what the app shows ───────────────────────────────────────────────────────
def status(data_dir: Path, keys_dir: Path, detector: Path) -> dict[str, Any]:
    """/system/release: GATE_PASSED only for a valid signature, a matching binding and a passed gate."""

    def assist(reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "mode": "ASSIST_PILOT",
            "gate_passed": False,
            "message": ASSIST_MESSAGE,
            "reason": reason,
            **(extra or {}),
        }

    path = data_dir / GATE_FILE
    if not path.is_file():
        return assist("no signed release-gate report for this site")
    if not (keys_dir / PUBLIC_FILE).is_file():
        return assist("release-gate public key missing (run `release_gate keygen` on this server)")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        ok = verify(doc, _public(keys_dir))
    except (OSError, ValueError):
        ok = False
    if not ok:
        return assist("release-gate report signature is INVALID (tampered or signed with another key)")
    p = doc["payload"]
    summary = {"site": p.get("site"), "created_at": p.get("created_at"), "criteria": p.get("criteria", [])}
    if p.get("binding") != current_binding(detector):
        return assist(
            "the signed report is for a different detector / thresholds / app version than this server runs", summary
        )
    signed_limit = (p.get("policy") or {}).get("max_false_alerts_per_camera_hour")
    if not isinstance(signed_limit, (int, float)) or not 0 <= signed_limit <= FALSE_ALERT_LIMIT_PER_CAMERA_HOUR:
        return assist(
            f"the signed report does not apply the false-alert policy "
            f"(<= {FALSE_ALERT_LIMIT_PER_CAMERA_HOUR:g} per camera-hour); re-sign it",
            summary,
        )
    if not p.get("gate_passed"):
        return assist("release gate FAILED: " + ", ".join(p.get("failed", [])), summary)
    return {
        "mode": "GATE_PASSED",
        "gate_passed": True,
        "message": f"Release gate passed for {p.get('site')} "
        f"on {p.get('created_at')}. Lifeguard supervision is still required.",
        "reason": "",
        **summary,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────
def _detector(settings: Any) -> Path:
    from lifeguard.vision import config as vision_config

    return (
        Path(settings.detector_model)
        if settings.detector_model
        else vision_config.MODELS_DIR / vision_config.DEFAULT_DET_MODEL
    )


def main(argv: list[str] | None = None) -> int:
    from lifeguard.core.config import load_settings

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen")
    k.add_argument("--machine", action="store_true", help="DPAPI machine scope instead of the current account")
    s = sub.add_parser("sign")
    s.add_argument("--report", type=Path, required=True, help="harness report.json on the TEST split")
    s.add_argument("--live", type=Path, default=None, help="harness report.json with --live-streams")
    s.add_argument(
        "--max-false-alerts-per-hour",
        type=float,
        default=None,
        help=f"stricter site limit (default and maximum: the policy, {FALSE_ALERT_LIMIT_PER_CAMERA_HOUR:g})",
    )
    s.add_argument("--site", required=True)
    s.add_argument("--out", type=Path, default=None)
    sub.add_parser("verify")
    a = ap.parse_args(argv)
    settings = load_settings()
    if a.cmd == "keygen":
        sys.stdout.write(f"public key: {keygen(settings.keys_dir, a.machine)}\n")
        return 0
    if a.cmd == "sign":
        test = json.loads(a.report.read_text(encoding="utf-8"))
        live = json.loads(a.live.read_text(encoding="utf-8")) if a.live else None
        doc = sign(
            build(test, live, a.max_false_alerts_per_hour, a.site, _detector(settings)), _private(settings.keys_dir)
        )
        out = a.out or settings.data_dir / GATE_FILE
        out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
        for c in doc["payload"]["criteria"]:
            sys.stdout.write(f"  {c['status']:13s} {c['key']:30s} {json.dumps(c['value'])}  target {c['target']}\n")
        sys.stdout.write(f"{'PASS' if doc['payload']['gate_passed'] else 'FAIL'} - signed {out}\n")
        return 0 if doc["payload"]["gate_passed"] else 3
    st = status(settings.data_dir, settings.keys_dir, _detector(settings))
    sys.stdout.write(json.dumps({k: v for k, v in st.items() if k != "criteria"}, indent=2) + "\n")
    return 0 if st["gate_passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
