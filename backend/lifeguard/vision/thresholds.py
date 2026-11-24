"""The versioned, audited thresholds file (plan Phase 3: "move thresholds into a versioned, audited file").

    backend/lifeguard/vision/thresholds.json      schema aquaguard-thresholds/1

Every tunable number and behaviour-changing switch of detection, tracking, scene and behaviour logic
lives in that file - never in code. The loader is strict on purpose:

  * every key of `_thresholds_spec.SPEC` must be present, no other key may be, and every value must
    have exactly its spec type (an `int` where a `float` is expected is refused, so `4` vs `4.0`
    cannot silently change integer arithmetic);
  * the active set carries its `version` and the sha256 of the exact file bytes; the server records
    both in the audit log at startup (`lifeguard.alerts.thresholds_audit`), and evaluation reports
    record them in their metadata, so every result is traceable to the thresholds that produced it.

`LIFEGUARD_THRESHOLDS_FILE` points the process at a candidate file (the evaluation harness uses this
to measure a proposed change before it is adopted). It is read once, on first use.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from lifeguard.vision._thresholds_spec import SPEC

SCHEMA = "aquaguard-thresholds/1"
DEFAULT_FILE = Path(__file__).with_name("thresholds.json")
ENV_VAR = "LIFEGUARD_THRESHOLDS_FILE"


class ThresholdsError(ValueError):
    """The thresholds file is missing, malformed or does not match the spec."""


@dataclass(frozen=True)
class ThresholdSet:
    version: str
    sha256: str
    path: str
    values: MappingProxyType[str, Any]

    def describe(self) -> dict[str, str]:
        return {"version": self.version, "sha256": self.sha256, "path": self.path}


def _coerce(key: str, value: Any, spec: Any) -> Any:
    if isinstance(spec, dict):
        if not isinstance(value, dict) or set(value) != set(spec):
            raise ThresholdsError(f"{key}: expected keys {sorted(spec)}, got {value!r}")
        # plain dict (callers deep-copy mode params); key order = spec order = the v2 order
        return {k: _coerce(f"{key}.{k}", value[k], s) for k, s in spec.items()}
    if isinstance(spec, tuple):
        kind, inner = spec
        if not isinstance(value, list):
            raise ThresholdsError(f"{key}: expected a list, got {value!r}")
        if kind == "tuple":
            if len(value) != len(inner):
                raise ThresholdsError(f"{key}: expected {len(inner)} items, got {len(value)}")
            return tuple(_coerce(f"{key}[{i}]", v, s) for i, (v, s) in enumerate(zip(value, inner, strict=True)))
        return [_coerce(f"{key}[{i}]", v, inner) for i, v in enumerate(value)]
    ok = {
        "bool": lambda v: isinstance(v, bool),
        "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "float": lambda v: isinstance(v, float),
        "str": lambda v: isinstance(v, str),
    }[spec](value)
    if not ok:
        raise ThresholdsError(f"{key}: expected {spec}, got {type(value).__name__} {value!r}")
    return value


def parse(raw: bytes, path: str = "<memory>") -> ThresholdSet:
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ThresholdsError(f"{path}: not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ThresholdsError(f"{path}: schema must be {SCHEMA!r}")
    version = doc.get("version")
    if not isinstance(version, str) or not version:
        raise ThresholdsError(f"{path}: missing version")
    log = doc.get("changelog")
    if not isinstance(log, list) or not log or not isinstance(log[-1], dict) or log[-1].get("version") != version:
        raise ThresholdsError(f"{path}: the last changelog entry must describe version {version}")
    groups = doc.get("values")
    if not isinstance(groups, dict):
        raise ThresholdsError(f"{path}: missing values")
    flat: dict[str, Any] = {}
    for section, entries in groups.items():
        if not isinstance(entries, dict):
            raise ThresholdsError(f"{path}: section {section!r} must be an object")
        for k, v in entries.items():
            if k in flat:
                raise ThresholdsError(f"{path}: {k} appears in more than one section")
            flat[k] = v
    missing, unknown = sorted(set(SPEC) - set(flat)), sorted(set(flat) - set(SPEC))
    if missing or unknown:
        raise ThresholdsError(f"{path}: missing keys {missing}, unknown keys {unknown}")
    values = {k: _coerce(k, flat[k], SPEC[k]) for k in SPEC}
    return ThresholdSet(version, hashlib.sha256(raw).hexdigest(), path, MappingProxyType(values))


def load(path: Path | None = None) -> ThresholdSet:
    p = Path(path) if path is not None else DEFAULT_FILE
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise ThresholdsError(f"cannot read thresholds file {p}: {exc}") from exc
    return parse(raw, str(p))


_lock = threading.Lock()
_active: ThresholdSet | None = None


def active() -> ThresholdSet:
    """The process-wide thresholds (default file, or $LIFEGUARD_THRESHOLDS_FILE), loaded once."""
    global _active
    with _lock:
        if _active is None:
            env = os.environ.get(ENV_VAR, "").strip()
            _active = load(Path(env) if env else None)
        return _active
