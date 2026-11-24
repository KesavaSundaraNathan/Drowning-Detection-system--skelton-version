"""Audit traceability gate (plan Phase 9 text, built Phase 10): every audit row has a resolution and evidence.

    python tools/check_traceability.py              # CI: exit 1 on any problem
    python tools/check_traceability.py --skeleton   # print a table skeleton for new rows (IDs + audit labels)

Source of truth for the rows: docs/audit/PRODUCTION_AUDIT_2026-10-07.md (frozen copy of the audit's 1.0-1.7
tables). IDs: C1..C10 for the ranked critical findings (1.0), "<section>-<n>" for the per-file rows (1.1-1 ...).
The trace table in docs/AUDIT_TRACEABILITY.md has the columns  | ID | Audit item | Status | Resolution | Evidence |.

Rules (each failure is reported; any failure fails CI):
  * every audit row appears exactly once, no unknown IDs, and "Audit item" equals the audit's own label;
  * Status is one of FIXED, MITIGATED, DROPPED, ACCEPTED, OPEN;
  * Evidence holds >= 1 reference in backticks: `path`, `path::name` (name must occur in the file: a test
    function, a describe/it title, a heading), or `path#text` (text must occur in the file);
  * every referenced path exists in the repository;
  * FIXED needs at least one TEST reference (tests/**, frontend/src/**/*.test.*, frontend/e2e/**) - a fix
    without a test that would fail if it regressed is not "fixed".
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = Path("docs/audit/PRODUCTION_AUDIT_2026-10-07.md")
TRACE = Path("docs/AUDIT_TRACEABILITY.md")
STATUSES = ("FIXED", "MITIGATED", "DROPPED", "ACCEPTED", "OPEN")
TEST_PREFIXES = ("tests/", "frontend/e2e/")
_REF = re.compile(r"`([^`]+)`")


@dataclass(frozen=True)
class AuditRow:
    id: str
    label: str


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _label(cell: str) -> str:
    """Normalise an audit 'File' cell: markdown links -> their text, no backticks / bold, single spaces."""
    cell = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", cell)
    cell = cell.replace("`", "").replace("**", "")
    return re.sub(r"\s+", " ", cell).strip()


def audit_rows(root: Path = ROOT) -> list[AuditRow]:
    rows: list[AuditRow] = []
    section = None
    count = 0
    header_seen = False
    for line in (root / AUDIT).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^### (1\.\d)\b", line)
        if m:
            section, count, header_seen = m.group(1), 0, False
            continue
        if line.startswith("## "):
            section = None
        if section is None or not line.startswith("|"):
            continue
        if re.match(r"^\|\s*-", line):
            continue
        if not header_seen:  # the first table line of a section is its header
            header_seen = True
            continue
        cells = _cells(line)
        if section == "1.0":
            rows.append(AuditRow(f"C{cells[0]}", _label(cells[2])))
        else:
            count += 1
            rows.append(AuditRow(f"{section}-{count}", _label(cells[0])))
    return rows


@dataclass
class TraceRow:
    id: str
    item: str
    status: str
    resolution: str
    evidence: str


def trace_rows(root: Path = ROOT) -> list[TraceRow]:
    out: list[TraceRow] = []
    for line in (root / TRACE).read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or re.match(r"^\|\s*-", line):
            continue
        cells = _cells(line)
        if len(cells) != 5 or cells[0] in ("ID", ""):
            continue
        out.append(TraceRow(*cells))
    return out


def _is_test(path: str) -> bool:
    return path.startswith(TEST_PREFIXES) or bool(re.search(r"\.test\.tsx?$", path))


def check_reference(ref: str, root: Path = ROOT) -> str | None:
    """None if the reference resolves, else the reason."""
    path, needle = ref, None
    for sep in ("::", "#"):
        if sep in ref:
            path, needle = ref.split(sep, 1)
            break
    target = root / path
    if not target.exists():
        return f"`{ref}`: {path} does not exist"
    if needle is not None:
        if target.is_dir():
            return f"`{ref}`: {path} is a directory"
        if needle not in target.read_text(encoding="utf-8", errors="replace"):
            return f"`{ref}`: {needle!r} not found in {path}"
    return None


def check(root: Path = ROOT) -> list[str]:
    problems: list[str] = []
    audit = {r.id: r for r in audit_rows(root)}
    if len(audit) < 10:
        problems.append(f"only {len(audit)} audit rows parsed from {AUDIT} - parser or file broken")
    seen: dict[str, int] = {}
    for t in trace_rows(root):
        seen[t.id] = seen.get(t.id, 0) + 1
        a = audit.get(t.id)
        if a is None:
            problems.append(f"{t.id}: not an audit row")
            continue
        if _label(t.item) != a.label:
            problems.append(f"{t.id}: audit item {t.item!r} != audit label {a.label!r}")
        if t.status.strip("*") not in STATUSES:
            problems.append(f"{t.id}: status {t.status!r} not one of {', '.join(STATUSES)}")
        if not t.resolution:
            problems.append(f"{t.id}: no resolution text")
        refs = _REF.findall(t.evidence)
        if not refs:
            problems.append(f"{t.id}: no evidence reference")
        for ref in refs:
            why = check_reference(ref, root)
            if why:
                problems.append(f"{t.id}: {why}")
        if t.status.strip("*") == "FIXED" and not any(_is_test(r.split("::")[0].split("#")[0]) for r in refs):
            problems.append(f"{t.id}: FIXED without a test reference")
    for rid in audit:
        if rid not in seen:
            problems.append(f"{rid}: audit row has no traceability row ({audit[rid].label})")
    for rid, n in seen.items():
        if n > 1:
            problems.append(f"{rid}: {n} traceability rows (must be exactly one)")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skeleton", action="store_true")
    a = ap.parse_args(argv)
    if a.skeleton:
        for r in audit_rows():
            sys.stdout.write(f"| {r.id} | {r.label} | OPEN | | |\n")
        return 0
    problems = check()
    rows = trace_rows()
    by_status = {s: sum(1 for t in rows if t.status.strip("*") == s) for s in STATUSES}
    for p in problems:
        sys.stdout.write(f"  FAIL  {p}\n")
    sys.stdout.write(
        f"traceability: {len(audit_rows())} audit rows, "
        + ", ".join(f"{k} {v}" for k, v in by_status.items())
        + f" - {'FAILED' if problems else 'every row has a resolution and resolving evidence'}\n"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
