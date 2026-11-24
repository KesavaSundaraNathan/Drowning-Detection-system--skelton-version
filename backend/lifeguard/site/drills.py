"""Drills, legal gate and drill log for the site session (Phase 11 WP6; pre-registered in
eval/site/levers_phase11.json "gates_wp6": drills, legal_gate, decisions 1 + 2; docs/DATA_PROTOCOL.md 4-5, 8-9).

* **Legal gate** (`legal_problems`): nothing is recorded until the DPIA, data-sharing agreement and counsel approval
  references, the retention period, signage, performer consent and (normal session) the public-notice reference
  are given. Only the references are stored, never the documents.
* **Adults only** (decision 1): every recording - each drill run and the normal session - needs the operator's
  attestation that nobody under 18 is in the water or in view; refused otherwise, and the attestation is logged.
* **Drill log**: marks on the SERVER clock (wall + monotonic), JSON with every mark and the DATA_PROTOCOL 9.2 CSV
  (one row per run). The replay aligns recordings to the same wall clock.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SAFETY = (
    "Adults only (18+). A qualified lifeguard supervises every drill; the performer can stop at any time with the "
    "agreed hand signal. Never swim over a submerged performer."
)


@dataclass(frozen=True)
class Drill:
    code: str
    title: str
    instructions: str
    marks: tuple[str, ...]  # the events marked per run, in order
    repeat_marks: bool = False  # D3: one mark per entry, as many as there are
    min_runs: int = 1
    lone_swimmer: bool = False  # exactly one person in the water (handoff / break metrics need it)
    submersion: bool = False


DRILLS: dict[str, Drill] = {
    d.code: d
    for d in (
        Drill("SYNC", "Sync flash", "One light flash (or clap) that EVERY camera sees. Mark at the flash.", ("flash",)),
        Drill("D1", "Many people", "6-15 adults spread over all lanes for 3 min.", ("start", "end")),
        Drill(
            "D2",
            "Lengths through every handoff",
            "ONE adult swims 4 lengths of 25 m; nobody else in the water (lifeguard on deck).",
            ("start", "end"),
            lone_swimmer=True,
        ),
        Drill(
            "D3",
            "Entries",
            "Adults enter one at a time: step in, ladder, jump. Mark AT each entry.",
            ("entry",),
            repeat_marks=True,
        ),
        Drill(
            "D4",
            "Zone crossings",
            "1-2 adults change lanes and step on / off the submerged teaching platform.",
            ("start", "end"),
        ),
        Drill(
            "D5",
            "Controlled submersion",
            "A trained adult sits on the bottom 10-15 s, rescuer in the water. Mark 'under' when the head goes under, "
            "'surfaced' when it is back up. 10 runs: the far end of every camera and the overlap areas.",
            ("under", "surfaced"),
            min_runs=10,
            submersion=True,
        ),
        Drill(
            "D5b",
            "Submersion with a passer-by",
            "As D5; 2-5 s after 'under' a second adult swims past in the ADJACENT lane within about 2 m - NEVER over "
            "the performer. Mark 'under', 'pass' (passer level with the performer), 'surfaced'. At least 5 runs.",
            ("under", "pass", "surfaced"),
            min_runs=5,
            submersion=True,
        ),
        Drill(
            "D6",
            "Instructor and learners (adults)",
            "A standing adult instructor with ADULT learners using floats / noodles in shallow water, 5 min.",
            ("start", "end"),
        ),
        Drill(
            "D7",
            "Glare passes",
            "An adult swims through the glare region repeatedly, 3 min (afternoon light).",
            ("start", "end"),
        ),
        Drill(
            "D7s",
            "Submersion in glare (optional)",
            "Only if the lifeguard agrees: one controlled submersion inside the glare, as D5.",
            ("under", "surfaced"),
            submersion=True,
        ),
        Drill(
            "NORMAL",
            "Normal session (adults only)",
            "An announced ADULTS-ONLY session (lap swimming / aqua aerobics); recorded unattended for N hours.",
            ("start", "end"),
        ),
    )
}


@dataclass
class SessionHeader:
    """DATA_PROTOCOL 9.3 + the legal references (gates_wp6.legal_gate)."""

    session_id: str
    site: str
    date: str
    cameras: list[str]
    lifeguard_in_water: str = ""
    deck_supervisor: str = ""
    conditions: str = ""
    public_present: bool = False
    signage_up: bool = False
    dpia_ref: str = ""
    dsa_ref: str = ""
    counsel_ref: str = ""
    retention: str = ""
    performer_ids: list[str] = field(default_factory=list)
    consent_on_file: bool = False
    public_notice_ref: str = ""  # normal session only


def legal_problems(h: SessionHeader, *, normal: bool) -> list[str]:
    """Empty list = recording allowed."""
    p: list[str] = []
    for name in ("dpia_ref", "dsa_ref", "counsel_ref", "retention"):
        if not str(getattr(h, name)).strip():
            p.append(f"{name} missing")
    if not h.signage_up:
        p.append("signage is not up")
    if not normal and not h.performer_ids:
        p.append("no performer ids")
    if not normal and not h.consent_on_file:
        p.append("performer consent forms not confirmed on file")
    if normal and not h.public_notice_ref.strip():
        p.append("public_notice_ref missing (the announced recording window)")
    return p


class GateError(RuntimeError):
    pass


@dataclass
class Mark:
    code: str
    run: int
    event: str
    wall: float  # server clock, seconds since the epoch (UTC)
    mono: float
    note: str = ""

    @property
    def iso(self) -> str:
        return datetime.fromtimestamp(self.wall, UTC).isoformat(timespec="milliseconds")


@dataclass
class Run:
    code: str
    run: int
    performer_ids: list[str]
    position: str
    adults_only_attested: bool
    attested_at: float
    marks: list[Mark] = field(default_factory=list)
    outcome: str = ""
    notes: str = ""


class DrillLog:
    """The session's drill log: JSON (every mark) + DATA_PROTOCOL 9.2 CSV, rewritten after every change."""

    CSV_COLUMNS = (
        "session_id",
        "date",
        "camera_ids",
        "code",
        "performer_id",
        "position",
        "start_time",
        "end_time",
        "outcome",
        "notes",
    )

    def __init__(
        self,
        folder: Path,
        header: SessionHeader,
        *,
        clock: Callable[[], float] = time.time,
        mono: Callable[[], float] = time.monotonic,
    ) -> None:
        self.folder = folder
        self.header = header
        self.clock, self.mono = clock, mono
        self.runs: list[Run] = []
        self.current: Run | None = None

    def begin(
        self, code: str, *, performer_ids: list[str], position: str, adults_only: bool, normal: bool = False
    ) -> Run:
        if code not in DRILLS:
            raise GateError(f"unknown drill {code!r}")
        problems = legal_problems(self.header, normal=normal or code == "NORMAL")
        if problems:
            raise GateError("legal gate: " + "; ".join(problems))
        if not adults_only:
            raise GateError("adults-only attestation refused: nothing is recorded (gates_wp6 decision 1)")
        unknown = [p for p in performer_ids if p not in self.header.performer_ids]
        if unknown and code != "NORMAL":
            raise GateError(f"performer ids without consent on file: {unknown}")
        if self.current is not None:
            raise GateError(f"drill {self.current.code} run {self.current.run} is still open")
        n = 1 + sum(1 for r in self.runs if r.code == code)
        self.current = Run(code, n, list(performer_ids), position, True, self.clock())
        self.runs.append(self.current)
        self.save()
        return self.current

    def mark(self, event: str, note: str = "") -> Mark:
        run = self.current
        if run is None:
            raise GateError("no drill run is open")
        drill = DRILLS[run.code]
        if event not in drill.marks:
            raise GateError(f"{run.code} has marks {drill.marks}, not {event!r}")
        m = Mark(run.code, run.run, event, self.clock(), self.mono(), note)
        run.marks.append(m)
        self.save()
        return m

    def end(self, outcome: str = "completed", notes: str = "") -> Run:
        run = self.current
        if run is None:
            raise GateError("no drill run is open")
        run.outcome, run.notes = outcome, notes
        self.current = None
        self.save()
        return run

    # ── files ──
    def to_dict(self) -> dict[str, Any]:
        return {"header": asdict(self.header), "runs": [asdict(r) for r in self.runs], "safety": SAFETY}

    def save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / "drill_log.json").write_bytes((json.dumps(self.to_dict(), indent=1) + "\n").encode("utf-8"))
        with (self.folder / "drill_log.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(self.CSV_COLUMNS)
            for r in self.runs:
                start = datetime.fromtimestamp(r.marks[0].wall, UTC).strftime("%H:%M:%S.%f")[:-5] if r.marks else ""
                end = datetime.fromtimestamp(r.marks[-1].wall, UTC).strftime("%H:%M:%S.%f")[:-5] if r.marks else ""
                marks = " ".join(
                    f"{m.event}@{datetime.fromtimestamp(m.wall, UTC).strftime('%H:%M:%S.%f')[:-5]}" for m in r.marks
                )
                w.writerow(
                    (
                        self.header.session_id,
                        self.header.date,
                        ";".join(self.header.cameras),
                        f"{r.code}#{r.run}",
                        ";".join(r.performer_ids),
                        r.position,
                        start,
                        end,
                        r.outcome or "open",
                        f"{marks} {r.notes}".strip(),
                    )
                )

    @classmethod
    def load(cls, folder: Path) -> DrillLog:
        data = json.loads((folder / "drill_log.json").read_text(encoding="utf-8"))
        log = cls(folder, SessionHeader(**data["header"]))
        for r in data["runs"]:
            marks = [Mark(**m) for m in r.pop("marks")]
            log.runs.append(Run(**r, marks=marks))
        return log


def run_drill(
    log: DrillLog,
    code: str,
    *,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
) -> Run:
    """Interactive: instructions + safety, performer ids, position, adults-only attestation, then one Enter per mark."""
    drill = DRILLS[code]
    say(f"\n=== {drill.code} {drill.title} ===\n{drill.instructions}\nSAFETY: {SAFETY}")
    performers = [p.strip() for p in ask("performer ids (e.g. P01;P02): ").split(";") if p.strip()]
    position = ask("position / lane / camera area: ").strip()
    adults = ask("Nobody under 18 is in the water or in view [type YES]: ").strip() == "YES"
    log.begin(code, performer_ids=performers, position=position, adults_only=adults)
    events = list(drill.marks)
    while events:
        event = events[0]
        reply = ask(
            f"  Enter = mark '{event}'" + ("  (e = end of entries)" if drill.repeat_marks else "") + ": "
        ).strip()
        if drill.repeat_marks and reply.lower() == "e":
            break
        m = log.mark(event)
        say(f"  {event} at {m.iso}")
        if not drill.repeat_marks:
            events.pop(0)
    outcome = ask("outcome [completed / aborted]: ").strip() or "completed"
    return log.end(outcome, ask("notes: ").strip())
