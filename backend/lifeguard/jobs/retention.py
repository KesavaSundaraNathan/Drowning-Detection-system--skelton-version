"""Retention: purge snapshots and recordings older than RETENTION_DAYS (default 30).

Footage of swimmers (often children) is personal data; keeping it forever is a liability. This job:
  * deletes only regular files strictly INSIDE the configured directories (resolved path
    containment; symlinks are never followed or deleted through);
  * also empties `<data_dir>/uploads`, left behind by the removed upload feature, on the same schedule;
  * prunes login-attempt rows older than the same window;
  * writes one audit event with the counts.
Run daily from a Windows scheduled task (Phase 8): ``python -m lifeguard retention run``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import Session

from lifeguard.core import audit
from lifeguard.core.config import Settings
from lifeguard.core.security import ratelimit
from lifeguard.db.models import AlertOutbox
from lifeguard.db.writer import DBWriter


@dataclass
class RetentionResult:
    cutoff: datetime
    files_deleted: dict[str, int] = field(default_factory=dict)
    login_attempts_deleted: int = 0
    outbox_rows_deleted: int = 0


def _purge_dir(root: Path, cutoff: datetime) -> int:
    if not root.exists():
        return 0
    base = root.resolve()
    deleted = 0
    for path in base.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(base):
            continue
        mtime = datetime.fromtimestamp(resolved.stat().st_mtime, tz=cutoff.tzinfo)
        if mtime < cutoff:
            resolved.unlink(missing_ok=True)
            deleted += 1
    return deleted


def run_retention(settings: Settings, writer: DBWriter, now: datetime) -> RetentionResult:
    cutoff = now - timedelta(days=settings.retention_days)
    result = RetentionResult(cutoff=cutoff)

    def purge_rows(db: Session) -> None:
        result.login_attempts_deleted = ratelimit.purge_older_than(db, cutoff)
        # delivered / dead outbox rows (pending rows are never purged: they are still being delivered)
        result.outbox_rows_deleted = int(
            db.execute(
                delete(AlertOutbox).where(AlertOutbox.status.in_(("sent", "dead")), AlertOutbox.created_at < cutoff)
            ).rowcount  # type: ignore[attr-defined]
            or 0
        )

    writer.run(purge_rows)
    for label, directory in (
        ("snapshots", settings.snapshots_dir),
        ("recordings", settings.recordings_dir),
        ("uploads", settings.data_dir / "uploads"),  # legacy: files of the removed upload feature
    ):
        result.files_deleted[label] = _purge_dir(directory, cutoff)

    writer.run(
        lambda db: audit.record(
            db,
            action="retention.purge",
            actor=None,
            actor_username="system",
            ip="local",
            target_type="storage",
            details={
                "cutoff": cutoff.isoformat(),
                "retention_days": settings.retention_days,
                "files_deleted": result.files_deleted,
                "login_attempts_deleted": result.login_attempts_deleted,
            },
        )
    )
    return result
