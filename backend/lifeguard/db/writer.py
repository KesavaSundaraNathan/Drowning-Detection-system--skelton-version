"""The single DB-writer thread: every write in the server goes through `DBWriter.run()`.

Why: SQLite allows one writer at a time; funnelling all writes through one thread removes
"database is locked" races between camera threads, request threads and background jobs, and
gives one place where every write is ordered, committed and (on failure) rolled back.

Each submitted function runs in its own transaction on the write engine: committed when it
returns, rolled back if it raises (the exception is re-raised in the caller). Functions must
return plain data (not live ORM objects) - their session is closed after the commit.
Calling `run()` from inside a writer job runs inline (no self-deadlock).
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, TypeVar

from sqlalchemy.orm import Session, sessionmaker

T = TypeVar("T")
log = logging.getLogger(__name__)
_STOP = object()


@dataclass
class _Job:
    fn: Callable[[Session], Any]
    future: Future[Any]


class WriterStopped(RuntimeError):
    pass


class DBWriter:
    def __init__(self, sessions: sessionmaker[Session], name: str = "db-writer") -> None:
        self._sessions = sessions
        self._queue: queue.Queue[_Job | object] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._active = threading.local()
        self._started = False
        self._stopped = False
        self.jobs_done = 0

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        if self._started and not self._stopped:
            self._stopped = True
            self._queue.put(_STOP)
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def run(self, fn: Callable[[Session], T], timeout: float = 30.0) -> T:
        if getattr(self._active, "session", None) is not None:
            return fn(self._active.session)  # nested call from inside a writer job
        if self._stopped or not self._started:
            raise WriterStopped("DB writer is not running")
        fut: Future[T] = Future()
        self._queue.put(_Job(fn, fut))
        return fut.result(timeout=timeout)

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is _STOP or not isinstance(job, _Job):
                return
            if not job.future.set_running_or_notify_cancel():
                continue
            with self._sessions() as session:
                self._active.session = session
                try:
                    result = job.fn(session)
                    session.commit()
                except BaseException as exc:  # noqa: BLE001 - forwarded to the caller's Future and re-raised there
                    session.rollback()
                    log.debug("db write rolled back: %s", type(exc).__name__)
                    job.future.set_exception(exc)
                else:
                    job.future.set_result(result)
                finally:
                    self._active.session = None
                    self.jobs_done += 1
