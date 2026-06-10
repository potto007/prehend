"""Cross-process admission gate for RequestScheduler (two-flock gate+pool).

Design: docs/superpowers/specs/2026-06-10-cross-process-coordination-design.md.

Two lock files per server key in a shared coordination directory:

    <dir>/<key>.gate  - doorway. Normal requests hold SH momentarily on the
                        way in; a p1 holds EX for its whole run, which freezes
                        new admissions machine-wide (the cross-process
                        _waiting_p1 rule) and serializes p1s globally.
    <dir>/<key>.pool  - the in-flight set. Normal requests hold SH for the
                        request duration; a p1 takes EX, granted only when
                        every holder drains (the cross-process _active == 0
                        rule).

Crash cleanup is the kernel's: flock drops when an fd closes, including on
process death. The gate distinguishes only p1 vs everything else; p2-p5
ordering stays in-process. Same-host processes only (flock does not span
machines, and network filesystems are explicitly out of scope).
"""

import fcntl
import logging
import os
import threading
from pathlib import Path

from lm_repl.clients.scheduler import Priority

log = logging.getLogger(__name__)


class CrossProcessGate:
    """Two-flock readers-writer gate with writer preference.

    enter()/aenter() acquire for one request; exit() releases one acquisition
    (non-blocking fd closes, so both sync and async paths use it). Normal
    requests' pool fds are fungible: exit(NORMAL) closes any one of this
    process's SH holds, which the kernel treats identically.
    """

    def __init__(self, coordination_dir: str | Path, server_key: str):
        self._dir = Path(coordination_dir)
        self._gate_path = self._dir / f"{server_key}.gate"
        self._pool_path = self._dir / f"{server_key}.pool"
        self._mu = threading.Lock()
        self._pool_fds: list[int] = []  # one SH fd per in-flight normal request
        self._p1_fds: tuple[int, int] | None = None  # (gate_fd, pool_fd) of the active p1
        # Fail fast: surface an unwritable dir or a no-flock filesystem at
        # construction, not on request N.
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            for path in (self._gate_path, self._pool_path):
                fd = self._open(path)
                try:
                    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except BlockingIOError:
                    pass  # held EX by a live p1 elsewhere: flock works here
                finally:
                    os.close(fd)
        except OSError as e:
            raise RuntimeError(
                f"cross-process coordination unavailable at {self._dir}: {e}"
            ) from e

    @staticmethod
    def _open(path: Path) -> int:
        return os.open(path, os.O_RDWR | os.O_CREAT, 0o644)

    def enter(self, priority: int) -> None:
        """Blocking acquisition for one request. Releases partial holds and
        re-raises on failure, leaving no lock behind."""
        if priority == Priority.CONTENTION_RETRY:
            gate_fd = self._open(self._gate_path)
            try:
                fcntl.flock(gate_fd, fcntl.LOCK_EX)
                pool_fd = self._open(self._pool_path)
                try:
                    fcntl.flock(pool_fd, fcntl.LOCK_EX)
                except BaseException:
                    os.close(pool_fd)
                    raise
            except BaseException:
                os.close(gate_fd)
                raise
            with self._mu:
                self._p1_fds = (gate_fd, pool_fd)
        else:
            gate_fd = self._open(self._gate_path)
            try:
                fcntl.flock(gate_fd, fcntl.LOCK_SH)
                pool_fd = self._open(self._pool_path)
                try:
                    fcntl.flock(pool_fd, fcntl.LOCK_SH)
                except BaseException:
                    os.close(pool_fd)
                    raise
            finally:
                # The gate is only the doorway: release it whether or not the
                # pool acquisition succeeded.
                os.close(gate_fd)
            with self._mu:
                self._pool_fds.append(pool_fd)

    def exit(self, priority: int) -> None:
        """Release one acquisition. Never raises: it sits in finally paths,
        and the locks are released by the fd close regardless."""
        try:
            if priority == Priority.CONTENTION_RETRY:
                with self._mu:
                    fds, self._p1_fds = self._p1_fds, None
                if fds is not None:
                    gate_fd, pool_fd = fds
                    os.close(pool_fd)
                    os.close(gate_fd)
            else:
                with self._mu:
                    pool_fd = self._pool_fds.pop() if self._pool_fds else None
                if pool_fd is not None:
                    os.close(pool_fd)
        except OSError as e:
            log.warning("gate exit failed (locks still released on close): %s", e)
