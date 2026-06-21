"""Tests for CrossProcessGate (two-flock gate+pool cross-process coordination)."""

import asyncio
import fcntl
import multiprocessing as mp
import os
import threading
import time

import pytest

from prehend.clients.coordination import CrossProcessGate
from prehend.clients.scheduler import Priority

KEY = "testkey0000000000"


# ---- module-level workers for spawn-context children ----

def _hold_pool_sh(dir_, key, acquired_evt, release_evt):
    gate = CrossProcessGate(dir_, key)
    gate.enter(Priority.NORMAL)
    acquired_evt.set()
    release_evt.wait(15)
    gate.exit(Priority.NORMAL)


def _hold_pool_sh_forever(dir_, key, acquired_evt):
    gate = CrossProcessGate(dir_, key)
    gate.enter(Priority.NORMAL)
    acquired_evt.set()
    time.sleep(60)


def _hold_gate_ex_raw(dir_, key, acquired_evt, release_evt):
    # Simulates a p1 that is WAITING (holds gate EX, not yet pool EX).
    fd = os.open(os.path.join(dir_, f"{key}.gate"), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    acquired_evt.set()
    release_evt.wait(15)
    os.close(fd)


# ---- same-process semantics ----

def test_normal_enters_are_shared(tmp_path):
    gate = CrossProcessGate(tmp_path, KEY)
    gate.enter(Priority.NORMAL)
    gate.enter(Priority.NORMAL)  # second SH must not block
    gate.exit(Priority.NORMAL)
    gate.exit(Priority.NORMAL)


def test_p1_enter_exit_roundtrip(tmp_path):
    gate = CrossProcessGate(tmp_path, KEY)
    gate.enter(Priority.CONTENTION_RETRY)
    gate.exit(Priority.CONTENTION_RETRY)
    # Reacquirable afterwards
    gate.enter(Priority.NORMAL)
    gate.exit(Priority.NORMAL)


def test_exit_without_enter_is_noop(tmp_path):
    gate = CrossProcessGate(tmp_path, KEY)
    gate.exit(Priority.NORMAL)
    gate.exit(Priority.CONTENTION_RETRY)


def test_unwritable_dir_raises(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, 0o500)
    try:
        with pytest.raises(RuntimeError, match="coordination unavailable"):
            CrossProcessGate(ro / "locks", KEY)
    finally:
        os.chmod(ro, 0o700)


# ---- cross-process semantics ----

def test_p1_waits_for_other_process_share(tmp_path):
    ctx = mp.get_context("spawn")
    acquired, release = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_hold_pool_sh, args=(str(tmp_path), KEY, acquired, release))
    child.start()
    try:
        assert acquired.wait(15)
        gate = CrossProcessGate(tmp_path, KEY)
        entered = threading.Event()

        def p1():
            gate.enter(Priority.CONTENTION_RETRY)
            entered.set()

        threading.Thread(target=p1, daemon=True).start()
        time.sleep(0.3)
        assert not entered.is_set()  # blocked: child holds pool SH
        release.set()
        assert entered.wait(15)
        gate.exit(Priority.CONTENTION_RETRY)
    finally:
        release.set()
        child.join(15)
    assert child.exitcode == 0


def test_waiting_p1_blocks_new_normal_admissions(tmp_path):
    ctx = mp.get_context("spawn")
    acquired, release = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_hold_gate_ex_raw, args=(str(tmp_path), KEY, acquired, release))
    child.start()
    try:
        assert acquired.wait(15)
        gate = CrossProcessGate(tmp_path, KEY)
        entered = threading.Event()

        def normal():
            gate.enter(Priority.NORMAL)
            entered.set()

        threading.Thread(target=normal, daemon=True).start()
        time.sleep(0.3)
        assert not entered.is_set()  # blocked at the gate doorway
        release.set()
        assert entered.wait(15)
        gate.exit(Priority.NORMAL)
    finally:
        release.set()
        child.join(15)
    assert child.exitcode == 0


def test_crash_releases_locks(tmp_path):
    ctx = mp.get_context("spawn")
    acquired = ctx.Event()
    child = ctx.Process(target=_hold_pool_sh_forever, args=(str(tmp_path), KEY, acquired))
    child.start()
    assert acquired.wait(15)
    child.kill()
    child.join(15)

    gate = CrossProcessGate(tmp_path, KEY)
    done = threading.Event()

    def p1():
        gate.enter(Priority.CONTENTION_RETRY)
        done.set()

    threading.Thread(target=p1, daemon=True).start()
    assert done.wait(15)  # the dead child's flock vanished with its fds
    gate.exit(Priority.CONTENTION_RETRY)


def test_different_server_keys_do_not_couple(tmp_path):
    ctx = mp.get_context("spawn")
    acquired, release = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_hold_gate_ex_raw, args=(str(tmp_path), "keyaaaa", acquired, release))
    child.start()
    try:
        assert acquired.wait(15)
        gate_b = CrossProcessGate(tmp_path, "keybbbb")
        entered = threading.Event()

        def normal():
            gate_b.enter(Priority.NORMAL)
            entered.set()

        threading.Thread(target=normal, daemon=True).start()
        assert entered.wait(5)  # different key: no coupling
        gate_b.exit(Priority.NORMAL)
    finally:
        release.set()
        child.join(15)


# ---- async path ----

def test_aenter_roundtrip(tmp_path):
    async def main():
        gate = CrossProcessGate(tmp_path, KEY)
        await gate.aenter(Priority.NORMAL)
        gate.exit(Priority.NORMAL)
        await gate.aenter(Priority.CONTENTION_RETRY)
        gate.exit(Priority.CONTENTION_RETRY)

    asyncio.run(main())


def test_aenter_p1_waits_for_share(tmp_path):
    async def main():
        gate = CrossProcessGate(tmp_path, KEY)
        gate.enter(Priority.NORMAL)
        task = asyncio.create_task(gate.aenter(Priority.CONTENTION_RETRY))
        await asyncio.sleep(0.2)
        assert not task.done()  # polling: pool SH still held
        gate.exit(Priority.NORMAL)
        await asyncio.wait_for(task, 5)
        gate.exit(Priority.CONTENTION_RETRY)

    asyncio.run(main())


def test_aenter_cancellation_releases_partial(tmp_path):
    async def main():
        # Hold gate EX via a raw fd (flock conflicts across fds even within
        # one process), so aenter(NORMAL) blocks polling at the doorway.
        raw_fd = os.open(tmp_path / f"{KEY}.gate", os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(raw_fd, fcntl.LOCK_EX)
        gate = CrossProcessGate(tmp_path, KEY)
        task = asyncio.create_task(gate.aenter(Priority.NORMAL))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert gate._pool_fds == []
        assert gate._p1_fds is None
        fcntl.flock(raw_fd, fcntl.LOCK_UN)
        os.close(raw_fd)
        # Gate still fully usable after the cancellation
        await gate.aenter(Priority.CONTENTION_RETRY)
        gate.exit(Priority.CONTENTION_RETRY)

    asyncio.run(main())
