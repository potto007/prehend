"""Priority-aware request scheduler for llama-server with unified KV.

Manages a priority queue of pending LLM requests so that context-contention
retries (p1) preempt normal traffic and the shared KV pool is not overwhelmed
by simultaneous large requests.

Priority levels:
    1  CONTENTION_RETRY  - router-reserved: retries after context-size 400
    2  HIGH              - caller-set via priority="high"
    3  NORMAL            - default
    4  LOW               - caller-set via priority="low"
    5  BACKGROUND        - router-reserved

Scheduling rules:
    - At most *max_concurrent* requests run simultaneously.
    - A p1 request runs ALONE: it is admitted only when nothing else is in
      flight, and p1 requests run one at a time. A contention retry exists to
      re-run with the entire KV pool; concurrent p1s would just re-exhaust it
      (the server's pool-exhaustion 500 kills every in-flight request, so all
      victims escalate to p1 together).
    - While any p1 is active OR waiting, no other priority may start - without
      the waiting clause a steady request stream would keep the pool occupied
      and starve the retry forever.
    - Otherwise requests dispatch by priority (lower number first), FIFO
      within the same level - subject to aging: every *aging_interval* seconds
      a p2-p5 waiter spends queued is worth one priority level, so a steady
      stream of HIGH traffic cannot starve NORMAL/LOW/BACKGROUND forever.
      Aging never crosses into p1: an aged waiter is admitted as a normal
      request, and a waiting p1 outranks any aged waiter.

Aging is implemented as a static sort key (priority * aging_interval +
enqueue_time): every waiter ages at the same rate, so relative order is fixed
at enqueue time and the min-heaps never need rebalancing.
"""

import asyncio
import heapq
import threading
import time
from enum import IntEnum


class Priority(IntEnum):
    CONTENTION_RETRY = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


PRIORITY_ALIASES: dict[str | int | None, int] = {
    None: Priority.NORMAL,
    "high": Priority.HIGH,
    "low": Priority.LOW,
    "normal": Priority.NORMAL,
    1: Priority.CONTENTION_RETRY,
    2: Priority.HIGH,
    3: Priority.NORMAL,
    4: Priority.LOW,
    5: Priority.BACKGROUND,
}


def resolve_priority(value: str | int | None) -> int:
    if isinstance(value, int) and 1 <= value <= 5:
        return value
    if value in PRIORITY_ALIASES:
        return PRIORITY_ALIASES[value]
    return Priority.NORMAL


class _Waiter:
    __slots__ = ("priority", "band", "vkey", "seq", "event", "cancelled", "dispatched")

    def __init__(self, priority: int, band: int, vkey: float, seq: int):
        self.priority = priority
        self.band = band
        self.vkey = vkey
        self.seq = seq
        self.event = threading.Event()
        self.cancelled = False
        self.dispatched = False

    def __lt__(self, other: "_Waiter") -> bool:
        if self.band != other.band:
            return self.band < other.band
        if self.vkey != other.vkey:
            return self.vkey < other.vkey
        return self.seq < other.seq


class _AsyncWaiter:
    __slots__ = ("priority", "band", "vkey", "seq", "event", "loop", "cancelled", "dispatched")

    def __init__(
        self, priority: int, band: int, vkey: float, seq: int, loop: asyncio.AbstractEventLoop
    ):
        self.priority = priority
        self.band = band
        self.vkey = vkey
        self.seq = seq
        self.loop = loop
        self.event = asyncio.Event()
        self.cancelled = False
        self.dispatched = False

    def __lt__(self, other: "_AsyncWaiter") -> bool:
        if self.band != other.band:
            return self.band < other.band
        if self.vkey != other.vkey:
            return self.vkey < other.vkey
        return self.seq < other.seq


class RequestScheduler:
    """Thread-safe priority scheduler with both sync and async acquire/release."""

    def __init__(self, max_concurrent: int = 8, aging_interval: float | None = 30.0):
        """
        Args:
            max_concurrent: Most requests allowed in flight at once. Match the
                server's slot count (llama-server --parallel).
            aging_interval: Seconds of queue wait worth one priority level for
                p2-p5 waiters (an old LOW eventually outranks a fresh HIGH).
                None disables aging (strict priority, FIFO within a level).
        """
        self._max_concurrent = max_concurrent
        self._aging_interval = aging_interval
        self._lock = threading.Lock()
        self._active = 0
        self._active_p1 = 0
        self._waiting_p1 = 0
        self._seq = 0
        self._sync_waiters: list[_Waiter] = []
        self._async_waiters: list[_AsyncWaiter] = []

    def _sort_fields(self, priority: int) -> tuple[int, float]:
        """(band, vkey) for a waiter enqueued now. Band 0 (p1) sorts before
        band 1 (p2-p5) unconditionally - aging never reaches p1. Within band 1
        the static vkey encodes aging; with aging disabled it is the bare
        priority, restoring strict (priority, seq) order."""
        if priority == Priority.CONTENTION_RETRY:
            return 0, float(priority)
        if self._aging_interval is not None:
            return 1, priority * self._aging_interval + time.monotonic()
        return 1, float(priority)

    def _can_dispatch(self, priority: int) -> bool:
        if priority == Priority.CONTENTION_RETRY:
            # Solo execution: the retry exists to run with the entire KV pool,
            # and that also serializes p1s (an active p1 keeps _active > 0).
            return self._active == 0
        if self._active >= self._max_concurrent:
            return False
        if self._active_p1 > 0 or self._waiting_p1 > 0:
            return False
        return True

    def _admit(self, priority: int) -> None:
        self._active += 1
        if priority == Priority.CONTENTION_RETRY:
            self._active_p1 += 1

    def _dispatch_next(self) -> None:
        while self._sync_waiters or self._async_waiters:
            top_sync = self._sync_waiters[0] if self._sync_waiters else None
            top_async = self._async_waiters[0] if self._async_waiters else None

            if top_sync and top_async:
                candidate = top_sync if top_sync < top_async else top_async
            else:
                candidate = top_sync or top_async

            if candidate.cancelled:
                # The waiting task was cancelled in aacquire; its counters were
                # already adjusted there. Just drop it and keep dispatching.
                if candidate is top_sync:
                    heapq.heappop(self._sync_waiters)
                else:
                    heapq.heappop(self._async_waiters)
                continue

            if not self._can_dispatch(candidate.priority):
                break

            if candidate is top_sync:
                heapq.heappop(self._sync_waiters)
            else:
                heapq.heappop(self._async_waiters)

            if candidate.priority == Priority.CONTENTION_RETRY:
                self._waiting_p1 -= 1
            self._admit(candidate.priority)

            if isinstance(candidate, _AsyncWaiter):
                candidate.dispatched = True
                try:
                    candidate.loop.call_soon_threadsafe(candidate.event.set)
                except RuntimeError:
                    # The waiter's event loop is closed, so its task can never
                    # resume (or release): take the slot back and move on.
                    self._active -= 1
                    if candidate.priority == Priority.CONTENTION_RETRY:
                        self._active_p1 -= 1
            else:
                candidate.dispatched = True
                candidate.event.set()

    # -- sync interface --

    def acquire(self, priority: int = Priority.NORMAL) -> None:
        with self._lock:
            if self._can_dispatch(priority):
                self._admit(priority)
                return
            self._seq += 1
            band, vkey = self._sort_fields(priority)
            waiter = _Waiter(priority, band, vkey, self._seq)
            if priority == Priority.CONTENTION_RETRY:
                self._waiting_p1 += 1
            heapq.heappush(self._sync_waiters, waiter)

        waiter.event.wait()

    def release(self, priority: int = Priority.NORMAL) -> None:
        with self._lock:
            self._active -= 1
            if priority == Priority.CONTENTION_RETRY:
                self._active_p1 -= 1
            self._dispatch_next()

    # -- async interface --

    async def aacquire(self, priority: int = Priority.NORMAL) -> None:
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._can_dispatch(priority):
                self._admit(priority)
                return
            self._seq += 1
            band, vkey = self._sort_fields(priority)
            waiter = _AsyncWaiter(priority, band, vkey, self._seq, loop)
            if priority == Priority.CONTENTION_RETRY:
                self._waiting_p1 += 1
            heapq.heappush(self._async_waiters, waiter)

        try:
            await waiter.event.wait()
        except asyncio.CancelledError:
            with self._lock:
                if waiter.dispatched:
                    # Dispatch already admitted us; give the slot back since
                    # this task will never run a request or release.
                    self._active -= 1
                    if priority == Priority.CONTENTION_RETRY:
                        self._active_p1 -= 1
                else:
                    # Still queued: mark for lazy removal by _dispatch_next.
                    waiter.cancelled = True
                    if priority == Priority.CONTENTION_RETRY:
                        self._waiting_p1 -= 1
                # Either branch can unblock other waiters (a freed slot, or a
                # vanished waiting-p1 that was gating admissions).
                self._dispatch_next()
            raise

    async def arelease(self, priority: int = Priority.NORMAL) -> None:
        with self._lock:
            self._active -= 1
            if priority == Priority.CONTENTION_RETRY:
                self._active_p1 -= 1
            self._dispatch_next()

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return len(self._sync_waiters) + len(self._async_waiters)
