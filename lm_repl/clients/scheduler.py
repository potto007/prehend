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
    - While any p1 request is active, ONLY p1 requests may start.
    - Otherwise requests dispatch by priority (lower number first), FIFO
      within the same level.
"""

import asyncio
import heapq
import threading
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
    __slots__ = ("priority", "seq", "event")

    def __init__(self, priority: int, seq: int):
        self.priority = priority
        self.seq = seq
        self.event = threading.Event()

    def __lt__(self, other: "_Waiter") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.seq < other.seq


class _AsyncWaiter:
    __slots__ = ("priority", "seq", "event", "loop")

    def __init__(self, priority: int, seq: int, loop: asyncio.AbstractEventLoop):
        self.priority = priority
        self.seq = seq
        self.loop = loop
        self.event = asyncio.Event()

    def __lt__(self, other: "_AsyncWaiter") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.seq < other.seq


class RequestScheduler:
    """Thread-safe priority scheduler with both sync and async acquire/release."""

    def __init__(self, max_concurrent: int = 8):
        self._max_concurrent = max_concurrent
        self._lock = threading.Lock()
        self._active = 0
        self._active_p1 = 0
        self._seq = 0
        self._sync_waiters: list[_Waiter] = []
        self._async_waiters: list[_AsyncWaiter] = []

    def _can_dispatch(self, priority: int) -> bool:
        if self._active >= self._max_concurrent:
            return False
        if self._active_p1 > 0 and priority != Priority.CONTENTION_RETRY:
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

            if not self._can_dispatch(candidate.priority):
                break

            if candidate is top_sync:
                heapq.heappop(self._sync_waiters)
            else:
                heapq.heappop(self._async_waiters)

            self._admit(candidate.priority)

            if isinstance(candidate, _AsyncWaiter):
                candidate.loop.call_soon_threadsafe(candidate.event.set)
            else:
                candidate.event.set()

    # -- sync interface --

    def acquire(self, priority: int = Priority.NORMAL) -> None:
        with self._lock:
            if self._can_dispatch(priority):
                self._admit(priority)
                return
            self._seq += 1
            waiter = _Waiter(priority, self._seq)
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
            waiter = _AsyncWaiter(priority, self._seq, loop)
            heapq.heappush(self._async_waiters, waiter)

        await waiter.event.wait()

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
