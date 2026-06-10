# RequestScheduler priority aging (fairness)

Date: 2026-06-10
Status: approved

## Problem

The priority scheduler dispatches strictly by level (p1 > p2 > ... > p5, FIFO within a
level). A sustained stream of HIGH (p2) requests therefore starves NORMAL/LOW/BACKGROUND
waiters forever. (p1-vs-everyone starvation is structurally bounded after commit 567e90f:
p1 retries only spawn from already-admitted requests, at most max_concurrent per
mass-kill wave, and no new requests are admitted while p1s wait - so it is explicitly
out of scope here.)

## Design: virtual-key aging

Each waiter gets a sort key computed once at enqueue: `(band, vkey, seq)`.

- `band` is 0 for p1 (CONTENTION_RETRY), 1 for everything else. p1 always sorts first;
  its admission rules (solo execution, waiting-p1 blocks lower admissions) are untouched.
- For p2-p5: `vkey = priority * aging_interval + enqueue_monotonic_time`. One interval
  of waiting equals one priority level: a p4 enqueued at t=0 (vkey = 4I) ranks equal to
  a p2 arriving at t=2I (vkey = 2I + 2I) and beats anything arriving later.
- With `aging_interval=None`: `vkey = priority`, i.e. exactly the previous
  `(priority, seq)` strict-priority behavior.
- `seq` keeps FIFO determinism for equal keys.

Because every waiter ages at the same rate, relative order between any two waiters is
fixed at enqueue time - the sort key is static. The existing min-heaps and
`_dispatch_next` logic work unchanged; only waiter construction and `__lt__` change.

Aging never crosses the band boundary: an aged p4 can outrank a fresh p2 in the queue,
but it is admitted as a normal request (`_active_p1` untouched, no solo semantics), and
no amount of aging outranks a waiting p1.

## API

- `RequestScheduler(max_concurrent=8, aging_interval=30.0)`; `None` disables aging.
- `LMHandler(..., scheduler_aging_interval=30.0)` and `RLM(..., scheduler_aging_interval=30.0)`
  pass-throughs (users only construct schedulers via `RLM(scheduler_max_concurrent=...)`).

Default 30s: a LOW request reaches parity with a fresh-HIGH stream after ~60s,
BACKGROUND after ~90s - matched to typical 5-120s LLM call durations.

## Rejected alternatives

- Dispatch-count aging: needs heap rebuilds; "fairness per dispatch" is a poor fit when
  call durations vary 5-120s.
- Weighted fair queuing: strongest guarantees but far more machinery, and redefines
  "high priority" from "goes first" to "gets a bigger share".

## Testing

- Aged LOW overtakes fresh HIGH (tiny interval).
- `aging_interval=None` keeps pure priority regardless of wait.
- p1 enqueued after a long-aged LOW still dispatches first (band).
- Aged non-p1 admission does not set `_active_p1`.
- Pass-through: `RLM`/`LMHandler` deliver `scheduler_aging_interval` to the scheduler.
- Existing tests (HIGH-beats-LOW contemporaneous, FIFO within level, p1 solo/serialize/
  no-barge) stay green with the 30s default.
