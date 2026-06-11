# Deadline-Aware Scheduling: Proposal

**Date:** 2026-06-10
**Status:** PROPOSED - not scheduled, not approved. Written to capture design
thinking after the 2026-06-10 tuning sweep so it is not relitigated later.
**Related:** docs/superpowers/specs/2026-06-10-cross-process-coordination-design.md
(shipped), docs/superpowers/specs/2026-06-10-scheduler-aging-design.md (shipped)

## Motivation: what the tuning sweep proved

The cross-process gate eliminated its target failure (surfaced ctx-500 errors:
14 -> 0 in every gated run). The residual failure mode is per-task timeouts on
heavy task families (agg, multihop), and the sweep isolated the cause:

- At server ctx 163840 / parallel 16 with benchmark c4 x mc4 (zero contention
  AND zero server-side queueing), agg still timed out 3 of 5. With nothing
  left to schedule around, the constraint is raw compute: an agg task needs
  somewhere between 25% and 100% of aggregate GPU throughput to finish inside
  its 600s budget, and equal-share scheduling at 4-8 concurrent tasks gives
  it less.
- Aggregate decode throughput is roughly fixed (memory-bandwidth bound), so a
  scheduler cannot create throughput; it can only redistribute it and stop
  wasting it.

The current RequestScheduler is deliberately a fairness machine: five static
bands, FIFO within band, time-based aging, p1 solo for contention retries.
Fairness is the wrong policy when heterogeneous tasks face a uniform deadline:
it spreads everyone thin instead of saving the tasks that can be saved.

The cheapest fix lives OUTSIDE lm-repl and is already shipped: rlm-trainer
benchmark.py 2a4f808 interleaves the task list by family so heavy tasks no
longer compete with each other. The enhancements below are the lm-repl-side
follow-ons, in descending value order.

## Enhancement 1: Laxity-based ordering (EDF within the existing key machinery)

**Idea.** Requests carry an optional deadline inherited from their parent task.
Within the p2-p5 band, waiters sort by laxity (time remaining before the
deadline minus estimated remaining work) instead of static priority + aging. A
request whose parent is at 550s of a 600s budget outranks one whose parent is
at 30s.

**Why it fits the existing design.** The scheduler already encodes ordering as
a static (band, vkey) sort key; aging is vkey = priority * aging_interval +
enqueue_time. Laxity is just a different vkey formula: vkey = deadline -
enqueue_time (absolute deadline works as a static key; true laxity needs an
estimate of remaining work, see open questions). Band 0 (p1) remains above
everything; deadline ordering never preempts contention recovery.

**API sketch.** `acquire(priority, deadline: float | None = None)` (monotonic
timestamp); RLM sets deadline = completion start + max_timeout and passes it
through LMHandler request metadata to every subcall. Requests without a
deadline sort as deadline = +inf (today's behavior among themselves).

**Limits.** Degenerates to FIFO when all concurrent tasks have the same
deadline pressure (e.g. a benchmark front-loading one family) - which is why
task interleaving ships first. Pays off only with a heterogeneous mix.

## Enhancement 2: Deadline propagation and doomed-work cancellation

**Idea.** When a task's deadline has passed, stop spending GPU on it. Today a
doomed task keeps consuming up to max_concurrent slots until the benchmark's
hard kill (timeout + 60s), throwing away up to ~660s x N-slots of work that
live tasks could have used.

**Mechanism.**
- RLM records its absolute deadline at completion() start (it already tracks
  max_timeout; this makes it a propagated value rather than a local check).
- Subcall paths (LMHandler request handling, llm_query_batched workers) check
  the deadline before dispatch: past-deadline requests fail fast with the
  existing timeout exception instead of entering the scheduler.
- In-flight requests: cancel the asyncio task / abandon the sync wait when the
  deadline passes. The cancellation-safety work shipped with the gate (poll-
  loop aenter, c316f2e slot rollback) already makes cancellation correct at
  the scheduler layer; what is missing is purely the RLM-side plumbing that
  triggers it.
- Server side: closing the HTTP connection is the only abort signal llama
  server gets; detached generations drain on their own (observed post-run).

**Expected effect.** Bounded: reclaims wasted tail work; does not save the
doomed task. Mainly improves neighbors' completion times near the deadline
cliff.

## Enhancement 3 (deferred): Cross-process weighted shares

A global per-task fair-share (weighted-fair queueing over the coordination
dir, or a shared token bucket) would prevent a wide-fanout task in one process
from crowding a narrow task in another. Nothing in the sweep data implicates
this as a current failure mode. Recorded for completeness; do not build
without evidence.

## Non-goals

- Preempting in-flight server requests (llama-server has no preemption API).
- Replacing the p1/gate machinery - it is correctness logic and stays as-is.
- Estimating per-request token counts client-side for admission control (the
  server's static 400 check plus the p1 retry already handle never-fits).

## Open questions (answer before implementation)

1. Laxity needs "estimated remaining work" to beat plain EDF. Is a usable
   estimate available (e.g. prompt length as a proxy), or do we ship plain
   EDF (deadline-only) first?
2. Static sort keys keep the heaps rebalance-free (the aging trick). Absolute
   deadline is static; true laxity is not (estimates change). Is plain EDF's
   static key worth the simplicity win? (Likely yes for v1.)
3. How does deadline ordering interact with aging? Proposal: deadline replaces
   aging within p2-p5 when present; aging remains the fallback for
   deadline-less traffic. Mixed queues need a defined comparison.
4. Does deadline propagation cross the LMHandler socket protocol (new request
   field) or ride in client state? Socket field is cleaner for environment
   subprocesses.
5. Benchmark evidence bar: re-run the c8 sweep with interleaving alone first;
   only build Enhancement 1/2 if timeouts remain material after interleaving.

## Revisit trigger

Pick this up if, after the interleaved-task benchmark run, heavy-family
timeouts remain >> the c1 baseline failure rate, and the timeout histogram
shows tasks dying with most of their work done (cancellation value) or light
tasks finishing with large slack while heavy ones die (EDF value).
