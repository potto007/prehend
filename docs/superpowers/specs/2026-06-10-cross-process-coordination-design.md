# Cross-Process Scheduler Coordination Design

**Date:** 2026-06-10
**Status:** Approved
**Related:** docs/superpowers/specs/2026-06-10-scheduler-aging-design.md, commits f3ed024 (p1 solo retry), 9f84fb6 (aging)

## Problem

`RequestScheduler` enforces p1 (contention-retry) solo execution with in-process
state only. In multi-process topologies (the rlm-trainer benchmark runs 8 isolated
worker processes against one llama-server with `--kv-unified`), a p1 retry in one
process still collides with other processes' in-flight requests and can fail again.
At benchmark concurrency 8, 382 server contention events were absorbed by retry
layers but 14 surfaced as task failures, all of them cross-process collisions.

Target: surfaced context-500 failures at benchmark concurrency 8 drop to ~0.

## Approach

Two-flock readers-writer locking with writer preference ("gate + pool"), chosen over
a single flock plus marker file (hand-rolled marker lifecycle, stale-pid cleanup,
polling) and over a broker process (a daemon to supervise; far more machinery than
the problem needs). The two-flock pattern gets writer preference, queueing, and
crash cleanup from the kernel.

Assumption: all coordinating processes run on the same host (true whenever
llama-server is on localhost). Cross-machine coordination is out of scope.

## Architecture

One new module, `lm_repl/clients/coordination.py`, containing `CrossProcessGate`.
`RequestScheduler` keeps owning all in-process policy (priorities, aging, local solo
rules) and calls the gate at its admission/release boundaries when one is
configured. Nothing else in the codebase learns about file locks.

- `CrossProcessGate(coordination_dir, server_key)` manages two lock files:
  `<dir>/<server_key>.gate` and `<dir>/<server_key>.pool`, where
  `server_key = sha256(base_url.encode()).hexdigest()[:16]`. Processes coordinate if and only if they
  target the same server.
- API: `enter(priority)` / `exit(priority)` (blocking, sync path) and
  `aenter(priority)` / `aexit(priority)` (async path). The gate distinguishes only
  p1 vs everything else; it knows nothing about p2-p5.
- `RequestScheduler` gains `gate: CrossProcessGate | None = None`.
  Acquire: local admission (unchanged logic), then `gate.enter(priority)`.
  Release: `gate.exit(priority)`, then local release and dispatch.
  With `gate=None`, behavior is bit-for-bit current behavior.

## Lock protocol

Normal request (p2-p5):

1. `flock(gate_fd, LOCK_SH)` - blocks only while some process's p1 holds the gate.
2. `flock(pool_fd, LOCK_SH)` - shared with all other normal requests everywhere.
3. Close the gate fd immediately (the gate is held only for the doorway).
4. Run the request; on exit, close the pool fd.

p1 retry:

1. `flock(gate_fd, LOCK_EX)` - from this instant no process anywhere admits a new
   request (all block at their step 1). This is the cross-process `_waiting_p1`
   rule, and it also serializes concurrent p1s from different processes.
2. `flock(pool_fd, LOCK_EX)` - grants only when every in-flight pool-SH holder
   drains. This is the cross-process `_active == 0` rule.
3. Run solo; on exit, release pool then gate.

Every `enter()` opens fresh fds (flock ownership is per open file description), so
concurrent requests within one process do not alias each other's locks. Lock files
are opened with `O_CREAT`; the directory is created at gate construction.

Properties, each inherited from the kernel rather than implemented:

- **Writer preference without starvation.** Readers hold the gate for microseconds,
  so a p1's `LOCK_EX` on the gate wins promptly even under a steady admission
  stream; after that, new readers queue behind it.
- **Crash cleanup.** A killed process's fds close and its locks vanish. No pid
  files, no stale-lock sweeper.
- **Deadlock-free.** Every path takes locks in the same order (local slot, gate,
  pool), and the existing retry loop in `lm_repl/clients/openai.py` fully releases
  at the old priority before re-acquiring at p1, so no request waits on pool-EX
  while holding pool-SH.
- The mass-kill cascade (the server 500s all slots and every victim escalates to
  p1) degrades to N sequential solo retries globally, which is the intended
  semantics: each retry runs with the entire KV pool; a retry that still fails is
  treated as never-fits and propagates.

## Async strategy

Sync path: plain blocking flock (sync callers cannot be cancelled).

Async path: non-blocking flock (`LOCK_NB`) plus an `asyncio.sleep` poll loop at
roughly 25 ms intervals, NOT `run_in_executor` with a blocking flock. Rationale: a
cancelled task cannot interrupt a blocking flock in an executor thread; the thread
eventually acquires a lock nobody will release, recreating the slot-leak class of
bug fixed in c316f2e but cross-process, where it would freeze every coordinating
process. With polling, cancellation is trivially safe: on `CancelledError`, release
whatever was partially acquired (gate but not yet pool) and re-raise;
`aacquire`'s existing cancellation handler then rolls back the local slot.

Costs accepted: up to ~25 ms added latency per acquisition under contention (noise
against multi-second LLM requests), and loss of kernel queueing fairness between two
simultaneous p1s from different processes (p1s are rare and either order is
correct).

## Integration and plumbing

Mirrors `scheduler_max_concurrent` exactly:

- `RLM(scheduler_coordination_dir: str | Path | None = None)` - stored, passed to
  `LMHandler`.
- `LMHandler`: when constructing the scheduler with `scheduler_coordination_dir`
  set, derive `server_key` from the default client's `base_url` and attach a
  `CrossProcessGate`. `other_backend_client` shares the gate, consistent with
  already sharing the in-process scheduler.
- `rlm-trainer/benchmark.py` later gains `--scheduler-coordination-dir` next to the
  uncommitted `--scheduler-max-concurrent` flag.
- Setting `scheduler_coordination_dir` without `scheduler_max_concurrent` raises at
  construction (no scheduler, no gate).
- Opt-in by design: implicit coupling between unrelated programs on the same
  machine, and filesystem fragility (flock over network mounts, container volume
  visibility), make auto-on surprising. The flock syscall cost itself is
  negligible (~1-2 us uncontended).

## Error handling

- Construction-time fail-fast: gate creation mkdirs and opens/locks/unlocks both
  files once; an unwritable dir or a filesystem without flock support raises
  immediately with a clear message, not on request N.
- `enter()` failing mid-acquisition releases partial holds and re-raises; the
  scheduler rolls back the local admission so counters never skew.
- `exit()` never raises (close failures are logged and swallowed); it sits in
  `finally` paths.

## Testing

- `tests/test_coordination.py`: gate unit tests with `multiprocessing` children
  against a `tmp_path` dir:
  - parent p1 `enter` blocks until the child's SH hold releases;
  - while the parent holds gate-EX, a child's normal `enter` blocks;
  - child SIGKILLed mid-hold: parent acquires (crash cleanup);
  - async cancellation during the poll loop releases partial locks.
- `tests/test_scheduler.py` additions: local-admission rollback when the gate
  raises; `gate=None` behavior untouched.
- Live: extend `scripts/test_scheduler_live.py` with a multi-process contention
  mode (N OS processes, unique ~12k-token prompts, shared coordination dir).
  Unique prompts are mandatory: identical prompts let llama-server's prefix cache
  mask KV pressure. Restart the server between contention runs.
- Final: re-run the rlm-trainer benchmark at concurrency 8 with
  `--scheduler-max-concurrent 4 --scheduler-coordination-dir <dir>`. Success:
  surfaced ctx-500 count ~0; watch the timeout count and wall-clock for regression
  (a p1 drain pauses admissions machine-wide, which can stretch heavy tasks).

## Out of scope

- Cross-machine coordination.
- Cross-process p2-p5 ordering and aging (stay per-process).
- Making the gate default-on.
- Cross-process `max_concurrent` enforcement: each process still admits up to its
  own local limit. The server's slot count is the real backpressure; the gate only
  needs to fix p1 exclusivity.
