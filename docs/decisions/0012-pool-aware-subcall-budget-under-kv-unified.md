---
status: "accepted"
date: "2026-06-22"
deciders: "potto"
---

# Pool-aware sub-call budget: divide the shared kv-unified pool across concurrent sub-calls

## Context and Problem Statement

ADR-0009 added a per-prompt input-size guard and ADR-0010 added auto-chunk map-reduce, both
bounding a SINGLE sub-call's input against `subcall_context_limit` (the served model's
context window). The eval passes `--subcall-context-limit 98304` to match the v13 model
(`gemma-4-12b-it-sft-kb-v13-sft`, served at `n_ctx = 98304`). Despite both guards, long
plain-multihop solves kept dying with task-level `500 Context size has been exceeded` and
600s timeouts that CONFOUNDED every memory A/B (cold/warm regressions on `002/008/011`,
`010` overflowing at 70s). The handoff hypothesis was "the orchestrator's own transcript
grows past the 98304 window."

The served-model log disproves that single-transcript theory. The server runs with
`--kv-unified` (`n_parallel = 4 and kv_unified = true`, `n_ctx = 98304`): under kv-unified
the 98304-token KV cache is ONE pool SHARED across all concurrent sequences, not a private
window per slot. The actual failure signature is `decode: failed to find a memory slot for
batch ... failed to find free space in the KV cache` (progressive batch shrink
256 -> 128 -> ... -> 16) then `Context size has been exceeded` returned to FOUR task ids at
once - i.e. collective pool exhaustion, not one sequence exceeding the window. With
`run-A3.sh` using `concurrency=1`, those four concurrent requests can only be ONE task's
map-reduce fan-out: `local_repl.py` runs the batched sub-calls in a
`ThreadPoolExecutor(max_workers=min(max_concurrent_subcalls, len(prompts)))` =
`min(slots, ...)`. So a single task issues up to `slots` (4) concurrent sub-calls, the guard
budgets EACH at the whole 98304 pool, and their SUM exhausts the shared cache. Memory
injection makes it worse because the injected block adds tokens to every concurrent call,
crossing the ceiling sooner. This also explains why ADR-0010 "closed the overflow in
isolation but not under load": in isolation one call fits 98304; concurrently the sum does
not. The root cause is a **budget-accounting bug**, not transcript growth.

## Decision Drivers

- The guard must bound the SUM of concurrent sub-calls against the shared pool, not each
  call independently against the whole pool.
- Preserve the single-big-REDUCE benefit kv-unified buys (a ~20K-token reduce over the whole
  pool must still fit): `98304 / 4 = 24576 >= 20K`.
- Keep the guard PURE and per-call (ADR-0009/0010 contract): the pool->per-call conversion
  belongs at the seam that knows the concurrency, not inside the guard.
- No serving change; no throttling of fan-out parallelism; robust when the runtime probe is
  ambiguous (server down / router reports `n_ctx = 0`).

## Considered Options

1. **Pool-aware client guard** (chosen): the effective per-call guard budget is
   `pool // slots`, computed once at the `Harness` resolution seam, where `pool` is the
   resolved shared limit (server `n_ctx`) and `slots = runtime.slots` is the concurrent
   sub-call count. The guard's existing 15% margin then reserves headroom for the resident
   root transcript.
2. **Cap fan-out concurrency** (+ scheduler) so `in_flight * per_call <= pool`. Simpler
   arithmetic but throttles parallelism (slower) and leaves the per-call over-budget intact.
3. **Disable `--kv-unified`** so `n_ctx` hard-partitions to `n_ctx / parallel = 24576` per
   slot. local-ai serving change; cost = a single reduce > 24576 tokens no longer fits.

## Decision Outcome

Chosen: **option 1**. Add a pure helper `token_utils.per_call_subcall_budget(pool, slots)`
returning `max(1, pool // max(1, slots))` (and `None` when `pool` is `None`, so a disabled
guard stays disabled). `Harness.__init__` resolves the shared pool via
`resolve_subcall_limit(...)` as before, then threads `per_call_subcall_budget(pool, slots)`
into the SRLM/RLM guard, the LocalREPL `environment_kwargs`, and the prompt's chunk-budget
wording. `slots` is the same value already used for `max_concurrent_subcalls`, so the divisor
provably equals the ThreadPoolExecutor fan-out width.

This holds for BOTH serving modes when the operator passes the server's total `n_ctx` as the
limit: kv-unified shares the pool so `pool / parallel` is the safe per-concurrent-call share;
non-unified already hard-partitions to `n_ctx / parallel`, which `pool / slots` reproduces.

### Consequences

- Good: `slots` concurrent sub-calls now sum to <= `0.85 * pool` (with the guard margin),
  leaving ~15% of the pool for the still-resident root orchestrator transcript; the
  collective KV-pool exhaustion is bounded out. The single big reduce (~20K) still fits
  `24576`, and `recommended_chunk_chars` (30% of the per-call budget) now targets ~7.4K-token
  chunks, so a 4-way map is ~29K << pool. Pure, no serving change, robust when the probe is
  ambiguous (explicit pool + fallback `slots = 4`).
- Bad / risks: (1) the per-call budget DROPS (98304 -> 24576 at slots=4), so a model that
  previously sent one near-pool call now gets reject-with-hint / auto-chunk - intended, but
  re-validate adoption. (2) This bounds the SUM of concurrent SUB-CALLS; it does NOT compact
  the root transcript itself. If the root prefix alone exceeds its ~15% reserve on very long
  multi-iteration solves, transcript compaction/trim remains a separate lever (deferred, see
  `project_orchestrator-transcript-overflow-followup`). (3) If a server runs kv-unified with
  `parallel` much larger than real concurrency, the divisor is conservative (smaller per-call
  budget than strictly necessary) - acceptable; correctness over tightness.

## More Information

- Evidence: served-model log `/tmp/llama-server.log` (PID 35953): `kv_unified = true`,
  `n_ctx = 98304`, "failed to find free space in the KV cache" bursts of 4 task ids under
  `run-A3.sh` `concurrency=1`.
- Builds on the per-call guard [ADR-0009](0009-subcall-input-context-guard.md) and auto-chunk
  [ADR-0010](0010-auto-chunk-enforcement-for-oversized-subcalls.md); resolved at the
  [ADR-0008](0008-high-level-harness-api.md) Harness seam.
- Stale doc to fix (out of scope here): CLAUDE.md "sft 32768/1" - the v13 sft section is
  actually `ctx-size 98304 / parallel 4`.
