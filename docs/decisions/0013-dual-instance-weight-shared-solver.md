---
status: "accepted"
date: "2026-06-23"
deciders: "potto"
---

# Dual-instance weight-shared solver: split orchestrator and sub-calls onto two processes sharing one weights copy

## Context and Problem Statement

ADR-0012 bounded the SUM of a task's concurrent sub-calls against the single
`--kv-unified` KV pool by budgeting each at `pool // slots` (98304 -> 24576 at
slots=4). That stopped the collective pool exhaustion, but the root cause it
worked around remains: the orchestrator and its sub-calls share ONE KV pool on
ONE server (`ctx-size 98304 / parallel 4`). Two consequences persist.

1. The orchestrator and sub-calls have OPPOSITE KV access patterns: the
   orchestrator is one long-lived big-context slot (CoT on, the ~20K-token
   REDUCE, a stable reused system-prompt prefix); the sub-calls are many
   short-lived bursty slots (CoT off, mechanical). Forcing both into one
   unified pool means the bursty sub-calls contend with and can evict the
   orchestrator's stable prefix - and the per-call budget must stay small
   (24576) precisely because the orchestrator shares the pool.
2. Prefix-cache reuse of the long orchestrator system prompt is our ~5-10x win
   (cache-reuse + swa-full). Sub-call bursts sharing the pool can evict that
   prefix, forcing re-prefill.

The only reason both roles share one server is VRAM: two un-shared llama-server
processes would each allocate the full v13 weights copy (Q4_0, ~6.5 GB), so a
two-process split costs ~13 GB of duplicated weights and crowds out the KV on a
32 GB card. `cuda-llm-weight-share.so` (LD_PRELOAD, CUDA IPC) removes that cost:
it intercepts the weights `cudaMalloc` so a worker process MAPS the master's
copy while keeping its own private KV, scratch, and CUDA-graph buffers.

## Decision Drivers

- Give each role a PRIVATE KV pool so sub-call bursts cannot contend with or
  evict the orchestrator (eliminate the ADR-0012 contention at the source, not
  just bound it).
- Keep each role's prompt prefix warm in its own process.
- Pay for only ONE weights copy in VRAM (the enabler).
- Backward compatibility: single-server / OpenRouter / vLLM callers must be
  unaffected when no worker endpoint is configured.
- Reuse the existing second-endpoint precedent (`MemoryConfig.reflect_url`) and
  the Harness resolution seam (ADR-0008); no new env hacks in core.

## Considered Options

1. **Two weight-shared processes** (chosen): orchestrator master (:8080, one big
   slot, CoT on) and sub-call worker (:8081, parallel N, CoT off) sharing one
   weights copy via `cuda-llm-weight-share.so`. The Harness routes sub-calls to
   the worker and derives the sub-call budget + fan-out from the WORKER runtime.
2. **Keep one server, only tune ADR-0012**: no contention elimination, no
   prefix isolation; the per-call budget stays squeezed by the shared pool.
3. **Two un-shared processes**: duplicates ~6.5 GB of weights, crowding KV on a
   32 GB card; the split it buys is the same as option 1 but without the VRAM
   headroom to size either pool well.
4. **Disable `--kv-unified`** so `n_ctx` hard-partitions per slot: still one
   process, still one shared prefix cache subject to eviction; does not separate
   the two access patterns.

## Decision Outcome

Chosen: **option 1**. Serving runs two plain llama-server processes (NOT router
mode) with `LD_PRELOAD=cuda-llm-weight-share.so`, `MODEL_SIZE` pinned to the
recon'd v13 weights allocation, and a shared `CUDA_VRAM_IPC_NAME`. The
orchestrator is the master; the worker maps its weights. Each carries its own
`ctx-size`/`parallel` plus the proven knobs (flash-attn, q4_0 KV, swa-full,
cache-reuse, jinja, temp 0).

In prehend, `Harness.__init__` gains `subcall_base_url` and `subcall_runtime`.
When `subcall_base_url` is set: the sub-call backend (`other_backend_kwargs[0]`)
points at the worker while the orchestrator backend keeps `base_url`; the
sub-call budget is `per_call_subcall_budget(resolve_subcall_limit(model,
explicit, runtime_ctx=subcall_runtime.ctx), subcall_runtime.slots)` and
`max_concurrent_subcalls = subcall_runtime.slots`. When `subcall_base_url` is
`None`, `self.subcall_runtime is self.runtime` and the path is byte-identical to
ADR-0012's single-server behavior. This extends ADR-0012: the pool-division is
now per the WORKER's dedicated pool and slots, and the orchestrator's REDUCE /
iterative context draws from its OWN pool with no division.

### Consequences

- Good: the two access patterns no longer share a pool, so the kv-unified
  contention is eliminated at the source; the orchestrator prefix cache is never
  evicted by sub-call bursts; only one weights copy occupies VRAM, freeing
  headroom to size each pool independently (orchestrator one big slot; worker
  pool divisible among only sub-call slots, raisable toward 98304).
- Bad / risks: (1) the reused prefix is now cached ONCE PER PROCESS (N x prefix
  KV, N cold warm-ups) instead of once in a shared pool - acceptable given the
  per-pool isolation, but it is extra KV. (2) Lifecycle: if the master frees the
  shared weights the worker's IPC mapping dangles, so `sleep-idle-seconds` MUST
  be OFF on the pair (or `CUDA_VRAM_IPC_SUPPRESS_MASTER_FREE=1`), and startup
  MUST be master-first or the worker self-elects master and allocates a second
  copy. (3) The pair runs OUTSIDE the router, so the router (kept on its own port
  for the 26B/19B training+benchmark models) must not load a large model
  concurrently with a live solve - an ops constraint, not a code one.
  (4) VRAM sizing is empirical: validate the chosen worker `ctx-size` under a
  SUSTAINED run, not a burst.

## More Information

- Extends [ADR-0012](0012-pool-aware-subcall-budget-under-kv-unified.md)
  (budget is now per-instance, not per-shared-pool); resolved at the
  [ADR-0008](0008-high-level-harness-api.md) Harness seam; sub-call guard
  contract from [ADR-0009](0009-subcall-input-context-guard.md) /
  [ADR-0010](0010-auto-chunk-enforcement-for-oversized-subcalls.md) unchanged.
- Library: `~/src/cuda-llm-weight-share` (LD_PRELOAD CUDA IPC weight sharing).
- Design + plan: `docs/superpowers/specs/2026-06-23-dual-instance-weight-shared-solver-design.md`,
  `docs/superpowers/plans/2026-06-23-dual-instance-weight-shared-solver.md`.
