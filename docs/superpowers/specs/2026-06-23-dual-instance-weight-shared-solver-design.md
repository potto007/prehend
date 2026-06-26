# Dual-instance weight-shared solver

Date: 2026-06-23
Status: design (pending implementation plan)
Repos touched: `prehend` (Harness API), `local-ai` (serving/ops), `cuda-llm-weight-share` (LD_PRELOAD lib, already built)

## Problem

The solver runs as a single llama-server with `ctx-size 98304 / parallel 4` under
`--kv-unified`, so `n_ctx` is **one KV pool shared across all 4 slots**. A single
task's concurrent map-reduce sub-calls plus the orchestrator's own iterative
context and the ~20K-tok REDUCE all draw from that one pool. This is the
ADR-0012 contention: budgeting each sub-call at the full pool exhausts the
shared cache (`failed to find free space in the KV cache` -> `Context size has
been exceeded` in bursts of `parallel` task ids). Separately, prefix-cache reuse
of the long orchestrator system prompt is our ~5-10x win, but bursty sub-calls
sharing the pool can evict that prefix.

The two roles have **opposite KV access patterns**: the orchestrator is one
long-lived big-context slot (CoT on, stable prefix, low concurrency); the
sub-calls are many short-lived bursty slots (CoT off). Forcing them into one
unified pool is what makes the bursts thrash the orchestrator's stable cache.

## Goals

1. **Kill the kv-unified shared-pool contention** by giving each role a private
   KV pool (ADR-0012 follow-through: budget becomes per-instance, not
   per-shared-pool).
2. **Protect the prefix cache** so each role's prompt prefix stays warm in its
   own process and is never evicted by the other role's bursts.

Achieved by running the orchestrator and sub-calls as **two persistent
llama-server processes that share one VRAM copy of the weights** via CUDA IPC
(`cuda-llm-weight-share.so`), so the split costs no duplicate weights.

## Non-goals

- Orchestrator transcript / root-prefix compaction (separate open lever).
- Changing the router for the 26B/19B training+benchmark models.
- Speculative decoding (disabled, see rlm-models.ini v13 notes).

## Architecture

Two plain llama-server processes (NOT router mode), same
`gemma-4-12B-it-sft-kb-v13-sft.Q4_0.gguf` (~6.5 GB), sharing weights via
`LD_PRELOAD=cuda-llm-weight-share.so` (README Step 2: Production mode):

| | Role | Port | KV pool (private) | CoT | parallel |
|---|---|---|---|---|---|
| **Master** | Orchestrator / REDUCE | 8080 | one big slot | ON | 1 |
| **Worker** | Map-reduce sub-calls | 8081 | many bursty slots | OFF | N |

The worker's weights `cudaMalloc` is intercepted and IPC-mapped to the master's
copy; **KV, scratch, and CUDA-graph buffers stay private per process** - exactly
the isolation we want. The existing router stays on its own port for the
26B/19B training+benchmark models (not concurrent with live solving; VRAM
coexistence is an ops note, not a design conflict).

## Serving / ops (local-ai)

**Reconnaissance once:** run the master with `LD_PRELOAD` but no `MODEL_SIZE` to
capture the exact v13 Q4_0 weights allocation in bytes; pin it as `MODEL_SIZE`
for both instances (tolerance 0; same gguf + backend -> identical size).

**Two new systemd user units** (sibling to `localai-llama-server.service`),
driven by an extended `llama-server.sh` with `start-pair`/`stop-pair`. Both export:

```
LD_PRELOAD=<repo>/cuda-llm-weight-share.so
LD_LIBRARY_PATH=/usr/local/cuda-13/lib64
CUDA_VISIBLE_DEVICES=0
MODEL_SIZE=<recon bytes>
CUDA_VRAM_IPC_NAME=/cuda_vram_ipc_v13_gpu0
CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC=3
```

Each instance launches as a plain `--model` server carrying per-role
`ctx-size`/`parallel` plus the proven global knobs: `flash-attn on`,
`cache-type-k/v q4_0`, `swa-full true`, `cache-reuse 256`, `cache-ram 4096`,
`jinja true`, `temp 0`.

Three config consequences:

1. **`sleep-idle-seconds` OFF on the pair.** Idle-unload frees the master's
   weights -> dangles the worker's IPC mapping -> crash. Use no idle-unload
   (recommended; the pair is the always-on live solver) or
   `CUDA_VRAM_IPC_SUPPRESS_MASTER_FREE=1`. Deliberate reversal of the current
   1800s setting for these two units only.
2. **Startup ordering:** master must publish its MASTER role before the worker
   starts, else the worker self-elects master and allocates a second weights
   copy. `start-pair` waits on the master's shm/health before launching worker.
3. **Stale/orphan hygiene:** before `start-pair`, assert
   `/dev/shm/cuda_vram_ipc_v13_gpu0` absent and keep the existing no-orphans guard.

Plan verification item: confirm the llama.cpp build uses `GGML_BACKEND_DL=ON`
(weight-share's lazy `dlsym` path targets that build; static-CUDA still works but
the hook firing must be verified first).

## Prehend Harness API

Extends the existing second-endpoint precedent (`reflect_url` routes
distillation to a separate server). New backward-compatible param:

```python
Harness(
    model,
    base_url,                  # orchestrator / master  -> :8080
    subcall_base_url=None,     # sub-call / worker       -> :8081 (defaults to base_url)
    ...
)
```

When `subcall_base_url` is set:
- `subcall_kwargs["base_url"] = subcall_base_url`; orchestrator backend keeps `base_url`.
- Probe both servers -> two `Runtime`s (orchestrator pool; worker `ctx` + `slots`).
- `max_concurrent_subcalls = subcall_runtime.slots` (fan-out matches the worker).
- `eff_subcall_limit = per_call_subcall_budget(subcall_ctx, subcall_slots)` - the
  division is now over the worker's dedicated pool among only sub-call slots; the
  orchestrator's REDUCE/iterative context draws from its own pool with no division.

When `subcall_base_url is None`: byte-identical to today's single-server path
(one probe, shared-pool division). `MemoryConfig.reflect_url` untouched/orthogonal.

Records as an ADR extending ADR-0012 (budget now per-instance, not
per-shared-pool); ADR-0008 stays the home for the new param.

## Sizing + VRAM (starting points, empirically validated)

Per CLAUDE.md, validate with a SUSTAINED run, not a burst. Starting points:

| Instance | ctx-size | parallel | per-call budget | Rationale |
|---|---|---|---|---|
| Orchestrator (8080) | 32768 | 1 | ~32K (whole pool) | ~20K REDUCE + CoT, long-lived slot, warm prefix |
| Sub-calls (8081) | 65536 | 4 | ~16K (pool / 4) | map leaves prefill ~12K doc text, CoT off |

VRAM envelope (32 GB): weights 6.5 (shared, once) + orchestrator KV(32768) +
worker KV(65536) at q4_0/swa-full + two compute buffers. Headroom exists to push
the worker to ctx 98304 (per-call 24.5K, now dedicated to sub-calls). Ramp:
prove the pair shares one weights copy, size the worker pool up under sustained
load until spill, back off one step. Spill alerting is host-side (local-ai ADR-0009).

## Test / validation

1. **Hook fires:** `nm -D` shows `T cudaMalloc`; worker log shows `WORKER: IPC
   handle mapped ... No duplicate weights allocation`; `nvidia-smi` shows the
   ~6.5 GB allocation ONCE (master weights+KV, worker KV only).
2. **Both endpoints serve:** `/props` on 8080 and 8081 return distinct
   `n_ctx`/`n_parallel`; Harness two-probe resolves two Runtimes.
3. **No-contention proof:** sustained multihop run that previously hit "failed
   to find free space in the KV cache" / "Context size has been exceeded" -
   confirm those vanish.
4. **Prefix-cache win:** server logs show prefix reuse firing independently on
   each instance; orchestrator prefix not evicted by sub-call bursts.
5. **Regression guard:** `subcall_base_url=None` path produces srlm_kwargs
   byte-identical to today (unit test).
6. **Lifecycle:** kill master, confirm worker holds mapping per README; confirm
   `start-pair` ordering prevents double-master.

## Open questions

- Final worker pool size (65536 vs 98304) - decided empirically in the ramp.
- Whether to also move the embed/reflect endpoints into the weight-share IPC
  group later (out of scope for v1).
