---
status: "accepted"
date: "2026-06-23"
deciders: "potto"
supersedes: "0013"
---

# Single-process dual-context solver: share weights via one llama_model backing two llama_context (supersedes the WSL2-impossible CUDA-IPC weight-share)

## Context and Problem Statement

[ADR-0013](0013-dual-instance-weight-shared-solver.md) split the solver into an
orchestrator and a sub-call worker as TWO OS processes sharing one VRAM weights
copy via CUDA IPC (`cuda-llm-weight-share`, LD_PRELOAD). The goal stands - give
each role a PRIVATE KV pool and prefix cache so sub-call bursts cannot contend
with or evict the orchestrator (kill the ADR-0012 `--kv-unified` shared-pool
contention) - but the IPC mechanism does NOT work on this WSL2 host.

Empirically settled 2026-06-23 on the RTX 5090 (32GB), CUDA 13, v13 model:

- **Legacy CUDA IPC is dead on WSL2.** Under CUDA 13 `cudaIpcGetMemHandle` now
  succeeds (the lib's original `code=2` was an older toolkit), but cross-process
  `cudaIpcOpenMemHandle` returns `400 (invalid resource handle)`. No flag fixes
  it; NVIDIA's CUDA-on-WSL stack does not implement IPC memory handles.
- **CUDA VMM POSIX-fd sharing (`cuMemCreate` + `cuMemExportToShareableHandle` ->
  SCM_RIGHTS -> `cuMemImportFromShareableHandle`) DOES work cross-process** on
  WSL2 - a viable two-process fallback, but more complex than needed.
- **One `llama_model` can back multiple `llama_context`, each with a PRIVATE KV
  cache, intra-process.** Proven (`wsl-experiments/multi_ctx_proof.c`): model
  load +7.1GB, ctxA(32k/1) KV +6.3GB, ctxB(8k/4) KV only +1.96GB (NOT a second
  weights copy); identical model pointer, distinct KV handles, independent
  decode. No IPC at all, so the WSL2 limitation never applies.

## Decision Drivers

- Eliminate the kv-unified contention at the source (private KV per role) and
  keep each role's prefix cache warm - same goals as ADR-0013.
- Pay for ONE weights copy in VRAM.
- Work on WSL2 (rules out CUDA IPC).
- Keep the prehend Harness `subcall_base_url` work (ADR-0013) unchanged.
- Reuse the server's existing slot scheduler + prompt cache rather than
  reimplement prefix reuse.

## Considered Options

1. **Single-process, two `server_context` sharing one `llama_model`** (chosen):
   a custom binary loads the model once and runs two `server_context`
   (orchestrator + worker), each with its own KV + prompt cache, on two ports.
2. Two-process + CUDA VMM POSIX-fd weight sharing: works on WSL2 but needs a
   rewritten LD_PRELOAD lib and cross-process fd plumbing; strictly more complex.
3. Two-process + legacy CUDA IPC (the ADR-0013 design): inoperable on WSL2.
4. Two un-shared processes (2x weights): ~13.2GB of weights crowds KV on 32GB.
5. Status quo: one server + ADR-0012 budget - bounds contention, does not
   eliminate it, and the shared prefix cache is still evictable.

## Decision Outcome

Chosen: **option 1**. Build `llama-dual-context-server` in the fork
`potto007/llama.cpp` (the `diffusion-gemma-server` example already establishes
the custom-OpenAI-server-target pattern). Refactor `server_context::load_model`
to accept an optional borrowed `llama_model*` (skip the load, do not free it in
teardown), so two `server_context` can be constructed against one shared model.
A new `main` loads the model once, constructs the orchestrator (`:8080`, big
ctx) and worker (`:8081`, parallel small slots) contexts, wires each to its own
`server_http_context`, and runs both `start_loop()` on threads.

Upstream `llama-server` cannot do this: single-model mode = one context; router
mode forks a SEPARATE child process per model entry
(`tools/server/server-models.cpp`) = weights loaded twice. Hence the custom
binary.

The prehend Harness is unchanged: orchestrator at `base_url` (`:8080`),
sub-calls at `subcall_base_url` (`:8081`). CoT stays per-request (the worker
backend sets `enable_thinking=false`), not a server-level difference.

### Consequences

- Good: one weights copy; each role has a private KV pool and prompt cache so
  the kv-unified contention is gone at the source and the orchestrator prefix is
  never evicted by sub-call bursts; reuses the server's slot scheduler + prompt
  cache (no reimplementation); works on WSL2.
- Bad / risks: (1) a custom binary tracks llama.cpp server internals
  (`server_context`, `server-http`) which are not a stable public API - upstream
  bumps may need rework; the shared-model refactor lives in our fork.
  Specifically, the OpenAI route table in `dual-context-server.cpp` is copied
  verbatim from `server.cpp`'s `llama_server()` (and the file-static
  `ex_wrapper`), so upstream route additions must be mirrored by hand - a
  known drift point flagged at build time. (2) Model
  lifetime: the borrowed model must outlive both contexts and be freed exactly
  once by `main`, not by either `server_context`. (3) VRAM sizing is empirical:
  at orchestrator ctx 98304, orchestrator KV + worker KV + one weights copy must
  fit 32GB - validate under a SUSTAINED run (CLAUDE.md). (4) The
  `cuda-llm-weight-share` LD_PRELOAD lib and the two-process systemd units from
  ADR-0013 are retired for this path; the VMM-fd result is kept on record as a
  fallback for a future multi-host design.

## More Information

- Supersedes [ADR-0013](0013-dual-instance-weight-shared-solver.md); builds on
  [ADR-0012](0012-pool-aware-subcall-budget-under-kv-unified.md) (the per-call
  budget guard stays as defense-in-depth) and the
  [ADR-0008](0008-high-level-harness-api.md) Harness seam.
- Evidence + prototypes: `~/src/cuda-llm-weight-share/wsl-experiments/`
  (`multi_ctx_proof.c`, `vmm_cross.cu`, `ipc_cross.cu`).
- Plan: `docs/superpowers/plans/2026-06-23-dual-context-server.md`.
