---
status: "proposed"
date: "2026-06-23"
deciders: "potto"
consulted: "research agents (vLLM, SGLang)"
---

# Inference-engine evaluation: spike vLLM and SGLang as single-engine replacements for the dual-context llama.cpp fork

## Context and Problem Statement

[ADR-0014](0014-single-process-dual-context-solver.md) built
`llama-dual-context-server` in the `potto007/llama.cpp` fork: one `llama_model`
backing two `llama_context` (orchestrator on `:8080`, worker on `:8081`), each
with a PRIVATE KV pool and prompt cache. That eliminated the
[ADR-0012](0012-pool-aware-subcall-budget-under-kv-unified.md) `--kv-unified`
shared-pool contention at the source. But the two contexts run SEPARATE
schedulers and CANNOT co-batch on the GPU - and the RLM pattern inherently
produces concurrent orchestrator+worker decode load (the orchestrator fans out
map-reduce sub-calls while it is itself decoding).

Empirically settled 2026-06-23 on the RTX 5090 (32GB), CUDA 13, v13 model
(`gemma-4-12b-it-sft-kb-v13-sft`):

- **~37% aggregate decode-throughput loss under concurrent load** (225 -> 142
  tok/s; each stream roughly halves), structural to having two contexts with
  independent schedulers that cannot share a batch.
- **Plain-multihop tasks that completed 15/15 on the prior single-endpoint
  `--kv-unified` server now TIME OUT** on the dual-context fork under that load.
- **Prefix caching itself works fine on the fork** (verified: a 17.6K-token
  prefix prefilled only ~150 new tokens, a 65x reuse). So the bottleneck is
  concurrent DECODE throughput, not caching - and it is structural to the
  two-context design, not a tuning miss.

Naming clarification (load-bearing for the whole evaluation): "gemma-4-12B-it"
and "gemma-4-E4B" are REAL Google Gemma 4 models (the family shipped ~early
2026; the 12B encoder-free "Unified" variant only ~2026-06-03), not internal
prehend shorthand. The solver is the bleeding-edge 12B unified arch
(`Gemma4UnifiedForConditionalGeneration`); the distiller `gemma-4-E4B` is a
different, second model (PLE / MatFormer lineage).

Question: should prehend keep the custom dual-context llama.cpp fork, or migrate
the served solver to a single-engine inference server (vLLM or SGLang) whose
continuous batching co-batches orchestrator and worker in ONE scheduler and
dissolves both the co-batching loss AND the per-slot KV budgeting?

## Decision Drivers

- Recover the ~37% concurrent-decode loss; make the timing-out multihop tasks
  complete again.
- Co-batch orchestrator and worker (the SAME 12B weights, two roles) in one
  scheduler - the thing two `llama_context`s structurally cannot do.
- Retire custom-fork maintenance (the fork tracks unstable llama.cpp server
  internals; ADR-0014 flags the verbatim-copied OpenAI route table as a known
  drift point).
- Drop the manual per-slot KV math (ADR-0009 `subcall_context_limit`, ADR-0012
  `per_call_subcall_budget`) in favor of a paged/dynamic KV pool.
- Keep the long shared orchestrator system-prompt prefix warm and reused across
  the fanned-out sub-calls (the RLM many-shared-prefix pattern).
- Run on the actual box: consumer RTX 5090 (Blackwell SM_120), 32GB, WSL2,
  CUDA 13 - the maturity gap is on consumer Blackwell, not datacenter Blackwell.
- Host the SECOND model (the `gemma-4-E4B` distiller) somewhere; it is offline
  and not latency-critical.
- Avoid betting the harness on a bleeding-edge stack that is nightly-only or
  currently broken for our exact 3-week-old model.

## Considered Options

1. **Keep the dual-context llama.cpp fork** (status quo, ADR-0014).
2. **Migrate the solver to vLLM** (single engine: continuous batching +
   PagedAttention + Automatic Prefix Caching + Hybrid KV Cache Manager).
3. **Migrate the solver to SGLang** (single engine: RadixAttention auto-prefix
   dedup + continuous batching, single scheduler).

## Decision Outcome

Chosen: **do NOT rip out llama.cpp yet; run a time-boxed (1-2 day) spike of BOTH
vLLM and SGLang, each gated, then decide.** This is a proposal, not an
implemented decision (hence `status: proposed`) - both single-engine options are
architecturally superior in principle but each carries a hardware/timing risk
that can only be settled empirically on this exact 5090/WSL2/CUDA-13 box.

Spike protocol (both engines), gated:

- **Gate #1 - clean load + decode on the 5090/WSL2.** Does the engine even launch
  and sustain decode on consumer Blackwell SM_120 under WSL2? This is the single
  most likely thing to sink either migration.
  - vLLM: reproduce a clean 12B-unified load on a nightly first; if shape-mismatch
    bug #44494 reproduces, vLLM is BLOCKED today.
  - SGLang: launch with `--attention-backend flashinfer` (the auto-selected
    `trtllm_mha` raises a ValueError on SM_120, #14814); build sgl-kernel from
    source if the prebuilt wheel lacks SM_120.
- **Gate #2 - sustained concurrent run + A/B.** Drive a concurrent
  orchestrator+worker RLM load (SUSTAINED, not a burst, per CLAUDE.md); confirm
  prefix-cache hit rate, no KV exhaustion under concurrent map-reduce, and that
  the plain-multihop tasks that time out on the dual-context fork now COMPLETE.
  Rerun the plain-multihop A/B against the dual-context baseline.

Direction if a spike passes: **lean SGLang** for the RLM many-shared-prefix fit
(RadixAttention auto-dedupes exactly the orchestrator-prefix + shared-sub-call
pattern; up to ~6.4x on shared-prefix workloads), with **vLLM as the fallback**
if SM_120 maturity blocks SGLang on the 5090. If the 12B-unified bug (#44494)
blocks vLLM now, re-evaluate vLLM in ~4-6 weeks when unified support stabilizes
into a stable release. A passing spike should be ratified by a FOLLOW-UP ADR that
supersedes [ADR-0013](0013-dual-instance-weight-shared-solver.md) and
[ADR-0014](0014-single-process-dual-context-solver.md).

### Consequences

- Good, because either single engine would let prehend RETIRE the custom
  `potto007/llama.cpp` fork and the entire ADR-0013/0014 two-context design: one
  engine, paged/radix KV, orchestrator and worker as per-request calls to the
  SAME engine (CoT on/off and mixed `max_tokens` are per-request knobs, not two
  servers), and NO `subcall_context_limit` / `per_call_subcall_budget` slot math
  (the paged or radix pool divides dynamically per token).
- Good, because not ripping out llama.cpp keeps a known-working solver while we
  de-risk; the fork stays the fallback if both gates fail.
- Bad, because both engines require re-quantizing off the existing Q4_0 GGUF to
  AWQ INT4 (a one-time calibration job, plus re-validating SFT quality at INT4);
  GGUF is the worst-supported path on both engines (vLLM: "highly experimental"
  and slow, ~93 tok/s with an explicit "consider llama.cpp instead"; SGLang:
  compatibility-only, no optimized kernels). Spikes may start on the existing
  GGUF to de-risk the architecture, but production numbers need AWQ.
- Bad, because the `gemma-4-E4B` distiller is a SECOND, different model and
  neither engine runs two different models in one process on one GPU; it becomes
  a second/on-demand process (start for distillation, stop to free VRAM). This is
  acceptable - the distiller is offline, out of the hot RLM loop - but it must be
  VRAM-tuned against the solver process on the 32GB card.
- Bad, because the spike costs 1-2 days and may conclude "stay on the fork" if
  consumer-Blackwell maturity (SM_120) or the 12B-unified bug blocks both engines.
- Neutral: WSL2 caps throughput at ~70% of native Linux; vLLM CUDA graphs need
  WSL2 >= 2.7.0 (current kernel 6.6.87 is older than the validated 6.6.114 -
  check/upgrade first), and FP8 tensor cores are not exposed through dxgkrnl on
  WSL2 (emulated/slow), so AWQ INT4 is preferred over FP8 regardless of engine.

## Pros and Cons of the Options

### Option 1 - Keep the dual-context llama.cpp fork (status quo, ADR-0014)

- Good, because it works TODAY: runs our existing Q4_0 GGUF directly with no
  re-quantization, no nightly tracking, no consumer-Blackwell maturity bet.
- Good, because each role already has a private KV pool and warm prompt cache;
  prefix caching is verified (65x on a 17.6K prefix).
- Good, because it is the de-risked baseline the spike is measured against.
- Bad, because the two `llama_context`s run SEPARATE schedulers and CANNOT
  co-batch - the structural ~37% aggregate decode loss (225 -> 142 tok/s) under
  the concurrent orchestrator+worker load the RLM pattern always produces.
- Bad, because plain-multihop tasks that passed 15/15 on the single-endpoint
  server now TIME OUT on the fork under that load.
- Bad, because the custom binary tracks unstable llama.cpp server internals
  (`server_context`, `server-http`) and copies the OpenAI route table verbatim
  from `server.cpp` - a hand-mirrored drift point on every upstream bump
  (ADR-0014).
- Bad, because the per-slot KV budgeting (ADR-0009/0012) stays as required
  machinery rather than being dissolved by a paged pool.

### Option 2 - Migrate the solver to vLLM

- Good, because single-engine continuous batching + PagedAttention co-batches
  orchestrator and worker in ONE scheduler - directly dissolving the 37% loss the
  fork's two schedulers cause. One 12B engine serves both roles as per-request
  calls; the fork's whole reason to exist disappears.
- Good, because Automatic Prefix Caching reuses the long shared system-prompt
  prefix automatically (~91% hit rate reported for Gemma 4 without speculative
  decoding), replacing manual `cache-reuse=256` tuning.
- Good, because the Hybrid KV Cache Manager handles Gemma's sliding-window +
  full-attention layers natively - the principled version of our `swa-full=true`
  + q8_0-KV workaround - and the paged allocator divides a single
  `--max-model-len` pool dynamically, removing ADR-0012's per-call budgeting.
- Good, because per-request params coexist natively (mixed `max_tokens`,
  per-request thinking budget via `--reasoning-parser gemma4`): CoT-on
  orchestrator vs CoT-off worker is one knob, not two servers.
- Good, because the 5090/WSL2/CUDA-graphs path is community-validated on a recent
  stack (vLLM 0.17.1, CUDA 12.8, WSL2 2.7.0, ~140 tok/s with CUDA graphs).
- Bad (HIGHEST RISK), because the 12B-Unified arch
  (`Gemma4UnifiedForConditionalGeneration`) is NIGHTLY-ONLY (landed in PR #44429,
  not in any stable release) with an OPEN shape-mismatch bug (#44494,
  `mat1/mat2 ... 2048x4096 and 8192x3840` during memory profiling) and a
  `transformers>=5.5` vs vLLM pin conflict (#39216). This can block the primary
  solver model outright today.
- Bad, because migrating now stacks bleeding-edge model support on a
  bleeding-edge Blackwell/WSL2 path - fragile, likely carrying patches.
- Bad, because GGUF is "highly experimental"/under-optimized and slow in vLLM;
  must re-quantize to AWQ INT4 (FP8 is emulated/slow on the 5090 under WSL2).
- Bad, because vLLM v1 pre-allocates a large non-reclaimable KV pool at startup,
  so the second E4B process must be hand-tuned via `gpu_memory_utilization` so
  the two preallocations fit 32GB without colliding.
- Bad, because speculative decoding must stay OFF: Gemma 4 + hybrid attention +
  DFlash spec-decoding gives 0% prefix-cache hits (#40624, open).

### Option 3 - Migrate the solver to SGLang

- Good, because RadixAttention auto-dedupes the shared-prefix RLM pattern via a
  radix tree (longest-prefix match, automatic cross-request, LRU eviction) -
  arguably the BEST fit for "one model issuing many calls that share an
  orchestrator prefix + sub-call context" (up to ~6.4x on shared-prefix
  workloads, ~29% over vLLM on mixed sets).
- Good, because the single scheduler + continuous batching co-batches
  orchestrator and worker - the same fix for the 37% loss - and thinking is a
  per-request `chat_template_kwargs.enable_thinking` toggle with per-request
  `max_tokens` against one dynamically-allocated pool (no per-slot budgeting;
  ADR-0012's footgun simply does not exist).
- Good, because the FULL Gemma 4 family is supported on SGLang MAIN, including
  the encoder-free unified 12B AND the E4B distiller (per the SGLang Gemma 4
  cookbook), so both target models are first-class - no nightly-only blocker like
  vLLM's unified bug.
- Good, because it serves the one-model-two-roles case natively: load the 12B
  once, vary sampling per request, one shared KV pool, continuous batching across
  both roles.
- Bad (TOP RISK), because consumer RTX 5090 (SM_120) kernel/backend maturity is
  unproven: SGLang's Blackwell work targets datacenter SM_100; the auto-selected
  `trtllm_mha` backend raises a ValueError on SM_120 (#14814, fix PR #14842 in
  flight) - mitigate by forcing `--attention-backend flashinfer` (or `triton`) -
  and earlier SM_120 kernel-image / RMSNorm gaps (#9542) plus FP8 block-wise
  unsupported on SM_120 (#9233). Must validate launch+decode on the exact box.
- Bad, because install is brittle: SGLang main + a PINNED transformers commit,
  and a from-source sgl-kernel build if the prebuilt wheel misses SM_120.
- Bad, because GGUF is compatibility-only (no optimized kernels); production
  needs the AWQ-4bit conversion.
- Bad, because two DIFFERENT models on one GPU in one process is not natively
  supported (#5507, #3265), so the E4B distiller is a second / on-demand process
  with a hand-split `--mem-fraction-static`.

## More Information

- Relates to and (on a passing spike) would supersede
  [ADR-0014](0014-single-process-dual-context-solver.md) and
  [ADR-0013](0013-dual-instance-weight-shared-solver.md); builds on
  [ADR-0012](0012-pool-aware-subcall-budget-under-kv-unified.md) (the per-call
  budget guard and `subcall_context_limit` would be removed, not just bypassed,
  once a paged/radix engine owns KV).
- Full research reports (this evaluation's source material):
  [`0015-research-vllm.md`](0015-research-vllm.md) and
  [`0015-research-sglang.md`](0015-research-sglang.md).

Key sources:

- Gemma 4 12B "Unified": https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/ ,
  https://ai.google.dev/gemma/docs/core/model_card_4
- vLLM Gemma 4 recipe: https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html ,
  https://recipes.vllm.ai/Google/gemma-4-12B-it
- vLLM 12B-unified bug #44494: https://github.com/vllm-project/vllm/issues/44494 ;
  transformers pin conflict #39216: https://github.com/vllm-project/vllm/issues/39216 ;
  Gemma4 + spec-decoding prefix-cache #40624: https://github.com/vllm-project/vllm/issues/40624
- vLLM prefix caching: https://docs.vllm.ai/en/stable/design/prefix_caching/ ;
  hybrid KV cache manager: https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/ ;
  GGUF caveat: https://docs.vllm.ai/en/stable/features/quantization/gguf/
- vLLM on RTX 5090 / WSL2: https://github.com/vllm-project/vllm/issues/37242
- SGLang Gemma 4 cookbook: https://docs.sglang.io/cookbook/autoregressive/Google/Gemma4
- RadixAttention: https://www.lmsys.org/blog/2024-01-17-sglang/
- SGLang SM_120 `trtllm_mha` ValueError #14814: https://github.com/sgl-project/sglang/issues/14814 ;
  SM_120 kernel compat #9542: https://github.com/sgl-project/sglang/issues/9542 ;
  FP8 block-wise on SM_120 #9233: https://github.com/sgl-project/sglang/issues/9233
- SGLang multi-model-on-one-GPU requests #5507 / #3265:
  https://github.com/sgl-project/sglang/issues/5507 ,
  https://github.com/sgl-project/sglang/issues/3265
- SGLang vs vLLM: https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison ,
  https://techsy.io/en/blog/vllm-vs-sglang
