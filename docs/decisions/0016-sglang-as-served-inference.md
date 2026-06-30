---
status: "superseded by 0021-vllm-as-served-inference"
date: "2026-06-24"
deciders: "potto"
consulted: "research agents (vLLM, SGLang); observability agent"
---

> **Superseded (2026-06-26) by [ADR-0021](0021-vllm-as-served-inference.md).** The
> SGLang serving path reached a working gate-1 but was never accepted; ADR-0015's
> lean flipped to vLLM-first and vLLM 0.23.0 was validated as the served inference.
> SGLang infra is retained for rollback only.

# SGLang as the served inference: retire the dual-context llama.cpp fork

## Context and Problem Statement

[ADR-0015](0015-inference-engine-evaluation-vllm-sglang.md) evaluated vLLM and
SGLang as single-engine replacements for the `llama-dual-context-server`
([ADR-0014](0014-single-process-dual-context-inference.md)), whose two
`llama_context`s run separate schedulers and **cannot co-batch** - costing ~37%
aggregate decode throughput under the concurrent orchestrator+worker load the RLM
pattern inherently produces, and timing out plain-multihop tasks. ADR-0015
recommended a time-boxed SGLang spike, gated on whether the bleeding-edge
`gemma4_unified` 12B arch even decodes on the consumer Blackwell RTX 5090 (SM_120)
under WSL2. This ADR records the outcome of running that spike.

## Status note (2026-06-24): PROPOSED, not yet accepted

The serving INFRA is validated; the end-to-end ACCURACY on the real workload is
NOT yet, so the production default is NOT flipped. Treat the "Decision" below as
the intended end state pending the GATE #2 accuracy A/B (see that section).

## Decision

**Adopt SGLang as the served inference.** One SGLang engine on `:8080` serves BOTH
the orchestrator and worker roles via continuous batching + RadixAttention,
co-batching what the two `llama_context`s could not. The roles differ only by
per-request params, not separate processes/ports. This **supersedes
[ADR-0013](0013-dual-instance-weight-shared-inference.md) and
[ADR-0014](0014-single-process-dual-context-inference.md)** (the two-process and
dual-context designs) and makes the [ADR-0012](0012-pool-aware-subcall-budget-under-kv-unified.md)
`--kv-unified` per-slot budget division unnecessary on this engine.

Spike outcome - GATE #1 clean PASS, GATE #2 partial (infra/latency yes,
accuracy A/B pending):

- **GATE #1 (does it decode on SM_120?): PASS.** The v13 model decodes correctly
  on the 5090/WSL2. Every blocker hit was config/version-skew, never a Blackwell
  kernel failure: cuda-graph capture and the triton attention kernels ran clean.
- **GATE #2 (do the timed-out multihop tasks complete + co-batch?): PARTIAL.**
  Co-batching is confirmed and scales near-linearly: per-stream ~43 tok/s held
  flat to concurrency 16 (aggregate ~680 tok/s), vs the fork's per-stream halving;
  saturation knee ~24, peak ~1461 tok/s at 48. The engine runs the real RLM
  workload (orchestrator + co-batched sub-calls, verified `#running-req > 1`).
  BUT on a small plain-multihop sample (no memory) ACCURACY was poor (give-up
  answers) and one task (multihop_053) persistently timed out. Root-caused and
  fixed a latency cliff: when total concurrent sub-call KV demand
  (`slots * subcall_context_limit`) approaches the pool, RadixAttention evicts the
  orchestrator's transcript prefix between iterations -> re-prefill every iteration
  (~365 s/iter, `#cached-token` ~0). Keeping demand well under the pool restored
  ~96-99% prefix reuse and ~3.5x faster iterations - but did NOT by itself fix
  accuracy/convergence. **Open before flipping the default:** a proper A/B vs the
  llama baseline (more tasks, with memory) to attribute the gap to (a) small-sample
  variance against the ~40%-cold / lower no-memory baseline, (b) fp8_e4m3 KV
  precision on long-context retrieval (vs llama's q8_0 KV - untested), or (c)
  chat-template / thinking-mode under sglang's auto `reasoning_parser=gemma4`.

### What it takes to serve the v13 model under SGLang (load-bearing specifics)

The v13 checkpoint is a Google **`gemma4_unified`** (text+vision+audio) merge from
`transformers 5.10.0.dev0`, but only used for text inference. Serving it required:

1. **Env** (isolated `~/src/local-ai/.venv-sglang`, py3.12): released
   `sglang==0.5.13.post1` (already ships `gemma4_unified`/`gemma4_causal` - the
   "sglang main + commit" advice in ADR-0015 is stale), the torch trio pinned to
   **cu130** (the resolver pulls torchvision cu128 -> its CUDA-major check aborts
   `import sglang`), and **`transformers >= 5.10`** for `gemma4_unified` (sglang's
   own pin 5.8.1 and 5.9.0 lack it; the RELEASED 5.10.0-5.12.1 line has it - verified
   serving on **5.12.1** - so no dev/git build is needed, despite an early spike
   pinning the `5.10.0.dev0` commit `1423d22`). `ninja` must be on PATH at serve time
   (fused-RoPE JIT). Captured in `scripts/setup-sglang.sh`.
2. **Text-only checkpoint extraction.** 666/677 tensors are the language model;
   11 are vision/audio stubs (no real vision encoder). Whole, it routes to
   sglang's `gemma4_mm` -> crashes building a vision tower. Fix: extract a
   `Gemma4ForCausalLM` checkpoint (`models/gemma-4-12B-it-sft-kb-v13-text`).
3. **Config field-name translation.** This gemma4 has dual head dims + dual
   KV-head counts per layer-type. transformers names base=sliding / `global_*`=full;
   sglang reads base=full / `swa_*`=sliding. Set `head_dim=512`(global),
   `swa_head_dim=256`, `num_key_value_heads=1`(global), `swa_num_key_value_heads=8`,
   `v_head_dim=512`, `swa_v_head_dim=256`.
4. **`--attention-backend triton`** is mandatory (gemma4 rejects flashinfer; the
   auto `trtllm_mha` ValueErrors on SM_120, issue #14814).
5. **fp8 is mandatory, not just for speed.** bf16 weights (24.6GB) leave
   `max_total_num_tokens=194` -> KV-starved scheduler spin, every request hangs.
   `--quantization fp8` (12.8GB) + `--kv-cache-dtype fp8_e4m3` -> ~98k-token pool.

### Client/harness changes (the "flip the default")

- **One endpoint:** `subcall_base_url == base_url` (pass `subcall_base_url=None`);
  orchestrator vs worker differ only by per-request `chat_template_kwargs.enable_thinking`
  + `max_tokens`, not separate servers.
- **`dynamic_kv_pool=True`** (new `Harness`/`Defaults` flag): bypasses the ADR-0012
  `per_call_subcall_budget` slot-division. SGLang's paged radix pool LRU-evicts
  under contention instead of 500ing, so each sub-call is budgeted against the
  full per-request context-length cap. Default stays `False` (llama.cpp path
  preserved for rollback).

### Infra & observability

`scripts/sglang-server.sh {start|stop|status|smoke}` + systemd unit
`localai-sglang.service` (fp8, triton, `--enable-metrics`, `:8080`),
mirroring the llama-server conventions. SGLang `/metrics` is scraped by Prometheus
(`sglang-inference` job), logs ship to Loki via promtail, and a Grafana dashboard
(`sglang-inference`) covers throughput / TTFT / ITL / KV usage / cache-hit / GPU.

## Consequences

- **Good:** co-batched single engine (recovers the ~37% loss); RadixAttention
  auto-dedupes the RLM shared prefix; no fork to maintain; no `--kv-unified`
  per-slot KV math; per-request thinking/max_tokens; standard OpenAI API.
- **Bad / watch:** depends on a hand-translated config (brittle across model
  revisions - re-translate per new vN checkpoint) + a recent transformers
  (released 5.12.1; pin will move as the model line evolves);
  triton backend (flashinfer is faster but gemma4-forbidden); the e4b distiller
  (`:8083`) and a second model can't co-reside in one SGLang process (run
  on-demand). fp8 weights are re-quantized online each start (~17s) - the only
  ONLINE option from bf16; every 4-bit weight path needs a pre-quantized checkpoint.
  4-bit weights (~7GB, ~2x KV pool) would relieve the GATE #2 KV-pressure failure,
  BUT (live-verified 2026-06-24) **NVFP4 on consumer Blackwell sm_120 is contested**:
  flashinfer #2577 (OPEN) reports all three NVFP4 GEMM backends broken on SM120
  (CUTLASS silent-zeros, cuDNN unsupported, TRT-LLM cap-120 error); vLLM #31085
  (open) requests sm_120 NVFP4 MoE kernels. One real-world report DID serve gemma4
  NVFP4 on a desktop-Blackwell (SM12.0) WSL2 box - but via **vLLM nightly**, forcing
  the flashinfer CUTLASS SM120 FP4 path (Marlin had negative-scale bugs), keeping
  self-attention in BF16, after 8 stacked blocker fixes. So 4-bit on the 5090 is
  "possible but painful / engine-and-version-specific", NOT a clean lever - needs
  real testing (try `awq`/`gptq` marlin first; NVFP4 likely means vLLM-nightly). A
  pre-quantized fp8/AWQ checkpoint on disk is a lighter follow-up lever (more KV + faster
  start). vLLM remains the fallback if a future model revision breaks SGLang's
  gemma4 support.
- **Rollback:** the llama.cpp dual-context path (ADR-0014) and its
  `localai-llama-*` units remain; flip the prometheus scrape + harness wiring back.
