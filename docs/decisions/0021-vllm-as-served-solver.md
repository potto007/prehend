---
status: "accepted"
date: "2026-06-26"
deciders: "potto"
consulted: "research agents (vLLM, SGLang); ADR-0015 evaluation"
supersedes: "0016-sglang-as-served-solver"
---

# vLLM as the served solver: gemma-4 v13 on vLLM 0.23.0, retire the SGLang/dual-context path

## Context and Problem Statement

[ADR-0015](0015-inference-engine-evaluation-vllm-sglang.md) evaluated vLLM and
SGLang as single-engine replacements for the dual-context llama.cpp fork
([ADR-0014](0014-single-process-dual-context-solver.md)), whose two `llama_context`
schedulers cannot co-batch and lose ~37% decode throughput under the concurrent
orchestrator+worker load the RLM pattern produces. The 2026-06-26 Update to
ADR-0015 flipped the lean to **vLLM-first** (vLLM 0.23.0 ships stable encoder-free
Gemma 4 Unified + MTP, WSL2 is on 2.7.8, TurboQuant KV is vLLM-only, and vLLM has
lower setup friction than SGLang's main + pinned-transformers + SM_120 dance).

[ADR-0016](0016-sglang-as-served-solver.md) (SGLang as the served solver) was the
prior direction and reached a WORKING SGLang serving setup (gate-1), but it was
never accepted. This ADR ratifies the vLLM-first outcome of ADR-0015 and
supersedes ADR-0016: **vLLM 0.23.0 is the served solver**.

## Decision Outcome

Chosen: **serve the v13 solver (`gemma-4-12B-it-sft-kb-v13-sft`) with vLLM 0.23.0
on `:8080`**, one engine for both orchestrator and worker roles (continuous
batching + paged-KV / prefix caching co-batch what the two `llama_context` could
not). The SGLang infra stays on disk for rollback but its systemd unit and
Prometheus scrape job are stopped/disabled.

Validated on the RTX 5090 (32GB, SM_120), CUDA 13, WSL2, 2026-06-26:

- **GATE #1 - load + decode.** vLLM loads the W4A16 compressed-tensors checkpoint
  (Marlin WNA16 kernel) with `--kv-cache-dtype fp8_e4m3`, auto-selects the
  **TRITON_ATTN** backend (it detects gemma-4's heterogeneous head dims -
  `head_dim=256` sliding / `global_head_dim=512` full - and forces triton to
  avoid mixed-backend divergence; no `VLLM_ATTENTION_BACKEND` override needed),
  and decodes correctly.
- **fp8 KV headroom is large, not starving.** GPU KV cache = **676,724 tokens**,
  **20.65x** max concurrency at full 32,768-token context. (Contrast: bf16 KV
  starved SGLang to `max_total_num_tokens=194` (ADR-0019); SGLang's own fp8 KV
  reached ~98k. vLLM's paged fp8 KV gives ~7x that.)
- **GATE #2 - the multihop tasks that TIMED OUT before now COMPLETE.** The plain
  large-context multihop set (`tasks/multihop.json`, ~315k chars / ~150k tokens
  per task, map-reduced) ran the memory-off control to completion: 3/3 tasks
  finished (no timeout), 2/3 correct (the third, `multihop_002`, is a known-hard
  conflicting-notes item the model reasons wrong - an answer-quality miss, not an
  engine failure). Latencies 26-56s for the chained answers (prefix-cache warming
  visibly cuts task 2 to 26s). Prefill throughput 5,800-7,800 tok/s.

### Config (production)

`localai-vllm-solver.service` ExecStart:

```
vllm serve <ckpt-vllm> \
  --served-model-name gemma-4-12B-it-sft-kb-v13-sft \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.80 \
  --host 127.0.0.1 --port 8080
```

(`--max-model-len` was raised 32768 -> 65536 by the tuning pass below.)

The `--served-model-name` is identical to the SGLang setup, so the prehend client
and harness are UNCHANGED. Subcall budgeting uses `PREHEND_DYNAMIC_KV_POOL=1`
(vLLM's paged KV is a single pool like SGLang's RadixAttention, so the llama.cpp
`--kv-unified` per-slot division of [ADR-0012](0012-pool-aware-subcall-budget-under-kv-unified.md)
is bypassed; each sub-call budgets against the full `subcall_context_limit`).

### The config.json field-name translation (load-bearing)

The W4A16 checkpoint's `config.json` was hand-translated to **SGLang's** gemma-4
field names (`head_dim`/`swa_head_dim` etc.), which SWAP the meaning of `head_dim`
vs the transformers-native schema. vLLM loads via the transformers config classes,
so it needs the **native** names: native `head_dim` = SLIDING head dim (256),
`global_head_dim` = FULL head dim (512), `num_key_value_heads` = sliding KV heads
(8), `num_global_key_value_heads` = full KV heads (1), `attention_k_eq_v=true`
(no separate `v_head_dim`). Confirmed against ground-truth weight shapes
(sliding layer q=16x256, k/v=8x256; full layer q=16x512, k=1x512, no v_proj).

Resolution: a vLLM-specific checkpoint dir
`gemma-4-12B-it-sft-kb-v13-w4a16-g32-ct-vllm/` with the transformers-native
`config.json` (taken from the `-text-native` checkpoint) + the `quantization_config`
block, and the 8GB `model.safetensors` / tokenizer SYMLINKED from the SGLang `-ct`
dir (no duplication). The SGLang `-ct` dir is left untouched for rollback.

### Consequences

- Good: the ~37% dual-context throughput loss and the multihop timeouts are gone;
  fp8 KV gives ~7x the token headroom; one engine, one port, client unchanged.
- Good: TurboQuant KV (arXiv:2504.19874, vLLM-only) is now reachable if we ever
  become capacity-bound (we are not - fp8 e4m3 is the proven floor, ADR-0019).
- Neutral: vLLM startup does a one-time torch.compile (~40s) + cudagraph capture;
  cached after first run.
- Rollback: re-enable the SGLang systemd unit + the `sglang-solver` Prometheus job
  (both left commented/stopped, not deleted).

## Infra delivered

- `~/src/local-ai/scripts/setup-vllm.sh` - reproducible env (`.venv-vllm`, py3.13.12,
  `vllm==0.23.0` cu130 torch trio, `transformers==5.12.1`).
- `~/src/local-ai/scripts/vllm-launch.sh` - drop-in `vllm serve` wrapper teeing to
  the canonical `/tmp/vllm-server.log` (monitoring-rail rule).
- `~/src/local-ai/scripts/vllm-server.sh {start|stop|status|smoke|logs|tail}`.
- `~/src/local-ai/scripts/localai-vllm-solver.service` (linked + enabled).
- Promtail `vllm` job (`/tmp/vllm-server.log` -> Loki `{job="vllm"}`).
- Prometheus `vllm-solver` scrape job (sglang-solver disabled); Grafana dashboard
  `local-ai (vLLM)` (uid `local-ai-vllm`).

## Attention backend: TRITON_ATTN is the only option (2026-06-26)

vLLM auto-forces TRITON_ATTN for gemma-4 Unified (heterogeneous head dims:
256 sliding / 512 full). We confirmed the alternatives are dead ends:
- **FA2/FlashAttention**: rejects head_size 512 (kernel limit 256) - the global
  layers can't run, so a flash+triton mix would diverge. Not built (conclusive).
- **FlashInfer**: forced it on via `--attention-config '{"backend":"FLASHINFER"}'`
  (the vLLM-0.23 mechanism; `VLLM_ATTENTION_BACKEND` env is GONE) - engine init
  FAILS with `ValueError: Selected backend FLASHINFER is not valid... Reason:
  ['partial multimodal token full attention not supported']` (gemma-4 sets
  `use_bidirectional_attention="vision"`). A separate reason from the 512 dim, same
  outcome. (llama.cpp runs flash on both dims only because ggml ships native 512
  kernels; not portable to vLLM.)

So TRITON_ATTN stays - and it's already optimal (676k-1.0M-token KV, correct).

## Tuning: max-model-len 65536 (the one lever that moved the needle)

Measured on plain-multihop (each task ~315k chars / ~150k tokens, map-reduced):
- **Prefix-cache reuse cannot be raised by a server knob.** Prefix caching works
  (control: identical prompt twice -> ~98% hit), but a real task shows 0 vLLM
  hits and **0 preemptions** - KV is NOT evicting (the pool dwarfs the task). The
  ~2.3x context re-prefill is (a) ~1.18x un-cacheable chunk overlap and (b) a
  ~2x boundary-misaligned second map scan that map_cache dedups IN-APP (so the
  flat vLLM hit counter is intended, not a bug). Reuse is a CLIENT property here,
  not a vLLM config.
- **max-model-len 32768 -> 65536** (client budgets sub-calls at ~60000 tokens):
  bigger chunks (~3/pass vs ~13) co-locate each multihop answer with its linking
  fact, so the extraction-map chains in ~1 pass not ~2. **Prefill 343k -> 194k
  tokens/task (-43%), 5/5 correct, and the sub-call overflow 400s disappear.**
  Tradeoff: ~5s higher single-task latency at concurrency=1 (revert to 32768 if
  that matters more than throughput/robustness).
- **Rejected: chunk overlap 0.15 -> 0.0** (`local_repl.py:193`). Saved ~18% prefill
  on paper but broke cross-chunk multihop chains: 5/5 -> 3/5, 31s -> 103s avg
  (a task timed out at 401s). The overlap is load-bearing; kept at 0.15.
- Open client-side lever (not done; needs a broader eval): trim the few-shot
  manual-slicing examples (`prompts.py:21,47-87`) to force every large read
  through the cached harness path, collapsing the residual second scan.

## More Information

- [ADR-0015](0015-inference-engine-evaluation-vllm-sglang.md) (evaluation + vLLM-first Update)
- [ADR-0016](0016-sglang-as-served-solver.md) (superseded)
- [ADR-0019](0019-fp8-e4m3-kv-cache-for-gemma4-solver.md) (fp8 e4m3 KV rationale)
- vLLM v0.23.0 release; compressed-tensors + fp8 KV docs.
