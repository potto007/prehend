# SGLang for the prehend dual-context RLM workload - research report

Date: 2026-06-23. All facts cited to primary/secondary sources at bottom. Honest
uncertainty flagged inline.

## TL;DR / bottom line

SGLang's **architecture support and programming model are an excellent fit**: both your
models - `google/gemma-4-12B-it` (12B dense) and `google/gemma-4-E4B-it` (effective-4B,
PLE) - are **first-class supported on SGLang main**, RadixAttention auto-dedupes exactly
the shared-prefix pattern your orchestrator + map-reduce sub-calls produce, and per-request
thinking on/off + mixed `max_tokens` need **no per-slot budgeting** (unlike your llama.cpp
`per_call_subcall_budget` guard from ADR-0012). One **single scheduler** with continuous
batching co-batches orchestrator and worker requests - directly fixing the "two
`llama_context`s can't co-batch → 37% decode loss" problem you measured.

**Two real blockers, both about your specific hardware/topology, not the engine:**

1. **One-model-two-roles is the natural SGLang case (good); two DIFFERENT models on one
   GPU in one process is NOT natively supported (bad for E4B co-residence).** Your
   orchestrator and worker are the SAME weights with different sampling params - SGLang
   serves that as ONE server and you just vary `chat_template_kwargs`/`max_tokens`
   per request. But the separate `gemma-4-E4B` distiller is a different model → second
   process (or CPU/disk offload workaround). See §5.
2. **Consumer Blackwell RTX 5090 (SM_120) is the highest risk.** SGLang's Blackwell work
   targets datacenter SM_100 (B200/GB300/RTX PRO 6000). SM_120 has had live kernel/backend
   gaps through the 25.10/25.11 cycle, incl. the auto-selected `trtllm_mha` backend
   raising a ValueError on SM_120 (issue #14814, Dec 2025; PR #14842 in flight). Workable
   with manual `--attention-backend flashinfer/triton`, but **you must validate on your
   exact box** before committing. See §4.

**Recommendation: run a time-boxed (1-2 day) spike** - install SGLang main on the 5090/WSL2,
launch `gemma-4-12B-it` with `--attention-backend flashinfer`, drive it with a concurrent
orchestrator+worker RLM run, confirm prefix-cache hit rates and that the multihop tasks that
timed out on the dual-context fork now complete. If the spike passes, **SGLang beats your
custom dual-context llama.cpp fork** for this workload (eliminates the fork maintenance, the
co-batching loss, and the manual KV budgeting) and is **a better fit than vLLM** for the
many-shared-prefix RLM pattern specifically. The risk is hardware-support maturity on
consumer Blackwell, not the architecture.

---

## 1. Architecture support (CRITICAL GATE) - PASS, with a naming clarification

Both target architectures are supported. Note these are the **real Google "Gemma 4"
generation** (launched ~early 2026, day-0 vLLM/SGLang/ROCm support), not internal naming -
your repo's "gemma-4" matches the public model IDs exactly.

The SGLang Gemma 4 cookbook lists five supported variants:
- `google/gemma-4-E2B-it` (~2B, dense)
- **`google/gemma-4-E4B-it` (~4B effective, dense, Per-Layer Embeddings)** ← your distiller
- **`google/gemma-4-12B-it` (12B dense, "encoder-free unified")** ← your solver
- `google/gemma-4-31B-it` (31B dense)
- `google/gemma-4-26B-A4B-it` (26B total / 4B active, MoE)

Cookbook quote: *"Gemma 4 (including the encoder-free unified 12B) is supported on SGLang
main."* Requires install from **main branch** plus a specific transformers commit
(`1423d22f7a3b62e8c70ad67b58ec25cd9b675897`). [sglang-gemma4]

**E4B / MatFormer / "3n" lineage:** The "E" = "effective" params; E4B uses **Per-Layer
Embeddings (PLE)** so total params exceed the 4B effective footprint. The MatFormer
(Matryoshka nested sub-models), AltUp, LAuReL, activation-sparsity Top-k, and KV-cache
sharing tricks originated in **Gemma 3n**; `gemma-4-E4B` is the Gemma-4-generation
descendant of `gemma-3n-E4B`. [gemma4-visual][matformer-hf][gemma3n-overview] The SGLang
cookbook describes Gemma 4 in terms of **Hybrid Attention (sliding-window + full)** and
**PLE**; it does NOT separately call out "MatFormer"/"Gemma 3n"/"selective activation" as
config knobs, because at serve time E4B is just a dense model with PLE. [sglang-gemma4]

> Uncertainty: I confirmed E4B is listed as supported, but did not find a benchmark/issue
> proving PLE runs at full speed under SGLang on 5090 specifically. Treat as
> "supported, verify perf in the spike." Note Gemma **3n** support had an open feature
> request (#6498) earlier in 2025; the Gemma **4** E4B is the one to target now.

---

## 2. RadixAttention + continuous batching fit - STRONG FIT

This is SGLang's headline strength and maps almost exactly onto the RLM pattern.

- **Automatic prefix dedup via a radix tree.** KV caches are stored as paths in a radix
  tree; on each new request the scheduler finds the **longest matching prefix**, reuses
  those KV tensors, and only computes the novel suffix. Fully automatic - no manual prefix
  declaration, no static config; LRU eviction when GPU KV memory is tight.
  [radix-mintlify][starlog][lmsys-radix] For your orchestrator (long reused system-prompt
  prefix every iteration) and the fanned-out sub-calls (shared sub-call context), this is
  the dedup you currently get from llama.cpp `cache-reuse=256` - but cross-request and
  automatic, not just same-slot prefix reuse.
- **Single scheduler + continuous batching = the fix for your co-batching loss.** SGLang
  runs ONE scheduler/engine; orchestrator and worker requests land in the **same continuous
  batch** and co-batch on the GPU. This is precisely what your two-`llama_context`
  design can't do (separate schedulers → 225→142 tok/s aggregate, per-stream halving). On
  shared-prefix/agentic workloads SGLang reports up to **6.4x** throughput vs engines
  without cross-request caching, and ~29% over vLLM on mixed H100 sets. [particula][techsy]
- **Per-request params, no per-slot budgeting.** Thinking is toggled **per request** via
  `extra_body={"chat_template_kwargs": {"enable_thinking": true|false}}` - orchestrator
  sends `true`, worker `false`, same server. [sglang-thinking] `max_tokens` and effective
  context vary per request against ONE shared KV pool managed by the radix tree; you do NOT
  pre-slice the pool into fixed slots. This **removes the ADR-0012 `per_call_subcall_budget`
  problem entirely** - there is no `--kv-unified` shared-pool-vs-private-window footgun
  because SGLang's pool is dynamically allocated per token, not partitioned into N fixed
  parallel windows.

---

## 3. Quantization path off Q4_0 GGUF - MIGRATE OFF GGUF

- **GGUF works but is the worst-supported path.** SGLang can load llama.cpp GGUF
  (`--quantization gguf`, full GGML types incl. Q4_0/Q8_0), dequantizing via
  `quantization/gguf.py`. But GGUF on SGLang is treated as compatibility, not the
  performance path, and lacks the optimized kernels the native formats get.
  [sglang-quant][sglang-gguf-disc]
- **Recommended path for an SFT'd model:** re-quantize to **AWQ or GPTQ (4-bit)** for
  quality-preserving 4-bit, or **FP8** if you want Blackwell-native throughput.
  - AWQ/GPTQ: small calibration step, generally best 4-bit quality retention for an SFT
    checkpoint; well-supported kernels. Effort: moderate (run calibration once).
  - FP8: best raw throughput on Blackwell, but **FP8 block-wise on consumer SM_120 had
    gaps** (issue #9233) - verify before relying on it. NVFP4 is supported in 25.10/25.11
    but again datacenter-Blackwell-leaning.
- **Practical recommendation:** start the spike with your **existing Q4_0 GGUF** to
  de-risk the architecture quickly, then convert to **AWQ-4bit** for the real deployment.
  Keep the BF16/QAT-unquantized SFT source so you can re-quantize cleanly. The cookbook
  notes Gemma 4 QAT checkpoints exist (`qat-q4_0-unquantized`). [sglang-gemma4]

---

## 4. Blackwell / RTX 5090 / WSL2 / CUDA - HIGHEST RISK, validate first

This is where SGLang is least proven for your exact box.

- **Datacenter Blackwell (SM_100: B200, GB300, RTX PRO 6000) is supported** in releases
  25.10 / 25.11, incl. NVFP4. **Consumer RTX 5090 (SM_120) is a different story.**
  [nvidia-2510][nvidia-2511]
- **Live SM_120 gaps (recent):**
  - Auto-selected `trtllm_mha` backend **raises ValueError on SM_120** because it's gated to
    SM_100 - issue #14814, **Dec 10 2025**, fix PR #14842 in flight. **Workaround: force
    `--attention-backend flashinfer` (or `triton`).** [issue-14814]
  - Earlier (`Aug 2025`) "no kernel image available for execution on the device" /
    RMSNorm failures on SM_120+CUDA 12.8 even with "blackwell" wheels - issue #9542 (marked
    inactive, not cleanly resolved in-thread). [issue-9542]
  - FP8 block-wise unsupported on SM_120 - issue #9233. [issue-9233]
- **PyTorch/kernel floor:** SM_120 needs recent PyTorch (the broader ecosystem only moved
  past sm_90-only stable builds via newer CUDA 12.8+ wheels). Use a **CUDA 12.8+ stack and
  a current PyTorch** matching SGLang main. Your box is CUDA 13, which is ahead of most
  cited setups - could help (newer) or surprise (less-tested); verify the sgl-kernel wheel
  matches. [pytorch-159207][sm120-wsl2]
- **WSL2:** I found no SGLang-specific WSL2 blocker, but also no positive SM_120-on-WSL2
  confirmation. Given you already run llama.cpp + CUDA 13 on this WSL2 box, the CUDA
  userspace is proven; the unknown is SGLang's prebuilt sgl-kernel for SM_120 under WSL2.

**Net:** plan for `--attention-backend flashinfer`, a from-source or main-branch sgl-kernel
build if prebuilt wheels miss SM_120, and **treat "does it even launch and decode on the
5090" as gate #1 of the spike.** Do not assume; this is the single most likely thing to
sink the migration.

---

## 5. VRAM / KV model - orchestrator+worker = ONE server; E4B = separate process

- **Memory model:** `--mem-fraction-static` = (weights + KV pool) / GPU capacity (default
  0.9; drop to 0.7-0.8 under pressure). KV is a single dynamically-allocated pool
  (`token_to_kv_pool`) the radix tree draws from - NOT fixed per-slot windows.
  [sglang-mem][deepwiki-mem]
- **Orchestrator + worker (same 12B weights, two roles): trivially ONE server.** Load the
  12B once; issue orchestrator requests (thinking on, long output) and worker requests
  (thinking off, short output) to the same endpoint. Single weights copy, single shared KV
  pool, continuous batching across both. This is the clean win vs your dual-`llama_context`
  fork - you get one-weights/co-batched for free, which the fork explicitly could not do.
- **Adding the separate `gemma-4-E4B` distiller is the constraint.** SGLang does **not**
  natively run two DIFFERENT models in one process on one GPU (open requests #5507, #3265).
  Options, roughly in order of preference:
  1. **Two processes on the one 5090.** 12B-AWQ4 ≈ 7-8 GB weights; E4B-4bit ≈ 3-4 GB.
     Combined weights ~11-12 GB leaves ~18-20 GB of the 32 GB for two KV pools - feasible,
     but you hand-split `--mem-fraction-static` between them and lose cross-process
     co-batching (acceptable: the distiller runs out-of-band, not in the hot RLM loop).
  2. **Distiller as a second process you start on demand** (it's an offline experience-
     distill step, not latency-critical) - start it, run distillation, stop it, freeing VRAM
     for the solver. Cleanest VRAM-wise.
  3. SGLang "Universal Memory" multi-model hosting with CPU/disk offload of the inactive
     model - exists but adds complexity; only if you truly need both hot simultaneously.
     [hf-sglang]
- **Bottom line on co-residence:** 12B + E4B **can** co-reside in 32 GB at 4-bit, but as
  **two SGLang processes**, not one - and easiest is to time-share (distiller on demand).

---

## 6. Migration steps, risks, and recommendation vs vLLM

### Suggested migration / spike steps
1. Install SGLang **main** + pinned transformers commit; CUDA 12.8+/13, current PyTorch,
   sgl-kernel matching SM_120 (build from source if prebuilt wheel lacks it).
2. Launch `gemma-4-12B-it` (start with existing Q4_0 GGUF, `--quantization gguf`) with
   **`--attention-backend flashinfer`**, `--mem-fraction-static 0.8`. Confirm it decodes on
   the 5090 at all (gate #1).
3. Point the prehend harness at the SGLang OpenAI-compatible endpoint; orchestrator vs
   worker = per-request `chat_template_kwargs.enable_thinking` + `max_tokens`. **Delete the
   `per_call_subcall_budget` slot math** - let the radix pool manage it.
4. Run the previously-timing-out large-context multihop tasks; verify completion +
   prefix-cache hit-rate (SGLang reports cache stats). This is the real acceptance test.
5. If green, re-quantize the SFT 12B to **AWQ-4bit** and re-measure. Stand up the E4B
   distiller as a second/on-demand process.

### Risks
- **(High)** Consumer SM_120 kernel/backend maturity under WSL2 - could block at step 2.
- **(Med)** GGUF perf is poor; need the AWQ conversion to land for production numbers.
- **(Med)** Two-process VRAM split between 12B and E4B requires tuning; lose cross-model
  batching (acceptable for an offline distiller).
- **(Low)** Install pins (main branch + exact transformers commit) are brittle; pin them.

### SGLang vs vLLM for the RLM pattern
Both support Gemma 4 day-0 and have prefix caching, but **SGLang is the better fit for the
*many-shared-prefix, structured-program* RLM workload specifically**: RadixAttention's
automatic cross-request radix-tree reuse and the co-batching single scheduler are tuned for
exactly "one model issuing many calls that share an orchestrator prefix + sub-call context"
(up to 6.4x on shared-prefix workloads; ~29% over vLLM on mixed sets). vLLM's prefix caching
is solid but the radix structure + structured-generation overlap make SGLang the natural
choice here. [particula][techsy][chatforest] vLLM would be the safer pick ONLY if SM_120
maturity blocks SGLang on your 5090 - keep it as the fallback.

### Verdict
**Yes - pending the hardware spike, SGLang beats the custom dual-context llama.cpp fork**
for this workload: it removes the fork, the co-batching throughput loss, and the manual KV
budgeting, while serving the one-model-two-roles pattern natively and deduping your shared
prefixes automatically. The gating uncertainty is **consumer-Blackwell support maturity**,
not architecture or programming model. Spike that first.

---

## Sources
- [sglang-gemma4] SGLang Gemma 4 cookbook: https://docs.sglang.io/cookbook/autoregressive/Google/Gemma4
- [gemma4-visual] Visual Guide to Gemma 4: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4
- [gemma4-family] Gemma 4 family overview: https://louiswang524.github.io/blog/gemma-family/
- [matformer-hf] MatFormer in Gemma 3n: https://huggingface.co/blog/rishiraj/matformer-in-gemma-3n
- [gemma3n-overview] Gemma 3n overview: https://ai.google.dev/gemma/docs/gemma-3n
- [gemma4-E4B] google/gemma-4-E4B: https://huggingface.co/google/gemma-4-E4B
- [amd-gemma4] Day-0 Gemma 4 on AMD (vLLM/SGLang/ROCm): https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-gemma-4-on-amd-processors-and-gpus.html
- [radix-mintlify] RadixAttention concept: https://sgl-project-sglang-93.mintlify.app/concepts/radix-attention
- [lmsys-radix] LMSYS RadixAttention/SGLang blog: https://www.lmsys.org/blog/2024-01-17-sglang/
- [starlog] RadixAttention explainer: https://starlog.is/articles/llm-engineering/sgl-project-sglang/
- [sglang-thinking] Per-request thinking toggle (chat_template_kwargs): https://docs.sglang.io/cookbook/autoregressive/Google/Gemma4 and https://qwen.readthedocs.io/en/latest/deployment/sglang.html
- [sglang-quant] SGLang quantization docs: https://docs.sglang.io/advanced_features/quantization.html
- [sglang-gguf-disc] GGUF quant discussion #2446: https://github.com/sgl-project/sglang/discussions/2446
- [sglang-mem] Hyperparameter/memory tuning: https://docs.sglang.io/advanced_features/hyperparameter_tuning.html
- [deepwiki-mem] Memory management & KV cache: https://deepwiki.com/sgl-project/sglang/2.3-memory-management-and-caching
- [issue-14814] trtllm_mha ValueError on SM_120 (RTX 5090): https://github.com/sgl-project/sglang/issues/14814
- [issue-9542] sgl-kernel Blackwell SM_120 compat: https://github.com/sgl-project/sglang/issues/9542
- [issue-9233] FP8 block-wise on SM_120: https://github.com/sgl-project/sglang/issues/9233
- [issue-5507] Multiple model instances on single GPU: https://github.com/sgl-project/sglang/issues/5507
- [issue-3265] Single GPU multiple servers: https://github.com/sgl-project/sglang/issues/3265
- [nvidia-2510] SGLang Release 25.10: https://docs.nvidia.com/deeplearning/frameworks/sglang-release-notes/rel-25-10.html
- [nvidia-2511] SGLang Release 25.11: https://docs.nvidia.com/deeplearning/frameworks/sglang-release-notes/rel-25-11.html
- [pytorch-159207] PyTorch sm_120 support: https://github.com/pytorch/pytorch/issues/159207
- [sm120-wsl2] PyTorch on RTX 5090 WSL2 (sm_120) guide: https://medium.com/@getnetdemil/getting-pytorch-to-actually-use-your-rtx-5090-a-complete-wsl2-setup-guide-for-blackwell-sm-120-61f86f64abc4
- [particula] SGLang vs vLLM 2026: https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison
- [techsy] vLLM vs SGLang H100 benchmarks: https://techsy.io/en/blog/vllm-vs-sglang
- [chatforest] SGLang prefix-heavy serving review: https://chatforest.com/reviews/sglang-structured-generation-llm-serving/
- [hf-sglang] HF SGLang engine docs: https://huggingface.co/docs/inference-endpoints/en/engines/sglang
