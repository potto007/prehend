# vLLM for the prehend RLM dual-context workload - research report

Date: 2026-06-23. All facts dated late-2025 / mid-2026. URLs cited inline.

## 0. CRITICAL NAMING CLARIFICATION (read first)

Your "gemma-4-12B-it" and "gemma-4-E4B" are **real Google models**, not internal shorthand for Gemma 3. Google shipped the Gemma 4 family in 2026:
- Gemma 4 family (E2B, E4B, 26B-A4B MoE, 31B dense) released **2026-03-31**.
- Gemma 4 MTP drafters **2026-04-16**.
- **Gemma 4 12B "Unified"** (dense, encoder-free multimodal) released **2026-06-03** - i.e. ~3 weeks ago.
  - https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/
  - https://ai.google.dev/gemma/docs/core/model_card_4

This matters enormously: your two models are **bleeding-edge June-2026 architectures**, and vLLM support for them is **nightly-only and actively buggy**, not the mature path that "Gemma 3 support" would imply. The 12B is the *unified/encoder-free* arch (`Gemma4UnifiedForConditionalGeneration`), distinct from the standard `Gemma4ForCausalLM` used by E2B/E4B/31B.

---

## 1. Architecture support (the gate) - PARTIAL / RISKY

### 12B dense unified (your orchestrator+worker model)
- Registered class: **`Gemma4UnifiedForConditionalGeneration`**. Encoder-free: raw pixel patches + audio frames projected directly into LM space (Dense+LayerNorm, factorized 2D positional embeddings) - a ~35M embedder replacing the ~550M SigLIP tower.
  - https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
- **Support landed in vLLM PR #44429 and has NOT shipped in any stable release** - nightly wheel or `vllm/vllm-openai:latest` Docker only.
  - https://github.com/vllm-project/vllm/pull/44429 (referenced via recipe + issue #44494)
- **Two open, blocking bugs as of mid-2026:**
  - **#44494** ([Bug] Gemma 4 12B is not working): on vLLM 0.21.0, `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2048x4096 and 8192x3840)` in the attention output projection during the dummy-run memory-profiling phase. **Open, no fix.** Implicates `transformers/models/gemma4_unified/modeling_gemma4_unified.py`.
    - https://github.com/vllm-project/vllm/issues/44494
  - **#39216**: vLLM 0.19.0 on PyPI pins `transformers<5`, but Gemma 4 needs `transformers>=5.5.0`. Dependency conflict; must use nightly built against the newer transformers.
    - https://github.com/vllm-project/vllm/issues/39216
- vLLM 0.17.x lacks the gemma4 model entirely; ≥0.19 needed even to begin, and the working path is a nightly ≥ the 0.21 line with the unified class - but #44494 shows 0.21.0 itself still broken. **You would be tracking nightlies and possibly carrying patches.**

### E4B (your distiller model) - supported, but it is the MatFormer/PLE arch
- Registered class: **`Gemma4ForCausalLM`** (example `google/gemma-4-E2B-it`); E4B is the same family. vLLM supported-models page lists Gemma 4 with LoRA + pipeline-parallel ticks.
  - https://docs.vllm.ai/en/stable/models/supported_models/
- E4B uses **Per-Layer Embeddings (PLE)** and the **MatFormer** "effective params" design (elastic E2B↔E4B Mix-n-Match). vLLM serves E4B as a fixed model; I found **no evidence vLLM exposes MatFormer elastic execution / dynamic E2B↔E4B switching** - you get a static E4B, not the slice-on-demand capability. For a distiller that is fine.
  - https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/ (3n MatFormer lineage that 4n/E-series inherits)
- The 3n-era multimodal encoders (`Gemma3nForConditionalGeneration`) depend on `timm>=1.0.17` and are "not yet fully optimized." Text-only use avoids that path.

**Gate verdict:** Both architectures are *nominally* registered, but the **12B unified is nightly-only with an open shape-mismatch bug and a transformers-version conflict**. This is the single biggest risk in the whole migration and it directly blocks your primary (orchestrator+worker) model. Contrast: llama.cpp already runs your GGUFs today.

---

## 2. Batching + prefix caching fit - STRONG in principle, one Gemma-4-specific bug

This is where vLLM is architecturally *better* than your dual-context llama.cpp fork:
- **One engine, one model copy, many concurrent requests.** PagedAttention + continuous batching means orchestrator and worker requests are **co-batched in a single scheduler** - exactly the co-batching your fork's two separate `llama_context` schedulers *cannot* do (your measured 37% aggregate decode loss, 225→142 tok/s). vLLM v1 enables chunked prefill by default and prioritizes decode requests, batching all pending decodes before prefills.
  - https://docs.vllm.ai/en/stable/configuration/optimization/
- **Per-request params coexist natively.** Different `max_tokens`, sampling, and reasoning/thinking budgets per request in the same batch - no manual per-slot KV budgeting. `thinking_token_budget` is a per-request sampling param (for models whose reasoning parser supports it; Gemma 4 has a `--reasoning-parser gemma4`). So "CoT-on orchestrator" vs "CoT-off worker" is a per-request knob, not two servers.
  - https://docs.vllm.ai/en/latest/features/reasoning_outputs/
  - reasoning-parser/tool-parser flags shown in recipe serve cmd below.
- **Automatic Prefix Caching (APC)** caches and reuses the long shared system-prompt prefix across requests automatically - replaces your manual cache-reuse=256 tuning. Hit rate ~91% reported for Gemma 4 without speculative decoding.
  - https://docs.vllm.ai/en/stable/design/prefix_caching/
- **Hybrid KV Cache Manager** handles Gemma's sliding-window + full-attention layers: full-attention layers keep all tokens; SWA layers keep only the last `sliding_window_size`. This is the principled version of your `swa-full=true` + q8_0-KV workaround.
  - https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/
- **Mixed context sizes coexist:** a single `--max-model-len` pool is shared and paged across requests; APC + paging give you mixed long-orchestrator / short-worker requests **without** the per-call budgeting your llama.cpp `kv-unified` shared pool forced (ADR-0012's `per_call_subcall_budget`). vLLM's paged allocator does that division dynamically.

**The one caveat (Gemma-4-specific bug):**
- **#40624** - Gemma4 **0% prefix-cache hits when hybrid-attention + DFlash speculative decoding are combined** (3 KV-cache specs → "block-dropping spiral"). **Open** (April 2026); the existing fix (#33524) only guards 2-group models. Workaround: `--disable-hybrid-kv-cache-manager` (costs memory, needs higher `--max-model-len`). **Without** DFlash/spec-decoding, Gemma 4 prefix caching is fine (~91%). So: **do not enable speculative decoding** with Gemma 4 + hybrid KV until this lands.
  - https://github.com/vllm-project/vllm/issues/40624

**Net:** vLLM's single-engine continuous batching is a genuinely better fit for the concurrent orchestrator+worker fan-out than two non-co-batching `llama_context`s. This is the strongest argument *for* vLLM.

---

## 3. Quantization path off Q4_0 GGUF - RECOMMEND MOVING OFF GGUF

- **GGUF in vLLM is officially experimental and slow.** vLLM docs verbatim: *"GGUF support in vLLM is highly experimental and under-optimized at the moment, it might be incompatible with other features."* Single-file only (use `gguf-split` to merge multi-file). Community measurement: ~93 tok/s in vLLM for GGUF, with the explicit recommendation *"consider llama.cpp instead of vLLM"* for GGUF.
  - https://docs.vllm.ai/en/stable/features/quantization/gguf/
  - https://discuss.vllm.ai/t/gguf-quantized-models-inference-support/234
  - **Running your existing Q4_0 GGUF in vLLM would likely be SLOWER than llama.cpp and may break prefix caching / hybrid-KV feature compat.** GGUF is the wrong format for vLLM.
- **Better path for your SFT model:** re-quantize the SFT weights with **llm-compressor**:
  - **AWQ INT4** - current best-practice INT4 for vLLM; activation-aware, protects salient channels; Marlin-AWQ kernel is the throughput leader. Best for VRAM-constrained INT4.
  - **FP8** - good accuracy/compression on Blackwell; native FP8 GEMM (SM120) exists in vLLM ≥0.17.0. BUT see §4: under WSL2 the 5090's FP8 tensor cores are **not exposed through dxgkrnl**, so FP8 falls back to an emulated/slow path (~45 tok/s, ~3x slower than AWQ). **On WSL2 today, prefer AWQ INT4 over FP8.**
  - Google also ships an **official QAT W4A16** Gemma 4 checkpoint (compressed-tensors, group_size=32) and a `gemma-4-12B-it-qat-q4_0-unquantized` HF repo - the W4A16 QAT is the highest-quality 4-bit option and is the recommended quant in the recipe.
  - https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-scheme/
  - https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized
- **Effort/quality:** re-quantizing an SFT model to AWQ via llm-compressor is a one-time calibration job (hours, needs a calibration set). Quality at INT4-AWQ ≈ your Q4_0 today or better; W4A16 QAT (if Google's checkpoint matches your SFT base) is better still but you'd need to re-apply your SFT on top of the QAT base or QAT-finetune - non-trivial.

---

## 4. Blackwell / RTX 5090 / WSL2 / CUDA - SUPPORTED on recent stack, with sharp edges

- **Works, confirmed late-2025/2026.** A community-validated config (issue #37242):
  - vLLM **0.17.1**, CUDA **12.8** (driver 581.80), **WSL2 2.7.0** (pre-release, Dec 2025), Ubuntu 22.04, kernel 6.6.114.1. **CUDA graphs WORK** (no `--enforce-eager`). ~140 tok/s Qwen3-14B-AWQ vs ~17 tok/s eager (8x) and 26% over Ollama.
    - https://github.com/vllm-project/vllm/issues/37242
  - Also: torch 2.9.0 cu128 working setup. https://discuss.vllm.ai/t/vllm-on-rtx5090-working-gpu-setup-with-torch-2-9-0-cu128/1492
- **Gotchas:**
  - **Pre-WSL2-2.7.0:** WDDM paravirtualization bug → unrecoverable GPU hangs on 5090/RTX PRO 6000; mitigation is `--enforce-eager` (kills throughput, 8x slower). **You need WSL2 ≥2.7.0 to get CUDA graphs.** Your current kernel is `6.6.87.2` - older than the validated `6.6.114.1`; **check/upgrade WSL2 first.**
  - **FP8 on WSL2:** native FP8 tensor cores **not exposed through dxgkrnl** yet → emulated, ~45 tok/s (3x slower than AWQ). **Skip FP8 on WSL2; use AWQ INT4.**
  - WSL2 ceiling ~70% of native-Linux throughput (~180-200 tok/s native).
  - Boot-stability fixes some report: remove Tailscale (interferes w/ CUDA init), mask `nvidia-cdi-refresh`, add GPU-service boot delays.
  - But note: the 12B-unified install needs **nightly cu129/cu130** wheels (see §6), newer than the 0.17.1 in the validated 5090 config - so you'd be combining "bleeding-edge model support" with "bleeding-edge Blackwell support." Compounded risk.
- **Gemma 4 NVFP4 on Blackwell/WSL2** has been demonstrated (third-party): https://allenkuo.medium.com/finishing-what-we-started-gemma-4-nvfp4-on-vllm-desktop-blackwell-wsl2-b2088c34815a

---

## 5. VRAM / KV model - 12B + E4B CO-RESIDENCE IS TIGHT; LIKELY SEPARATE PROCESSES

- vLLM v1 **pre-allocates** a large, **non-reclaimable** KV pool at startup sized by `gpu_memory_utilization` × VRAM (for `max_model_len`). nvidia-smi shows high usage even when idle; memory is never released until shutdown.
  - https://docs.vllm.ai/en/stable/configuration/optimization/
  - https://discuss.vllm.ai/t/vllm-v1-forces-me-to-pre-allocate-a-huge-non-reclaimable-gpu-kv-cache-for-long-contexts-and-none-of-the-current-offload-or-quantization-options-solve-the-resulting-vram-bloat-without-crippling-speed/1502
- **One engine cannot host two different models.** To run 12B and E4B you need **two vLLM processes**, each preallocating its own `gpu_memory_utilization` slice. On 32GB:
  - 12B at INT4-AWQ ≈ ~6.5-7GB weights + KV pool. E4B effective-4B ≈ ~2.5-3GB INT4 + its KV. Both *can* fit, but you must hand-tune `gpu_memory_utilization` per process (e.g. 12B at 0.65-0.70, E4B at 0.15-0.20) so they don't collide - vLLM won't negotiate between processes, and each grabs its share greedily at startup.
  - This **loses the single-weights-copy benefit** entirely for the 12B↔E4B split (they're different models anyway, so that was never shared). The win that matters - co-batching the orchestrator and worker, which are the *same* 12B model - is fully realized in ONE 12B engine. That is the core advantage over your fork.
- **Important nuance vs your fork's goal:** your dual-context fork's whole point was "one weights copy, two contexts" for the *same* 12B serving two roles. **vLLM gives you that for free in a single engine via continuous batching** - you don't need two contexts at all; you submit orchestrator and worker requests to the same engine. So vLLM doesn't just match the fork, it **dissolves the problem the fork was built to solve.**

---

## 6. Migration steps, risks, bottom line

### Migration steps
1. **Upgrade WSL2 to ≥2.7.0** and verify kernel ≥6.6.114 (you're on 6.6.87 - likely pre-CUDA-graph-fix). Confirm CUDA graphs work without `--enforce-eager` on a small model first.
2. **Stand up a single vLLM engine for the 12B** via nightly:
   `uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/cu129 --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match`
   or Docker `vllm/vllm-openai:latest`.
3. **Re-quantize the SFT 12B to AWQ INT4** (llm-compressor) - do NOT load your Q4_0 GGUF into vLLM.
4. Serve (recipe baseline; raise `--max-model-len` for your 32K orchestrator slots, keep APC on, leave spec-decoding OFF):
   `vllm serve <your-awq-12b> --max-model-len 32768 --gpu-memory-utilization 0.68 --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice --async-scheduling` (APC is on by default in v1; `--enable-prefix-caching` if needed).
   - https://recipes.vllm.ai/Google/gemma-4-12B-it
5. **Rewire the harness** so orchestrator and worker hit the SAME engine as per-request calls (CoT via `--reasoning-parser gemma4` + per-request thinking budget; worker = no/low thinking). Drop the manual per-slot KV budgeting (ADR-0012) - let paging handle it.
6. **Second process for E4B distiller** at `--gpu-memory-utilization ~0.18`, INT4-AWQ. Tune both util values so combined preallocation fits 32GB.
7. **Do NOT enable speculative decoding / DFlash** with Gemma 4 + hybrid KV (bug #40624).

### Risks (ranked)
1. **12B-unified is nightly-only + open shape-mismatch bug #44494 + transformers≥5.5 conflict #39216.** This can block you outright today. **Highest risk.** Reproduce a clean 12B load on a nightly BEFORE committing.
2. **Bleeding-edge stack stacking:** nightly vLLM + nightly Blackwell/WSL2 path simultaneously → fragile, frequent breakage, you may carry patches.
3. **GGUF dead-end:** must re-quantize (calibration effort + revalidate SFT quality at AWQ).
4. **Two-process VRAM tuning** is manual and brittle on 32GB; an OOM at startup if util values overlap.
5. **#40624** removes spec-decoding as an option for now.

### Bottom line / recommendation
**Architecturally, vLLM is the RIGHT engine for this workload** - its single-engine continuous batching + PagedAttention + APC + Hybrid-KV directly solves the exact pain your dual-context llama.cpp fork has (two schedulers can't co-batch → 37% loss; manual per-slot KV budgeting; multihop timeouts). One 12B engine serving orchestrator+worker as concurrent per-request calls is cleaner and faster *in principle* than two `llama_context`s, and removes the need for the custom fork.

**BUT the timing is bad.** Your specific model - Gemma 4 **12B Unified**, ~3 weeks old - is **nightly-only and currently broken** in vLLM (#44494 open, transformers pin conflict), on top of an already-bleeding-edge Blackwell/WSL2 path. Migrating now means tracking nightlies and likely patching.

**Recommendation: do NOT rip out llama.cpp yet.** Instead:
- **Spike it (1-2 days):** on a nightly + Docker, attempt a clean 12B-AWQ load and a sustained concurrent orchestrator+worker run. If #44494 reproduces, vLLM is blocked - stay on the fork and re-evaluate in ~4-6 weeks when 12B-unified support stabilizes into a stable release.
- **If the spike loads cleanly:** vLLM very likely beats the dual-context fork on the concurrent RLM pattern (co-batching alone recovers the 37% you lose), and you can retire the custom fork. Validate with a SUSTAINED run (your CLAUDE.md rule), not a burst, and confirm APC hit-rate and no KV-exhaustion under concurrent map-reduce.
- Keep the E4B distiller as a separate small vLLM (or leave it on llama.cpp) - it's not on the hot path.

Net: **vLLM is the better long-term architecture; the blocker is the 3-week-old model's nightly-only/buggy support, not vLLM's batching model.** Gate the migration on a 12B-unified clean-load spike.
