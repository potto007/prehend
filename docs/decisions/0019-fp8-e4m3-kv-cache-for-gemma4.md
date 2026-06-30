---
status: "accepted"
date: "2026-06-24"
deciders: "potto"
consulted: "debugging session (KV-dtype validation: SGLang scale path + gemma4 norm gammas)"
---

# fp8_e4m3 (not fp8_e5m2) for the gemma4 v13 KV cache

## Context and Problem Statement

The served inference ([ADR-0016](0016-sglang-as-served-inference.md)) runs SGLang with
`--kv-cache-dtype fp8_e4m3`. fp8 KV is mandatory for capacity, not chosen for
accuracy (bf16 weights leave `max_total_num_tokens=194`, a KV-starved hang; fp8
weights + fp8 KV -> ~98k-token pool). INT4 KV was considered as a further
compression but is a dead end: the installed SGLang (0.5.13.post1) exposes no
`int4`/`int8` KV dtype at all (only `auto`, `fp8_e4m3`, `fp8_e5m2`, `bf16`,
`fp4_e2m1`), and we are not capacity-bound, so the iceboxed-TurboQuant conclusion
holds (fp8 Pareto-dominates 4-bit KV for our prefill-heavy, not-capacity-bound
profile).

That leaves one real KV-dtype question: **`fp8_e4m3` vs `fp8_e5m2`?** The two
trade off the same 8 bits differently:

- `e4m3`: 3 mantissa bits, range +/-448. More precision, less range.
- `e5m2`: 2 mantissa bits, range +/-57344. More range, less precision.

The concern: could anything about what Gnosis-MedPolicy-13
(`gemma-4-12B-it-sft-kb-v13`) is SFT'd on push KV activations into a dynamic
range wide enough that e4m3 clips, making e5m2's wider range necessary?

## Decision

**Keep `fp8_e4m3`.** e5m2 is not necessary and would be strictly worse here: it
buys dynamic range the gemma4 architecture makes entirely unused, at the cost of
one mantissa bit of precision on every stored value.

## Rationale (the mechanism, end to end)

1. **No calibrated KV scales in the checkpoint.** SGLang's fp8 KV path applies a
   per-tensor scale before the cast (`k = k / k_scale`), but the scales are loaded
   from the checkpoint; if absent they default to **1.0**
   (`sglang/srt/layers/quantization/kv_cache.py`). The served checkpoint
   (`gemma-4-12B-it-sft-kb-v13-text`) has **0 scale-named tensors out of 666** and
   no `kv_cache_scaling` config field. So the cast is **raw at scale=1.0** -> the
   absolute +/-448 e4m3 ceiling applies directly to the raw cached K/V magnitudes.
   (This is the non-obvious part: the usual "e4m3 wins for KV" wisdom assumes
   calibrated scales; without them, clipping is a real risk worth checking.)

2. **gemma4 bounds cached K/V far below 448, by construction.** SGLang's serving
   path (`sglang/srt/models/gemma4_causal.py:466-475`) applies
   `q_norm`/`k_norm`/`v_norm` before RoPE and the cache:
   - **V**: `v_norm` is `Gemma4RMSNorm(..., with_scale=False)` (weight = ones) ->
     cached V is pure unit-RMS. Hard ceiling sqrt(head_dim)=sqrt(512)=**22.6**;
     realistically < 5.
   - **K**: `k_norm` output = unit-RMS * learned gamma. Read from the checkpoint,
     `|k_norm.weight|` max = **0.1318** (range 0.06-0.13) across all 48 layers, so
     K element <= 0.1318 * 22.6 ~= **3.0**; RoPE preserves the per-pair norm, so the
     bound survives the rotation. Realistically < 1.
   - Attention logits are additionally tanh-softcapped at 50.

   Cached **K ≲ 3** and **V ≲ 23** worst case -> **20-150x headroom** under e4m3's
   448. e4m3 uses essentially none of its range, so its extra mantissa bit is pure
   precision gain with zero clipping risk.

3. **The SFT corpus cannot change this.** v13 is moderate-length (prompts p95
   ~4.7k tok, max ~6.1k; CPT verbosity explicitly removed), but more
   fundamentally the QK/V-norm makes cached-KV dynamic range **independent of the
   text content** - the outlier mechanism (massive activations / attention sinks
   exceeding 448 under a scale=1.0 raw cast) that would justify e5m2 is structurally
   prevented upstream of the cache.

This argument is a hard static bound derived from (a) the checkpoint's own learned
gammas and (b) the definition of RMSNorm, confirmed against SGLang's actual serving
code - stronger than a spot measurement, which could only show the values are
smaller still.

## Consequences

- **Good:** Production stays on `fp8_e4m3` with documented, evidence-backed
  rationale; the KV-dtype question is closed. No accuracy gap is attributable to
  KV-dtype range clipping (the AMBER "validate fp8 KV" item from the 2026-06-24
  serving report is resolved on the range axis).
- **Caveat / when to revisit:** This bound depends on the checkpoint having **no
  calibrated KV scales** and `k_norm` gammas staying small. A future checkpoint
  that (a) ships calibrated kv_scales, or (b) trains k_norm gammas materially
  larger, or a different (non-gemma4) base without `v_norm`/`k_norm`, would need
  re-checking - the raw-cast + unnormalized-V combination is what makes e5m2
  relevant elsewhere.
- **Not addressed here:** e4m3's *precision* effect on long-context retrieval
  accuracy (mantissa, not range) remains the open question; if a KV-precision A/B
  is ever wanted it is `fp8_e4m3` vs `bf16` KV, not vs e5m2.
