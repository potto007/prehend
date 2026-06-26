---
status: "accepted"
date: "2026-06-24"
deciders: "potto"
consulted: "debugging session (sglang prefix-cache root-cause)"
---

# Data-first sub-call layout: stable context leads, instruction trails

## Context and Problem Statement

The served solver ([ADR-0016](0016-sglang-as-served-solver.md), SGLang on `:8080`)
reuses a request's leading tokens via RadixAttention/prefix caching: a new
request that shares a byte-identical PREFIX with a cached one re-prefills only
the diverging suffix. The RLM multihop workload re-queries the SAME large context
(or the same chunks of it) across hops and map-reduce sub-calls, so prefix reuse
on those chunks is the dominant latency lever.

Measured on the plain-multihop bench (hybrid SWA pool, fp8 KV, default config):
**2.73M tokens prefilled vs ~423k unique context tokens = ~6.4x re-prefilling**;
58% of prefill batches were ZERO-cached (`#cached-token: 0`) with the SWA pool
showing essentially no eviction. Zero-from-token-0 + no eviction is the signature
of **prefix misalignment**, not pool pressure.

Root cause: the harness composed every sub-call as
`{instruction}\n\n{label}:\n{data}` (`prehend/utils/mapreduce.py::_compose`, and
the `context=` path in `local_repl.py`), and the system-prompt few-shot examples
taught the same instruction-first manual layout (`f"...question? Here is the
chunk: {chunk}"`). Because the per-hop instruction VARIES, the request prefix
diverges at token 0 and the large identical chunk that follows is recomputed every
query, even though it is bit-identical to a chunk already in the cache.

(Note: this was discovered while investigating `--disable-hybrid-swa-memory`,
which is a **dead end for gemma-4** - a `hybrid_swa_compress` model whose sliding
layers are 4x wider than its full layers (8x256 vs 1x512), so a single uniform
non-hybrid pool cannot represent both widths and `store_cache` aborts cuda-graph
capture with a batch-size shape mismatch. The hybrid SWA pool is retained.)

## Decision

**Compose every sub-call DATA-FIRST: the large, stable data leads and the varying
instruction trails** - `{label}:\n{data}\n\n{instruction}`. Applied in two places:

1. **The harness seam** - `_compose` (the `context=` / map-reduce path, the
   dominant big-context route).
2. **The system-prompt few-shot examples** (`prehend/utils/prompts.py`) - rewritten
   so model-authored manual-slice sub-calls also lead with the chunk, plus an
   explicit "Cache-friendly sub-call layout" principle telling the orchestrator to
   put context first and its question last.

The cache invariant is unit-tested (`tests/test_mapreduce.py`): for a fixed chunk,
two different instructions must share a common prefix that already contains the
whole chunk.

## Consequences

- **Good:** identical chunks re-prefill once instead of every query; expected to
  collapse the ~6.4x re-prefill toward ~1x on the multihop workload, on the
  throughput-safe hybrid pool (cuda-graph capture works at a real `cuda-graph-max-bs`).
- **Risk / validation pending:** this is a prompt-FORMAT change to the trained
  Gnosis solver (instruction position moved). Correctness is gated on a GATE #2
  accuracy A/B (data-first vs instruction-first) on plain-multihop tasks; the
  layout is only "done" once reuse is measured UP and accuracy is measured NEUTRAL.
  Until then this ADR is accepted on the root-cause reasoning (instruction-first
  provably breaks token-0 matching) but the empirical magnitude is unconfirmed.
- Reduce-step partials (small, varying) gain little from this and are unaffected in
  spirit; the change is harmless there.
