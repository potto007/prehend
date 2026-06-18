---
status: "accepted"
date: "2026-06-18"
deciders: "potto"
---

# Hard per-generation decode-token ceiling (`max_decode_tokens`)

## Context and Problem Statement

The teacher serves the librarian with a **unified KV cache** (local-ai ADR-0007):
slots share one context pool. A single heavy trajectory that fans out batched
subcalls, or any one generation that decodes too many tokens, can oversubscribe the
pool. When no KV slot is free the decode returns null logits ->
`GGML_ASSERT(logits != nullptr)` -> `ggml_abort` kills the child process (a
non-recoverable crash, not a recoverable 500). How do we keep a single generation
from exhausting the shared pool?

## Considered Options

- **Client-side hard ceiling** on decode tokens per generation (`max_decode_tokens`),
  enforced in the lm-repl handler/RLM/SRLM paths.
- Serving-side only: lower `parallel` / raise ctx headroom.
- Rely on the advisory soft-budget wrap-up message.

## Decision Outcome

Chosen option: a **hard per-generation decode-token ceiling** (`max_decode_tokens`),
enforced in the client (`lm_handler.py` / `rlm.py` / `srlm.py`). Every generation,
including batched subcalls, is bounded below the server's per-slot capacity so no
single trajectory can drive the unified pool to the `failed to find a memory slot`
state. This is the client companion to local-ai ADR-0007 (unified KV serving): the
serving choice is only safe because the client caps decode length.

### Consequences

- Good, because the non-recoverable `ggml_abort` child crash (rlm-trainer #7) is
  prevented at the source rather than recovered after the fact.
- Good, because it composes with the runaway-generation guards (lm-repl ADR-0003).
- Bad, because a too-tight ceiling truncates a legitimately long REDUCE; the ceiling
  must be set against the longest real generation (the ~20K-tok multihop REDUCE).

## More Information

- rlm-trainer #7 (crash signature: `decode: failed to find a memory slot` ->
  `ggml_abort`; recovery runbook).
- local-ai ADR-0007 (unified KV cache — the serving decision this cap makes safe).
- Companion: lm-repl ADR-0003 (subcall caps, soft-budget, contention-retry).
