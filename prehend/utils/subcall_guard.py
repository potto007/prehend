"""
Deterministic input-size guard for sub-calls (reject-with-hint).

The RLM premise is context-by-reference: large context lives in a REPL variable
and is sliced/queried, never fed whole to a sub-model. When a sub-call prompt
exceeds the sub-model's context window the server 400s ("exceeds available
context size") and the trajectory spins to a hard timeout. This module provides
a PURE, arithmetic guard that the orchestrator can act on: rather than failing
open like an LM verifier, it returns an actionable rejection string telling the
model to chunk the context and map-reduce via rlm_query_batched.

Wording deliberately mirrors the strategy-verifier rejection style
("... rejected this call: <reason>") but is self-contained and actionable.
"""

import math

from prehend.utils.token_utils import (
    CONSERVATIVE_CHARS_PER_TOKEN,
    count_tokens,
)

# Fraction of the sub-model window to TARGET per chunk in guidance (distinct from
# the guard ceiling below). A chunk near the full window prefills slowly and
# serializes the map-reduce; a chunk at ~this fraction lets several chunks fan
# out across the server's parallel slots and each prefills fast. ~0.30 -> a large
# context becomes a handful of chunks instead of 1-2 giant ones. This is advisory
# (speed); it is NOT the hard limit -- a fitting chunk above it is never rejected.
RECOMMENDED_CHUNK_FRAC = 0.30


def safe_chunk_chars(limit: int, model: str, margin_frac: float = 0.15) -> int:
    """
    Return the guard CEILING in CHARACTERS: the max chunk that safely fits one
    sub-call without 400ing the server.

    Derived from the safe token budget (limit minus a margin reserved for the
    system+user prompt envelope and tokenizer skew) converted to chars with the
    conservative chars-per-token. This is the hard reject threshold, NOT the
    recommended chunk size (see recommended_chunk_chars). Pure. Always >= 1.
    """
    safe_tokens = math.floor(limit * (1 - margin_frac))
    chars = int(safe_tokens * CONSERVATIVE_CHARS_PER_TOKEN)
    return max(1, chars)


def recommended_chunk_chars(
    limit: int, model: str, frac: float = RECOMMENDED_CHUNK_FRAC
) -> int:
    """
    Return the RECOMMENDED chunk size in CHARACTERS for prompt/hint guidance.

    Smaller than safe_chunk_chars (the hard ceiling): targeting ~frac of the
    window per chunk keeps each sub-call fast to prefill and lets several chunks
    run in parallel, which is the latency lever for heavy map-reduce tasks. Pure.
    Clamped to [1, safe_chunk_chars] so it can never exceed the hard ceiling.
    """
    target_tokens = math.floor(limit * frac)
    chars = int(target_tokens * CONSERVATIVE_CHARS_PER_TOKEN)
    return max(1, min(chars, safe_chunk_chars(limit, model)))


def oversize_rejection(
    prompt: str,
    *,
    limit: int,
    model: str,
    margin_frac: float = 0.15,
) -> str | None:
    """
    Return None if the prompt fits the safe budget, else an actionable rejection.

    Fits when count_tokens([{user: prompt}], model) <= floor(limit*(1-margin_frac)).
    Otherwise returns a string that (a) names the limit and the prompt's estimated
    token size and (b) instructs the model to split the context into chunks of
    <= K characters and map-reduce via rlm_query_batched.
    """
    est_tokens = count_tokens([{"role": "user", "content": prompt}], model)
    safe_tokens = math.floor(limit * (1 - margin_frac))
    if est_tokens <= safe_tokens:
        return None
    # Recommend the SMALLER target chunk (not the hard ceiling): smaller chunks
    # prefill fast and fan out across slots, which is the latency lever.
    chunk_chars = recommended_chunk_chars(limit, model)
    return (
        f"Sub-call input guard rejected this call: the prompt is ~{est_tokens} "
        f"tokens, which exceeds the safe budget of {safe_tokens} tokens "
        f"(sub-model context limit {limit} tokens, with a {int(margin_frac * 100)}% "
        f"margin reserved for the prompt envelope and tokenizer skew). Do NOT pass "
        f"this much context to a single sub-call. Instead, split the context into "
        f"several chunks of <= {chunk_chars} characters each and map-reduce them via "
        f"rlm_query_batched (smaller chunks run in parallel and are faster), then "
        f"combine the per-chunk results."
    )
