"""
Token counting and model context limits for compaction and context sizing.

Uses tiktoken for OpenAI-style models when available; otherwise estimates
with ~4 characters per token.
"""

import math
from typing import Any

# Default context limit when model is unknown (tokens)
DEFAULT_CONTEXT_LIMIT = 128_000

# Characters per token when tokenizer is unavailable (rough average).
CHARS_PER_TOKEN_ESTIMATE = 4

# Conservative chars-per-token for the non-tiktoken fallback path.
# Real (non-OpenAI) models like gemma tokenize DENSER than tiktoken's cl100k for
# structured/code text, so the cl100k fallback (and a 4.0 char/token average)
# can UNDERESTIMATE. An undercount would let an oversized prompt slip past an
# input-size guard.
#
# The estimate is tokens = chars / CONSERVATIVE_CHARS_PER_TOKEN, so to bias
# toward OVER-counting (more tokens = the safe direction for a guard) this
# constant must sit at or BELOW the real density. Measured gemma-4 density on
# the rlm-trainer multihop KB contexts is ~2.07 chars/token (5 tasks, 2.069-
# 2.073); the prior 3.0 UNDER-counted that dense text by ~45%, letting a chunk
# the guard scored at <=20400 est-tokens reach ~29.5k REAL tokens - right at the
# 32768-ctx served window, so any output reservation 400'd it (the 2026-06-24
# GATE #2 oversized-chunk failure). 2.0 sits below the measured floor so the
# estimate over-counts this text and a guard-passing prompt provably fits the
# served window with output headroom.
CONSERVATIVE_CHARS_PER_TOKEN = 2.0

# Substrings that mark a model as clearly NOT OpenAI/tiktoken-compatible.
# For these we prefer the conservative char-based estimate over cl100k, which
# would undercount their denser tokenizers.
_NON_OPENAI_MODEL_MARKERS = ("gemma", "gemini", "qwen", "kimi", "glm", "llama", "mistral")

# Model context limits (max input context in tokens).
# Match: key contained in model_name (e.g. "gpt-4o" matches "@openai/gpt-4o").
# Longest matching key wins.
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # OpenAI (GPT-5: 272k input, 128k reasoning+output)
    "gpt-5-nano": 272_000,
    "gpt-5": 272_000,
    "gpt-4o-mini": 128_000,
    "gpt-4o-2024": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo-preview": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4-32k": 32_768,
    "gpt-4": 8_192,
    "gpt-3.5-turbo-16k": 16_385,
    "gpt-3.5-turbo": 16_385,
    "o1-mini": 128_000,
    "o1-preview": 128_000,
    "o1": 200_000,
    # Anthropic
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-2.1": 200_000,
    "claude-2": 100_000,
    # Gemini
    "gemini-2.5-flash": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
    "gemini-1.5-pro": 1_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-1.0-pro": 30_720,
    # Qwen (Alibaba)
    "qwen3-max": 256_000,
    "qwen3-72b": 128_000,
    "qwen3-32b": 128_000,
    "qwen3-8b": 32_768,
    "qwen3": 128_000,
    # Kimi (Moonshot)
    "kimi-k2.5": 262_000,
    "kimi-k2-0905": 256_000,
    "kimi-k2-thinking": 256_000,
    "kimi-k2": 128_000,
    "kimi": 128_000,
    # GLM (Zhipu)
    "glm-4.6": 200_000,
    "glm-4-9b": 1_000_000,
    "glm-4": 128_000,
    "glm": 128_000,
    # Gemma (Google) - true trained window (262144). Fixes the silent 128000
    # default for e.g. gemma-4-12b-it-sft-kb-v13-sft. "gemma-4" is longer than
    # "gemma" so longest-key-wins picks it for gemma-4 variants; both map to the
    # same honest model property.
    "gemma-4": 262_144,
    "gemma": 262_144,
}


def get_context_limit(model_name: str) -> int:
    """
    Return max context size in tokens for a model.

    Matches when the dict key is contained in model_name (e.g. "gpt-4o" matches
    "@openai/gpt-4o"). Longest matching key wins. Falls back to DEFAULT_CONTEXT_LIMIT
    for unknown models.
    """
    if not model_name or model_name == "unknown":
        return DEFAULT_CONTEXT_LIMIT
    exact = MODEL_CONTEXT_LIMITS.get(model_name)
    if exact is not None:
        return exact
    best = 0
    best_limit = DEFAULT_CONTEXT_LIMIT
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if key in model_name and len(key) > best:
            best = len(key)
            best_limit = limit
    return best_limit


def _count_tokens_tiktoken(messages: list[dict[str, Any]], model_name: str) -> int | None:
    """Count tokens with tiktoken if available. Returns None on failure."""
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None
    total = 0
    # Approximate OpenAI message format overhead per message
    tokens_per_message = 3
    tokens_per_name = 1
    for m in messages:
        total += tokens_per_message
        content = m.get("content")
        if isinstance(content, str):
            total += len(enc.encode(content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(enc.encode(part.get("text", "") or ""))
        elif content is not None and content != "":
            total += len(enc.encode(str(content)))
        if m.get("name"):
            total += tokens_per_name
    return total


def _is_non_openai_model(model_name: str) -> bool:
    """True for models whose tokenizer is denser than cl100k (gemma, etc.)."""
    lowered = model_name.lower()
    return any(marker in lowered for marker in _NON_OPENAI_MODEL_MARKERS)


def count_tokens(messages: list[dict[str, Any]], model_name: str) -> int:
    """
    Count tokens in a list of message dicts (role, content).

    Uses tiktoken for OpenAI-style models when the package is available.
    For clearly-non-OpenAI models (gemma, gemini, qwen, ...) we deliberately
    DO NOT use the cl100k fallback: those tokenizers are denser, so cl100k (and
    a 4.0 char/token average) UNDERCOUNT them, which would let oversized prompts
    slip past an input-size guard. Instead we use a CONSERVATIVE char-based
    estimate (CONSERVATIVE_CHARS_PER_TOKEN ~3.0) that biases toward
    over-counting. tiktoken behavior for real OpenAI models is unchanged.
    """
    if not messages:
        return 0
    use_conservative = _is_non_openai_model(model_name)
    if model_name and model_name != "unknown" and not use_conservative:
        n = _count_tokens_tiktoken(messages, model_name)
        if n is not None:
            return n
    # Char-based estimate (stringify in case content is not str, e.g. list).
    total_chars = 0
    for m in messages:
        raw = m.get("content", "") or ""
        total_chars += len(raw) if isinstance(raw, str) else len(str(raw))
    if use_conservative:
        # Conservative: fewer chars per token => more tokens => over-estimate.
        return math.ceil(total_chars / CONSERVATIVE_CHARS_PER_TOKEN)
    return (total_chars + CHARS_PER_TOKEN_ESTIMATE - 1) // CHARS_PER_TOKEN_ESTIMATE


def resolve_subcall_limit(
    model: str,
    *,
    explicit: int | None = None,
    runtime_ctx: int | None = None,
) -> int:
    """
    Resolve the effective sub-call context limit (in tokens).

    Returns the first non-None of (explicit, runtime_ctx, get_context_limit(model)).
    Pure and never raises.
    """
    if explicit is not None:
        return explicit
    if runtime_ctx is not None:
        return runtime_ctx
    return get_context_limit(model)


def per_call_subcall_budget(pool: int | None, slots: int) -> int | None:
    """Convert a SHARED context pool into a safe PER-CALL sub-call budget.

    Under llama-server ``--kv-unified`` the server's ``n_ctx`` is one KV pool
    shared across all concurrent sequences, not a private window per slot. A
    single task's map-reduce fans out into up to ``slots`` concurrent sub-calls
    (the LocalREPL ThreadPoolExecutor is bounded by ``max_concurrent_subcalls``
    = ``slots``), so budgeting each sub-call against the WHOLE pool lets their
    combined footprint exhaust the cache - the server then logs "failed to find
    free space in the KV cache" and returns "Context size has been exceeded" to
    every concurrent sub-call at once.

    Dividing the pool by ``slots`` bounds the sum of ``slots`` concurrent calls
    to the pool; the guard's existing margin (``oversize_rejection`` reserves
    ~15%) then leaves headroom for the still-resident root orchestrator
    transcript. ``None`` pool means the guard is off and stays off. A
    non-positive ``slots`` is treated as 1 (no division, never divide-by-zero).
    Result is floored and never below 1. Pure.
    """
    if pool is None:
        return None
    slots = max(1, slots)
    return max(1, pool // slots)
