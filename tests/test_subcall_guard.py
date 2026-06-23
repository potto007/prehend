"""Tests for the oversize sub-call input guard (pure, deterministic)."""

import math

from prehend.utils.subcall_guard import (
    oversize_rejection,
    recommended_chunk_chars,
    safe_chunk_chars,
)

MODEL = "gemma-4-12b-it-sft-kb-v13-sft"


class TestOversizeRejection:
    """oversize_rejection returns None under limit, an actionable string over."""

    def test_under_limit_returns_none(self):
        prompt = "Summarize this short sentence."
        assert oversize_rejection(prompt, limit=98_304, model=MODEL) is None

    def test_over_limit_returns_string(self):
        # ~98304 tokens * ~3 chars/token is ~295K chars; build well over the limit.
        prompt = "x " * 400_000
        rej = oversize_rejection(prompt, limit=98_304, model=MODEL)
        assert rej is not None
        assert isinstance(rej, str)

    def test_rejection_names_limit_and_size(self):
        prompt = "word " * 300_000
        rej = oversize_rejection(prompt, limit=98_304, model=MODEL)
        assert rej is not None
        assert "98304" in rej or "98,304" in rej
        # Must mention the prompt's estimated token size somewhere.
        est = count_estimate(prompt)
        assert str(est) in rej or "token" in rej.lower()

    def test_rejection_mentions_rlm_query_batched_and_chunk(self):
        prompt = "word " * 300_000
        rej = oversize_rejection(prompt, limit=98_304, model=MODEL)
        assert rej is not None
        assert "rlm_query_batched" in rej
        # The hint recommends the SMALLER target chunk (latency lever), not the
        # hard ceiling; it must stay strictly below safe_chunk_chars.
        chars = recommended_chunk_chars(98_304, MODEL)
        assert str(chars) in rej
        assert chars < safe_chunk_chars(98_304, MODEL)
        assert "char" in rej.lower()

    def test_margin_applied_between_safe_and_full(self):
        # A prompt sized between limit*(1-margin) and limit must be REJECTED.
        margin_frac = 0.15
        limit = 98_304
        safe_tokens = math.floor(limit * (1 - margin_frac))
        # Build a prompt whose estimated tokens sits between safe_tokens and limit.
        # Use a target of (safe_tokens + limit)//2 tokens; convert to chars via
        # the conservative chars/token used by the estimator (~3.0).
        target_tokens = (safe_tokens + limit) // 2
        from prehend.utils.token_utils import CONSERVATIVE_CHARS_PER_TOKEN

        target_chars = int(target_tokens * CONSERVATIVE_CHARS_PER_TOKEN) + 10
        prompt = "a" * target_chars
        est = count_estimate(prompt)
        assert safe_tokens < est <= limit  # sanity: in the margin band
        rej = oversize_rejection(prompt, limit=limit, model=MODEL, margin_frac=margin_frac)
        assert rej is not None

    def test_custom_margin_zero_lets_more_through(self):
        prompt = "a" * 200_000
        # With a large margin it is rejected; with zero margin (more headroom for
        # raw prompt) the same prompt may pass if under the raw limit.
        strict = oversize_rejection(prompt, limit=98_304, model=MODEL, margin_frac=0.15)
        loose = oversize_rejection(prompt, limit=98_304, model=MODEL, margin_frac=0.0)
        # Strict should be at least as likely to reject as loose.
        if loose is not None:
            assert strict is not None


class TestSafeChunkChars:
    """safe_chunk_chars is a positive char budget below the raw limit-in-chars."""

    def test_positive(self):
        assert safe_chunk_chars(98_304, MODEL) > 0

    def test_below_raw_limit_in_chars(self):
        # Raw limit in chars (using the average 4 chars/token) is an upper bound
        # the safe chunk must stay under.
        limit = 98_304
        assert safe_chunk_chars(limit, MODEL) < limit * 4

    def test_smaller_with_larger_margin(self):
        a = safe_chunk_chars(98_304, MODEL, margin_frac=0.10)
        b = safe_chunk_chars(98_304, MODEL, margin_frac=0.30)
        assert b < a


class TestRecommendedChunkChars:
    """recommended_chunk_chars is the small, latency-friendly advisory target -
    strictly below the safe_chunk_chars hard ceiling, positive, and scaling."""

    def test_positive(self):
        assert recommended_chunk_chars(98_304, MODEL) > 0

    def test_strictly_below_ceiling(self):
        assert recommended_chunk_chars(98_304, MODEL) < safe_chunk_chars(98_304, MODEL)

    def test_scales_with_limit(self):
        assert recommended_chunk_chars(196_608, MODEL) > recommended_chunk_chars(98_304, MODEL)

    def test_clamped_to_ceiling(self):
        # A frac at/above the ceiling fraction can never exceed safe_chunk_chars.
        assert recommended_chunk_chars(98_304, MODEL, frac=0.99) <= safe_chunk_chars(98_304, MODEL)

    def test_fills_most_of_the_budget(self):
        # After ADR-0012 the limit handed here is already the per-call share
        # (pool // slots), so the recommended chunk must fill a MAJORITY of the
        # safe budget - else map-reduce splits a context into many needless
        # tiny chunks that saturate the slots and inflate latency. Companion to
        # the pool-aware budget: large chunks, few rounds.
        assert recommended_chunk_chars(24_576, MODEL) > 0.5 * safe_chunk_chars(24_576, MODEL)


def count_estimate(prompt: str) -> int:
    """Helper mirroring how the guard counts the prompt's tokens."""
    from prehend.utils.token_utils import count_tokens

    return count_tokens([{"role": "user", "content": prompt}], MODEL)
