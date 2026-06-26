"""Tests for token counting, context limits, and subcall limit resolution."""

from prehend.utils.token_utils import (
    CONSERVATIVE_CHARS_PER_TOKEN,
    count_tokens,
    get_context_limit,
    per_call_subcall_budget,
    resolve_subcall_limit,
)


class TestPerCallSubcallBudget:
    """The input-size guard limit must be the SHARED kv-unified pool (server
    n_ctx) divided by the number of concurrent sub-calls, not the whole pool.

    Under --kv-unified the server's n_ctx is ONE pool shared across all
    concurrent sequences. A single task's map-reduce fans out into up to
    `slots` concurrent sub-calls, so budgeting each at the full pool lets their
    SUM exhaust the shared cache ("failed to find free space in the KV cache").
    Dividing the pool by `slots` bounds the sum; the guard's own margin then
    reserves headroom for the still-resident root transcript.
    """

    def test_none_pool_stays_off(self):
        # No pool -> guard disabled; never start guarding as a side effect.
        assert per_call_subcall_budget(None, 4) is None

    def test_divides_shared_pool_across_slots(self):
        assert per_call_subcall_budget(98304, 4) == 24576

    def test_single_slot_uses_whole_pool(self):
        assert per_call_subcall_budget(98304, 1) == 98304

    def test_nonpositive_slots_treated_as_one(self):
        # A bad/absent slot count must never divide-by-zero or inflate budget.
        assert per_call_subcall_budget(98304, 0) == 98304
        assert per_call_subcall_budget(98304, -3) == 98304

    def test_floors_and_never_below_one(self):
        assert per_call_subcall_budget(100, 8) == 12   # floor(100 / 8)
        assert per_call_subcall_budget(3, 8) == 1      # floor 0 -> clamped to 1


class TestGetContextLimit:
    """Tests for get_context_limit, especially gemma keys."""

    def test_gemma_4_sft_kb_v13_not_default_128k(self):
        # The v13 sft model name must NOT silently fall back to 128000.
        limit = get_context_limit("gemma-4-12b-it-sft-kb-v13-sft")
        assert limit != 128_000
        assert limit == 262_144

    def test_bare_gemma_key(self):
        assert get_context_limit("gemma") == 262_144

    def test_gemma_4_key(self):
        assert get_context_limit("gemma-4") == 262_144

    def test_unknown_model_still_default(self):
        assert get_context_limit("totally-unknown-model-xyz") == 128_000

    def test_empty_and_unknown_sentinel(self):
        assert get_context_limit("") == 128_000
        assert get_context_limit("unknown") == 128_000

    def test_longest_key_wins_preserved(self):
        # gpt-4o-mini (longer key) must beat gpt-4 / gpt-4o.
        assert get_context_limit("@openai/gpt-4o-mini") == 128_000


class TestCountTokensConservative:
    """count_tokens must NOT undercount for gemma (dense tokenizer)."""

    def test_gemma_does_not_undercount_vs_naive_char4(self):
        # Dense structured text: gemma estimate must be strictly larger than the
        # naive char/4 count (the old undercount path).
        text = "def f(x): return {'a': [1,2,3], 'b': x*x}  # dense_code_$%^&*()"
        text = text * 50
        messages = [{"role": "user", "content": text}]
        naive_char4 = (len(text) + 3) // 4
        est = count_tokens(messages, "gemma-4-12b-it-sft-kb-v13-sft")
        assert est > naive_char4

    def test_gemma_estimate_at_least_conservative_lower_bound(self):
        text = "The quick brown fox jumps over the lazy dog. " * 100
        messages = [{"role": "user", "content": text}]
        est = count_tokens(messages, "gemma-4-12b-it-sft-kb-v13-sft")
        # Conservative lower bound: chars / CONSERVATIVE_CHARS_PER_TOKEN.
        lower = int(len(text) / CONSERVATIVE_CHARS_PER_TOKEN)
        assert est >= lower

    def test_conservative_constant_is_below_average(self):
        # Must over-estimate vs the 4.0 average to bias toward over-counting.
        assert CONSERVATIVE_CHARS_PER_TOKEN < 4.0
        assert CONSERVATIVE_CHARS_PER_TOKEN > 0

    def test_conservative_constant_not_above_real_gemma_density(self):
        # The constant converts a CHAR budget into a token estimate; if it claims
        # MORE chars per token than the real served tokenizer delivers, it
        # UNDERCOUNTS and an oversized prompt slips past the guard to a 400.
        # Measured gemma-4 density on the rlm-trainer multihop KB contexts is
        # 2.069-2.073 chars/token; the conservative constant must sit at/below
        # that floor so the estimate is an over-count (the safe direction).
        MEASURED_GEMMA_MIN_CHARS_PER_TOKEN = 2.069
        assert CONSERVATIVE_CHARS_PER_TOKEN <= MEASURED_GEMMA_MIN_CHARS_PER_TOKEN

    def test_empty_messages_zero(self):
        assert count_tokens([], "gemma-4-12b-it-sft-kb-v13-sft") == 0


class TestResolveSubcallLimit:
    """resolve_subcall_limit precedence: explicit > runtime_ctx > table."""

    def test_explicit_wins(self):
        assert (
            resolve_subcall_limit(
                "gemma-4-12b-it-sft-kb-v13-sft", explicit=98_304, runtime_ctx=50_000
            )
            == 98_304
        )

    def test_runtime_ctx_when_no_explicit(self):
        assert (
            resolve_subcall_limit(
                "gemma-4-12b-it-sft-kb-v13-sft", explicit=None, runtime_ctx=50_000
            )
            == 50_000
        )

    def test_falls_back_to_table_when_all_none(self):
        assert (
            resolve_subcall_limit("gemma-4-12b-it-sft-kb-v13-sft")
            == 262_144
        )

    def test_fallback_uses_get_context_limit_for_unknown(self):
        assert resolve_subcall_limit("unknown-model") == 128_000

    def test_never_raises_on_weird_input(self):
        # Should not raise even with empty model name.
        assert resolve_subcall_limit("") == 128_000
