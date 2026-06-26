"""Tests for the pure map-reduce engine (prehend/utils/mapreduce.py).

The engine is pure: all LM I/O is injected via ``run_batch``; ``fits`` and
``compose`` are injectable so structure can be controlled without real token math.
See docs/superpowers/specs/2026-06-22-auto-chunk-enforcement-design.md (source of truth).
"""

import math

from prehend.utils.mapreduce import (
    _EXTRACTION_MAP_INSTRUCTION,
    _MAP_SENTINEL_DIRECTIVE,
    _NO_INFO_SENTINEL,
    MapReduceResult,
    _compose,
    _is_control,
    map_reduce,
)


class RecordingBatch:
    """Fake context-free run_batch: records each batch of prompts, returns responses.

    Default responder returns a distinct marker per prompt. A custom responder
    receives the list of prompts and returns a same-length list of responses.
    """

    def __init__(self, responder=None):
        self.calls: list[list[str]] = []
        self._responder = responder

    def __call__(self, prompts):
        prompts = list(prompts)
        self.calls.append(prompts)
        if self._responder is not None:
            return self._responder(prompts)
        return [f"<ans{i}>" for i in range(len(prompts))]


# Identity-ish compose for structural tests: just the data, ignore instruction/label.
def _data_only(instr, data, label="Text"):
    return data


def _len_fits(limit):
    return lambda t: len(t) <= limit


class TestComposeAndControl:
    def test_compose_frames_data_first_instruction_last(self):
        # Data-first layout: the large, stable data leads; the varying
        # instruction trails. This is what makes the radix prefix cache reuse
        # the chunk across sub-calls (ADR-0017). Instruction-first broke the
        # match at token 0 and caused ~6.4x re-prefill.
        assert _compose("Q?", "DATA", "Text") == "Text:\nDATA\n\nQ?"

    def test_compose_is_prefix_stable_across_instructions(self):
        # The cache invariant: for a FIXED chunk, two different instructions
        # must share a common prefix that already contains the whole chunk, so
        # the server re-prefills only the short trailing instruction.
        import os

        a = _compose("first question?", "STABLE_CHUNK_DATA", "Text")
        b = _compose("a completely different instruction", "STABLE_CHUNK_DATA", "Text")
        common = os.path.commonprefix([a, b])
        assert "STABLE_CHUNK_DATA" in common
        assert common.startswith("Text:\nSTABLE_CHUNK_DATA")

    def test_is_control_detects_guard_prefix(self):
        assert _is_control("Sub-call input guard rejected this call: too big")

    def test_is_control_detects_error_and_budget(self):
        assert _is_control("Error: boom")
        assert _is_control("Error: retrieval budget exhausted - STOP searching")

    def test_is_control_false_for_real_answer(self):
        assert not _is_control("Dave owns a bicycle and a kayak.")


class TestSingleChunk:
    def test_context_fits_single_chunk_one_map_no_reduce(self):
        rb = RecordingBatch()
        res = map_reduce(
            "Q?", "short context", run_batch=rb, fits=_len_fits(10_000),
            chunk_chars=1000,
        )
        assert isinstance(res, MapReduceResult)
        assert res.n_chunks == 1
        assert res.reduce_levels == 0
        assert res.truncated is False
        assert res.answer == "<ans0>"
        assert len(rb.calls) == 1  # one map batch, no reduce

    def test_empty_context_yields_one_empty_chunk(self):
        rb = RecordingBatch()
        res = map_reduce("Q?", "", run_batch=rb, fits=_len_fits(10_000), chunk_chars=1000)
        assert res.n_chunks == 1
        assert len(rb.calls) == 1
        assert len(rb.calls[0]) == 1


class TestSplitting:
    def test_oversized_context_splits_into_expected_chunk_count(self):
        rb = RecordingBatch()
        ctx = "x" * 2500
        res = map_reduce(
            "Q?", ctx, run_batch=rb, fits=_len_fits(10_000), chunk_chars=1000,
        )
        assert res.n_chunks == math.ceil(2500 / 1000) == 3
        assert len(rb.calls[0]) == 3  # map batch has one prompt per chunk

    def test_map_runs_prompt_over_each_chunk(self):
        rb = RecordingBatch()
        ctx = "A" * 1000 + "B" * 1000 + "C" * 500
        map_reduce("FINDIT", ctx, run_batch=rb, fits=_len_fits(10_000), chunk_chars=1000)
        map_prompts = rb.calls[0]
        assert all("FINDIT" in p for p in map_prompts)
        assert "A" * 1000 in map_prompts[0]
        assert "B" * 1000 in map_prompts[1]
        assert "C" * 500 in map_prompts[2]

    def test_boundary_chunk_counts(self):
        for n_chars, expected in [(0, 1), (1, 1), (1000, 1), (1001, 2)]:
            rb = RecordingBatch()
            res = map_reduce(
                "Q?", "z" * n_chars, run_batch=rb, fits=_len_fits(10_000),
                chunk_chars=1000,
            )
            assert res.n_chunks == expected, f"{n_chars} chars -> {expected} chunks"


class TestReduce:
    def test_reduce_combines_partials_one_level(self):
        # 3 chunks -> 3 partials; fits is generous so all collapse in one group.
        rb = RecordingBatch(lambda prompts: [f"part{i}" for i in range(len(prompts))]
                            if len(prompts) > 1 else ["FINAL"])
        res = map_reduce(
            "Q?", "y" * 3000, run_batch=rb, fits=_len_fits(10_000),
            chunk_chars=1000, compose=_data_only,
        )
        assert res.reduce_levels == 1
        assert res.answer == "FINAL"
        # reduce batch (calls[1]) had a single grouped prompt containing all partials.
        assert len(rb.calls) == 2
        assert len(rb.calls[1]) == 1
        assert "part0" in rb.calls[1][0] and "part2" in rb.calls[1][0]

    def test_reduce_uses_reduce_prompt_when_given(self):
        rb = RecordingBatch(lambda prompts: ["P"] * len(prompts) if len(prompts) > 1
                            else ["DONE"])
        map_reduce(
            "MAPQ", "y" * 2000, run_batch=rb, fits=_len_fits(10_000),
            chunk_chars=1000, reduce_prompt="COMBINE_NOW",
        )
        # map prompts use MAPQ; reduce prompt uses COMBINE_NOW.
        assert all("MAPQ" in p for p in rb.calls[0])
        assert "COMBINE_NOW" in rb.calls[1][0]

    def test_reduce_defaults_to_prompt(self):
        rb = RecordingBatch(lambda prompts: ["P"] * len(prompts) if len(prompts) > 1
                            else ["DONE"])
        map_reduce("ONLYQ", "y" * 2000, run_batch=rb, fits=_len_fits(10_000),
                   chunk_chars=1000)
        assert "ONLYQ" in rb.calls[1][0]

    def test_tree_reduce_multiple_levels(self):
        # 4 partials, fits allows at most 2 per group -> level0: 2 groups,
        # level1: 1 group -> reduce_levels == 2.
        def responder(prompts):
            return [f"r{i}" for i in range(len(prompts))]
        rb = RecordingBatch(responder)
        # fits permits joining 2 of the small "r#"/data tokens but not 3.
        # With _data_only compose and chunks of 200 chars each: partials are "r0".."r3".
        res = map_reduce(
            "Q?", "z" * 800, run_batch=rb, fits=lambda t: t.count("\n\n") <= 1,
            chunk_chars=200, compose=_data_only,
        )
        assert res.n_chunks == 4
        assert res.reduce_levels == 2
        # level0 reduce batch produced 2 groups, level1 produced 1.
        assert len(rb.calls[1]) == 2
        assert len(rb.calls[2]) == 1

    def test_reduce_levels_zero_for_single_chunk(self):
        rb = RecordingBatch()
        res = map_reduce("Q?", "tiny", run_batch=rb, fits=_len_fits(10_000),
                         chunk_chars=1000)
        assert res.reduce_levels == 0


class TestTruncation:
    def test_max_reduce_depth_truncates_and_warns(self):
        # 4 partials, fits allows 2 per group, max_reduce_depth=1.
        # level0 groups -> 2 partials, level=1 >= max -> truncate (one final reduce).
        rb = RecordingBatch(lambda prompts: [f"r{i}" for i in range(len(prompts))])
        res = map_reduce(
            "Q?", "z" * 800, run_batch=rb, fits=lambda t: t.count("\n\n") <= 1,
            chunk_chars=200, compose=_data_only, max_reduce_depth=1,
        )
        assert res.truncated is True
        assert res.reduce_levels == 2  # max_reduce_depth + 1
        # final reduce is a single batch of one prompt at the bound.
        assert len(rb.calls[-1]) == 1
        assert "[note: reduce truncated at max depth" in rb.calls[-1][0]


class TestHardCut:
    def test_single_partial_too_large_is_hardcut_before_reduce(self):
        # 3 chunks -> 3 partials; one is larger than chunk_chars and must be cut
        # to chunk_chars before being composed into the reduce group.
        big = "Q" * 1500
        rb = RecordingBatch(lambda prompts: [big, "s1", "s2"] if len(prompts) == 3
                            else ["FINAL"])
        map_reduce(
            "Q?", "z" * 3000, run_batch=rb, fits=_len_fits(100_000),
            chunk_chars=1000, compose=_data_only,
        )
        reduce_input = rb.calls[1][0]
        # the 1500-char partial appears cut to 1000 chars, not in full.
        assert big not in reduce_input
        assert "Q" * 1000 in reduce_input


class TestControlFiltering:
    def test_control_string_partial_is_filtered(self):
        rb = RecordingBatch(lambda prompts: ["Error: boom", "real answer"]
                            if len(prompts) == 2 else ["FINAL"])
        res = map_reduce(
            "Q?", "z" * 2000, run_batch=rb, fits=_len_fits(10_000),
            chunk_chars=1000, compose=_data_only,
        )
        assert res.dropped == 1
        # one real partial remains -> returned directly, no reduce needed.
        assert res.answer == "real answer"

    def test_all_partials_control_returns_first_verbatim(self):
        rb = RecordingBatch(lambda prompts: ["Error: a", "Error: b"])
        res = map_reduce(
            "Q?", "z" * 2000, run_batch=rb, fits=_len_fits(10_000),
            chunk_chars=1000, compose=_data_only,
        )
        assert res.answer == "Error: a"
        assert res.reduce_levels == 0
        assert len(rb.calls) == 1  # no reduce batch

    def test_budget_message_stops_and_flags(self):
        budget = ("Error: retrieval budget exhausted - STOP searching")
        rb = RecordingBatch(lambda prompts: ["real", budget])
        res = map_reduce(
            "Q?", "z" * 2000, run_batch=rb, fits=_len_fits(10_000),
            chunk_chars=1000, compose=_data_only,
        )
        assert res.budget_exhausted is True
        assert res.answer == "real"


class TestOverlap:
    def test_overlap_chunks_share_boundary_text(self):
        # With overlap, consecutive chunks share `overlap_chars` of text so a
        # multi-hop link straddling a boundary survives in both chunks.
        rb = RecordingBatch()
        ctx = "abcdefghij" * 300  # 3000 chars
        map_reduce("Q?", ctx, run_batch=rb, fits=_len_fits(100_000),
                   chunk_chars=1000, overlap_chars=200, compose=_data_only)
        chunks = rb.calls[0]  # _data_only -> map prompts ARE the chunks
        assert len(chunks) >= 2
        # tail of chunk i equals head of chunk i+1
        assert chunks[0][-200:] == chunks[1][:200]

    def test_overlap_increases_chunk_count_via_smaller_step(self):
        rb = RecordingBatch()
        ctx = "z" * 3000
        map_reduce("Q?", ctx, run_batch=rb, fits=_len_fits(100_000),
                   chunk_chars=1000, overlap_chars=200, compose=_data_only)
        # step = chunk_chars - overlap = 800 -> starts at 0,800,1600,2400 = 4 chunks
        assert len(rb.calls[0]) == 4

    def test_overlap_zero_is_contiguous(self):
        rb = RecordingBatch()
        ctx = "z" * 3000
        map_reduce("Q?", ctx, run_batch=rb, fits=_len_fits(100_000),
                   chunk_chars=1000, overlap_chars=0, compose=_data_only)
        assert len(rb.calls[0]) == 3  # contiguous: 0,1000,2000

    def test_overlap_clamped_below_chunk_chars_terminates(self):
        # overlap >= chunk_chars would make step <= 0 (infinite); must be clamped.
        rb = RecordingBatch()
        ctx = "z" * 3000
        res = map_reduce("Q?", ctx, run_batch=rb, fits=_len_fits(100_000),
                         chunk_chars=1000, overlap_chars=5000, compose=_data_only)
        assert res.n_chunks >= 1  # finite, did not hang


class TestOrder:
    def test_run_batch_order_preserved_into_reduce(self):
        # distinct chunk contents -> partials echo chunk first char; reduce must
        # see them in chunk order.
        def responder(prompts):
            if len(prompts) > 1:
                return [p[0] for p in prompts]  # first char of each chunk
            return ["FINAL"]
        rb = RecordingBatch(responder)
        ctx = "A" * 1000 + "B" * 1000 + "C" * 1000
        map_reduce("Q?", ctx, run_batch=rb, fits=_len_fits(10_000),
                   chunk_chars=1000, compose=_data_only)
        reduce_input = rb.calls[1][0]
        assert reduce_input.index("A") < reduce_input.index("B") < reduce_input.index("C")


class TestNoInfoSentinel:
    """MAP chunks with nothing relevant return a droppable NO_RELEVANT_INFO
    sentinel instead of a verbose 'no information' answer. Without this, on a
    multihop task the one chunk that holds the answer is statistically outvoted
    by the many 'no info' partials and the reduce loses it (the diagnosed
    multihop_053 reduce-loss: 1 correct partial vs 15 noise partials)."""

    def test_is_control_drops_sentinel(self):
        assert _is_control(_NO_INFO_SENTINEL)
        assert _is_control("  " + _NO_INFO_SENTINEL + "  ")
        assert _is_control(_NO_INFO_SENTINEL.lower())

    def test_is_control_false_for_real_answer_mentioning_info(self):
        # A genuine answer that merely contains the word "information" is NOT a
        # sentinel; only the explicit token is dropped.
        assert not _is_control("The relevant information is that Alice owns a key.")

    def test_map_instruction_carries_sentinel_directive(self):
        # The per-chunk MAP prompts must instruct the model to emit the sentinel;
        # the REDUCE prompt must NOT (a reduce that finds nothing is a real
        # answer, not a droppable chunk).
        rb = RecordingBatch(lambda prompts: (
            [_NO_INFO_SENTINEL] * len(prompts) if len(prompts) > 1 else ["FINAL"]
        ))
        ctx = "A" * 1000 + "B" * 1000 + "C" * 1000
        map_reduce("What does Alice own?", ctx, run_batch=rb,
                   fits=_len_fits(10_000), chunk_chars=1000)
        map_prompts = rb.calls[0]
        assert all(_MAP_SENTINEL_DIRECTIVE in p for p in map_prompts)

    def test_sentinel_partials_excluded_from_reduce(self):
        # Multihop: chunk 0 holds hop-1, chunk 2 holds hop-2, chunk 1 is noise
        # and returns the sentinel. The reduce must see ONLY the two real hops.
        def responder(prompts):
            if _MAP_SENTINEL_DIRECTIVE in prompts[0]:  # MAP step
                out = []
                for p in prompts:
                    if "ALICE_CHICAGO" in p:
                        out.append("Alice moved to Chicago")
                    elif "CHICAGO_KEY" in p:
                        out.append("The Chicago person owns a golden key")
                    else:
                        out.append(_NO_INFO_SENTINEL)
                return out
            return ["a golden key"]  # REDUCE step
        rb = RecordingBatch(responder)
        ctx = ("ALICE_CHICAGO" + "x" * 987
               + "noisenoise" + "y" * 990
               + "CHICAGO_KEY" + "z" * 989)
        res = map_reduce("What does Alice own?", ctx, run_batch=rb,
                         fits=_len_fits(100_000), chunk_chars=1000)
        assert len(rb.calls) >= 2
        reduce_input = "\n".join(rb.calls[1])
        assert "Alice moved to Chicago" in reduce_input
        assert "The Chicago person owns a golden key" in reduce_input
        assert _NO_INFO_SENTINEL not in reduce_input
        assert res.dropped == 1
        assert res.answer == "a golden key"

    def test_all_sentinel_returns_clean_message_not_raw_token(self):
        # Every chunk irrelevant -> all sentinels -> the answer is a readable
        # 'no info' message, never the bare NO_RELEVANT_INFO token.
        rb = RecordingBatch(lambda prompts: [_NO_INFO_SENTINEL] * len(prompts))
        ctx = "A" * 1000 + "B" * 1000 + "C" * 1000
        res = map_reduce("What does Alice own?", ctx, run_batch=rb,
                         fits=_len_fits(10_000), chunk_chars=1000)
        assert _NO_INFO_SENTINEL not in res.answer
        assert "no relevant information" in res.answer.lower()


class TestExtractionMap:
    """Extraction-map mode (multihop CHAINING fix). The legacy per-query MAP uses
    the user query as the per-chunk instruction, so the model judges a background
    hop ("Alice moved to Chicago") irrelevant to "what does Alice own?" and drops
    it - the Alice->Chicago->key chain is never connected (diagnosed multihop_053).
    Extraction mode makes the MAP query-INDEPENDENT (extract every fact about every
    named entity) so all hops survive to a global REDUCE, where the user query
    drives the actual chaining."""

    def test_extraction_map_uses_query_independent_map_instruction(self):
        # The per-chunk MAP must NOT contain the user query (so the model has no
        # basis to filter a background hop), and MUST carry the fixed extraction
        # instruction instead.
        rb = RecordingBatch(lambda prompts: (
            [_NO_INFO_SENTINEL] * len(prompts) if len(prompts) > 1 else ["FINAL"]
        ))
        ctx = "A" * 1000 + "B" * 1000 + "C" * 1000
        map_reduce("WHAT_DOES_ALICE_OWN_UNIQUEQ", ctx, run_batch=rb,
                   fits=_len_fits(10_000), chunk_chars=1000, extraction_map=True)
        map_prompts = rb.calls[0]
        assert all(_EXTRACTION_MAP_INSTRUCTION in p for p in map_prompts)
        assert all("WHAT_DOES_ALICE_OWN_UNIQUEQ" not in p for p in map_prompts)
        # The legacy per-query sentinel directive must NOT be used in this mode.
        assert all(_MAP_SENTINEL_DIRECTIVE not in p for p in map_prompts)

    def test_extraction_map_applies_user_query_at_reduce(self):
        # The user query drives the REDUCE, where the extracted facts are chained
        # into an answer.
        def responder(prompts):
            if _EXTRACTION_MAP_INSTRUCTION in prompts[0]:
                return ["fact A", "fact B", "fact C"]
            return ["CHAINED"]
        rb = RecordingBatch(responder)
        ctx = "A" * 1000 + "B" * 1000 + "C" * 1000
        map_reduce("WHAT_DOES_ALICE_OWN_UNIQUEQ", ctx, run_batch=rb,
                   fits=_len_fits(100_000), chunk_chars=1000, extraction_map=True)
        reduce_input = "\n".join(rb.calls[1])
        assert "WHAT_DOES_ALICE_OWN_UNIQUEQ" in reduce_input

    def test_extraction_map_off_by_default_keeps_per_query_map(self):
        # Backward-compat: without the flag the legacy per-query map (query +
        # sentinel directive) is unchanged.
        rb = RecordingBatch(lambda prompts: (
            [_NO_INFO_SENTINEL] * len(prompts) if len(prompts) > 1 else ["FINAL"]
        ))
        ctx = "A" * 1000 + "B" * 1000 + "C" * 1000
        map_reduce("WHAT_DOES_ALICE_OWN_UNIQUEQ", ctx, run_batch=rb,
                   fits=_len_fits(10_000), chunk_chars=1000)
        map_prompts = rb.calls[0]
        assert all("WHAT_DOES_ALICE_OWN_UNIQUEQ" in p for p in map_prompts)
        assert all(_MAP_SENTINEL_DIRECTIVE in p for p in map_prompts)
        assert all(_EXTRACTION_MAP_INSTRUCTION not in p for p in map_prompts)

    def test_extraction_map_preserves_background_hop_and_chains(self):
        # The headline behavior. Because the MAP is query-independent, the chunk
        # holding the BACKGROUND hop ("Alice lives in Chicago") extracts it rather
        # than dropping it as irrelevant to "owns". The REDUCE then chains
        # Alice->Chicago->golden key. Models the multihop_053 chain.
        def responder(prompts):
            if _EXTRACTION_MAP_INSTRUCTION in prompts[0]:  # MAP step
                out = []
                for p in prompts:
                    if "ALICE_CHICAGO" in p:
                        out.append("Alice lives in Chicago.")
                    elif "CHICAGO_KEY" in p:
                        out.append("The person who lives in Chicago owns a golden key.")
                    else:
                        out.append(_NO_INFO_SENTINEL)
                return out
            return ["Alice owns a golden key."]  # REDUCE chains the two hops
        rb = RecordingBatch(responder)
        ctx = ("ALICE_CHICAGO" + "x" * 987
               + "fillerfiller" + "y" * 988
               + "CHICAGO_KEY" + "z" * 989)
        res = map_reduce("What does Alice own?", ctx, run_batch=rb,
                         fits=_len_fits(100_000), chunk_chars=1000,
                         extraction_map=True)
        reduce_input = "\n".join(rb.calls[1])
        assert "Alice lives in Chicago." in reduce_input
        assert "The person who lives in Chicago owns a golden key." in reduce_input
        assert _NO_INFO_SENTINEL not in reduce_input
        assert res.dropped == 1  # the filler chunk
        assert res.answer == "Alice owns a golden key."

    def test_extraction_map_all_filler_returns_clean_message(self):
        # Extraction mode still emits the sentinel for entity-free filler chunks,
        # so an all-filler context yields the clean 'no info' message.
        rb = RecordingBatch(lambda prompts: [_NO_INFO_SENTINEL] * len(prompts))
        ctx = "A" * 1000 + "B" * 1000 + "C" * 1000
        res = map_reduce("What does Alice own?", ctx, run_batch=rb,
                         fits=_len_fits(10_000), chunk_chars=1000,
                         extraction_map=True)
        assert _NO_INFO_SENTINEL not in res.answer
        assert "no relevant information" in res.answer.lower()


def _fact_or_answer(prompts):
    """Responder: MAP prompts (carry the extraction instruction) return a fact;
    everything else (REDUCE) returns a single converging answer."""
    return ["FACT" if _EXTRACTION_MAP_INSTRUCTION in p else "ANSWER" for p in prompts]


def _count_map_sends(rb):
    """Count extraction-MAP prompts sent across every recorded batch."""
    return sum(1 for batch in rb.calls for p in batch if _EXTRACTION_MAP_INSTRUCTION in p)


class TestMapCacheMemoization:
    """Re-scan fix: the orchestrator re-issues llm_query(context=BIG) across
    iterations, and each call re-ran the full map-reduce (live evidence:
    ~8-9x re-prefill of a 150k-token context, 42% radix reuse). Because the
    extraction-MAP is query-INDEPENDENT (ADR-0018), its per-chunk output depends
    only on (context, chunk_chars, overlap), so a caller-owned ``map_cache`` lets
    repeated same-context calls reuse the MAP and re-run only the cheap REDUCE."""

    def test_extraction_map_memoized_across_calls_with_shared_cache(self):
        rb = RecordingBatch(_fact_or_answer)
        ctx = "A" * 1000 + "B" * 1000  # 2 chunks at chunk_chars=1000
        cache = {}
        map_reduce("QUERY_ONE", ctx, run_batch=rb, fits=_len_fits(100_000),
                   chunk_chars=1000, extraction_map=True, map_cache=cache)
        assert _count_map_sends(rb) == 2  # both chunks mapped on the first call
        map_reduce("A_DIFFERENT_QUERY_TWO", ctx, run_batch=rb, fits=_len_fits(100_000),
                   chunk_chars=1000, extraction_map=True, map_cache=cache)
        # Second call adds ZERO new MAP sends: it reused the cached partials.
        assert _count_map_sends(rb) == 2

    def test_no_map_cache_reruns_map_each_call(self):
        rb = RecordingBatch(_fact_or_answer)
        ctx = "A" * 1000 + "B" * 1000
        for q in ("Q1", "Q2"):
            map_reduce(q, ctx, run_batch=rb, fits=_len_fits(100_000),
                       chunk_chars=1000, extraction_map=True)  # no cache passed
        assert _count_map_sends(rb) == 4  # 2 chunks x 2 calls (opt-in only)

    def test_legacy_map_not_memoized_even_with_cache(self):
        # Legacy (non-extraction) MAP instruction IS the user query, so caching
        # across different queries would be WRONG; the cache must be ignored.
        rb = RecordingBatch(_fact_or_answer)
        ctx = "A" * 1000 + "B" * 1000
        cache = {}
        for q in ("Q1", "Q2"):
            map_reduce(q, ctx, run_batch=rb, fits=_len_fits(100_000),
                       chunk_chars=1000, extraction_map=False, map_cache=cache)
        legacy_map_sends = sum(
            1 for batch in rb.calls for p in batch if _MAP_SENTINEL_DIRECTIVE in p
        )
        assert legacy_map_sends == 4  # both calls re-map

    def test_map_cache_miss_on_different_context(self):
        rb = RecordingBatch(_fact_or_answer)
        cache = {}
        map_reduce("Q", "A" * 1000 + "B" * 1000, run_batch=rb, fits=_len_fits(100_000),
                   chunk_chars=1000, extraction_map=True, map_cache=cache)
        map_reduce("Q", "C" * 1000 + "D" * 1000, run_batch=rb, fits=_len_fits(100_000),
                   chunk_chars=1000, extraction_map=True, map_cache=cache)
        assert _count_map_sends(rb) == 4  # different context -> re-map

    def test_cached_map_still_applies_new_query_at_reduce(self):
        def responder(prompts):
            if _EXTRACTION_MAP_INSTRUCTION in prompts[0]:
                return ["fact A", "fact B"]
            return ["CHAINED"]
        rb = RecordingBatch(responder)
        ctx = "A" * 1000 + "B" * 1000
        cache = {}
        map_reduce("FIRSTQ", ctx, run_batch=rb, fits=_len_fits(100_000),
                   chunk_chars=1000, extraction_map=True, map_cache=cache)
        res = map_reduce("SECOND_UNIQUE_QUERY", ctx, run_batch=rb, fits=_len_fits(100_000),
                         chunk_chars=1000, extraction_map=True, map_cache=cache)
        reduce_prompts = [p for batch in rb.calls for p in batch
                          if _EXTRACTION_MAP_INSTRUCTION not in p]
        # the reused-MAP second call still drives the REDUCE with the NEW query
        assert any("SECOND_UNIQUE_QUERY" in p for p in reduce_prompts)
        assert res.answer  # non-empty
