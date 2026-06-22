"""Tests for the pure map-reduce engine (prehend/utils/mapreduce.py).

The engine is pure: all LM I/O is injected via ``run_batch``; ``fits`` and
``compose`` are injectable so structure can be controlled without real token math.
See docs/superpowers/specs/2026-06-22-auto-chunk-enforcement-design.md (source of truth).
"""

import math

from prehend.utils.mapreduce import (
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
    def test_compose_frames_instruction_and_data(self):
        assert _compose("Q?", "DATA", "Text") == "Q?\n\nText:\nDATA"

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
