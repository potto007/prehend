"""Seam-level tests for auto-chunk enforcement via context= (ADR-0010).

These cover the LocalREPL dispatch: context= that fits is inlined; context= that
exceeds the recommended size is map-reduced through the context-free send helpers;
the no-context path is byte-for-byte unchanged. No sockets: send_lm_request /
send_lm_request_batched are patched, and the rlm path uses a fake subcall_fn.
See docs/superpowers/specs/2026-06-22-auto-chunk-enforcement-design.md.
"""

from types import SimpleNamespace
from unittest.mock import patch

from prehend.core.types import RLMChatCompletion, UsageSummary
from prehend.environments.local_repl import LocalREPL
from prehend.utils.subcall_guard import recommended_chunk_chars

MODEL = "gemma-4-12b-it-sft-kb-v13-sft"
R = recommended_chunk_chars(98304, MODEL)  # ~88473 chars


def _ok(_addr, req):
    return SimpleNamespace(
        success=True,
        chat_completion=SimpleNamespace(response="DOC TEXT", _req=req),
    )


def _ok_batched(_addr, prompts, **kw):
    return [
        SimpleNamespace(success=True, chat_completion=SimpleNamespace(response=f"R{i}"))
        for i, _ in enumerate(prompts)
    ]


def _env(**kw):
    return LocalREPL(lm_handler_address=("127.0.0.1", 1), model_name=MODEL, **kw)


class TestLLMQueryContextDispatch:
    def test_small_context_inlined_single_call(self):
        env = _env(subcall_context_limit=98304)
        with patch("prehend.environments.local_repl.send_lm_request",
                   side_effect=_ok) as send, \
             patch("prehend.environments.local_repl.send_lm_request_batched",
                   side_effect=_ok_batched) as sendb:
            result = env._llm_query("Which items?", context="a small context")
        assert result == "DOC TEXT"
        assert send.call_count == 1          # one inline send
        assert sendb.call_count == 0         # no map-reduce
        # the composed prompt carried both the instruction and the context
        sent_prompt = send.call_args[0][1].prompt
        assert "Which items?" in sent_prompt and "a small context" in sent_prompt

    def test_midband_context_triggers_mapreduce(self):
        # composed length is in the R..ceiling band: it FITS the window but
        # prefills slowly, so it must map-reduce, not inline (latency band).
        env = _env(subcall_context_limit=98304)
        big = "x" * 150_000  # > R (~88K) but < safe ceiling (~250K)
        with patch("prehend.environments.local_repl.send_lm_request",
                   side_effect=_ok) as send, \
             patch("prehend.environments.local_repl.send_lm_request_batched",
                   side_effect=_ok_batched) as sendb:
            result = env._llm_query("find", context=big)
        assert send.call_count == 0          # never an inline single send
        assert sendb.call_count == 2         # map batch + reduce batch
        assert result == "R0"

    def test_no_context_bare_oversized_still_reject_with_hint(self):
        env = _env(subcall_context_limit=98304)
        with patch("prehend.environments.local_repl.send_lm_request",
                   side_effect=_ok) as send:
            result = env._llm_query("word " * 300_000)
        assert send.call_count == 0
        assert "rlm_query_batched" in result

    def test_context_coerced_to_str(self):
        env = _env(subcall_context_limit=98304)
        with patch("prehend.environments.local_repl.send_lm_request",
                   side_effect=_ok) as send:
            env._llm_query("q", context=12345)
        assert "12345" in send.call_args[0][1].prompt

    def test_priority_preserved_and_forwarded(self):
        env = _env(subcall_context_limit=98304)
        with patch("prehend.environments.local_repl.send_lm_request",
                   side_effect=_ok) as send:
            env._llm_query("q", priority="high")
        assert send.call_args[0][1].priority == "high"

    def test_huge_instruction_skips_mapreduce_and_rejects(self):
        # instruction alone exceeds R -> chunk budget <= 0 -> no map-reduce; the
        # composed prompt is oversized and the inner guard reject-with-hints it.
        env = _env(subcall_context_limit=98304)
        with patch("prehend.environments.local_repl.send_lm_request",
                   side_effect=_ok) as send, \
             patch("prehend.environments.local_repl.send_lm_request_batched",
                   side_effect=_ok_batched) as sendb:
            result = env._llm_query("word " * 300_000, context="tiny")
        assert send.call_count == 0
        assert sendb.call_count == 0
        assert "rlm_query_batched" in result

    def test_guard_disabled_inlines_context_no_mapreduce(self):
        env = _env()  # no subcall_context_limit
        big = "x" * 150_000
        with patch("prehend.environments.local_repl.send_lm_request",
                   side_effect=_ok) as send, \
             patch("prehend.environments.local_repl.send_lm_request_batched",
                   side_effect=_ok_batched) as sendb:
            result = env._llm_query("find", context=big)
        assert result == "DOC TEXT"
        assert send.call_count == 1
        assert sendb.call_count == 0

    def test_mapreduce_passed_nonzero_overlap(self):
        # the seam threads a positive chunk overlap into the engine so multi-hop
        # links straddling a chunk boundary survive partition.
        env = _env(subcall_context_limit=98304)
        captured = {}

        def fake_map_reduce(prompt, context, **kw):
            captured.update(kw)
            from prehend.utils.mapreduce import MapReduceResult
            return MapReduceResult(answer="X", n_chunks=2, reduce_levels=1,
                                   truncated=False, dropped=0, budget_exhausted=False)

        with patch("prehend.environments.local_repl.map_reduce", side_effect=fake_map_reduce):
            env._llm_query("find", context="x" * 150_000)
        assert captured.get("overlap_chars", 0) > 0
        assert captured["overlap_chars"] < captured["chunk_chars"]

    def test_mapreduce_uses_extraction_map(self):
        # The seam drives multihop CHAINING via extraction_map: the per-chunk MAP
        # is query-INDEPENDENT so a background hop survives to the reduce instead
        # of being filtered as irrelevant to the query. Validated live on the
        # multihop subset: legacy 1/5 -> extraction 5/5 (ADR-0018).
        env = _env(subcall_context_limit=98304)
        captured = {}

        def fake_map_reduce(prompt, context, **kw):
            captured.update(kw)
            from prehend.utils.mapreduce import MapReduceResult
            return MapReduceResult(answer="X", n_chunks=2, reduce_levels=1,
                                   truncated=False, dropped=0, budget_exhausted=False)

        with patch("prehend.environments.local_repl.map_reduce", side_effect=fake_map_reduce):
            env._llm_query("find", context="x" * 150_000)
        assert captured.get("extraction_map") is True

    def test_chunk_sized_for_extraction_map_instruction(self):
        # In extraction_map mode the MAP uses the fixed (longer) extraction
        # instruction, not the short user query. chunk_chars must leave room for
        # IT, else a composed map prompt overshoots the recommended size and the
        # inner guard rejects it. So chunk_chars + extraction overhead <= R.
        from prehend.utils.mapreduce import _EXTRACTION_MAP_INSTRUCTION, _compose
        env = _env(subcall_context_limit=98304)
        captured = {}

        def fake_map_reduce(prompt, context, **kw):
            captured.update(kw)
            from prehend.utils.mapreduce import MapReduceResult
            return MapReduceResult(answer="X", n_chunks=2, reduce_levels=1,
                                   truncated=False, dropped=0, budget_exhausted=False)

        with patch("prehend.environments.local_repl.map_reduce", side_effect=fake_map_reduce):
            env._llm_query("q", context="x" * 150_000)  # 1-char query
        overhead = len(_compose(_EXTRACTION_MAP_INSTRUCTION, "", "Text"))
        assert captured["chunk_chars"] + overhead <= R

    def test_seam_passes_persistent_map_cache_across_calls(self):
        # Re-scan fix: the orchestrator re-issues llm_query(context=SAME_BIG) across
        # iterations; the seam must hand map_reduce a PERSISTENT cache so the
        # query-independent extraction MAP is computed once and reused (live
        # evidence: ~8-9x re-prefill of a fixed 150k-token context otherwise).
        env = _env(subcall_context_limit=98304)
        big = "x" * 150_000
        seen = []

        def fake_map_reduce(prompt, context, **kw):
            seen.append(kw.get("map_cache"))
            from prehend.utils.mapreduce import MapReduceResult
            return MapReduceResult(answer="X", n_chunks=2, reduce_levels=1,
                                   truncated=False, dropped=0, budget_exhausted=False)

        with patch("prehend.environments.local_repl.map_reduce", side_effect=fake_map_reduce):
            env._llm_query("q1", context=big)
            env._llm_query("q2", context=big)
        assert len(seen) == 2
        assert seen[0] is not None              # a cache is provided to the engine
        assert seen[0] is seen[1]               # the SAME dict persists across calls

    def test_no_synthetic_pending_call_on_mapreduce_branch(self):
        env = _env(subcall_context_limit=98304)
        big = "x" * 150_000
        with patch("prehend.environments.local_repl.send_lm_request_batched",
                   side_effect=_ok_batched):
            env._llm_query("find", context=big)
        # 2 map prompts + 1 reduce prompt = 3 underlying sends, no extra synthetic.
        assert len(env._pending_llm_calls) == 3

    def test_placeholder_filled_in_context_before_send(self):
        env = _env(subcall_context_limit=98304, repair_unfilled_placeholders=True)
        env._active_namespace = {"myvar": "FILLED"}
        with patch("prehend.environments.local_repl.send_lm_request",
                   side_effect=_ok) as send:
            env._llm_query("q", context="see {myvar}")
        assert "FILLED" in send.call_args[0][1].prompt
        assert "{myvar}" not in send.call_args[0][1].prompt


class TestBatchedContextDispatch:
    def test_no_context_delegates_unchanged(self):
        env = _env(subcall_context_limit=98304)
        with patch("prehend.environments.local_repl.send_lm_request_batched",
                   side_effect=_ok_batched) as sendb:
            results = env._llm_query_batched(["a", "b"])
        assert sendb.call_count == 1
        assert sendb.call_args[0][1] == ["a", "b"]
        assert results == ["R0", "R1"]

    def test_context_applies_to_every_prompt_order_preserved(self):
        env = _env(subcall_context_limit=98304)
        with patch("prehend.environments.local_repl.send_lm_request",
                   side_effect=_ok) as send:
            results = env._llm_query_batched(["q1", "q2"], context="small ctx")
        assert len(results) == 2
        sent = [c.args[1].prompt for c in send.call_args_list]
        assert "q1" in sent[0] and "q2" in sent[1]  # order preserved
        assert all("small ctx" in p for p in sent)


class TestRLMQueryContextDispatch:
    def _fake_subcall(self):
        calls = []

        def subcall_fn(prompt, model=None):
            calls.append(prompt)
            return RLMChatCompletion(
                root_model=MODEL, prompt=prompt, response="PARTIAL",
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=0.0,
            )
        return subcall_fn, calls

    def test_oversized_context_fans_out_to_children(self):
        subcall_fn, calls = self._fake_subcall()
        env = _env(subcall_context_limit=98304, subcall_fn=subcall_fn)
        big = "x" * 150_000
        result = env._rlm_query("find", context=big)
        # 2 chunk children (map) + 1 reduce child = 3 subcall_fn invocations.
        assert len(calls) == 3
        assert result == "PARTIAL"

    def test_small_context_inlined_single_child(self):
        subcall_fn, calls = self._fake_subcall()
        env = _env(subcall_context_limit=98304, subcall_fn=subcall_fn)
        result = env._rlm_query("find", context="small")
        assert len(calls) == 1
        assert "small" in calls[0]
        assert result == "PARTIAL"


class TestSubcallSignatureUnchanged:
    def test_subcall_has_no_context_param(self):
        # _subcall stays single-shot; the REPL composes before calling it (R2-6).
        import inspect

        from prehend.core.rlm import RLM
        params = inspect.signature(RLM._subcall).parameters
        assert "context" not in params
        assert "reduce" not in params


class TestPromptTeaching:
    def test_prompt_documents_context_arg_for_large_text(self):
        from prehend.utils.prompts import RLM_SYSTEM_PROMPT
        assert "context=" in RLM_SYSTEM_PROMPT
        assert "reduce=" in RLM_SYSTEM_PROMPT

    def test_prompt_keeps_manual_slicing_fallback(self):
        # the ADR-0009 "sub-calls do NOT see your context" guidance still stands.
        from prehend.utils.prompts import RLM_SYSTEM_PROMPT
        assert "sub-calls do NOT see your" in RLM_SYSTEM_PROMPT
