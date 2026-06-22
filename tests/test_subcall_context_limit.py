"""Integration tests for the sub-call context-limit fix (Units B, C, D).

The pure-logic foundation (token_utils.resolve_subcall_limit / get_context_limit
and subcall_guard.oversize_rejection / safe_chunk_chars) is tested elsewhere
(tests/test_subcall_guard.py). These tests cover the WIRING:
- Unit B: subcall_context_limit threads Harness -> SRLM -> RLM -> LocalREPL and
  into _subcall + the prompt build.
- Unit C: the reject-with-hint guard fires at BOTH seams (LocalREPL llm_query /
  llm_query_batched, and RLM._subcall).
- Unit D: the RLM_SYSTEM_PROMPT is realigned (no "500K", capacity parameterized,
  guidance points large context at rlm_query_batched and reserves llm_query for
  SHORT text).
"""

from types import SimpleNamespace
from unittest.mock import patch

from prehend.harness import Defaults, Harness, Runtime
from prehend.core.rlm import RLM
from prehend.core.types import RLMChatCompletion, UsageSummary
from prehend.environments.local_repl import LocalREPL
from prehend.utils.prompts import RLM_SYSTEM_PROMPT, build_rlm_system_prompt
from prehend.utils.token_utils import get_context_limit
from prehend.utils.subcall_guard import recommended_chunk_chars, safe_chunk_chars

MODEL = "gemma-4-12b-it-sft-kb-v13-sft"


def _ok(_addr, _req):
    return SimpleNamespace(success=True, chat_completion=SimpleNamespace(response="DOC TEXT"))


def _ok_batched(_addr, prompts, **kw):
    return [
        SimpleNamespace(success=True, chat_completion=SimpleNamespace(response=f"R{i}"))
        for i, _ in enumerate(prompts)
    ]


def _h(**kw):
    return Harness(model=MODEL, base_url="http://localhost:9999/v1",
                   runtime=Runtime(slots=4, ctx=98304), **kw)


# --------------------------------------------------------------------------- #
# Unit B: limit plumbing
# --------------------------------------------------------------------------- #
class TestLimitPlumbing:
    def test_subcall_context_limit_defaults_to_resolved(self):
        # No explicit param, no Defaults override -> resolve_subcall_limit uses
        # runtime.ctx (98304) since that is the first non-None after explicit.
        h = _h()
        assert h.srlm.subcall_context_limit == 98304

    def test_falls_back_to_get_context_limit_when_ctx_unknown(self):
        # runtime.ctx None and no explicit -> get_context_limit(model) (gemma 262144).
        h = Harness(model=MODEL, base_url="http://localhost:9999/v1",
                    runtime=Runtime(slots=4, ctx=None))
        assert h.srlm.subcall_context_limit == get_context_limit(MODEL)
        assert h.srlm.subcall_context_limit == 262144

    def test_explicit_param_overrides(self):
        h = _h(subcall_context_limit=12345)
        assert h.srlm.subcall_context_limit == 12345

    def test_defaults_field_used_when_no_param(self):
        d = Defaults(subcall_context_limit=55555)
        h = Harness(model=MODEL, base_url="http://localhost:9999/v1",
                    runtime=Runtime(slots=4, ctx=98304), defaults=d)
        assert h.srlm.subcall_context_limit == 55555

    def test_param_wins_over_defaults_field(self):
        d = Defaults(subcall_context_limit=55555)
        h = Harness(model=MODEL, base_url="http://localhost:9999/v1",
                    runtime=Runtime(slots=4, ctx=98304), defaults=d,
                    subcall_context_limit=77777)
        assert h.srlm.subcall_context_limit == 77777

    def test_runtime_ctx_threaded_into_limit(self):
        h = Harness(model=MODEL, base_url="http://localhost:9999/v1",
                    runtime=Runtime(slots=4, ctx=65536))
        assert h.srlm.subcall_context_limit == 65536

    def test_threads_into_environment_kwargs(self):
        # LocalREPL must receive the limit AND the model name so it can guard.
        h = _h(subcall_context_limit=98304)
        env_kw = h.srlm.environment_kwargs
        assert env_kw["subcall_context_limit"] == 98304
        assert env_kw["model_name"] == MODEL

    def test_subcall_default_field_is_none(self):
        assert Defaults().subcall_context_limit is None


class TestRLMThreading:
    def _rlm(self, **kw):
        return RLM(backend_kwargs={"model_name": MODEL}, max_depth=2, **kw)

    def test_rlm_stores_subcall_context_limit(self):
        rlm = self._rlm(subcall_context_limit=98304)
        assert rlm.subcall_context_limit == 98304

    def test_rlm_default_is_none(self):
        rlm = self._rlm()
        assert rlm.subcall_context_limit is None

    def test_spawned_localrepl_gets_limit_and_model(self):
        # _spawn_completion_context builds env_kwargs that include the limit and
        # the model name (derived from backend_kwargs). Patch get_client so no
        # real OpenAI client (needs an api_key) is constructed.
        rlm = self._rlm(subcall_context_limit=98304)
        from tests.mock_lm import MockLM
        with patch("prehend.core.rlm.get_client", return_value=MockLM(model_name=MODEL)):
            with rlm._spawn_completion_context("ctx") as (_handler, env):
                assert isinstance(env, LocalREPL)
                assert env.subcall_context_limit == 98304
                assert env.model_name == MODEL

    def test_child_inherits_subcall_context_limit(self):
        # _subcall builds a child RLM; the limit must follow. Patch the child's
        # completion so no network is hit; capture the constructed child.
        rlm = self._rlm(subcall_context_limit=98304)
        captured = {}
        real_init = RLM.__init__

        def spy_init(self, *a, **k):
            real_init(self, *a, **k)
            captured["child"] = self

        done = RLMChatCompletion(
            root_model=MODEL, prompt="p", response="ok",
            usage_summary=UsageSummary(model_usage_summaries={}), execution_time=0.0,
        )
        with patch.object(RLM, "completion", return_value=done):
            with patch.object(RLM, "__init__", spy_init):
                rlm._subcall("short prompt")
        assert captured["child"].subcall_context_limit == 98304


# --------------------------------------------------------------------------- #
# Unit C: reject-with-hint guard at both seams
# --------------------------------------------------------------------------- #
class TestLocalREPLGuard:
    def _env(self, **kw):
        return LocalREPL(lm_handler_address=("127.0.0.1", 1),
                         model_name=MODEL, **kw)

    def test_oversized_llm_query_returns_hint_not_sent(self):
        env = self._env(subcall_context_limit=98304)
        prompt = "word " * 300_000  # well over the safe budget
        with patch("prehend.environments.local_repl.send_lm_request",
                   side_effect=_ok) as send:
            result = env._llm_query(prompt)
        assert send.call_count == 0  # never reached the server
        assert "rlm_query_batched" in result
        # hint recommends the smaller latency-friendly chunk, not the hard ceiling
        assert str(recommended_chunk_chars(98304, MODEL)) in result

    def test_under_limit_llm_query_passes_through(self):
        env = self._env(subcall_context_limit=98304)
        with patch("prehend.environments.local_repl.send_lm_request",
                   side_effect=_ok) as send:
            result = env._llm_query("short prompt")
        assert result == "DOC TEXT"
        assert send.call_count == 1

    def test_no_limit_means_no_guard(self):
        # subcall_context_limit None -> guard is inert even for a huge prompt.
        env = self._env()  # no limit
        prompt = "word " * 300_000
        with patch("prehend.environments.local_repl.send_lm_request",
                   side_effect=_ok) as send:
            result = env._llm_query(prompt)
        assert result == "DOC TEXT"
        assert send.call_count == 1

    def test_batched_guards_per_prompt(self):
        env = self._env(subcall_context_limit=98304)
        big = "word " * 300_000
        prompts = ["short a", big, "short b"]
        with patch("prehend.environments.local_repl.send_lm_request_batched",
                   side_effect=_ok_batched) as send:
            results = env._llm_query_batched(prompts)
        # The two short prompts were sent; the big one was rejected with a hint.
        assert send.call_args[0][1] == ["short a", "short b"]
        assert results[0] == "R0"
        assert "rlm_query_batched" in results[1]
        assert results[2] == "R1"


class TestSubcallGuard:
    def _rlm(self, **kw):
        # max_depth=1 so _subcall's leaf path would call get_client - but the
        # guard returns BEFORE that, so no client/network is needed.
        return RLM(backend_kwargs={"model_name": MODEL}, max_depth=1, **kw)

    def test_rlm_query_leaf_oversize_rejected(self):
        rlm = self._rlm(subcall_context_limit=98304)
        prompt = "word " * 300_000
        result = rlm._subcall(prompt)
        assert isinstance(result, RLMChatCompletion)
        assert "rlm_query_batched" in result.response
        assert result.execution_time == 0.0
        # zero usage like the verifier-rejection completion
        assert result.usage_summary.model_usage_summaries == {}

    def test_subcall_under_limit_not_rejected(self):
        # With no limit, the guard never fires; the leaf path runs (mock client).
        rlm = self._rlm()  # no limit
        with patch("prehend.core.rlm.get_client") as gc:
            gc.return_value = SimpleNamespace(
                model_name=MODEL,
                completion=lambda p, **kw: "leaf answer",
                get_last_usage=lambda: SimpleNamespace(),
            )
            result = rlm._subcall("short prompt")
        assert result.response == "leaf answer"

    def test_subcall_oversize_rejected_before_client(self):
        # Even when a client would be available, an oversized prompt is rejected
        # with no client call.
        rlm = self._rlm(subcall_context_limit=98304)
        with patch("prehend.core.rlm.get_client") as gc:
            result = rlm._subcall("word " * 300_000)
        assert gc.call_count == 0
        assert "rlm_query_batched" in result.response


# --------------------------------------------------------------------------- #
# Unit D: prompt realignment
# --------------------------------------------------------------------------- #
class TestPromptRealignment:
    def test_no_hardcoded_500k_in_prompt(self):
        assert "500K" not in RLM_SYSTEM_PROMPT
        assert "500k" not in RLM_SYSTEM_PROMPT

    def test_prompt_has_subcall_char_budget_field(self):
        assert "{subcall_char_budget}" in RLM_SYSTEM_PROMPT

    def test_build_fills_capacity_field(self):
        from prehend.core.types import QueryMetadata
        meta = QueryMetadata("some context string")
        msgs = build_rlm_system_prompt(RLM_SYSTEM_PROMPT, meta,
                                       subcall_char_budget=88_000)
        system = msgs[0]["content"]
        assert "{subcall_char_budget}" not in system  # field was filled
        assert "88,000" in system

    def test_build_uses_safe_default_when_unset(self):
        from prehend.core.types import QueryMetadata
        meta = QueryMetadata("some context string")
        msgs = build_rlm_system_prompt(RLM_SYSTEM_PROMPT, meta)
        system = msgs[0]["content"]
        assert "{subcall_char_budget}" not in system  # still filled with default

    def test_prompt_warns_llm_query_short_only_large_to_rlm_query(self):
        from prehend.core.types import QueryMetadata
        meta = QueryMetadata("ctx")
        system = build_rlm_system_prompt(RLM_SYSTEM_PROMPT, meta,
                                         subcall_char_budget=90_000)[0]["content"]
        lowered = system.lower()
        assert "rlm_query_batched" in system
        assert "short" in lowered

    def test_rlm_build_passes_recommended_chunk_chars(self):
        # When an RLM has a subcall_context_limit, its system prompt's chunk
        # guidance reflects recommended_chunk_chars (the small latency-friendly
        # target), which is strictly below the safe_chunk_chars hard ceiling.
        rlm = RLM(backend_kwargs={"model_name": MODEL}, max_depth=2,
                  subcall_context_limit=98304)
        msgs = rlm._setup_prompt("context string")
        system = msgs[0]["content"]
        expected = f"{recommended_chunk_chars(98304, MODEL):,}"
        assert expected in system
        assert recommended_chunk_chars(98304, MODEL) < safe_chunk_chars(98304, MODEL)
