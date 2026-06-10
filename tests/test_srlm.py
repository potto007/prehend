"""Tests for the SRLM subclass - context-length routing, direct mode, and selection."""
from unittest.mock import MagicMock, patch

from lm_repl.core.rlm import RLM
from lm_repl.core.srlm import SRLM, _choose_mode, _build_direct_messages, _select_best
from lm_repl.core.types import RLMChatCompletion, UsageSummary


class TestChooseMode:
    def test_rlm_when_threshold_zero(self):
        assert _choose_mode(100, 0) == "rlm"

    def test_rlm_when_threshold_none(self):
        assert _choose_mode(100, None) == "rlm"

    def test_direct_when_below(self):
        assert _choose_mode(5000, 30000) == "direct"

    def test_rlm_when_at_threshold(self):
        assert _choose_mode(30000, 30000) == "rlm"

    def test_rlm_when_above(self):
        assert _choose_mode(50000, 30000) == "rlm"


class TestBuildDirectMessages:
    def test_roles(self):
        msgs = _build_direct_messages("ctx", "q")
        assert [m["role"] for m in msgs] == ["system", "user"]

    def test_content(self):
        msgs = _build_direct_messages("my data", "find X")
        user = msgs[1]["content"]
        assert "my data" in user
        assert "find X" in user


class TestSRLMInit:
    def test_inherits_rlm(self):
        from lm_repl import RLM
        assert issubclass(SRLM, RLM)

    def test_accepts_srlm_params(self):
        srlm = SRLM(
            backend="openai",
            backend_kwargs={"model_name": "test", "base_url": "http://localhost:9999/v1"},
            direct_threshold=30000,
            n_candidates=4,
        )
        assert srlm.direct_threshold == 30000
        assert srlm.n_candidates == 4

    def test_defaults(self):
        srlm = SRLM(
            backend="openai",
            backend_kwargs={"model_name": "test", "base_url": "http://localhost:9999/v1"},
        )
        assert srlm.direct_threshold == 0
        assert srlm.n_candidates == 1


def _make_completion(response: str, exec_time: float) -> RLMChatCompletion:
    return RLMChatCompletion(
        root_model="test",
        prompt="test prompt",
        response=response,
        usage_summary=UsageSummary(model_usage_summaries={}),
        execution_time=exec_time,
    )


def _make_completion_with_tokens(
    response: str, exec_time: float, out_tokens: int
) -> RLMChatCompletion:
    from lm_repl.core.types import ModelUsageSummary

    return RLMChatCompletion(
        root_model="test",
        prompt="test prompt",
        response=response,
        usage_summary=UsageSummary(
            model_usage_summaries={
                "test": ModelUsageSummary(
                    total_calls=1, total_input_tokens=0, total_output_tokens=out_tokens
                )
            }
        ),
        execution_time=exec_time,
    )


class TestTraceLen:
    def test_uses_output_tokens_when_available(self):
        from lm_repl.core.srlm import _trace_len

        c = _make_completion_with_tokens("42", exec_time=9.0, out_tokens=350)
        assert _trace_len(c) == 350

    def test_falls_back_to_execution_time_without_usage(self):
        from lm_repl.core.srlm import _trace_len

        c = _make_completion("42", exec_time=2.5)
        assert _trace_len(c) == 2.5

    def test_no_confidence_tiebreak_prefers_fewer_tokens(self):
        """Without confidence, the consistent-set tiebreak is trace length in
        tokens - not wall clock, which is confounded by cache hits and slots."""
        slow_but_short = _make_completion_with_tokens("42", exec_time=9.0, out_tokens=200)
        fast_but_long = _make_completion_with_tokens("42", exec_time=1.0, out_tokens=4000)
        result = _select_best([slow_but_short, fast_but_long])
        assert result is slow_but_short


class TestSelectBest:
    def test_single_candidate(self):
        c = _make_completion("42", 1.0)
        assert _select_best([c]) is c

    def test_majority_vote(self):
        c1 = _make_completion("42", 1.0)
        c2 = _make_completion("42", 2.0)
        c3 = _make_completion("99", 0.5)
        result = _select_best([c1, c2, c3])
        assert result.response == "42"

    def test_picks_shortest_trace_among_consistent(self):
        c1 = _make_completion("42", 3.0)
        c2 = _make_completion("42", 1.0)
        c3 = _make_completion("42", 2.0)
        assert _select_best([c1, c2, c3]) is c2

    def test_all_different_picks_any(self):
        c1 = _make_completion("a", 1.0)
        c2 = _make_completion("b", 2.0)
        c3 = _make_completion("c", 3.0)
        result = _select_best([c1, c2, c3])
        assert result in [c1, c2, c3]

    def test_case_insensitive_consistency(self):
        c1 = _make_completion("YES", 2.0)
        c2 = _make_completion("yes", 1.0)
        result = _select_best([c1, c2])
        assert result.execution_time == 1.0


class TestCandidateTemperature:
    def test_default_is_none(self):
        srlm = SRLM(
            backend="openai",
            backend_kwargs={"model_name": "test", "base_url": "http://localhost:9999/v1"},
        )
        assert srlm.candidate_temperature is None

    def test_accepts_temperature(self):
        srlm = SRLM(
            backend="openai",
            backend_kwargs={"model_name": "test", "base_url": "http://localhost:9999/v1"},
            candidate_temperature=0.7,
        )
        assert srlm.candidate_temperature == 0.7

    def test_temperature_injected_during_multi_trajectory(self):
        """When candidate_temperature is set, backend_kwargs should get temperature
        injected into default_extra_body during multi-trajectory runs, then restored."""
        srlm = SRLM(
            backend="openai",
            backend_kwargs={"model_name": "test", "base_url": "http://localhost:9999/v1"},
            n_candidates=2,
            candidate_temperature=0.8,
        )
        original_extra = dict(srlm.backend_kwargs.get("default_extra_body", {}))

        captured_temps = []
        original_completion = RLM.completion

        def mock_completion(self_inner, prompt, root_prompt=None):
            extra = self_inner.backend_kwargs.get("default_extra_body", {})
            captured_temps.append(extra.get("temperature"))
            from lm_repl.core.types import RLMChatCompletion, UsageSummary
            return RLMChatCompletion(
                root_model="test", prompt=prompt, response="42",
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=1.0,
            )

        import unittest.mock
        with unittest.mock.patch.object(RLM, 'completion', mock_completion):
            srlm.completion("test prompt")

        assert all(t == 0.8 for t in captured_temps), f"Expected 0.8, got {captured_temps}"
        assert srlm.backend_kwargs.get("default_extra_body", {}) == original_extra

    def test_no_temperature_injection_when_none(self):
        """When candidate_temperature is None, no temperature is injected."""
        srlm = SRLM(
            backend="openai",
            backend_kwargs={"model_name": "test", "base_url": "http://localhost:9999/v1"},
            n_candidates=2,
        )

        captured_temps = []
        def mock_completion(self_inner, prompt, root_prompt=None):
            extra = self_inner.backend_kwargs.get("default_extra_body", {})
            captured_temps.append(extra.get("temperature"))
            from lm_repl.core.types import RLMChatCompletion, UsageSummary
            return RLMChatCompletion(
                root_model="test", prompt=prompt, response="42",
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=1.0,
            )

        import unittest.mock
        with unittest.mock.patch.object(RLM, 'completion', mock_completion):
            srlm.completion("test prompt")

        assert all(t is None for t in captured_temps)


# --- Verbalized confidence & joint scoring tests ---

import math
from lm_repl.core.srlm import (
    _compute_vc_score,
    _extract_step_texts,
    _parse_confidence_scores,
)


def _make_trajectory_metadata(step_responses: list[str]) -> dict:
    """Build metadata in the real shape RLM attaches: logger.get_trajectory()."""
    return {
        "run_metadata": {"root_model": "test"},
        "iterations": [
            {
                "type": "iteration",
                "iteration": i + 1,
                "prompt": "step prompt",
                "response": resp,
                "code_blocks": [],
                "final_answer": None,
                "iteration_time": 1.0,
            }
            for i, resp in enumerate(step_responses)
        ],
    }


class TestExtractStepTexts:
    def test_none_metadata(self):
        assert _extract_step_texts(None) == []

    def test_empty_dict(self):
        assert _extract_step_texts({}) == []

    def test_real_rlm_trajectory_shape(self):
        meta = _make_trajectory_metadata(["step one text", "step two text"])
        assert _extract_step_texts(meta) == ["step one text", "step two text"]

    def test_legacy_trajectory_text(self):
        assert _extract_step_texts({"trajectory_text": "blob"}) == ["blob"]

    def test_iterations_missing_response(self):
        meta = {"iterations": [{"iteration": 1}, {"iteration": 2, "response": "ok"}]}
        assert _extract_step_texts(meta) == ["", "ok"]


class TestParseConfidenceScores:
    def test_single_score(self):
        text = 'I found the answer. {"confidence": 85}'
        assert _parse_confidence_scores(text) == [85.0]

    def test_multiple_scores(self):
        text = '{"confidence": 90}\nsome code\n{"confidence": 70}'
        assert _parse_confidence_scores(text) == [90.0, 70.0]

    def test_no_scores(self):
        assert _parse_confidence_scores("just regular text") == []

    def test_handles_whitespace_variants(self):
        text = '{"confidence" : 75}'
        assert _parse_confidence_scores(text) == [75.0]

    def test_handles_integer_and_float(self):
        text = '{"confidence": 80}\n{"confidence": 92.5}'
        scores = _parse_confidence_scores(text)
        assert scores == [80.0, 92.5]

    def test_clamps_to_range(self):
        text = '{"confidence": 0}\n{"confidence": 100}\n{"confidence": 150}'
        scores = _parse_confidence_scores(text)
        assert scores[0] == 0.0
        assert scores[1] == 100.0
        assert scores[2] == 100.0  # clamped


class TestComputeVCScore:
    def test_perfect_confidence(self):
        steps = ['{"confidence": 100}', '{"confidence": 100}']
        assert _compute_vc_score(steps) == 0.0  # log(1) + log(1) = 0

    def test_partial_confidence(self):
        steps = ['{"confidence": 50}']
        score = _compute_vc_score(steps)
        assert score < 0  # log(0.5) is negative
        assert abs(score - math.log(0.5)) < 1e-6

    def test_no_scores_returns_neg_inf(self):
        assert _compute_vc_score(["no confidence here", "still none"]) == float('-inf')

    def test_empty_steps_returns_neg_inf(self):
        assert _compute_vc_score([]) == float('-inf')

    def test_zero_confidence_clamps(self):
        steps = ['{"confidence": 0}']
        score = _compute_vc_score(steps)
        assert score == float('-inf')  # log(0) is -inf, use floor

    def test_missing_steps_imputed_with_trajectory_mean(self):
        """A step without a confidence report is filled with the mean of the
        reported steps (per the paper), so skipping reports cannot inflate VC."""
        steps = ['code... {"confidence": 80}', 'code without any report']
        score = _compute_vc_score(steps)
        expected = 2 * math.log(0.8)  # second step imputed with mean 80
        assert abs(score - expected) < 1e-6

    def test_imputation_prevents_underreporting_gaming(self):
        """Reporting on 1 of 3 steps must NOT beat honestly reporting all 3."""
        underreporter = ['{"confidence": 90}', 'no report', 'no report']
        honest = ['{"confidence": 90}', '{"confidence": 90}', '{"confidence": 90}']
        assert abs(_compute_vc_score(underreporter) - _compute_vc_score(honest)) < 1e-6

    def test_last_score_per_step_wins(self):
        """When a step contains several confidence lines, use the final one."""
        steps = ['draft {"confidence": 20} ... revised {"confidence": 90}']
        assert abs(_compute_vc_score(steps) - math.log(0.9)) < 1e-6


class TestSelectBestWithConfidence:
    def test_confidence_mode_prefers_high_vc(self):
        """High VC score (closer to 0) wins over low VC score."""
        c1 = _make_completion("42", 2.0)
        c1.metadata = {"trajectory_text": '{"confidence": 95}\n{"confidence": 90}'}
        c2 = _make_completion("42", 2.0)
        c2.metadata = {"trajectory_text": '{"confidence": 40}\n{"confidence": 30}'}

        result = _select_best([c1, c2], use_confidence=True)
        assert result is c1

    def test_confidence_mode_joint_score(self):
        """Joint score VC*Len: high confidence + short trace beats low confidence + short trace."""
        c1 = _make_completion("42", 1.0)
        c1.metadata = {"trajectory_text": '{"confidence": 95}'}
        c2 = _make_completion("42", 1.0)
        c2.metadata = {"trajectory_text": '{"confidence": 50}'}

        result = _select_best([c1, c2], use_confidence=True)
        assert result is c1

    def test_confidence_off_ignores_metadata(self):
        """Without confidence mode, selection uses execution_time only."""
        c1 = _make_completion("42", 2.0)
        c1.metadata = {"trajectory_text": '{"confidence": 95}'}
        c2 = _make_completion("42", 1.0)
        c2.metadata = {"trajectory_text": '{"confidence": 30}'}

        result = _select_best([c1, c2], use_confidence=False)
        assert result is c2  # shorter time wins

    def test_falls_back_to_time_when_no_confidence_data(self):
        """If metadata has no trajectory_text, fall back to time-based selection."""
        c1 = _make_completion("42", 2.0)
        c2 = _make_completion("42", 1.0)

        result = _select_best([c1, c2], use_confidence=True)
        assert result is c2

    def test_joint_score_uses_tokens_not_wall_clock(self):
        """Len(p) is trace tokens per the paper. With equal confidence, the
        candidate with fewer output tokens must win even if prefix-cache
        effects made its wall clock slower."""
        few_tokens = _make_completion_with_tokens("42", exec_time=9.0, out_tokens=200)
        few_tokens.metadata = _make_trajectory_metadata(['{"confidence": 80}'])
        many_tokens = _make_completion_with_tokens("42", exec_time=1.0, out_tokens=4000)
        many_tokens.metadata = _make_trajectory_metadata(['{"confidence": 80}'])

        result = _select_best([few_tokens, many_tokens], use_confidence=True)
        assert result is few_tokens

    def test_confidence_scoring_reads_real_rlm_metadata(self):
        """RLM attaches logger.get_trajectory() as metadata - the iterations
        shape, NOT a trajectory_text key. VC selection must work on it.

        Regression: _joint_score looked up metadata["trajectory_text"], which
        no real RLM run ever sets, so confidence selection silently fell back
        to execution_time for every real trajectory."""
        confident = _make_completion("42", 5.0)
        confident.metadata = _make_trajectory_metadata(
            ['x = ctx.find("v") {"confidence": 95}', 'FINAL(42) {"confidence": 95}']
        )
        unsure = _make_completion("42", 1.0)
        unsure.metadata = _make_trajectory_metadata(
            ['hmm {"confidence": 30}', 'FINAL(42) {"confidence": 25}']
        )

        # Time-based selection would pick `unsure` (1.0s < 5.0s). Confidence
        # selection must pick `confident` despite the slower wall clock.
        result = _select_best([confident, unsure], use_confidence=True)
        assert result is confident
