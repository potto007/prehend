"""Tests for the optional Prometheus instrumentation module."""

from __future__ import annotations

import pytest

pytest.importorskip("prometheus_client")

from lm_repl import metrics
from lm_repl.core.types import (
    CodeBlock,
    ModelUsageSummary,
    REPLResult,
    RLMChatCompletion,
    RLMIteration,
    RLMMetadata,
    UsageSummary,
)


def _sample_repl_result(input_tokens: int = 10, output_tokens: int = 20) -> REPLResult:
    usage = UsageSummary(
        model_usage_summaries={
            "test-model": ModelUsageSummary(
                total_calls=1,
                total_input_tokens=input_tokens,
                total_output_tokens=output_tokens,
            ),
        },
    )
    call = RLMChatCompletion(
        root_model="test-model",
        prompt="child prompt",
        response="child response",
        usage_summary=usage,
        execution_time=0.1,
    )
    return REPLResult(
        stdout="ok",
        stderr="",
        locals={},
        execution_time=0.1,
        rlm_calls=[call],
    )


def _read(counter, **labels) -> float:
    """Read the current value of a counter/gauge (labeled or unlabeled)."""
    if labels:
        return counter.labels(**labels)._value.get()
    return counter._value.get()


class TestClassifyProgram:
    def test_rlm_query_batched(self):
        assert metrics._classify_program("xs = rlm_query_batched(['a'])") == "rlm_query_batched"

    def test_rlm_query(self):
        assert metrics._classify_program("rlm_query('a')") == "rlm_query"

    def test_llm_query(self):
        assert metrics._classify_program("llm_query('a')") == "llm_query"

    def test_slice_when_unknown(self):
        assert metrics._classify_program("docs['001'][:1000]") == "slice"

    def test_other_when_empty(self):
        assert metrics._classify_program("") == "other"
        assert metrics._classify_program("   \n") == "other"


class TestPrometheusLogger:
    def test_log_metadata_picks_up_model(self):
        logger = metrics.PrometheusLogger()
        meta = RLMMetadata(
            root_model="kb-model",
            max_depth=3,
            max_iterations=10,
            backend="openai",
            backend_kwargs={},
            environment_type="local",
            environment_kwargs={},
        )
        logger.log_metadata(meta)
        assert logger.model_label == "kb-model"

    def test_log_iteration_records_program_and_tokens(self):
        logger = metrics.PrometheusLogger(model_label="kb-model")
        before = _read(metrics.iterations_total, program="rlm_query_batched", depth="0")
        tokens_before = _read(metrics.tokens_total, role="worker", kind="child", direction="prompt")
        iteration = RLMIteration(
            prompt="root prompt",
            response="root response",
            code_blocks=[
                CodeBlock(code="rlm_query_batched(['x','y'])", result=_sample_repl_result(7, 13)),
            ],
            iteration_time=0.5,
        )
        logger.log(iteration)
        assert _read(metrics.iterations_total, program="rlm_query_batched", depth="0") == before + 1
        assert (
            _read(metrics.tokens_total, role="worker", kind="child", direction="prompt")
            == tokens_before + 7
        )

    def test_log_iteration_swallows_internal_errors(self):
        logger = metrics.PrometheusLogger()
        before = _read(metrics.callback_failures_total)
        # An iteration with a code_blocks attribute that's not a list won't iterate.
        bad_iteration = RLMIteration(
            prompt="x", response="y", code_blocks=None  # type: ignore[arg-type]
        )
        logger.log(bad_iteration)  # must not raise
        # code_blocks=None gracefully iterates to no-op; failures counter shouldn't move.
        # Force a real error: monkey-patch attribute to trigger TypeError.
        class Boom:
            def __iter__(self):
                raise RuntimeError("boom")

        bad2 = RLMIteration(prompt="x", response="y", code_blocks=Boom())  # type: ignore[arg-type]
        logger.log(bad2)
        assert _read(metrics.callback_failures_total) >= before + 1


class TestBind:
    def test_bind_attaches_all_four_callbacks(self):
        class FakeRLM:
            on_subcall_start = None
            on_subcall_complete = None
            on_iteration_start = None
            on_iteration_complete = None
            logger = None
            backend_kwargs = {"model_name": "kb-model"}

        rlm = FakeRLM()
        metrics.bind(rlm, model_label="kb-model")
        assert callable(rlm.on_subcall_start)
        assert callable(rlm.on_subcall_complete)
        assert callable(rlm.on_iteration_start)
        assert callable(rlm.on_iteration_complete)
        assert isinstance(rlm.logger, metrics.PrometheusLogger)

    def test_bind_does_not_overwrite_user_logger(self):
        class FakeLogger:
            pass

        class FakeRLM:
            on_subcall_start = None
            on_subcall_complete = None
            on_iteration_start = None
            on_iteration_complete = None
            logger = FakeLogger()
            backend_kwargs = None

        rlm = FakeRLM()
        original_logger = rlm.logger
        metrics.bind(rlm)
        assert rlm.logger is original_logger


class TestConcurrencyTracker:
    def test_subcall_start_complete_pair(self):
        before_in_flight = metrics.concurrent_children._value.get()
        before_total = _read(
            metrics.calls_total, kind="child", model="kb-model", outcome="success"
        )
        metrics._tracker.on_subcall_start(2, "kb-model", "prompt preview")
        assert metrics.concurrent_children._value.get() == before_in_flight + 1
        metrics._tracker.on_subcall_complete(2, "kb-model", 0.42, None)
        assert metrics.concurrent_children._value.get() == before_in_flight
        assert (
            _read(metrics.calls_total, kind="child", model="kb-model", outcome="success")
            == before_total + 1
        )

    def test_subcall_complete_with_timeout(self):
        before = _read(metrics.timeouts_total, kind="child")
        metrics._tracker.on_subcall_start(1, "kb-model", "")
        metrics._tracker.on_subcall_complete(
            1, "kb-model", 1.0, "TimeoutExceededError: ran past deadline"
        )
        assert _read(metrics.timeouts_total, kind="child") >= before + 1


class TestCallScope:
    def test_records_duration_and_success(self):
        class FakeRLM:
            backend_kwargs = {"model_name": "kb-model"}

        before_total = _read(
            metrics.calls_total, kind="root", model="kb-model", outcome="success"
        )
        with metrics.CallScope(FakeRLM(), prompt="x" * 200):
            pass
        assert (
            _read(metrics.calls_total, kind="root", model="kb-model", outcome="success")
            == before_total + 1
        )

    def test_records_error_outcome(self):
        class FakeRLM:
            backend_kwargs = {"model_name": "kb-model"}

        before = _read(
            metrics.calls_total, kind="root", model="kb-model", outcome="error"
        )
        with pytest.raises(RuntimeError):
            with metrics.CallScope(FakeRLM(), prompt="x"):
                raise RuntimeError("boom")
        assert (
            _read(metrics.calls_total, kind="root", model="kb-model", outcome="error")
            >= before + 1
        )

    def test_calls_in_flight_returns_to_zero(self):
        class FakeRLM:
            backend_kwargs = {"model_name": "kb-model"}

        before = _read(metrics.calls_in_flight, kind="root")
        with metrics.CallScope(FakeRLM()):
            assert _read(metrics.calls_in_flight, kind="root") == before + 1
        assert _read(metrics.calls_in_flight, kind="root") == before
