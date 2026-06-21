"""Unit tests for RLM._subcall() method.

Tests for the parameter propagation to child RLM instances:
1. max_timeout (remaining time) is passed to child
2. max_tokens is passed to child
3. max_errors is passed to child
4. model= parameter overrides child's backend model
"""

import time
from unittest.mock import Mock, patch

import prehend.core.rlm as rlm_module
from prehend import RLM
from prehend.core.types import ModelUsageSummary, RLMChatCompletion, UsageSummary


def create_mock_lm(responses: list[str], model_name: str = "mock-model") -> Mock:
    """Create a mock LM that returns responses in order."""
    mock = Mock()
    mock.model_name = model_name
    mock.completion.side_effect = list(responses)
    mock.get_usage_summary.return_value = UsageSummary(
        model_usage_summaries={
            model_name: ModelUsageSummary(
                total_calls=1, total_input_tokens=100, total_output_tokens=50
            )
        }
    )
    mock.get_last_usage.return_value = mock.get_usage_summary.return_value
    return mock


def final(content: str) -> str:
    """Render a model response that submits ``content`` as the final answer."""
    return f"```repl\nanswer['content'] = {content!r}\nanswer['ready'] = True\n```"


class TestSubcallTimeoutPropagation:
    """Tests for max_timeout propagation to child RLM."""

    def test_child_receives_remaining_timeout(self):
        """When parent has max_timeout=60 and 10s have elapsed, child should get max_timeout approx 50."""
        captured_child_params = {}

        # Create a fake child RLM class to capture initialization params
        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                # Capture the kwargs before calling parent
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            # Create parent RLM with max_timeout
            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,  # Need depth > 1 to allow child spawning
                max_timeout=60.0,
            )

            # Simulate that 10 seconds have elapsed since completion started
            parent._completion_start_time = time.perf_counter() - 10.0

            # Patch RLM class to capture child creation
            with patch.object(rlm_module, "RLM", CapturingRLM):
                # Call _subcall which should spawn a child RLM
                parent._subcall("test prompt")

            # Verify child received remaining timeout (approximately 50 seconds)
            assert "max_timeout" in captured_child_params
            remaining = captured_child_params["max_timeout"]
            # Allow some tolerance for test execution time
            assert 45.0 < remaining < 55.0, f"Expected ~50s remaining, got {remaining}"

            parent.close()

    def test_child_receives_none_timeout_when_parent_has_none(self):
        """When parent has no max_timeout, child should also have None."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_timeout=None,  # No timeout
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            assert captured_child_params.get("max_timeout") is None

            parent.close()

    def test_subcall_returns_error_when_timeout_exhausted(self):
        """When timeout is already exhausted, _subcall should return error message."""
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_timeout=10.0,
            )

            # Simulate that more time has elapsed than the timeout
            parent._completion_start_time = time.perf_counter() - 15.0

            result = parent._subcall("test prompt")

            assert "Error: Timeout exhausted" in result.response

            parent.close()


class TestSubcallTokensPropagation:
    """Tests for max_tokens propagation to child RLM."""

    def test_child_receives_max_tokens(self):
        """Child RLM should get same max_tokens as parent."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_tokens=50000,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            assert captured_child_params.get("max_tokens") == 50000

            parent.close()

    def test_child_receives_none_tokens_when_parent_has_none(self):
        """When parent has no max_tokens, child should also have None."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_tokens=None,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            assert captured_child_params.get("max_tokens") is None

            parent.close()


class TestSubcallErrorsPropagation:
    """Tests for max_errors propagation to child RLM."""

    def test_child_receives_max_errors(self):
        """Child RLM should get same max_errors as parent."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_errors=5,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            assert captured_child_params.get("max_errors") == 5

            parent.close()

    def test_child_receives_none_errors_when_parent_has_none(self):
        """When parent has no max_errors, child should also have None."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_errors=None,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            assert captured_child_params.get("max_errors") is None

            parent.close()


class TestSubcallModelOverride:
    """Tests for model= parameter override in _subcall."""

    def test_model_override_sets_child_backend_kwargs(self):
        """When llm_query(prompt, model='test-model') is called, child's backend_kwargs should have model_name='test-model'."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model", "api_key": "test-key"},
                max_depth=3,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                # Call _subcall with model override
                parent._subcall("test prompt", model="override-model")

            # Verify child received overridden model in backend_kwargs
            child_backend_kwargs = captured_child_params.get("backend_kwargs", {})
            assert child_backend_kwargs.get("model_name") == "override-model"
            # Original kwargs should be preserved
            assert child_backend_kwargs.get("api_key") == "test-key"

            parent.close()

    def test_model_override_does_not_mutate_parent_kwargs(self):
        """Model override should not mutate parent's backend_kwargs."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
            )

            original_model = parent.backend_kwargs["model_name"]

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt", model="override-model")

            # Parent's backend_kwargs should be unchanged
            assert parent.backend_kwargs["model_name"] == original_model

            parent.close()

    def test_no_model_override_uses_parent_kwargs(self):
        """When no model override is provided, child uses parent's backend_kwargs."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                # Call _subcall without model override
                parent._subcall("test prompt")

            # Child should use parent's backend_kwargs
            child_backend_kwargs = captured_child_params.get("backend_kwargs", {})
            assert child_backend_kwargs.get("model_name") == "parent-model"

            parent.close()


class TestSubcallModelOverrideAtLeafDepth:
    """Tests for model override at max_depth (leaf LM completion)."""

    def test_model_override_at_leaf_depth_uses_overridden_model(self):
        """When at max_depth, the leaf LM completion should use the overridden model."""
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["leaf response"])
            mock_get_client.return_value = mock_lm

            # Parent at depth 1, max_depth 2 means next depth (2) will be at max_depth
            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                depth=1,
                max_depth=2,
            )

            # Call _subcall with model override - should trigger leaf LM completion
            result = parent._subcall("test prompt", model="leaf-override-model")

            # Verify get_client was called with overridden model in backend_kwargs
            # The call should be: get_client("openai", {"model_name": "leaf-override-model"})
            call_args = mock_get_client.call_args_list
            # Find the call that has the overridden model
            found_override_call = False
            for call in call_args:
                args, kwargs = call
                if len(args) >= 2:
                    backend_kwargs = args[1]
                    if (
                        isinstance(backend_kwargs, dict)
                        and backend_kwargs.get("model_name") == "leaf-override-model"
                    ):
                        found_override_call = True
                        break

            assert found_override_call, (
                f"Expected get_client to be called with model_name='leaf-override-model', got calls: {call_args}"
            )
            assert result.response == "leaf response"

            parent.close()

    def test_leaf_depth_without_model_override_uses_parent_model(self):
        """When at max_depth without model override, uses parent's model."""
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")] * 2 + ["leaf response"])
            mock_get_client.return_value = mock_lm

            # Parent at depth 1, max_depth 2 means next depth (2) will be at max_depth
            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                depth=1,
                max_depth=2,
            )

            # Call _subcall without model override
            parent._subcall("test prompt")

            # Verify get_client was called with parent's model
            # The last call should use the parent's backend_kwargs
            call_args = mock_get_client.call_args_list
            # Check the most recent call (for leaf completion)
            last_call = call_args[-1]
            args, _ = last_call
            if len(args) >= 2:
                backend_kwargs = args[1]
                assert backend_kwargs.get("model_name") == "parent-model"

            parent.close()


def _capture_child_kwargs(parent_kwargs: dict, elapsed: float | None = None) -> dict:
    """Run _subcall on a parent built with parent_kwargs and return the kwargs
    the child RLM was constructed with. The child's completion() is stubbed so
    these tests exercise only propagation, never a child run."""
    captured = {}

    class CapturingRLM(rlm_module.RLM):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

        def completion(self, prompt, root_prompt=None):
            return RLMChatCompletion(
                root_model="mock-model",
                prompt=prompt,
                response="ok",
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=0.0,
            )

    with patch.object(rlm_module, "get_client") as mock_get_client:
        mock_get_client.return_value = create_mock_lm([final("answer")])
        parent = RLM(
            backend="openai",
            backend_kwargs={"model_name": "parent-model"},
            max_depth=3,
            **parent_kwargs,
        )
        if elapsed is not None:
            parent._completion_start_time = time.perf_counter() - elapsed
        with patch.object(rlm_module, "RLM", CapturingRLM):
            parent._subcall("test prompt")
        parent.close()
    return captured


class TestSubcallGuardPropagation:
    """The runaway-generation guards must follow rlm_query children. On
    2026-06-11 a whole-question delegation spawned a child whose sub-calls
    were uncapped and unscheduled because _subcall dropped these settings."""

    def test_child_receives_subcall_max_tokens(self):
        captured = _capture_child_kwargs({"subcall_max_tokens": 4096})
        assert captured.get("subcall_max_tokens") == 4096

    def test_child_receives_subcall_extra_body(self):
        """No-think (or any per-sub-call body extras) must follow children:
        a child's llm_query calls are as thought-channel-prone as the
        parent's (2026-06-12 empty-content incident)."""
        nothink = {"chat_template_kwargs": {"enable_thinking": False}}
        captured = _capture_child_kwargs({"subcall_extra_body": nothink})
        assert captured.get("subcall_extra_body") == nothink

    def test_child_receives_root_max_tokens(self):
        """A child's root iterations are as runaway-prone as the parent's."""
        captured = _capture_child_kwargs({"root_max_tokens": 8192})
        assert captured.get("root_max_tokens") == 8192

    def test_child_receives_scheduler_settings(self, tmp_path):
        captured = _capture_child_kwargs(
            {
                "scheduler_max_concurrent": 8,
                "scheduler_aging_interval": 15.0,
                "scheduler_coordination_dir": tmp_path,
            }
        )
        assert captured.get("scheduler_max_concurrent") == 8
        assert captured.get("scheduler_aging_interval") == 15.0
        assert captured.get("scheduler_coordination_dir") == tmp_path


class TestSubcallMaxTimeout:
    """subcall_max_timeout bounds each rlm_query child's slice of the budget:
    one delegated child must not be able to starve the parent's whole run."""

    def test_cap_bounds_child_share_of_remaining_budget(self):
        captured = _capture_child_kwargs(
            {"max_timeout": 600.0, "subcall_max_timeout": 240.0}, elapsed=10.0
        )
        assert captured.get("max_timeout") == 240.0

    def test_remaining_budget_smaller_than_cap_wins(self):
        captured = _capture_child_kwargs(
            {"max_timeout": 60.0, "subcall_max_timeout": 240.0}, elapsed=10.0
        )
        remaining = captured.get("max_timeout")
        assert 45.0 < remaining < 55.0, f"Expected ~50s remaining, got {remaining}"

    def test_cap_applies_even_without_parent_deadline(self):
        captured = _capture_child_kwargs(
            {"max_timeout": None, "subcall_max_timeout": 240.0}
        )
        assert captured.get("max_timeout") == 240.0

    def test_cap_propagates_to_grandchildren(self):
        captured = _capture_child_kwargs(
            {"max_timeout": 600.0, "subcall_max_timeout": 240.0}, elapsed=10.0
        )
        assert captured.get("subcall_max_timeout") == 240.0

    def test_no_cap_keeps_full_remaining(self):
        captured = _capture_child_kwargs({"max_timeout": 600.0}, elapsed=10.0)
        remaining = captured.get("max_timeout")
        assert 585.0 < remaining < 595.0, f"Expected ~590s remaining, got {remaining}"


class TestSubcallCombinedParameters:
    """Tests for combined parameter propagation."""

    def test_all_parameters_propagate_together(self):
        """All parameters (timeout, tokens, errors, model) should propagate correctly together."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model", "api_key": "test-key"},
                max_depth=3,
                max_timeout=120.0,
                max_tokens=100000,
                max_errors=10,
            )

            # Simulate 30 seconds elapsed
            parent._completion_start_time = time.perf_counter() - 30.0

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt", model="override-model")

            # Verify all parameters
            assert captured_child_params.get("max_tokens") == 100000
            assert captured_child_params.get("max_errors") == 10

            # Remaining timeout should be around 90 seconds
            remaining_timeout = captured_child_params.get("max_timeout")
            assert 85.0 < remaining_timeout < 95.0

            # Model should be overridden
            child_backend_kwargs = captured_child_params.get("backend_kwargs", {})
            assert child_backend_kwargs.get("model_name") == "override-model"
            assert child_backend_kwargs.get("api_key") == "test-key"

            parent.close()
