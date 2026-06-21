"""TimeoutExceededError must carry _best_partial_answer from EVERY raise path.

Born from the 2026-06-11 null-salvage incident: a kb-librarian ask burned its
10 iterations, fell into _default_answer, and the run deadline aborted that
final generation mid-stream. The client's TimeoutExceededError propagated
without the 10 iterations of accumulated notes - the caller got
partial_answer: null despite minutes of completed work.
"""

from unittest.mock import patch

import pytest

from prehend import RLM
from prehend.utils.exceptions import TimeoutExceededError
from tests.mock_lm import MockLM

ITERATION_NOTES = "Working notes: doc 006 covers the licensing rule."


def _rlm(max_iterations: int) -> RLM:
    return RLM(
        backend="openai",
        backend_kwargs={"model_name": "test", "base_url": "http://localhost:9999/v1"},
        environment="local",
        max_iterations=max_iterations,
        max_depth=1,
    )


def test_timeout_during_default_answer_carries_partial():
    """Deadline abort inside _default_answer (iterations exhausted) must still
    attach the best partial answer from the completed iterations."""

    def respond(prompt):
        if "Please provide a final answer" in str(prompt):
            raise TimeoutExceededError(elapsed=301.0, timeout=300.0)
        return ITERATION_NOTES

    mock = MockLM(response_fn=respond)
    with patch("prehend.core.rlm.get_client", return_value=mock):
        with pytest.raises(TimeoutExceededError) as excinfo:
            _rlm(max_iterations=1).completion("hard question")
    assert excinfo.value.partial_answer == ITERATION_NOTES


def test_timeout_mid_iteration_carries_partial():
    """Deadline abort during a root iteration call keeps the previous
    iteration's response as the partial answer."""
    calls = {"n": 0}

    def respond(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return ITERATION_NOTES
        raise TimeoutExceededError(elapsed=301.0, timeout=300.0)

    mock = MockLM(response_fn=respond)
    with patch("prehend.core.rlm.get_client", return_value=mock):
        with pytest.raises(TimeoutExceededError) as excinfo:
            _rlm(max_iterations=3).completion("hard question")
    assert excinfo.value.partial_answer == ITERATION_NOTES
