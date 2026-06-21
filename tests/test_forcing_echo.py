"""The forced-final prompt is injected as an assistant turn, so the model often
ECHOES it verbatim before the real answer (observed across the 2026-06-16
kb-librarian eval: 3/28 completions began with "Please provide a final answer to
the user's question based on the information provided."). _strip_forcing_echo
removes that leading echo; _default_answer returns the cleaned text.
"""

from unittest.mock import patch

from prehend import RLM
from prehend.core.rlm import _FORCE_FINAL_MSG, _strip_forcing_echo
from tests.mock_lm import MockLM


def test_strip_removes_leading_echo():
    out = _strip_forcing_echo(_FORCE_FINAL_MSG + "\n\nNine services are required [002].")
    assert out == "Nine services are required [002]."


def test_strip_is_case_and_whitespace_tolerant():
    out = _strip_forcing_echo("  " + _FORCE_FINAL_MSG.upper() + "   The answer is 42 [006].")
    assert out == "The answer is 42 [006]."


def test_strip_noop_without_echo():
    ans = "The answer is 42 [006]."
    assert _strip_forcing_echo(ans) == ans


def test_strip_keeps_echo_only_response():
    # degenerate: model returned nothing but the echo - keep it, don't return ""
    assert _strip_forcing_echo(_FORCE_FINAL_MSG) == _FORCE_FINAL_MSG
    assert _strip_forcing_echo("") == ""


def _rlm():
    return RLM(
        backend="openai",
        backend_kwargs={"model_name": "test", "base_url": "http://localhost:9999/v1"},
        environment="local",
        max_iterations=1,
        max_depth=1,
    )


def test_default_answer_returns_stripped_answer_end_to_end():
    echoed = _FORCE_FINAL_MSG + "\n\nNine services are required [002, 012]."

    def respond(prompt):
        if _FORCE_FINAL_MSG[:20] in str(prompt):
            return echoed
        return "working notes"

    mock = MockLM(response_fn=respond)
    with patch("prehend.core.rlm.get_client", return_value=mock):
        result = _rlm().completion("hard question")
    assert result.response == "Nine services are required [002, 012]."
