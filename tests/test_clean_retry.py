"""clean_retry_on_error: on a REPL error, the failed turn (broken code) is dropped
from the next prompt and replaced with a compact error note, so the model retries
fresh instead of escalating its broken attempt."""

from unittest.mock import Mock, patch

import lm_repl.core.rlm as rlm_module
from lm_repl import RLM
from lm_repl.core.types import ModelUsageSummary, UsageSummary

BROKEN = "```repl\nundefined_variable_xyz_marker\n```"  # NameError -> stderr


def _final(content: str) -> str:
    return f"```repl\nanswer['content'] = {content!r}\nanswer['ready'] = True\n```"


def _mock_lm(responses):
    m = Mock()
    m.completion.side_effect = list(responses)
    usage = UsageSummary(model_usage_summaries={
        "mock": ModelUsageSummary(total_calls=1, total_input_tokens=10, total_output_tokens=5)})
    m.get_usage_summary.return_value = usage
    m.get_last_usage.return_value = usage
    return m


def _second_prompt_text(mock_lm):
    return str(mock_lm.completion.call_args_list[1].args[0])


def test_clean_retry_drops_failed_turn():
    with patch.object(rlm_module, "get_client") as mgc:
        mock_lm = _mock_lm([BROKEN, _final("done")])
        mgc.return_value = mock_lm
        with RLM(backend="openai", backend_kwargs={"model_name": "t"},
                 clean_retry_on_error=True, max_iterations=4) as rlm:
            rlm.completion("ctx")
        text = _second_prompt_text(mock_lm)
    # the format_iteration code-echo ("Code executed:") and the assistant turn that
    # re-emits the ```repl block are both gone; only the compact error note remains.
    assert "Code executed:" not in text
    assert "```repl\\nundefined_variable_xyz_marker" not in text
    assert "discarded" in text


def test_default_keeps_failed_turn():
    with patch.object(rlm_module, "get_client") as mgc:
        mock_lm = _mock_lm([BROKEN, _final("done")])
        mgc.return_value = mock_lm
        with RLM(backend="openai", backend_kwargs={"model_name": "t"},
                 max_iterations=4) as rlm:  # clean_retry_on_error defaults False
            rlm.completion("ctx")
        text = _second_prompt_text(mock_lm)
    assert "Code executed:" in text                       # broken code echo retained (legacy behavior)
