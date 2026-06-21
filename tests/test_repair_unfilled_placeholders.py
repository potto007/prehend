"""repair_unfilled_placeholders: when the model builds a variable (e.g. `combined_data`)
and then references it as a literal ``{combined_data}`` inside a PLAIN (non-f) string passed
to llm_query / llm_query_batched, the data is never substituted and the sub-call sees an empty
placeholder ("No information was found..."). This guard detects a ``{name}`` that names a live,
model-created REPL variable and interpolates it before the sub-call runs. Opt-in (default off);
mirrors repair_doubled_calls.
"""

import prehend.environments.local_repl as local_repl
from prehend.core.comms_utils import LMResponse
from prehend.core.types import RLMChatCompletion, UsageSummary
from prehend.environments.local_repl import LocalREPL


def _echo(prompt: str) -> RLMChatCompletion:
    """A completion whose response IS the prompt - lets a test assert what the sub-call saw."""
    return RLMChatCompletion(
        root_model="echo",
        prompt=prompt,
        response=prompt,
        usage_summary=UsageSummary(model_usage_summaries={}),
        execution_time=0.0,
    )


def _echo_with(prompt: str, fn) -> RLMChatCompletion:
    """A completion whose response is fn(prompt) - lets a test react to the post-repair prompt."""
    return RLMChatCompletion(
        root_model="echo",
        prompt=prompt,
        response=fn(prompt),
        usage_summary=UsageSummary(model_usage_summaries={}),
        execution_time=0.0,
    )


def _patch_echo(monkeypatch):
    """Make llm_query / llm_query_batched echo the (post-repair) prompt back as the response."""
    def fake_single(address, request, *a, **k):
        return LMResponse.success_response(_echo(request.prompt))

    def fake_batched(address, prompts, *a, **k):
        return [LMResponse.success_response(_echo(p)) for p in prompts]

    monkeypatch.setattr(local_repl, "send_lm_request", fake_single)
    monkeypatch.setattr(local_repl, "send_lm_request_batched", fake_batched)


def _repl(**kw):
    return LocalREPL(lm_handler_address=("127.0.0.1", 0), **kw)


def test_fills_missing_fstring_placeholder(monkeypatch):
    _patch_echo(monkeypatch)
    repl = _repl(repair_unfilled_placeholders=True)
    r = repl.execute_code('data = "REAL DATA"\nout = llm_query("Synthesize this: {data}")')
    assert not (r.stderr or "").strip(), f"unexpected stderr: {r.stderr!r}"
    assert "REAL DATA" in repl.locals["out"]
    assert "{data}" not in repl.locals["out"]
    repl.cleanup()


def test_default_off_leaves_placeholder_literal(monkeypatch):
    _patch_echo(monkeypatch)
    repl = _repl()  # default off -> legacy behavior, no interpolation
    repl.execute_code('data = "REAL DATA"\nout = llm_query("Synthesize this: {data}")')
    assert "{data}" in repl.locals["out"]
    assert "REAL DATA" not in repl.locals["out"]
    repl.cleanup()


def test_unknown_placeholder_untouched(monkeypatch):
    _patch_echo(monkeypatch)
    repl = _repl(repair_unfilled_placeholders=True)
    # `result` is never defined -> nothing to fill, must be left literal (no false positive).
    repl.execute_code('out = llm_query("Return JSON like {result} here")')
    assert "{result}" in repl.locals["out"]
    repl.cleanup()


def test_scaffold_name_not_filled(monkeypatch):
    _patch_echo(monkeypatch)
    repl = _repl(repair_unfilled_placeholders=True)
    # llm_query is scaffold (a function in globals), not a model-created data var -> leave it.
    repl.execute_code('out = llm_query("please call {llm_query} now")')
    assert "{llm_query}" in repl.locals["out"]
    repl.cleanup()


def test_batched_fills_placeholder(monkeypatch):
    _patch_echo(monkeypatch)
    repl = _repl(repair_unfilled_placeholders=True)
    repl.execute_code('d = "X-CONTENT"\nouts = llm_query_batched(["use {d}", "no placeholder"])')
    outs = repl.locals["outs"]
    assert "X-CONTENT" in outs[0]
    assert outs[1] == "no placeholder"
    repl.cleanup()


def test_synthesis_dropped_data_regression(monkeypatch):
    """The captured v8 give-up (q1 rep5): extraction succeeded, but the synthesis prompt
    was a plain (non-f) string with a literal {combined_data}, so the sub-call saw no data
    and returned 'No information was found...'. With the guard the data is interpolated and
    the sub-call synthesizes a real answer instead."""
    def respond(prompt):
        # The synthesis sub-call: gives up iff the data placeholder was never filled.
        if "{combined_data}" in prompt:
            return "No information was found in the provided documents."
        return "Synthesis: penalties apply [083]."

    def fake_single(address, request, *a, **k):
        return LMResponse.success_response(_echo_with(request.prompt, respond))

    def fake_batched(address, prompts, *a, **k):
        # Map step returns real extracted notes per doc.
        return [LMResponse.success_response(_echo_with(p, lambda _p: "Doc note: ...")) for p in prompts]

    monkeypatch.setattr(local_repl, "send_lm_request", fake_single)
    monkeypatch.setattr(local_repl, "send_lm_request_batched", fake_batched)

    repl = _repl(repair_unfilled_placeholders=True)
    code = (
        'extracted_data = llm_query_batched(["extract from doc 083", "extract from doc 085"])\n'
        'combined_data = "\\n\\n".join(extracted_data)\n'
        'synthesis_query = """Using the extracted data, answer the question.\n\n'
        'Extracted Data:\n'
        '{combined_data}\n"""\n'  # plain string - the missing-f-string bug
        'final_answer = llm_query(synthesis_query)\n'
        'print(final_answer)'
    )
    r = repl.execute_code(code)
    assert not (r.stderr or "").strip(), f"unexpected stderr: {r.stderr!r}"
    assert repl.locals["final_answer"] == "Synthesis: penalties apply [083]."
    assert "No information was found" not in repl.locals["final_answer"]
    repl.cleanup()


def test_repair_is_noted_in_stdout(monkeypatch):
    _patch_echo(monkeypatch)
    repl = _repl(repair_unfilled_placeholders=True)
    r = repl.execute_code('data = "REAL DATA"\nout = llm_query("Synthesize: {data}")')
    assert "data" in (r.stdout or "")
    assert "placeholder" in (r.stdout or "").lower()
    repl.cleanup()
