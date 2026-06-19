"""Repeat-guard escalation (rlm-trainer #6 part 3 follow-up).

The lm-repl repeat-guard aborts a single looping leaf completion fast, but an ask
can re-enter the loop many times - each abort cheap, the ask still degenerate. Once
the cumulative guard-abort count for a completion crosses repeat_guard_abort_limit,
inject the SAME one-time wrap-up message the soft-budget uses, so a persistent
looper is forced to answer-from-context or refuse cleanly instead of producing a
fast-but-empty degeneration. Generic + default-off; the librarian supplies the
policy message (its exact no-coverage sentence) and the limit.
"""

from unittest.mock import patch

from lm_repl import RLM
from lm_repl.core.rlm import _guard_escalation_due
from tests.mock_lm import MockLM

# ---- pure decision function ---------------------------------------------------

def test_not_due_below_limit():
    assert _guard_escalation_due(aborts=2, limit=3, already_fired=False) is False


def test_due_at_and_above_limit():
    assert _guard_escalation_due(3, 3, False) is True
    assert _guard_escalation_due(9, 3, False) is True


def test_noop_when_limit_none():
    assert _guard_escalation_due(99, None, False) is False


def test_noop_when_already_fired():
    assert _guard_escalation_due(99, 3, already_fired=True) is False


def test_noop_when_limit_non_positive():
    assert _guard_escalation_due(99, 0, False) is False
    assert _guard_escalation_due(99, -1, False) is False


# ---- injection side effect ----------------------------------------------------

def _rlm(**kw):
    return RLM(
        backend="openai",
        backend_kwargs={"model_name": "test", "base_url": "http://localhost:9999/v1"},
        environment="local",
        max_iterations=1,
        max_depth=1,
        max_timeout=100.0,
        **kw,
    )


def test_inject_appends_wrapup_once_and_sets_flag():
    rlm = _rlm(repeat_guard_abort_limit=3, soft_timeout_message="WRAP UP or refuse.")
    rlm._guard_escalation_fired = False
    hist = [{"role": "user", "content": "q"}]

    assert rlm._maybe_inject_guard_escalation(hist, aborts=3) is True
    assert rlm._guard_escalation_fired is True
    assert hist[-1] == {"role": "user", "content": "WRAP UP or refuse."}
    n = len(hist)

    # at most once per completion
    assert rlm._maybe_inject_guard_escalation(hist, aborts=5) is False
    assert len(hist) == n


def test_disabled_by_default():
    rlm = _rlm(soft_timeout_message="WRAP")  # no repeat_guard_abort_limit
    rlm._guard_escalation_fired = False
    hist = []
    assert rlm._maybe_inject_guard_escalation(hist, aborts=999) is False
    assert hist == []


# ---- end-to-end: the wrap-up reaches the model once the abort count crosses -----

def test_escalation_message_reaches_model_e2e():
    """A model that loops forever until it sees the wrap-up message, then answers.
    The handler reports a guard-abort count over the limit (as the client would
    after repeated stream aborts), so the escalation fires and the model wraps up."""
    msg = "WRAP UP: answer from what you have or say you cannot."
    seen = {"wrapup": False}

    def respond(prompt):
        if msg[:20] in str(prompt):
            seen["wrapup"] = True
            return "answer['content'] = 'Final: refused'\nanswer['ready'] = True"
        return "x = 1  # still looping"

    mock = MockLM(response_fn=respond)
    mock.repeat_guard_aborts = 5  # what the OpenAI client would report after 5 aborts
    with patch("lm_repl.core.rlm.get_client", return_value=mock):
        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "test", "base_url": "http://localhost:9999/v1"},
            environment="local",
            max_iterations=5,
            max_depth=1,
            max_timeout=100.0,
            repeat_guard_abort_limit=3,
            soft_timeout_message=msg,
        )
        result = rlm.completion("looping question")

    assert seen["wrapup"] is True
    assert "Final: refused" in result.response
