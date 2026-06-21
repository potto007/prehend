"""Soft-budget early wrap-up (rlm-trainer eval finding #4, 2026-06-16).

On uncovered or hard questions the model keeps searching until the HARD deadline
(max_timeout) elapses - the run then 504s with no answer, or its reasoning
degenerates into a token loop. The soft budget injects a ONE-TIME wrap-up message
once a configurable fraction of max_timeout has elapsed, telling the model to
answer from what it has gathered or refuse cleanly while time remains. This turns
the slow tail into clean completions/refusals and caps tail latency.

Mechanism lives in lm-repl (generic, default-off); the librarian supplies the
policy message (its exact no-coverage refusal sentence).
"""

from unittest.mock import patch

from lm_repl import RLM
from lm_repl.core.rlm import _SOFT_BUDGET_MSG, _soft_budget_due
from tests.mock_lm import MockLM

# ---- pure decision function ---------------------------------------------------

def test_not_due_before_threshold():
    assert _soft_budget_due(elapsed=40.0, max_timeout=100.0, soft_pct=0.7, already_fired=False) is False


def test_due_at_and_after_threshold():
    assert _soft_budget_due(70.0, 100.0, 0.7, False) is True
    assert _soft_budget_due(95.0, 100.0, 0.7, False) is True


def test_noop_when_pct_none():
    assert _soft_budget_due(99.0, 100.0, None, False) is False


def test_noop_when_max_timeout_none():
    assert _soft_budget_due(99.0, None, 0.7, False) is False


def test_noop_when_already_fired():
    assert _soft_budget_due(99.0, 100.0, 0.7, already_fired=True) is False


def test_noop_when_pct_out_of_range():
    assert _soft_budget_due(99.0, 100.0, 0.0, False) is False
    assert _soft_budget_due(99.0, 100.0, 1.0, False) is False
    assert _soft_budget_due(99.0, 100.0, -0.5, False) is False


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
    rlm = _rlm(soft_timeout_pct=0.7)
    rlm._soft_budget_fired = False
    hist = [{"role": "user", "content": "q"}]

    fired = rlm._maybe_inject_soft_budget(hist, elapsed=80.0)
    assert fired is True
    assert rlm._soft_budget_fired is True
    assert hist[-1]["role"] == "user"
    assert hist[-1]["content"] == _SOFT_BUDGET_MSG
    n = len(hist)

    # second call is a no-op (fires at most once per completion)
    assert rlm._maybe_inject_soft_budget(hist, elapsed=95.0) is False
    assert len(hist) == n


def test_inject_uses_custom_message():
    rlm = _rlm(soft_timeout_pct=0.5, soft_timeout_message="WRAP UP NOW or say you cannot.")
    rlm._soft_budget_fired = False
    hist = []
    assert rlm._maybe_inject_soft_budget(hist, elapsed=60.0) is True
    assert hist[-1]["content"] == "WRAP UP NOW or say you cannot."


def test_disabled_by_default():
    rlm = _rlm()  # no soft_timeout_pct
    rlm._soft_budget_fired = False
    hist = []
    assert rlm._maybe_inject_soft_budget(hist, elapsed=999.0) is False
    assert hist == []


# ---- end-to-end: the wrap-up message reaches the model ------------------------

def test_soft_budget_message_reaches_model_e2e():
    """With a fake clock past the soft threshold, the model that loops forever
    until it sees the wrap-up message then produces a final answer."""
    seen = {"soft": False}

    def respond(prompt):
        text = str(prompt)
        if _SOFT_BUDGET_MSG[:25] in text:
            seen["soft"] = True
            return "answer['content'] = 'Final: 9 services [002]'\nanswer['ready'] = True"
        # before the wrap-up: keep producing non-final working notes
        return "x = 1  # still searching"

    # Fake monotonic clock: time_start reads 0, every later read returns 80
    # (past 0.7*100=70 but below the 100 hard deadline), so _check_timeout
    # passes and the soft budget fires on iteration 0.
    clock = iter([0.0] + [80.0] * 50)

    mock = MockLM(response_fn=respond)
    with patch("lm_repl.core.rlm.get_client", return_value=mock), \
         patch("lm_repl.core.rlm.time.perf_counter", lambda: next(clock, 80.0)):
        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "test", "base_url": "http://localhost:9999/v1"},
            environment="local",
            max_iterations=5,
            max_depth=1,
            max_timeout=100.0,
            soft_timeout_pct=0.7,
        )
        result = rlm.completion("hard question")

    assert seen["soft"] is True
    assert "Final: 9 services [002]" in result.response
