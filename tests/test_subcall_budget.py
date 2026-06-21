"""Retrieval circuit-breaker (rlm-trainer eval finding, issue #4, 2026-06-17).

The RLM orchestrator's retrieval loop was unbounded: a single ask could issue
hundreds of llm_query sub-calls (one observed run made 579) before the time
budget stopped it - collapsing answerable asks into refusals, inflating tail
latency, and escaping the soft-budget. A hard per-completion sub-call cap
(max_subcalls) short-circuits further reads once breached, returning a wrap-up
instruction instead of hitting the server, so the model must answer from what it
has already gathered. Default off (None) in lm-repl; the librarian sets the value.
"""

from types import SimpleNamespace
from unittest.mock import patch

from lm_repl.environments.local_repl import (
    _SUBCALL_BUDGET_MSG,
    LocalREPL,
    _subcall_budget_remaining,
)

# ---- pure decision function ---------------------------------------------------

def test_remaining_unlimited_when_disabled():
    assert _subcall_budget_remaining(999, None) is None
    assert _subcall_budget_remaining(999, 0) is None  # <=0 treated as off


def test_remaining_counts_down_and_floors_at_zero():
    assert _subcall_budget_remaining(0, 50) == 50
    assert _subcall_budget_remaining(49, 50) == 1
    assert _subcall_budget_remaining(50, 50) == 0
    assert _subcall_budget_remaining(80, 50) == 0  # never negative


# ---- _llm_query gate ----------------------------------------------------------

def _env(**kw):
    return LocalREPL(lm_handler_address=("127.0.0.1", 1), **kw)


def _ok(_addr, _req):
    return SimpleNamespace(success=True, chat_completion=SimpleNamespace(response="DOC TEXT"))


def test_llm_query_blocks_after_cap():
    env = _env(max_subcalls=2)
    with patch("lm_repl.environments.local_repl.send_lm_request", side_effect=_ok) as send:
        r1 = env._llm_query("read doc 1")
        r2 = env._llm_query("read doc 2")
        r3 = env._llm_query("read doc 3")  # over budget -> blocked
    assert r1 == "DOC TEXT"
    assert r2 == "DOC TEXT"
    assert _SUBCALL_BUDGET_MSG[:20] in r3
    assert send.call_count == 2  # the 3rd never hit the server


def test_llm_query_unlimited_when_disabled():
    env = _env()  # max_subcalls default None
    with patch("lm_repl.environments.local_repl.send_lm_request", side_effect=_ok) as send:
        for _ in range(10):
            assert env._llm_query("q") == "DOC TEXT"
    assert send.call_count == 10


# ---- _llm_query_batched gate --------------------------------------------------

def _ok_batched(_addr, prompts, **kw):
    return [
        SimpleNamespace(success=True, chat_completion=SimpleNamespace(response=f"R{i}"))
        for i, _ in enumerate(prompts)
    ]


def test_llm_query_batched_partitions_at_cap():
    env = _env(max_subcalls=2)
    with patch(
        "lm_repl.environments.local_repl.send_lm_request_batched", side_effect=_ok_batched
    ) as send:
        results = env._llm_query_batched(["a", "b", "c", "d"])
    # first 2 dispatched, last 2 blocked with the budget message, order preserved
    assert results[0] == "R0"
    assert results[1] == "R1"
    assert _SUBCALL_BUDGET_MSG[:20] in results[2]
    assert _SUBCALL_BUDGET_MSG[:20] in results[3]
    # only the 2 allowed prompts reached the server
    assert send.call_args[0][1] == ["a", "b"]


def test_llm_query_batched_all_blocked_when_already_exhausted():
    env = _env(max_subcalls=1)
    with patch("lm_repl.environments.local_repl.send_lm_request", side_effect=_ok):
        env._llm_query("burn the one allowed call")
    with patch(
        "lm_repl.environments.local_repl.send_lm_request_batched", side_effect=_ok_batched
    ) as send:
        results = env._llm_query_batched(["a", "b"])
    assert all(_SUBCALL_BUDGET_MSG[:20] in r for r in results)
    assert send.call_count == 0  # nothing dispatched once budget is spent


def test_count_is_shared_across_query_and_batched():
    env = _env(max_subcalls=3)
    with patch("lm_repl.environments.local_repl.send_lm_request", side_effect=_ok), patch(
        "lm_repl.environments.local_repl.send_lm_request_batched", side_effect=_ok_batched
    ) as sendb:
        env._llm_query("one")  # count -> 1
        results = env._llm_query_batched(["two", "three", "four"])  # remaining 2
    assert results[0] == "R0"
    assert results[1] == "R1"
    assert _SUBCALL_BUDGET_MSG[:20] in results[2]
    assert sendb.call_args[0][1] == ["two", "three"]
