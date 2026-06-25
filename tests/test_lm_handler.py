"""Tests for LMHandler using MockLM (no real LM required)."""

from prehend.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from prehend.core.lm_handler import LMHandler
from tests.mock_lm import MockLM


def test_lm_handler_single_request():
    """Single prompt request returns success and echo-style content."""
    mock = MockLM(responses=["hello back"])
    with LMHandler(client=mock) as handler:
        request = LMRequest(prompt="hello")
        response = send_lm_request(handler.address, request)
    assert response.success
    assert response.chat_completion is not None
    assert response.chat_completion.response == "hello back"


def test_lm_handler_batched_request():
    """Batched prompts return one response per prompt in order."""
    responses = [f"r{i}" for i in range(5)]
    mock = MockLM(responses=responses)
    with LMHandler(client=mock, batch_max_concurrent=3) as handler:
        prompts = [f"prompt-{i}" for i in range(5)]
        result = send_lm_request_batched(handler.address, prompts)
    assert len(result) == 5
    for i, resp in enumerate(result):
        assert resp.success, resp.error
        assert resp.chat_completion is not None
        assert resp.chat_completion.response == f"r{i}"


def test_lm_handler_batched_one_failure_does_not_poison_siblings():
    """One sub-call raising must NOT turn the whole batch into errors.

    Regression (2026-06-24 multihop epic-fail RCA): _handle_batched used
    asyncio.gather without return_exceptions, so a single oversized chunk's 400
    tore down the event loop and every in-flight sibling came back as
    APIConnectionError ("Connection error."). The good chunks must still return
    their answers; the bad one degrades to an "Error:" string that map_reduce
    already filters out of the reduce.
    """
    def fn(prompt):
        if "POISON" in str(prompt):
            raise RuntimeError("Connection error.")
        return f"ok:{prompt}"

    mock = MockLM(response_fn=fn)
    with LMHandler(client=mock, batch_max_concurrent=4) as handler:
        prompts = ["p-0", "p-1", "POISON", "p-3", "p-4"]
        result = send_lm_request_batched(handler.address, prompts)

    assert len(result) == 5
    # Good siblings preserved (the poison must not cascade onto them).
    for i in (0, 1, 3, 4):
        assert result[i].success, result[i].error
        assert result[i].chat_completion.response == f"ok:p-{i}"
    # The failing chunk surfaces as a per-prompt error string, not a batch-wide kill.
    assert result[2].chat_completion is not None
    assert result[2].chat_completion.response.startswith("Error:")


def test_lm_handler_batched_many_prompts_semaphore_cap():
    """Many prompts complete successfully with semaphore limiting concurrency."""
    # 50 prompts, max 4 concurrent: should still all complete
    count = 50
    responses = [f"resp-{i}" for i in range(count)]
    mock = MockLM(responses=responses)
    with LMHandler(client=mock, batch_max_concurrent=4) as handler:
        prompts = [f"p-{i}" for i in range(count)]
        result = send_lm_request_batched(handler.address, prompts)
    assert len(result) == count
    for i, resp in enumerate(result):
        assert resp.success, (i, resp.error)
        assert resp.chat_completion.response == f"resp-{i}"


def test_lm_handler_batches_share_one_persistent_loop():
    """Two batches on the same handler run on the SAME event loop (task #6).

    The old asyncio.run()-per-batch path created+destroyed a loop each batch,
    churning the AsyncOpenAI httpx transport (sglang keepalive-reuse races ->
    APIConnectionError). The persistent loop must be a single running loop reused
    across batches.
    """
    mock = MockLM(response_fn=lambda p: f"ok:{p}")
    with LMHandler(client=mock, batch_max_concurrent=4) as handler:
        loop1 = handler._loop
        assert loop1 is not None and loop1.is_running()
        send_lm_request_batched(handler.address, ["a-0", "a-1"])
        loop2 = handler._loop
        send_lm_request_batched(handler.address, ["b-0", "b-1"])
        loop3 = handler._loop
        # Same loop object across both batches: no per-batch teardown/churn.
        assert loop1 is loop2 is loop3
        assert loop3.is_running()


def test_lm_handler_stop_tears_down_loop_cleanly():
    """stop() must close the persistent loop and join its thread (bounded).

    The test simply COMPLETING is the proof that the bounded join did not hang
    (the one teardown risk Option A introduces).
    """
    mock = MockLM(responses=["x"])
    handler = LMHandler(client=mock)
    handler.start()
    loop, thread = handler._loop, handler._loop_thread
    assert loop is not None and loop.is_running()
    assert thread is not None and thread.is_alive()
    handler.stop()
    assert handler._loop is None
    assert handler._loop_thread is None
    assert not thread.is_alive()      # joined within the 5s bound
    assert not loop.is_running()
