"""Tests for the priority RequestScheduler and its OpenAIClient integration."""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import httpx
import openai as openai_sdk
import pytest

from lm_repl.clients.openai import OpenAIClient, _is_context_contention
from lm_repl.clients.scheduler import Priority, RequestScheduler, resolve_priority
from lm_repl.core.comms_utils import LMRequest
from lm_repl.core.lm_handler import LMHandler


def wait_until(cond, timeout=2.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


# =============================================================================
# resolve_priority
# =============================================================================


def test_resolve_priority_mappings():
    assert resolve_priority(None) == Priority.NORMAL
    assert resolve_priority("high") == Priority.HIGH
    assert resolve_priority("normal") == Priority.NORMAL
    assert resolve_priority("low") == Priority.LOW
    assert resolve_priority(1) == Priority.CONTENTION_RETRY
    assert resolve_priority(5) == Priority.BACKGROUND
    # Out-of-range / unknown values fall back to NORMAL
    assert resolve_priority(0) == Priority.NORMAL
    assert resolve_priority(99) == Priority.NORMAL
    assert resolve_priority("urgent") == Priority.NORMAL


# =============================================================================
# Scheduler core (sync)
# =============================================================================


def test_acquire_release_respects_max_concurrent():
    s = RequestScheduler(max_concurrent=2)
    s.acquire(Priority.NORMAL)
    s.acquire(Priority.NORMAL)
    assert s.active == 2

    third_admitted = threading.Event()

    def third():
        s.acquire(Priority.NORMAL)
        third_admitted.set()

    t = threading.Thread(target=third, daemon=True)
    t.start()
    assert wait_until(lambda: s.waiting == 1)
    assert not third_admitted.is_set()

    s.release(Priority.NORMAL)
    assert third_admitted.wait(2)
    assert s.active == 2

    s.release(Priority.NORMAL)
    s.release(Priority.NORMAL)
    assert s.active == 0


def test_p1_exclusivity_blocks_other_priorities():
    s = RequestScheduler(max_concurrent=4)
    s.acquire(Priority.CONTENTION_RETRY)

    normal_admitted = threading.Event()

    def normal():
        s.acquire(Priority.NORMAL)
        normal_admitted.set()

    t = threading.Thread(target=normal, daemon=True)
    t.start()
    assert wait_until(lambda: s.waiting == 1)
    # Capacity is available (1 of 4 used) but p1 is active, so NORMAL must wait.
    assert not normal_admitted.is_set()

    # Another p1 may still start while a p1 is active.
    s.acquire(Priority.CONTENTION_RETRY)
    assert s.active == 2
    s.release(Priority.CONTENTION_RETRY)

    s.release(Priority.CONTENTION_RETRY)
    assert normal_admitted.wait(2)
    s.release(Priority.NORMAL)
    assert s.active == 0


def test_priority_ordering_high_before_normal():
    s = RequestScheduler(max_concurrent=1)
    s.acquire(Priority.NORMAL)  # occupy the only slot

    order = []

    def waiter(priority, tag):
        s.acquire(priority)
        order.append(tag)
        s.release(priority)

    # Enqueue LOW first so FIFO alone would dispatch it first.
    t_low = threading.Thread(target=waiter, args=(Priority.LOW, "low"), daemon=True)
    t_low.start()
    assert wait_until(lambda: s.waiting == 1)
    t_high = threading.Thread(target=waiter, args=(Priority.HIGH, "high"), daemon=True)
    t_high.start()
    assert wait_until(lambda: s.waiting == 2)

    s.release(Priority.NORMAL)
    t_low.join(2)
    t_high.join(2)
    assert order == ["high", "low"]
    assert s.active == 0


def test_fifo_within_same_priority():
    s = RequestScheduler(max_concurrent=1)
    s.acquire(Priority.NORMAL)

    order = []

    def waiter(tag):
        s.acquire(Priority.NORMAL)
        order.append(tag)
        s.release(Priority.NORMAL)

    threads = []
    for i in range(3):
        t = threading.Thread(target=waiter, args=(i,), daemon=True)
        t.start()
        assert wait_until(lambda n=i + 1: s.waiting == n)
        threads.append(t)

    s.release(Priority.NORMAL)
    for t in threads:
        t.join(2)
    assert order == [0, 1, 2]
    assert s.active == 0


# =============================================================================
# Scheduler core (async + mixed)
# =============================================================================


def test_async_acquire_release():
    async def main():
        s = RequestScheduler(max_concurrent=1)
        await s.aacquire(Priority.NORMAL)

        async def waiter():
            await s.aacquire(Priority.NORMAL)
            await s.arelease(Priority.NORMAL)
            return "done"

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        assert s.waiting == 1

        await s.arelease(Priority.NORMAL)
        assert await asyncio.wait_for(task, 2) == "done"
        assert s.active == 0

    asyncio.run(main())


def test_sync_release_wakes_async_waiter():
    async def main():
        s = RequestScheduler(max_concurrent=1)
        s.acquire(Priority.NORMAL)  # sync holder

        async def waiter():
            await s.aacquire(Priority.NORMAL)
            await s.arelease(Priority.NORMAL)
            return "done"

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        assert s.waiting == 1

        s.release(Priority.NORMAL)
        assert await asyncio.wait_for(task, 2) == "done"
        assert s.active == 0

    asyncio.run(main())


def test_async_priority_beats_sync_fifo():
    async def main():
        s = RequestScheduler(max_concurrent=1)
        s.acquire(Priority.NORMAL)

        order = []

        def sync_waiter():
            s.acquire(Priority.LOW)
            order.append("sync-low")
            s.release(Priority.LOW)

        t = threading.Thread(target=sync_waiter, daemon=True)
        t.start()
        assert wait_until(lambda: s.waiting == 1)

        async def async_waiter():
            await s.aacquire(Priority.HIGH)
            order.append("async-high")
            await s.arelease(Priority.HIGH)

        task = asyncio.create_task(async_waiter())
        await asyncio.sleep(0.05)
        assert s.waiting == 2

        s.release(Priority.NORMAL)
        await asyncio.wait_for(task, 2)
        await asyncio.get_running_loop().run_in_executor(None, t.join, 2)
        assert order == ["async-high", "sync-low"]
        assert s.active == 0

    asyncio.run(main())


# =============================================================================
# Context-contention detection
# =============================================================================


def _contention_error() -> openai_sdk.BadRequestError:
    request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
    response = httpx.Response(400, request=request)
    body = {
        "error": {
            "code": 400,
            "message": (
                "request (40069 tokens) exceeds the available context size "
                "(22016 tokens), try increasing it"
            ),
            "type": "exceed_context_size_error",
        }
    }
    return openai_sdk.BadRequestError(
        body["error"]["message"], response=response, body=body
    )


def _other_400_error() -> openai_sdk.BadRequestError:
    request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
    response = httpx.Response(400, request=request)
    body = {"error": {"code": 400, "message": "invalid request", "type": "invalid_request_error"}}
    return openai_sdk.BadRequestError("invalid request", response=response, body=body)


def test_is_context_contention():
    assert _is_context_contention(_contention_error())
    assert not _is_context_contention(_other_400_error())


# =============================================================================
# OpenAIClient integration
# =============================================================================


def _ok_response(content="ok"):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = MagicMock(
        prompt_tokens=5, completion_tokens=5, total_tokens=10, cost=None, model_extra=None
    )
    return response


def _make_client(scheduler=None):
    with patch("lm_repl.clients.openai.openai.OpenAI"):
        with patch("lm_repl.clients.openai.openai.AsyncOpenAI"):
            return OpenAIClient(api_key="test", model_name="test-model", scheduler=scheduler)


def test_completion_contention_retries_at_p1():
    scheduler = RequestScheduler(max_concurrent=2)
    client = _make_client(scheduler)
    client.client.chat.completions.create = MagicMock(
        side_effect=[_contention_error(), _ok_response()]
    )

    with patch.object(scheduler, "acquire", wraps=scheduler.acquire) as acquire_spy:
        result = client.completion("hello")

    assert result == "ok"
    assert client.client.chat.completions.create.call_count == 2
    priorities = [call.args[0] for call in acquire_spy.call_args_list]
    assert priorities == [Priority.NORMAL, Priority.CONTENTION_RETRY]
    # No leaked slots or p1 markers after the retry path (double-release regression).
    assert scheduler.active == 0
    assert scheduler._active_p1 == 0
    assert scheduler.waiting == 0


def test_completion_contention_retry_fails_propagates():
    scheduler = RequestScheduler(max_concurrent=2)
    client = _make_client(scheduler)
    client.client.chat.completions.create = MagicMock(
        side_effect=[_contention_error(), _contention_error()]
    )

    with pytest.raises(openai_sdk.BadRequestError):
        client.completion("hello")

    # Retried exactly once: a p1 failure means the request is genuinely too large.
    assert client.client.chat.completions.create.call_count == 2
    assert scheduler.active == 0
    assert scheduler._active_p1 == 0


def test_completion_non_contention_400_no_retry():
    scheduler = RequestScheduler(max_concurrent=2)
    client = _make_client(scheduler)
    client.client.chat.completions.create = MagicMock(side_effect=_other_400_error())

    with pytest.raises(openai_sdk.BadRequestError):
        client.completion("hello")

    assert client.client.chat.completions.create.call_count == 1
    assert scheduler.active == 0


def test_completion_without_scheduler_unchanged():
    client = _make_client(scheduler=None)
    client.client.chat.completions.create = MagicMock(return_value=_ok_response())
    assert client.completion("hello") == "ok"

    # Without a scheduler there is no p1 exclusivity, so contention is not retried.
    client.client.chat.completions.create = MagicMock(side_effect=_contention_error())
    with pytest.raises(openai_sdk.BadRequestError):
        client.completion("hello")
    assert client.client.chat.completions.create.call_count == 1


def test_completion_priority_strings_are_resolved():
    scheduler = RequestScheduler(max_concurrent=2)
    client = _make_client(scheduler)
    client.client.chat.completions.create = MagicMock(return_value=_ok_response())

    with patch.object(scheduler, "acquire", wraps=scheduler.acquire) as acquire_spy:
        client.completion("hello", priority="high")
        client.completion("hello", priority="low")
        client.completion("hello")

    priorities = [call.args[0] for call in acquire_spy.call_args_list]
    assert priorities == [Priority.HIGH, Priority.LOW, Priority.NORMAL]
    assert scheduler.active == 0


def test_acompletion_contention_retries_at_p1():
    async def main():
        scheduler = RequestScheduler(max_concurrent=2)
        client = _make_client(scheduler)

        calls = []

        async def fake_create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise _contention_error()
            return _ok_response()

        client.async_client.chat.completions.create = fake_create

        acquired = []
        real_aacquire = scheduler.aacquire

        async def spy_aacquire(priority):
            acquired.append(priority)
            await real_aacquire(priority)

        scheduler.aacquire = spy_aacquire

        result = await client.acompletion("hello")
        assert result == "ok"
        assert len(calls) == 2
        assert acquired == [Priority.NORMAL, Priority.CONTENTION_RETRY]
        assert scheduler.active == 0
        assert scheduler._active_p1 == 0

    asyncio.run(main())


def test_acompletion_without_scheduler_unchanged():
    async def main():
        client = _make_client(scheduler=None)

        async def fake_create(**kwargs):
            return _ok_response("async-ok")

        client.async_client.chat.completions.create = fake_create
        assert await client.acompletion("hello") == "async-ok"

    asyncio.run(main())


# =============================================================================
# LMRequest protocol round-trip
# =============================================================================


def test_lm_request_priority_round_trip():
    request = LMRequest(prompt="hi", priority="high", depth=0)
    data = request.to_dict()
    assert data["priority"] == "high"
    assert LMRequest.from_dict(data).priority == "high"

    # Omitted when unset, and absent keys round-trip to None.
    data = LMRequest(prompt="hi", depth=0).to_dict()
    assert "priority" not in data
    assert LMRequest.from_dict(data).priority is None


# =============================================================================
# LMHandler scheduler creation
# =============================================================================


def test_lm_handler_creates_and_attaches_scheduler():
    default = _make_client()
    other = _make_client()
    handler = LMHandler(
        default, other_backend_client=other, scheduler_max_concurrent=4
    )
    assert isinstance(handler.scheduler, RequestScheduler)
    # Both clients share the one scheduler so the priority queue spans all traffic.
    assert default.scheduler is handler.scheduler
    assert other.scheduler is handler.scheduler


def test_lm_handler_no_scheduler_by_default():
    client = _make_client()
    handler = LMHandler(client)
    assert handler.scheduler is None
    assert client.scheduler is None
