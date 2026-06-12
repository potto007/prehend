"""Tests for the runaway-generation guards: subcall_max_tokens, run deadlines,
and in-flight cancellation (born from the 2026-06-11 zombie-generation incident:
a timed-out ask left sub-calls generating 35K+ tokens server-side)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lm_repl.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from lm_repl.core.lm_handler import LMHandler
from lm_repl.utils.exceptions import CancellationError, TimeoutExceededError
from tests.mock_lm import MockLM


# ---------------------------------------------------------------------------
# subcall_max_tokens plumbing (handler -> client)
# ---------------------------------------------------------------------------


def test_single_subcall_gets_max_tokens_cap():
    mock = MockLM(responses=["ok"])
    with LMHandler(client=mock, subcall_max_tokens=2048) as handler:
        request = LMRequest(prompt="hi")
        response = send_lm_request(handler.address, request)
    assert response.success
    assert mock.seen_max_tokens == [2048]


def test_batched_subcalls_get_max_tokens_cap():
    mock = MockLM(responses=["a", "b", "c"])
    with LMHandler(client=mock, subcall_max_tokens=512) as handler:
        responses = send_lm_request_batched(handler.address, ["p1", "p2", "p3"])
    assert all(r.success for r in responses)
    assert mock.seen_max_tokens == [512, 512, 512]


def test_no_cap_by_default():
    mock = MockLM(responses=["ok"])
    with LMHandler(client=mock) as handler:
        response = send_lm_request(handler.address, LMRequest(prompt="hi"))
    assert response.success
    assert mock.seen_max_tokens == [None]


def test_root_completion_not_capped_by_subcall_limit():
    """subcall_max_tokens does not touch the root orchestrator path."""
    mock = MockLM(responses=["ok"])
    handler = LMHandler(client=mock, subcall_max_tokens=128)
    assert handler.completion("root prompt") == "ok"
    assert mock.seen_max_tokens == [None]


def test_root_completion_capped_by_root_limit():
    """root_max_tokens bounds root orchestrator generations: the forced final
    REDUCE on 2026-06-11 ran away to ~50K tokens (n_tokens 65024 at deadline
    cancel) because the root path had no cap at all."""
    mock = MockLM(responses=["ok"])
    handler = LMHandler(client=mock, subcall_max_tokens=128, root_max_tokens=8192)
    assert handler.completion("root prompt") == "ok"
    assert mock.seen_max_tokens == [8192]


def test_root_limit_does_not_leak_into_subcalls():
    """Sub-calls keep their own (tighter) cap when both are set."""
    mock = MockLM(responses=["ok"])
    with LMHandler(client=mock, subcall_max_tokens=128, root_max_tokens=8192) as handler:
        response = send_lm_request(handler.address, LMRequest(prompt="hi"))
    assert response.success
    assert mock.seen_max_tokens == [128]


def test_rlm_wires_root_max_tokens_through_to_root_calls():
    import lm_repl.core.rlm as rlm_module
    from lm_repl import RLM
    from tests.test_subcall import create_mock_lm, final

    with patch.object(rlm_module, "get_client") as mock_get_client:
        mock_lm = create_mock_lm([final("answer")])
        mock_get_client.return_value = mock_lm
        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "m"},
            root_max_tokens=8192,
        )
        rlm.completion("context", root_prompt="q")
        root_call_kwargs = mock_lm.completion.call_args_list[0].kwargs
        assert root_call_kwargs.get("max_tokens") == 8192


# ---------------------------------------------------------------------------
# cancel_inflight / set_run_deadline fan-out
# ---------------------------------------------------------------------------


def _patched_openai_client(**kwargs):
    from lm_repl.clients.openai import OpenAIClient

    with patch("lm_repl.clients.openai.openai.OpenAI"), patch(
        "lm_repl.clients.openai.openai.AsyncOpenAI"
    ):
        return OpenAIClient(api_key="test-key", model_name="m", **kwargs)


def test_cancel_inflight_sets_event_on_all_clients():
    a = _patched_openai_client()
    b = _patched_openai_client()
    handler = LMHandler(client=a, other_backend_client=b)
    assert not a.cancel_event.is_set() and not b.cancel_event.is_set()
    handler.cancel_inflight()
    assert a.cancel_event.is_set() and b.cancel_event.is_set()


def test_cancel_inflight_tolerates_clients_without_event():
    mock = MockLM()
    handler = LMHandler(client=mock)
    handler.cancel_inflight()  # must not raise


def test_set_run_deadline_arms_clients():
    a = _patched_openai_client()
    handler = LMHandler(client=a)
    handler.set_run_deadline(300.0)
    assert a._deadline is not None
    handler.set_run_deadline(None)
    assert a._deadline is None


# ---------------------------------------------------------------------------
# OpenAIClient abort behavior
# ---------------------------------------------------------------------------


def _chunk(content=None, usage=None):
    choices = []
    if content is not None:
        choices = [SimpleNamespace(delta=SimpleNamespace(content=content))]
    return SimpleNamespace(choices=choices, usage=usage)


_USAGE = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)


class _FakeStream:
    """Iterable chat-completion stream that records close()."""

    def __init__(self, chunks, on_yield=None):
        self._chunks = list(chunks)
        self._on_yield = on_yield
        self.closed = False

    def __iter__(self):
        for i, chunk in enumerate(self._chunks):
            if self._on_yield:
                self._on_yield(i)
            yield chunk

    def close(self):
        self.closed = True


def test_stream_completion_assembles_chunks_and_tracks_usage():
    client = _patched_openai_client(stream=True)
    stream = _FakeStream([_chunk("hel"), _chunk("lo"), _chunk(usage=_USAGE)])
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = stream

    assert client.completion("hi") == "hello"
    assert stream.closed
    assert client.client.chat.completions.create.call_args.kwargs["stream"] is True
    assert client.model_output_tokens["m"] == 5


def test_stream_aborts_on_cancel_event():
    client = _patched_openai_client(stream=True)
    # Set the event after the first chunk is yielded
    stream = _FakeStream(
        [_chunk("a"), _chunk("b"), _chunk("c")],
        on_yield=lambda i: client.cancel_event.set() if i == 1 else None,
    )
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = stream

    with pytest.raises(CancellationError):
        client.completion("hi")
    assert stream.closed  # the server-side generation is torn down


def test_stream_aborts_when_deadline_expires_mid_generation():
    client = _patched_openai_client(stream=True)
    stream = _FakeStream(
        [_chunk("a"), _chunk("b"), _chunk("c")],
        on_yield=lambda i: client.set_deadline(0.0) if i == 1 else None,
    )
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = stream

    with pytest.raises(TimeoutExceededError):
        client.completion("hi")
    assert stream.closed


def test_expired_deadline_prevents_new_request():
    client = _patched_openai_client(stream=True)
    client.client = MagicMock()
    client.set_deadline(0.0)  # already expired before the call
    with pytest.raises(TimeoutExceededError):
        client.completion("hi")
    client.client.chat.completions.create.assert_not_called()


def test_cancelled_call_never_starts_new_generation():
    """A call that was queued past cancellation must not hit the backend."""
    client = _patched_openai_client(stream=True)
    client.client = MagicMock()
    client.cancel_event.set()
    with pytest.raises(CancellationError):
        client.completion("hi")
    client.client.chat.completions.create.assert_not_called()


def test_nonstream_call_also_checks_abort_before_request():
    client = _patched_openai_client()  # stream=False
    client.client = MagicMock()
    client.cancel_event.set()
    with pytest.raises(CancellationError):
        client.completion("hi")
    client.client.chat.completions.create.assert_not_called()


def test_completion_max_tokens_lands_in_request():
    client = _patched_openai_client()
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=_USAGE,
    )
    client.completion("hi", max_tokens=777)
    assert client.client.chat.completions.create.call_args.kwargs["max_tokens"] == 777


# ---------------------------------------------------------------------------
# subcall_extra_body plumbing (handler -> client): per-sub-call request body
# extras, e.g. {"chat_template_kwargs": {"enable_thinking": False}} so gemma
# sub-calls skip the thought channel (2026-06-12: the kb fine-tune ruminated
# in-channel to the token cap on triage prompts, returning empty content).
# ---------------------------------------------------------------------------

_NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


def test_single_subcall_gets_extra_body():
    mock = MockLM(responses=["ok"])
    with LMHandler(client=mock, subcall_extra_body=_NOTHINK) as handler:
        response = send_lm_request(handler.address, LMRequest(prompt="hi"))
    assert response.success
    assert mock.seen_extra_body == [_NOTHINK]


def test_batched_subcalls_get_extra_body():
    mock = MockLM(responses=["a", "b"])
    with LMHandler(client=mock, subcall_extra_body=_NOTHINK) as handler:
        responses = send_lm_request_batched(handler.address, ["p1", "p2"])
    assert all(r.success for r in responses)
    assert mock.seen_extra_body == [_NOTHINK, _NOTHINK]


def test_no_extra_body_by_default():
    mock = MockLM(responses=["ok"])
    with LMHandler(client=mock) as handler:
        response = send_lm_request(handler.address, LMRequest(prompt="hi"))
    assert response.success
    assert mock.seen_extra_body == [None]


def test_root_completion_not_affected_by_subcall_extra_body():
    """The root orchestrator keeps thinking ON (gemma reasons better with it);
    only sub-calls are mechanical."""
    mock = MockLM(responses=["ok"])
    handler = LMHandler(client=mock, subcall_extra_body=_NOTHINK)
    assert handler.completion("root prompt") == "ok"
    assert mock.seen_extra_body == [None]


def test_completion_extra_body_lands_in_request():
    client = _patched_openai_client()
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=_USAGE,
    )
    client.completion("hi", extra_body=_NOTHINK)
    sent = client.client.chat.completions.create.call_args.kwargs["extra_body"]
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


# ---------------------------------------------------------------------------
# Reasoning-only responses must not surface as silent empty strings
# (2026-06-12: jinja routes gemma thought-channel tokens to reasoning_content;
# a response that never exits the channel arrived as content="" and the
# orchestrator retried the same call for 20 iterations).
# ---------------------------------------------------------------------------


def test_reasoning_only_response_returns_tagged_fallback():
    from lm_repl.clients.openai import _resolve_content

    out = _resolve_content("", "step 1... step 2... conclusion: None.")
    assert out.startswith("[reasoning-only response")
    assert "conclusion: None." in out


def test_reasoning_fallback_is_bounded():
    from lm_repl.clients.openai import _resolve_content

    out = _resolve_content("", "x" * 100_000)
    assert len(out) < 3000


def test_content_wins_over_reasoning():
    from lm_repl.clients.openai import _resolve_content

    assert _resolve_content("the answer", "thinking...") == "the answer"


def test_empty_content_no_reasoning_stays_empty():
    from lm_repl.clients.openai import _resolve_content

    assert _resolve_content("", None) == ""


def test_nonstream_completion_falls_back_to_reasoning_content():
    client = _patched_openai_client()
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="", reasoning_content="only thoughts here")
            )
        ],
        usage=_USAGE,
    )
    out = client.completion("hi")
    assert "only thoughts here" in out
    assert out.startswith("[reasoning-only response")


# ---------------------------------------------------------------------------
# Max-depth leaf fallback guards (2026-06-12: the fallback built a fresh
# client with no cap, no deadline, no scheduler - watched it generate 60K+
# tokens five minutes past its run's expired deadline).
# ---------------------------------------------------------------------------


def test_leaf_fallback_applies_subcall_guards():
    import time

    import lm_repl.core.rlm as rlm_module
    from lm_repl import RLM

    with patch.object(rlm_module, "get_client") as mock_get_client:
        mock_lm = MockLM(responses=["leaf response"])
        mock_get_client.return_value = mock_lm
        parent = RLM(
            backend="openai",
            backend_kwargs={"model_name": "m"},
            depth=1,
            max_depth=2,
            subcall_max_tokens=2048,
            subcall_extra_body=_NOTHINK,
            max_timeout=600.0,
        )
        parent._completion_start_time = time.perf_counter() - 100.0
        result = parent._subcall("decomposed subtask")
        parent.close()

    assert result.response == "leaf response"
    assert mock_lm.seen_max_tokens == [2048]
    assert mock_lm.seen_extra_body == [_NOTHINK]
    # Deadline = remaining run budget (600 - ~100 elapsed), not unlimited.
    assert len(mock_lm.seen_deadlines) == 1
    assert mock_lm.seen_deadlines[0] is not None
    assert 400.0 < mock_lm.seen_deadlines[0] <= 510.0
