"""Tests for the OpenAI-compatible reflect function (distiller's LLM call)."""
from __future__ import annotations

from types import SimpleNamespace

from prehend.memory.reflect import OpenAIReflectFn


class FakeChatCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, *, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        msg = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class FakeClient:
    def __init__(self, content):
        self.chat = SimpleNamespace(completions=FakeChatCompletions(content))


def test_returns_message_content():
    client = FakeClient('{"key_insight": "k"}')
    reflect = OpenAIReflectFn(client, model="judge-model")
    assert reflect("distill this") == '{"key_insight": "k"}'


def test_sends_prompt_as_user_message_to_model():
    client = FakeClient("ok")
    reflect = OpenAIReflectFn(client, model="judge-model")
    reflect("the reflect prompt")
    call = client.chat.completions.create.__self__.calls[0]
    assert call["model"] == "judge-model"
    assert call["messages"][-1]["role"] == "user"
    assert call["messages"][-1]["content"] == "the reflect prompt"


def test_returns_empty_string_when_content_is_none():
    client = FakeClient(None)
    reflect = OpenAIReflectFn(client, model="m")
    assert reflect("x") == ""


def _last_kwargs(client):
    return client.chat.completions.create.__self__.calls[0]["kwargs"]


def test_forwards_extra_body_to_disable_thinking():
    # The reflect call is mechanical JSON extraction; with gemma CoT left on it
    # can degenerate into an unbounded thinking trace. Callers disable it via
    # chat_template_kwargs.enable_thinking=False, which must reach the client.
    nothink = {"chat_template_kwargs": {"enable_thinking": False}}
    client = FakeClient("ok")
    reflect = OpenAIReflectFn(client, model="m", extra_body=nothink)
    reflect("x")
    assert _last_kwargs(client)["extra_body"] == nothink


def test_forwards_max_tokens_when_set():
    client = FakeClient("ok")
    reflect = OpenAIReflectFn(client, model="m", max_tokens=512)
    reflect("x")
    assert _last_kwargs(client)["max_tokens"] == 512


def test_omits_extra_body_and_max_tokens_when_unset():
    client = FakeClient("ok")
    reflect = OpenAIReflectFn(client, model="m")
    reflect("x")
    kw = _last_kwargs(client)
    assert "extra_body" not in kw
    assert "max_tokens" not in kw


def test_from_config_forwards_extra_body_and_max_tokens_kwargs():
    nothink = {"chat_template_kwargs": {"enable_thinking": False}}
    reflect = OpenAIReflectFn.from_config(
        base_url="http://x/v1", model="m", extra_body=nothink, max_tokens=256
    )
    assert reflect.extra_body == nothink
    assert reflect.max_tokens == 256
