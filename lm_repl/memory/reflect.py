"""A reflect function backed by an OpenAI-compatible chat endpoint.

The :class:`~lm_repl.memory.distill.TraceDistiller` needs a ``prompt -> text``
callable to distill experiences. This adapter implements it over the same kind
of OpenAI-compatible server lm-repl drives, so distillation can use a dedicated
judge/memory model. Injected client keeps it unit-testable without network.
"""
from __future__ import annotations

from typing import Any


class OpenAIReflectFn:
    """Callable ``prompt -> str`` over ``client.chat.completions.create``."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        temperature: float = 0.3,
        system_prompt: str | None = None,
        extra_body: dict | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        # Distillation is mechanical JSON extraction, not reasoning. On a thinking
        # model (e.g. gemma with CoT on) leave it unbounded and it can degenerate
        # into a huge thought trace; callers pass
        # {"chat_template_kwargs": {"enable_thinking": False}} to disable it, and
        # max_tokens as a belt-and-suspenders cap.
        self.extra_body = extra_body
        self.max_tokens = max_tokens

    def __call__(self, prompt: str) -> str:
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {}
        if self.extra_body is not None:
            kwargs["extra_body"] = self.extra_body
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    @classmethod
    def from_config(
        cls, *, base_url: str, model: str, api_key: str = "EMPTY", **kwargs: Any
    ) -> OpenAIReflectFn:
        """Build a reflect fn backed by a real ``openai.OpenAI`` client."""
        import openai

        client = openai.OpenAI(base_url=base_url, api_key=api_key)
        return cls(client, model=model, **kwargs)
