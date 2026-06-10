import logging
import os
from collections import defaultdict
from typing import Any

import openai
from dotenv import load_dotenv

from lm_repl.clients.base_lm import BaseLM
from lm_repl.clients.scheduler import Priority, RequestScheduler, resolve_priority
from lm_repl.core.types import ModelUsageSummary, UsageSummary

log = logging.getLogger(__name__)

load_dotenv()

# Load API keys from environment variables
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_VERCEL_API_KEY = os.getenv("AI_GATEWAY_API_KEY")
DEFAULT_PRIME_API_KEY = os.getenv("PRIME_API_KEY")
DEFAULT_PRIME_INTELLECT_BASE_URL = "https://api.pinference.ai/api/v1/"


def _is_context_contention(e: openai.BadRequestError) -> bool:
    """True when llama-server rejected the request for context size.

    With ``--kv-unified`` the available context is a shared pool, so this 400 usually means
    another slot is occupying the pool right now - not that the request can never fit. The
    caller retries once at p1 (exclusive access); if that retry also fails, the request is
    genuinely too large and the error propagates. Example body:
    ``{"error":{"code":400,"message":"request (40069 tokens) exceeds the available context
    size (22016 tokens), try increasing it","type":"exceed_context_size_error"}}``
    """
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("type") == "exceed_context_size_error":
            return True
    text = str(e)
    return "exceed_context_size_error" in text or "exceeds the available context size" in text


def _merge_consecutive_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent messages that share a role, concatenating their content.

    The RLM REPL can emit consecutive same-role turns - e.g. two assistant iterations with no
    code block between them produce no intervening REPL `user` message, leaving `[..., assistant,
    assistant]`. Strict OpenAI-compatible servers (llama.cpp `/chat/completions` applying a gemma
    jinja template) reject 2+ trailing assistant messages ("Cannot have 2 or more assistant
    messages at the end of the list"). Merging is semantically identical - the model sees the same
    text - and yields the alternating sequence those templates require. Lenient backends (e.g. the
    retired LM Studio endpoint) are unaffected. Only string content is merged; dict/list content
    (multimodal parts) is left as its own message so structured payloads are never corrupted.
    """
    merged: list[dict[str, Any]] = []
    for msg in messages:
        if (
            merged
            and merged[-1].get("role") == msg.get("role")
            and isinstance(merged[-1].get("content"), str)
            and isinstance(msg.get("content"), str)
        ):
            merged[-1] = {**merged[-1], "content": merged[-1]["content"] + "\n\n" + msg["content"]}
        else:
            merged.append(msg)
    return merged


class OpenAIClient(BaseLM):
    """
    LM Client for running models with the OpenAI API. Works with vLLM as well.

    Any additional keyword arguments (e.g. default_headers, default_query, max_retries)
    are passed through to the underlying openai.OpenAI and openai.AsyncOpenAI constructors.
    Only model_name is excluded, since it is not a client constructor argument.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        default_extra_body: dict[str, Any] | None = None,
        scheduler: RequestScheduler | None = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        # Per-request body fields merged into every call (e.g. {"chat_template_kwargs":
        # {"enable_thinking": False}} to disable gemma CoT on the recursive sub-call backend
        # while the root orchestrator keeps thinking). Not an OpenAI client ctor arg, so it is a
        # named param here and never leaks into client_kwargs below.
        self.default_extra_body = default_extra_body or {}
        self.scheduler = scheduler

        if api_key is None:
            if base_url == "https://api.openai.com/v1" or base_url is None:
                api_key = DEFAULT_OPENAI_API_KEY
            elif base_url == "https://openrouter.ai/api/v1":
                api_key = DEFAULT_OPENROUTER_API_KEY
            elif base_url == "https://ai-gateway.vercel.sh/v1":
                api_key = DEFAULT_VERCEL_API_KEY
            elif base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
                api_key = DEFAULT_PRIME_API_KEY

        # Pass through arbitrary kwargs to the OpenAI client (e.g. default_headers, default_query, max_retries).
        # Exclude model_name since it is not an OpenAI client constructor argument.
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": self.timeout,
            **{k: v for k, v in self.kwargs.items() if k != "model_name"},
        }
        self.client = openai.OpenAI(**client_kwargs)
        self.async_client = openai.AsyncOpenAI(**client_kwargs)
        self.model_name = model_name
        self.base_url = base_url  # Track for cost extraction

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)
        self.model_costs: dict[str, float] = defaultdict(float)  # Cost in USD

    def completion(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
        response_format: dict[str, Any] | None = None,
        priority: str | int | None = None,
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = _merge_consecutive_roles(prompt)
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OpenAI client.")

        extra_body = dict(self.default_extra_body)
        if self.client.base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
            extra_body["usage"] = {"include": True}

        create_kwargs: dict[str, Any] = {}
        if response_format is not None:
            create_kwargs["response_format"] = response_format

        p = resolve_priority(priority)
        return self._do_completion(model, messages, extra_body, create_kwargs, p)

    def _do_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        extra_body: dict[str, Any],
        create_kwargs: dict[str, Any],
        priority: int,
    ) -> str:
        while True:
            # The finally must release at the priority we acquired with, not the one the
            # retry branch escalates to for the next iteration.
            acquired = priority
            if self.scheduler:
                self.scheduler.acquire(acquired)
            try:
                response = self.client.chat.completions.create(
                    model=model, messages=messages, extra_body=extra_body, **create_kwargs
                )
                self._track_cost(response, model)
                return response.choices[0].message.content
            except openai.BadRequestError as e:
                if (
                    self.scheduler
                    and priority != Priority.CONTENTION_RETRY
                    and _is_context_contention(e)
                ):
                    log.info("context contention (%s), retrying at p1", str(e)[:80])
                    priority = Priority.CONTENTION_RETRY
                    continue
                raise
            finally:
                if self.scheduler:
                    self.scheduler.release(acquired)

    async def acompletion(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
        priority: str | int | None = None,
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = _merge_consecutive_roles(prompt)
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OpenAI client.")

        extra_body = dict(self.default_extra_body)
        if self.client.base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
            extra_body["usage"] = {"include": True}

        p = resolve_priority(priority)
        return await self._ado_completion(model, messages, extra_body, p)

    async def _ado_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        extra_body: dict[str, Any],
        priority: int,
    ) -> str:
        while True:
            acquired = priority
            if self.scheduler:
                await self.scheduler.aacquire(acquired)
            try:
                response = await self.async_client.chat.completions.create(
                    model=model, messages=messages, extra_body=extra_body
                )
                self._track_cost(response, model)
                return response.choices[0].message.content
            except openai.BadRequestError as e:
                if (
                    self.scheduler
                    and priority != Priority.CONTENTION_RETRY
                    and _is_context_contention(e)
                ):
                    log.info("context contention (%s), retrying at p1", str(e)[:80])
                    priority = Priority.CONTENTION_RETRY
                    continue
                raise
            finally:
                if self.scheduler:
                    await self.scheduler.arelease(acquired)

    def _track_cost(self, response: openai.ChatCompletion, model: str):
        self.model_call_counts[model] += 1

        usage = getattr(response, "usage", None)
        if usage is None:
            raise ValueError("No usage data received. Tracking tokens not possible.")

        self.model_input_tokens[model] += usage.prompt_tokens
        self.model_output_tokens[model] += usage.completion_tokens
        self.model_total_tokens[model] += usage.total_tokens

        # Track last call for handler to read
        self.last_prompt_tokens = usage.prompt_tokens
        self.last_completion_tokens = usage.completion_tokens

        # Extract cost from OpenRouter responses (cost is in USD)
        # OpenRouter returns cost in usage.model_extra for pydantic models
        self.last_cost: float | None = None
        cost = None

        # Try direct attribute first
        if hasattr(usage, "cost") and usage.cost:
            cost = usage.cost
        # Then try model_extra (OpenRouter uses this)
        elif hasattr(usage, "model_extra") and usage.model_extra:
            extra = usage.model_extra
            # Primary cost field (may be 0 for BYOK)
            if extra.get("cost"):
                cost = extra["cost"]
            # Fallback to upstream cost details
            elif extra.get("cost_details", {}).get("upstream_inference_cost"):
                cost = extra["cost_details"]["upstream_inference_cost"]

        if cost is not None and cost > 0:
            self.last_cost = float(cost)
            self.model_costs[model] += self.last_cost

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            cost = self.model_costs.get(model)
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
                total_cost=cost if cost else None,
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
            total_cost=getattr(self, "last_cost", None),
        )
