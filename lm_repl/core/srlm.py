"""SRLM - Self-Reflective Language Model.

Extends RLM with context-length routing, multi-trajectory generation,
and uncertainty-guided selection per the Apple SRLM paper
(arxiv.org/abs/2603.15653).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from lm_repl.clients import get_client
from lm_repl.core.rlm import RLM
from lm_repl.core.types import RLMChatCompletion, UsageSummary


def _choose_mode(context_len: int, direct_threshold: int | None) -> str:
    if not direct_threshold or direct_threshold <= 0:
        return "rlm"
    return "direct" if context_len < direct_threshold else "rlm"


def _build_direct_messages(context: str, query: str) -> list[dict]:
    return [
        {"role": "system", "content": "Answer the question using only the provided context. Be concise."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]


@dataclass
class SRLMChatCompletion:
    """Extends RLMChatCompletion with SRLM selection metadata."""
    completion: RLMChatCompletion
    mode: str = "rlm"
    n_candidates: int = 1
    n_consistent: int = 1
    trace_tokens: int = 0
    selected_index: int = 0


class SRLM(RLM):
    """Self-Reflective Language Model.

    Subclasses RLM to add:
    - Context-length routing (direct LLM call for short contexts)
    - Multi-trajectory generation (K candidates)
    - Uncertainty-guided selection (self-consistency + trace length)
    """

    def __init__(
        self,
        *,
        direct_threshold: int = 0,
        n_candidates: int = 1,
        candidate_temperature: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.direct_threshold = direct_threshold
        self.n_candidates = n_candidates
        self.candidate_temperature = candidate_temperature

    def _direct_completion(self, prompt: str | dict[str, Any]) -> RLMChatCompletion:
        """Bypass REPL - direct LLM chat completion for short contexts."""
        context = prompt if isinstance(prompt, str) else str(prompt)
        root_prompt = ""
        if isinstance(prompt, dict):
            context = prompt.get("context", str(prompt))
            root_prompt = prompt.get("query", "")

        backend_name = (
            self.other_backends[0] if self.other_backends else self.backend
        )
        backend_kw = (
            self.other_backend_kwargs[0] if self.other_backend_kwargs else self.backend_kwargs
        ) or {}
        client = get_client(backend_name, backend_kw)
        messages = _build_direct_messages(context, root_prompt)

        start = time.perf_counter()
        response_text = client.completion(messages)
        elapsed = time.perf_counter() - start

        model_name = backend_kw.get("model_name", "unknown")
        return RLMChatCompletion(
            root_model=model_name,
            prompt=prompt,
            response=response_text,
            usage_summary=client.get_usage_summary(),
            execution_time=elapsed,
            metadata={"mode": "direct"},
        )

    def completion(
        self, prompt: str | dict[str, Any], root_prompt: str | None = None
    ) -> RLMChatCompletion:
        context_str = prompt if isinstance(prompt, str) else str(prompt)
        mode = _choose_mode(len(context_str), self.direct_threshold)

        if mode == "direct":
            return self._direct_completion(prompt)

        if self.n_candidates <= 1:
            result = super().completion(prompt, root_prompt)
            if result.metadata is None:
                result.metadata = {}
            if isinstance(result.metadata, dict):
                result.metadata["mode"] = "rlm"
                result.metadata["n_candidates"] = 1
            return result

        return self._multi_trajectory_completion(prompt, root_prompt)

    def _multi_trajectory_completion(
        self, prompt: str | dict[str, Any], root_prompt: str | None = None
    ) -> RLMChatCompletion:
        """Generate K candidates and select the best by uncertainty signals."""
        saved_extra = None
        if self.candidate_temperature is not None and self.backend_kwargs:
            saved_extra = self.backend_kwargs.get("default_extra_body")
            merged = dict(saved_extra) if saved_extra else {}
            merged["temperature"] = self.candidate_temperature
            self.backend_kwargs["default_extra_body"] = merged

        try:
            candidates: list[RLMChatCompletion] = []
            for _ in range(self.n_candidates):
                result = super().completion(prompt, root_prompt)
                candidates.append(result)
        finally:
            if saved_extra is not None and self.backend_kwargs:
                self.backend_kwargs["default_extra_body"] = saved_extra
            elif self.candidate_temperature is not None and self.backend_kwargs and "default_extra_body" in self.backend_kwargs:
                self.backend_kwargs["default_extra_body"].pop("temperature", None)

        best = _select_best(candidates)
        if best.metadata is None:
            best.metadata = {}
        if isinstance(best.metadata, dict):
            best.metadata["mode"] = "rlm"
            best.metadata["n_candidates"] = self.n_candidates
        return best


def _select_best(candidates: list[RLMChatCompletion]) -> RLMChatCompletion:
    """Select the best candidate using self-consistency + trace length.

    1. Majority vote on final answer (self-consistency).
    2. Among the consistent set, pick the shortest trace (fewest tokens).
    """
    if len(candidates) == 1:
        return candidates[0]

    answers = [c.response.strip().lower() for c in candidates]
    counts: dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1

    majority = max(counts, key=counts.get)
    consistent = [c for c, a in zip(candidates, answers) if a == majority]

    return min(consistent, key=lambda c: c.execution_time)
