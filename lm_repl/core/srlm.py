"""SRLM - Self-Reflective Language Model.

Extends RLM with context-length routing, multi-trajectory generation,
and uncertainty-guided selection per the Apple SRLM paper
(arxiv.org/abs/2603.15653).
"""
from __future__ import annotations

import math
import re
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
        confidence_elicitation: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.direct_threshold = direct_threshold
        self.n_candidates = n_candidates
        self.candidate_temperature = candidate_temperature
        self.confidence_elicitation = confidence_elicitation

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

        best = _select_best(candidates, use_confidence=self.confidence_elicitation)
        if best.metadata is None:
            best.metadata = {}
        if isinstance(best.metadata, dict):
            best.metadata["mode"] = "rlm"
            best.metadata["n_candidates"] = self.n_candidates
        return best


def _parse_confidence_scores(text: str) -> list[float]:
    """Extract {"confidence": N} values from model response text."""
    pattern = r'\{\s*"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*\}'
    matches = re.findall(pattern, text)
    return [min(float(m), 100.0) for m in matches]


def _extract_step_texts(metadata: dict | None) -> list[str]:
    """Per-step response texts from RLMChatCompletion.metadata.

    RLM attaches logger.get_trajectory(): {"run_metadata": ..., "iterations":
    [{"response": ..., ...}, ...]}. Also accepts a legacy {"trajectory_text":
    str} blob as a single step.
    """
    if not isinstance(metadata, dict):
        return []
    iterations = metadata.get("iterations")
    if isinstance(iterations, list) and iterations:
        return [str(it.get("response", "") or "") for it in iterations]
    legacy = metadata.get("trajectory_text")
    if legacy:
        return [str(legacy)]
    return []


def _compute_vc_score(step_texts: list[str]) -> float:
    """Verbalized confidence score: sum over steps of log(confidence/100).

    Uses the last reported confidence in each step. Steps with no report are
    imputed with the mean of the reported steps (per the SRLM paper), so a
    trajectory cannot inflate its score by skipping reports.

    Returns a value <= 0; closer to 0 = higher confidence. Returns -inf when
    no step reports any confidence, or a reported confidence is 0.
    """
    per_step: list[float | None] = []
    for text in step_texts:
        scores = _parse_confidence_scores(text)
        per_step.append(scores[-1] if scores else None)

    reported = [v for v in per_step if v is not None]
    if not reported:
        return float('-inf')
    mean = sum(reported) / len(reported)

    total = 0.0
    for v in per_step:
        value = v if v is not None else mean
        if value <= 0:
            return float('-inf')
        total += math.log(value / 100.0)
    return total


def _trace_len(c: RLMChatCompletion) -> float:
    """Len(p): trace length in output tokens, per the paper.

    Falls back to execution_time when the backend reported no token usage.
    Tokens are preferred because wall clock is confounded by prefix-cache
    hits and server slot contention.
    """
    tokens = c.usage_summary.total_output_tokens if c.usage_summary else 0
    return float(tokens) if tokens and tokens > 0 else c.execution_time


def _select_best(
    candidates: list[RLMChatCompletion], use_confidence: bool = False
) -> RLMChatCompletion:
    """Select the best candidate using self-consistency + uncertainty signals.

    1. Majority vote on final answer (self-consistency).
    2. Among the consistent set:
       - If use_confidence: joint score VC(p) * Len(p), pick argmax (closest to 0)
       - Otherwise: pick shortest trace (output tokens, else execution time)
    """
    if len(candidates) == 1:
        return candidates[0]

    answers = [c.response.strip().lower() for c in candidates]
    counts: dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1

    majority = max(counts, key=counts.get)
    consistent = [c for c, a in zip(candidates, answers) if a == majority]

    if not use_confidence:
        return min(consistent, key=_trace_len)

    def _joint_score(c: RLMChatCompletion) -> float:
        vc = _compute_vc_score(_extract_step_texts(c.metadata))
        if vc == float('-inf'):
            return float('-inf')
        return vc * _trace_len(c)

    scored = [(c, _joint_score(c)) for c in consistent]
    has_scores = any(s != float('-inf') for _, s in scored)

    if not has_scores:
        return min(consistent, key=_trace_len)

    return max(scored, key=lambda x: x[1])[0]
