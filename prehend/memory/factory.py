"""Wire a context-offloading solver into a memory-backed MemoryHarness.

``build_memory_harness`` assembles the prehend loop end-to-end: a Bank, an
embedding backend, a reflect function, and a TraceDistiller around any solver
exposing ``completion(prompt, root_prompt)`` (in practice an SRLM). Components
may be injected directly (for tests) or built from connection config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from prehend.memory.bank import Bank
from prehend.memory.distill import TraceDistiller
from prehend.memory.embed import EmbeddingBackend
from prehend.memory.embed_openai import OpenAIEmbeddingBackend
from prehend.memory.harness import (
    Distiller,
    MemoryHarness,
    MemoryObserver,
    Solver,
)
from prehend.memory.reflect import OpenAIReflectFn
from prehend.memory.retrieve import DEFAULT_K_MAX, DEFAULT_MIN_COSINE
from prehend.memory.tagger import Tagger


def build_memory_harness(
    solver: Solver,
    bank_dir: Path | str,
    *,
    embed_backend: EmbeddingBackend | None = None,
    embed_client: Any = None,
    embed_model: str | None = None,
    reflect_fn: Distiller | None = None,
    reflect_client: Any = None,
    reflect_model: str | None = None,
    source: str = "prehend",
    k_max: int = DEFAULT_K_MAX,
    min_cosine: float = DEFAULT_MIN_COSINE,
    tagger: Tagger | None = None,
    defer_collect: bool = False,
    learn_from_failure: bool = False,
    max_inject_negatives: int = 2,
    context_signature: bool = False,
    observer: MemoryObserver | None = None,
) -> MemoryHarness:
    """Assemble a memory-backed harness.

    Provide either a ready ``embed_backend`` or an ``embed_client`` + ``embed_model``;
    likewise either a ready ``reflect_fn`` or a ``reflect_client`` + ``reflect_model``.

    ``defer_collect`` defers distillation until ``collect_pending(correct)`` so a
    caller that scores the answer can learn only from correct solves.

    ``observer`` receives retrieve/collect telemetry; pass
    ``prehend.metrics.memory_observer()`` to emit the localai_prehend_memory_*
    Prometheus series. Defaults to a no-op.
    """
    backend = embed_backend
    if backend is None:
        if embed_client is None or embed_model is None:
            raise ValueError(
                "provide embed_backend, or both embed_client and embed_model"
            )
        backend = OpenAIEmbeddingBackend(embed_client, model=embed_model)

    reflect = reflect_fn
    if reflect is None:
        if reflect_client is None or reflect_model is None:
            raise ValueError(
                "provide reflect_fn, or both reflect_client and reflect_model"
            )
        reflect = OpenAIReflectFn(reflect_client, model=reflect_model)

    distiller = TraceDistiller(reflect, backend, source=source)
    return MemoryHarness(
        solver, Bank(bank_dir), backend,
        k_max=k_max, min_cosine=min_cosine, distiller=distiller, tagger=tagger,
        defer_collect=defer_collect, learn_from_failure=learn_from_failure,
        max_inject_negatives=max_inject_negatives,
        context_signature=context_signature, observer=observer,
    )


def build_memory_harness_from_config(
    solver: Solver,
    bank_dir: Path | str,
    *,
    base_url: str,
    embed_model: str,
    reflect_model: str,
    api_key: str = "EMPTY",
    embed_base_url: str | None = None,
    embed_api_key: str | None = None,
    reflect_base_url: str | None = None,
    reflect_api_key: str | None = None,
    source: str = "prehend",
    k_max: int = DEFAULT_K_MAX,
    min_cosine: float = DEFAULT_MIN_COSINE,
    tagger: Tagger | None = None,
    reflect_enable_thinking: bool = False,
    reflect_max_tokens: int | None = 512,
    reflect_temperature: float = 1.0,
    defer_collect: bool = False,
    learn_from_failure: bool = False,
    max_inject_negatives: int = 2,
    context_signature: bool = False,
    observer: MemoryObserver | None = None,
) -> MemoryHarness:
    """Convenience: build embedding + reflect against OpenAI-compatible servers.

    Reflect (trace distillation) runs against ``reflect_base_url`` when given,
    else ``base_url``; embedding against ``embed_base_url`` else ``base_url``.
    Both can be split off the solver endpoint - the common local setup where a
    small embedding model (e.g. bge-m3 on :8084) and a small/neutral distill
    model (e.g. Gemma 4 e4b on :8082) run on their own ports while the solver
    model is swapped on the single-model router (:8080), which must not swap.

    Distillation is mechanical JSON extraction, not reasoning, so by default
    reflect runs with thinking OFF and a bounded ``reflect_max_tokens``. Without
    this, a reasoning ``reflect_model`` (e.g. a gemma sft-kb with CoT on) emits a
    full thought trace per solve -- the dominant memory-layer overhead and a prime
    source of single-GPU contention. Override the knobs to restore CoT/raise the
    cap when the reflect model genuinely needs it.
    """
    backend = OpenAIEmbeddingBackend.from_config(
        base_url=embed_base_url or base_url,
        model=embed_model,
        api_key=embed_api_key or api_key,
    )
    reflect = OpenAIReflectFn.from_config(
        base_url=reflect_base_url or base_url, model=reflect_model,
        api_key=reflect_api_key or api_key,
        temperature=reflect_temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": reflect_enable_thinking}},
        max_tokens=reflect_max_tokens,
    )
    return build_memory_harness(
        solver, bank_dir,
        embed_backend=backend, reflect_fn=reflect,
        source=source, k_max=k_max, min_cosine=min_cosine, tagger=tagger,
        defer_collect=defer_collect, learn_from_failure=learn_from_failure,
        max_inject_negatives=max_inject_negatives,
        context_signature=context_signature, observer=observer,
    )
