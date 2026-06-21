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
) -> MemoryHarness:
    """Assemble a memory-backed harness.

    Provide either a ready ``embed_backend`` or an ``embed_client`` + ``embed_model``;
    likewise either a ready ``reflect_fn`` or a ``reflect_client`` + ``reflect_model``.
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
    source: str = "prehend",
    k_max: int = DEFAULT_K_MAX,
    min_cosine: float = DEFAULT_MIN_COSINE,
    tagger: Tagger | None = None,
) -> MemoryHarness:
    """Convenience: build embedding + reflect against OpenAI-compatible servers.

    Reflect (trace distillation) runs against ``base_url``. Embedding runs
    against ``embed_base_url`` when given, else ``base_url`` -- this is the
    common local setup where a small embedding model (e.g. bge-m3) is served on
    its own port while the chat model is swapped on a single-model router.
    """
    backend = OpenAIEmbeddingBackend.from_config(
        base_url=embed_base_url or base_url,
        model=embed_model,
        api_key=embed_api_key or api_key,
    )
    reflect = OpenAIReflectFn.from_config(
        base_url=base_url, model=reflect_model, api_key=api_key
    )
    return build_memory_harness(
        solver, bank_dir,
        embed_backend=backend, reflect_fn=reflect,
        source=source, k_max=k_max, min_cosine=min_cosine, tagger=tagger,
    )
