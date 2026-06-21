"""prehend: self-evolving experience memory for prehend.

A domain-agnostic port of FinAcumen's FM subsystem. Wraps a context-offloading
solver (an :class:`~prehend.SRLM`) so it accumulates and reuses verified,
polarity-tagged experience across tasks: retrieve -> inject -> solve -> collect.

Quick start::

    from prehend import SRLM
    from prehend.memory import build_memory_harness_from_config

    srlm = SRLM(backend="openai", backend_kwargs={...})
    harness = build_memory_harness_from_config(
        srlm, "memory_bank",
        base_url="http://localhost:8080/v1",
        embed_model="nv-embed-v2", reflect_model="my-judge",
    )
    result = harness.answer(context=long_context, question="...")
"""
from prehend.memory.bank import Bank
from prehend.memory.distill import TraceDistiller
from prehend.memory.embed import EmbeddingBackend, HashingEmbeddingBackend, cosine
from prehend.memory.embed_openai import OpenAIEmbeddingBackend
from prehend.memory.factory import (
    build_memory_harness,
    build_memory_harness_from_config,
)
from prehend.memory.harness import Distiller, MemoryHarness, Solver
from prehend.memory.inject import render_memory_block
from prehend.memory.pruning_rules import is_anti_give_up
from prehend.memory.reflect import OpenAIReflectFn
from prehend.memory.retrieve import RetrievalResult, retrieve
from prehend.memory.tagger import NullTagger, Tagger

__all__ = [
    "Bank",
    "EmbeddingBackend",
    "HashingEmbeddingBackend",
    "OpenAIEmbeddingBackend",
    "OpenAIReflectFn",
    "cosine",
    "MemoryHarness",
    "Distiller",
    "Solver",
    "Tagger",
    "NullTagger",
    "TraceDistiller",
    "build_memory_harness",
    "build_memory_harness_from_config",
    "is_anti_give_up",
    "render_memory_block",
    "RetrievalResult",
    "retrieve",
]
