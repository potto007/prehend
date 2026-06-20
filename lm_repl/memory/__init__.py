"""mnemex: self-evolving experience memory for lm-repl.

A domain-agnostic port of FinAcumen's FM subsystem. Wraps a context-offloading
solver (an :class:`~lm_repl.SRLM`) so it accumulates and reuses verified,
polarity-tagged experience across tasks: retrieve -> inject -> solve -> collect.
"""
from lm_repl.memory.bank import Bank
from lm_repl.memory.distill import TraceDistiller
from lm_repl.memory.embed import EmbeddingBackend, cosine
from lm_repl.memory.embed_openai import OpenAIEmbeddingBackend
from lm_repl.memory.harness import Distiller, MemoryHarness, Solver
from lm_repl.memory.inject import render_memory_block
from lm_repl.memory.pruning_rules import is_anti_give_up
from lm_repl.memory.retrieve import RetrievalResult, retrieve

__all__ = [
    "Bank",
    "EmbeddingBackend",
    "OpenAIEmbeddingBackend",
    "cosine",
    "MemoryHarness",
    "Distiller",
    "Solver",
    "TraceDistiller",
    "is_anti_give_up",
    "render_memory_block",
    "RetrievalResult",
    "retrieve",
]
