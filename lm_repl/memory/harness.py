"""mnemex MemoryHarness: wrap a solver with self-evolving experience memory.

This is the integration seam. It mirrors FinAcumen's ``MemoryAgentVariant``
(retrieve -> delegate solve -> collect) but delegates to anything exposing the
lm-repl ``completion(prompt, root_prompt)`` interface -- in practice an
:class:`~lm_repl.SRLM`. ``prompt`` is the context offloaded into the REPL;
``root_prompt`` is the question the orchestrator attends to directly, and where
the retrieved ``<Memory_Block>`` is injected.

Design invariants carried over from FinAcumen:
  * No-memory baseline integrity -- when retrieval is empty, ``root_prompt`` is
    byte-identical to the bare question; no memory tokens leak in.
  * Graceful degradation -- any retrieval failure falls back to no-memory; a
    single failure never crashes the solve.
  * Collect is best-effort -- a distillation failure never breaks the answer.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from lm_repl.memory.bank import Bank
from lm_repl.memory.embed import EmbeddingBackend
from lm_repl.memory.inject import render_memory_block
from lm_repl.memory.retrieve import (
    DEFAULT_K_MAX,
    DEFAULT_MIN_COSINE,
    retrieve,
)

# (question, context, solver_result) -> a new entry dict, or None to write nothing.
Distiller = Callable[[str, str, Any], dict | None]


class Solver(Protocol):
    def completion(self, prompt: str, root_prompt: str | None = None) -> Any:
        ...


class MemoryHarness:
    """Adds retrieve/inject/collect around a context-offloading solver."""

    def __init__(
        self,
        solver: Solver,
        bank: Bank,
        backend: EmbeddingBackend,
        *,
        k_max: int = DEFAULT_K_MAX,
        min_cosine: float = DEFAULT_MIN_COSINE,
        distiller: Distiller | None = None,
    ) -> None:
        self.solver = solver
        self.bank = bank
        self.backend = backend
        self.k_max = k_max
        self.min_cosine = min_cosine
        self.distiller = distiller

    def answer(self, context: str, question: str) -> Any:
        """Solve ``question`` over ``context``, using and growing memory."""
        entries, scores = self._retrieve(question)

        if entries:
            block = render_memory_block(entries)
            root_prompt = f"{block}\n{question}" if block else question
            for entry in entries:
                eid = entry.get("id")
                if eid is not None:
                    try:
                        self.bank.bump_stats(eid, use_delta=1)
                    except Exception:
                        pass
        else:
            root_prompt = question

        result = self.solver.completion(context, root_prompt)

        self._collect(question, context, result)
        return result

    def _retrieve(self, question: str) -> tuple[list[dict], list[float]]:
        """Retrieve experiences, degrading to empty on any failure."""
        try:
            res = retrieve(
                question, self.bank, self.backend,
                k_max=self.k_max, min_cosine=self.min_cosine,
            )
            return res.entries, res.scores
        except Exception:
            return [], []

    def _collect(self, question: str, context: str, result: Any) -> None:
        """Best-effort distillation of a new experience; never raises."""
        if self.distiller is None:
            return
        try:
            entry = self.distiller(question, context, result)
            if not entry:
                return
            eid = entry.get("id")
            if eid is not None and any(e.get("id") == eid for e in self.bank.load()):
                return  # one experience per id; do not append a duplicate
            self.bank.append(entry)
        except Exception:
            pass
