"""prehend MemoryHarness: wrap a solver with self-evolving experience memory.

This is the integration seam. It mirrors FinAcumen's ``MemoryAgentVariant``
(retrieve -> delegate solve -> collect) but delegates to anything exposing the
prehend ``completion(prompt, root_prompt)`` interface -- in practice an
:class:`~prehend.SRLM`. ``prompt`` is the context offloaded into the REPL;
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

import time
from collections.abc import Callable
from typing import Any, Protocol

from prehend.memory.bank import Bank
from prehend.memory.embed import EmbeddingBackend
from prehend.memory.inject import render_memory_block
from prehend.memory.retrieve import (
    DEFAULT_K_MAX,
    DEFAULT_MIN_COSINE,
    retrieve,
)
from prehend.memory.tagger import NullTagger, Tagger

# (question, context, solver_result) -> a new entry dict, or None to write nothing.
Distiller = Callable[[str, str, Any], dict | None]


class Solver(Protocol):
    def completion(self, prompt: str, root_prompt: str | None = None) -> Any:
        ...


class MemoryObserver(Protocol):
    """Observe MemoryHarness internals for metrics/telemetry.

    Both methods are best-effort: the harness wraps every call so an observer
    bug never breaks a solve. The default :class:`NullObserver` no-ops, keeping
    the harness free of any metrics dependency. ``prehend.metrics`` ships a
    ``PrometheusMemoryObserver`` that maps these events onto Prometheus series.
    """

    def on_retrieve(
        self, *, entries: int, top_score: float | None, block_chars: int,
        seconds: float, error: bool,
    ) -> None: ...

    def on_collect(
        self, *, outcome: str, seconds: float, bank_size: int | None,
    ) -> None: ...


class NullObserver:
    """No-op :class:`MemoryObserver`; the default when none is supplied."""

    def on_retrieve(self, **_: Any) -> None:
        pass

    def on_collect(self, **_: Any) -> None:
        pass


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
        tagger: Tagger | None = None,
        defer_collect: bool = False,
        observer: MemoryObserver | None = None,
    ) -> None:
        self.solver = solver
        self.bank = bank
        self.backend = backend
        self.k_max = k_max
        self.min_cosine = min_cosine
        self.distiller = distiller
        self.tagger = tagger or NullTagger()
        self.observer = observer or NullObserver()
        # When True, answer() solves but does NOT distill; the caller invokes
        # collect_pending(correct) once it knows the outcome, so a wrong solve
        # never poisons the bank (the dominant no-upside cause on the v13
        # plain-multihop eval). Default off keeps the drop-in Solver contract:
        # a bare completion() call still learns immediately.
        self.defer_collect = defer_collect
        self._pending: tuple[str, str, Any, dict] | None = None

    def completion(self, prompt: str, root_prompt: str | None = None) -> Any:
        """Transparent :class:`Solver` adapter over :meth:`answer`.

        Lets a memory-wrapped solver drop into any call site that drives a bare
        ``Solver`` via ``completion(context, question)``. ``prompt`` is the
        offloaded context; ``root_prompt`` is the question (defaulting to
        ``prompt`` when omitted, matching the bare-solver convention).
        """
        return self.answer(prompt, root_prompt if root_prompt is not None else prompt)

    def _observe(self, method: str, **kw: Any) -> None:
        """Call an observer hook best-effort; a telemetry bug never propagates."""
        try:
            getattr(self.observer, method)(**kw)
        except Exception:
            pass

    def answer(self, context: str, question: str) -> Any:
        """Solve ``question`` over ``context``, using and growing memory."""
        try:
            query_tags = self.tagger.tag(question)
        except Exception:
            query_tags = {}
        entries, scores, error, retrieve_seconds = self._retrieve(question, query_tags)

        block = ""
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

        self._observe(
            "on_retrieve",
            entries=len(entries),
            top_score=(max(scores) if scores else None),
            block_chars=len(block),
            seconds=retrieve_seconds,
            error=error,
        )

        result = self.solver.completion(context, root_prompt)

        if self.defer_collect:
            self._pending = (question, context, result, query_tags)
            self._observe("on_collect", outcome="deferred", seconds=0.0, bank_size=None)
        else:
            self._collect(question, context, result, query_tags)
        return result

    def collect_pending(self, correct: bool | None = True) -> None:
        """Distill the last deferred solve, gated on its outcome.

        ``correct is False`` drops it (do not learn from a wrong solve);
        ``True`` or ``None`` (unscored) distills. No-op when nothing is pending
        or ``defer_collect`` was off. Best-effort: never raises.
        """
        pending = self._pending
        self._pending = None
        if pending is None:
            return
        if correct is False:
            self._observe("on_collect", outcome="dropped", seconds=0.0, bank_size=None)
            return
        question, context, result, query_tags = pending
        self._collect(question, context, result, query_tags)

    def _retrieve(
        self, question: str, query_tags: dict
    ) -> tuple[list[dict], list[float], bool, float]:
        """Retrieve experiences, degrading to empty on any failure.

        Returns ``(entries, scores, error, seconds)`` -- ``error`` distinguishes
        a real retrieval failure from a clean miss, and ``seconds`` is the
        embed+search wall time, both for the observer.
        """
        t0 = time.perf_counter()
        try:
            res = retrieve(
                question, self.bank, self.backend,
                k_max=self.k_max, min_cosine=self.min_cosine,
                query_tags=query_tags,
            )
            return res.entries, res.scores, False, time.perf_counter() - t0
        except Exception:
            return [], [], True, time.perf_counter() - t0

    def _collect(self, question: str, context: str, result: Any, query_tags: dict) -> None:
        """Best-effort distillation of a new experience; never raises."""
        if self.distiller is None:
            return
        t0 = time.perf_counter()
        outcome = "error"
        bank_size: int | None = None
        try:
            entry = self.distiller(question, context, result)
            if not entry:
                outcome = "empty"
                return
            if query_tags and not entry.get("tags"):
                entry["tags"] = dict(query_tags)
            existing = self.bank.load()
            eid = entry.get("id")
            if eid is not None and any(e.get("id") == eid for e in existing):
                outcome = "duplicate"  # one experience per id; do not append
                return
            self.bank.append(entry)
            outcome = "written"
            bank_size = len(existing) + 1
        except Exception:
            outcome = "error"
        finally:
            self._observe(
                "on_collect", outcome=outcome,
                seconds=time.perf_counter() - t0, bank_size=bank_size,
            )
