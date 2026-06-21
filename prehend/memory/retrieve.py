"""prehend retrieval: rank bank entries against a live query embedding.

Single-stage cosine ranking (matching FinAcumen's shipped ``fm/retrieve.py``,
not the unimplemented 3-stage tagger/rerank in its docs), with the key change
that the query is embedded live through an injected backend rather than resolved
from a pre-baked, id-keyed matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from prehend.memory.bank import Bank
from prehend.memory.embed import EmbeddingBackend, cosine

DEFAULT_K_MAX = 3
DEFAULT_MIN_COSINE = 0.65


@dataclass
class RetrievalResult:
    """Outcome of a retrieval call.

    ``mode`` is ``"with-memory"`` when at least one entry cleared the threshold,
    else ``"no-memory"`` (the agent then runs with no experience injected).
    """

    mode: str
    entries: list[dict] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)


def _no_memory() -> RetrievalResult:
    return RetrievalResult(mode="no-memory", entries=[], scores=[])


def _passes_tag_gate(entry: dict, query_tags: dict | None) -> bool:
    """Permissive hard-gate: an entry is excluded only when a tag it defines
    conflicts with the query. Entries lacking a queried key are kept, so
    untagged banks behave exactly as if no gate were applied.
    """
    if not query_tags:
        return True
    entry_tags = entry.get("tags") or {}
    for key, value in query_tags.items():
        if key in entry_tags and entry_tags[key] != value:
            return False
    return True


def retrieve(
    query: str,
    bank: Bank,
    backend: EmbeddingBackend,
    *,
    k_max: int = DEFAULT_K_MAX,
    min_cosine: float = DEFAULT_MIN_COSINE,
    query_tags: dict | None = None,
) -> RetrievalResult:
    """Return up to ``k_max`` bank entries most similar to ``query``.

    Entries scoring below ``min_cosine`` are dropped; duplicates sharing an
    ``id`` are collapsed to their highest-scoring occurrence. When ``query_tags``
    is given, entries whose own tags conflict with it are gated out first.
    """
    entries = bank.load()
    if not entries:
        return _no_memory()

    query_vec = backend.embed(query)

    scored: list[tuple[dict, float]] = []
    for entry in entries:
        if not _passes_tag_gate(entry, query_tags):
            continue
        emb = entry.get("embedding")
        if not emb:
            continue
        score = cosine(query_vec, emb)
        if score >= min_cosine:
            scored.append((entry, score))

    if not scored:
        return _no_memory()

    scored.sort(key=lambda pair: pair[1], reverse=True)

    selected: list[tuple[dict, float]] = []
    seen_ids: set[str] = set()
    for entry, score in scored:
        eid = entry.get("id")
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        selected.append((entry, score))
        if len(selected) >= k_max:
            break

    return RetrievalResult(
        mode="with-memory",
        entries=[e for e, _ in selected],
        scores=[s for _, s in selected],
    )
