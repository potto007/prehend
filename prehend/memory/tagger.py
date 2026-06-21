"""Pluggable query tagger -- the prehend genericization seam (ADR-0005).

FinAcumen hardcoded a finance taxonomy (question_class / question_type /
tool_tags) to hard-gate retrieval. prehend abstracts that behind a ``Tagger``:
``tag(query) -> {key: value}`` structured keys, used both to gate retrieval and
to label collected experiences. The default :class:`NullTagger` returns no tags,
so retrieval is purely embedding-based (matching FinAcumen's shipped behavior).
Domain taggers (finance, code, research) plug in without touching the core.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tagger(Protocol):
    def tag(self, query: str) -> dict[str, str]:
        ...


class NullTagger:
    """A tagger that assigns no tags (embedding-only retrieval)."""

    def tag(self, query: str) -> dict[str, str]:
        return {}
