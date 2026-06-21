"""Tests for the pluggable Tagger seam."""
from __future__ import annotations

from prehend.memory.tagger import NullTagger, Tagger


def test_null_tagger_returns_empty_tags():
    assert NullTagger().tag("anything") == {}


def test_null_tagger_satisfies_protocol():
    assert isinstance(NullTagger(), Tagger)
