"""Tests for the prehend embedding helpers."""
from __future__ import annotations

import math

from prehend.memory.embed import EmbeddingBackend, HashingEmbeddingBackend, cosine


def test_cosine_of_identical_vectors_is_one():
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_is_scale_invariant():
    # Direction matters, magnitude does not.
    assert math.isclose(cosine([1.0, 1.0], [3.0, 3.0]), 1.0)


def test_cosine_of_opposite_vectors_is_minus_one():
    assert math.isclose(cosine([1.0, 0.0], [-1.0, 0.0]), -1.0)


def test_hashing_backend_is_deterministic():
    backend = HashingEmbeddingBackend(dim=64)
    assert backend.embed("six times seven") == backend.embed("six times seven")


def test_hashing_backend_respects_dimension():
    assert len(HashingEmbeddingBackend(dim=32).embed("anything")) == 32


def test_hashing_backend_different_text_yields_different_vector():
    backend = HashingEmbeddingBackend(dim=64)
    assert backend.embed("alpha") != backend.embed("beta")


def test_hashing_backend_identical_text_is_cosine_one():
    # The retrieval guarantee: re-asking the same question matches itself.
    backend = HashingEmbeddingBackend(dim=64)
    v = backend.embed("repeat me")
    assert math.isclose(cosine(v, backend.embed("repeat me")), 1.0)


def test_hashing_backend_satisfies_embedding_protocol():
    assert isinstance(HashingEmbeddingBackend(), EmbeddingBackend)
