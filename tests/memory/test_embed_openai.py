"""Tests for the live OpenAI-compatible embedding backend."""
from __future__ import annotations

from types import SimpleNamespace

from lm_repl.memory.embed import EmbeddingBackend
from lm_repl.memory.embed_openai import OpenAIEmbeddingBackend


class FakeEmbeddings:
    def __init__(self, vector):
        self.vector = vector
        self.calls = []

    def create(self, *, model, input):
        self.calls.append({"model": model, "input": input})
        return SimpleNamespace(data=[SimpleNamespace(embedding=self.vector)])


class FakeClient:
    def __init__(self, vector):
        self.embeddings = FakeEmbeddings(vector)


def test_embed_returns_vector_from_response():
    client = FakeClient([0.1, 0.2, 0.3])
    backend = OpenAIEmbeddingBackend(client, model="nv-embed-v2")
    assert backend.embed("hello") == [0.1, 0.2, 0.3]


def test_embed_passes_model_and_input_through():
    client = FakeClient([0.0])
    backend = OpenAIEmbeddingBackend(client, model="my-embed-model")
    backend.embed("the query text")
    assert client.embeddings.calls == [
        {"model": "my-embed-model", "input": "the query text"}
    ]


def test_embed_coerces_to_float_list():
    client = FakeClient((1, 2, 3))  # ints in a tuple
    backend = OpenAIEmbeddingBackend(client, model="m")
    result = backend.embed("x")
    assert result == [1.0, 2.0, 3.0]
    assert all(isinstance(v, float) for v in result)


def test_satisfies_embedding_backend_protocol():
    backend = OpenAIEmbeddingBackend(FakeClient([0.0]), model="m")
    assert isinstance(backend, EmbeddingBackend)
