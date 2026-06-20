"""Live embedding backend over an OpenAI-compatible endpoint.

Reuses the same OpenAI-compatible server lm-repl already drives for completions
(set ``base_url``/``api_key`` to the local llama-server). This is the production
:class:`~lm_repl.memory.embed.EmbeddingBackend`, replacing FinAcumen's pre-baked
id-keyed ``datasets/*_emb.npy`` lookup with live per-query embedding.
"""
from __future__ import annotations

from typing import Any


class OpenAIEmbeddingBackend:
    """Embeds text via ``client.embeddings.create`` (OpenAI SDK shape)."""

    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def embed(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.model, input=text)
        return [float(x) for x in resp.data[0].embedding]

    @classmethod
    def from_config(
        cls, *, base_url: str, model: str, api_key: str = "EMPTY"
    ) -> OpenAIEmbeddingBackend:
        """Build a backend backed by a real ``openai.OpenAI`` client.

        ``api_key`` defaults to a placeholder for local servers that ignore it.
        """
        import openai

        client = openai.OpenAI(base_url=base_url, api_key=api_key)
        return cls(client, model=model)
