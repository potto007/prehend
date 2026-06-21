"""mnemex embedding helpers and the pluggable embedding backend.

FinAcumen resolved embeddings from pre-baked ``datasets/*_emb.npy`` keyed by a
benchmark ``target_id`` (zero API calls during eval). mnemex instead embeds
arbitrary queries live through an injected :class:`EmbeddingBackend`, so the
memory layer works on open-ended inputs rather than a fixed benchmark set.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

Vector = Sequence[float]


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity between two vectors. Scale-invariant.

    Returns 0.0 when either vector has zero magnitude (no defined direction).
    """
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Anything that turns text into a fixed-dimension embedding vector."""

    def embed(self, text: str) -> list[float]:
        ...


class HashingEmbeddingBackend:
    """Deterministic, dependency-free embedding from a hash of the text.

    For smoke tests and offline runs where no embedding endpoint exists (the
    local llama-server serves only chat GGUFs). The same text always maps to the
    same vector, so re-asking a question retrieves its own bank entry; distinct
    texts almost always map to distinct vectors. It carries no semantic meaning,
    so it cannot match paraphrases - use :class:`OpenAIEmbeddingBackend` for that.
    """

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        data = text.encode("utf-8")
        out: list[float] = []
        counter = 0
        while len(out) < self.dim:
            digest = hashlib.sha256(data + counter.to_bytes(8, "big")).digest()
            for i in range(0, len(digest), 4):
                if len(out) >= self.dim:
                    break
                word = int.from_bytes(digest[i : i + 4], "big")
                # Map to [-1, 1) so vectors spread across orthants like real embeddings.
                out.append(word / 2**31 - 1.0)
            counter += 1
        return out
