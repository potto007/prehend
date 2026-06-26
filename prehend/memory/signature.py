"""Deterministic document signature for experience-memory retrieval gating.

The bank embeds the BARE question (paraphrase-friendly ranking; see
``distill.py``), so two tasks that ask the same question over DIFFERENT
documents land at cosine ~= 1.0 and an experience distilled for document A
self-injects when solving document B. ``context_signature`` reduces a document
to a short stable token that ``MemoryHarness`` stamps onto the entry as a
``ctx_sig`` tag and passes as a query tag; the existing permissive
``_passes_tag_gate`` then excludes an experience whose document signature
conflicts with the one being solved, without touching the embedding key.

The signature is content-only (a normalized hash), so it is stable across runs
and processes - no Date/random, safe under forked benchmark workers.
"""
from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")

# 16 hex chars (64 bits) of SHA-1 over the normalized context: collision-free in
# practice for a per-run bank while staying short in the serialized entry.
_SIG_LEN = 16


def context_signature(context: str) -> str:
    """Return a short stable signature for ``context``.

    Normalization (strip, lowercase, whitespace-collapse) absorbs incidental
    formatting differences so the SAME document always hashes identically while
    DIFFERENT documents (the cross-document collision we gate out) differ.
    """
    normalized = _WS.sub(" ", (context or "").strip().lower())
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return digest[:_SIG_LEN]
