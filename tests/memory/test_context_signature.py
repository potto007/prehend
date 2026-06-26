"""Tests for the opt-in document-signature retrieval gate (context_signature)."""
from __future__ import annotations

from types import SimpleNamespace

from prehend.memory.bank import Bank
from prehend.memory.harness import MemoryHarness
from prehend.memory.signature import context_signature


class FakeSolver:
    def __init__(self, answer="42"):
        self.answer = answer
        self.calls: list[tuple] = []

    def completion(self, prompt, root_prompt=None):
        self.calls.append((prompt, root_prompt))
        return SimpleNamespace(final_answer=self.answer)


class FakeBackend:
    """Maps a text to a fixed vector; the bare question always self-matches."""

    def __init__(self, table=None):
        self.table = table or {}

    def embed(self, text):
        return self.table.get(text, [0.0, 0.0])


def _distiller(backend):
    """Distills an entry whose embedding is the question's, so it self-matches."""

    def distill(question, context, result, failed=False):
        return {
            "id": f"e::{question}",
            "polarity": "positive",
            "key_insight": f"insight for {question}",
            "embedding": backend.embed(question),
            "stats": {"use_count": 0, "hit_count": 0},
        }

    return distill


# --- the signature helper itself ---------------------------------------------


def test_signature_is_deterministic_and_normalizes():
    a = context_signature("Doc A: the filing limit is 90 days.")
    assert a == context_signature("Doc A: the filing limit is 90 days.")
    # whitespace + case are normalized away
    assert a == context_signature("  doc a:   THE filing LIMIT is 90 days.  ")
    # a genuinely different document differs
    assert a != context_signature("Doc B: the filing limit is 30 days.")
    assert len(a) == 16


# --- the gate behavior --------------------------------------------------------


def _harness(tmp_path, *, gate: bool):
    backend = FakeBackend({"Q": [1.0, 0.0]})
    bank = Bank(tmp_path / "mem")
    return MemoryHarness(
        FakeSolver(), bank, backend, min_cosine=0.5,
        distiller=_distiller(backend), context_signature=gate,
    ), bank


def test_gate_blocks_same_question_different_document(tmp_path):
    harness, bank = _harness(tmp_path, gate=True)
    # Learn an experience over document A.
    harness.answer(context="DOCUMENT A", question="Q")
    # The stored entry carries A's signature as a ctx_sig tag.
    entry = bank.load()[0]
    assert entry["tags"]["ctx_sig"] == context_signature("DOCUMENT A")

    # Solving the SAME question over a DIFFERENT document must NOT inject A's
    # experience: same bare-question embedding, but the ctx_sig conflicts.
    solver = FakeSolver()
    harness.solver = solver
    harness.answer(context="DOCUMENT B", question="Q")
    _, root_prompt = solver.calls[0]
    assert root_prompt == "Q"
    assert "<Memory_Block>" not in root_prompt


def test_gate_allows_same_document(tmp_path):
    harness, _ = _harness(tmp_path, gate=True)
    harness.answer(context="DOCUMENT A", question="Q")
    solver = FakeSolver()
    harness.solver = solver
    # Same question, same document -> ctx_sig matches -> injects.
    harness.answer(context="DOCUMENT A", question="Q")
    _, root_prompt = solver.calls[0]
    assert "<Memory_Block>" in root_prompt


def test_gate_off_injects_across_documents(tmp_path):
    # Opt-in: with the gate off, cross-document self-injection still happens
    # (the pre-existing behavior), and no ctx_sig tag is stamped.
    harness, bank = _harness(tmp_path, gate=False)
    harness.answer(context="DOCUMENT A", question="Q")
    assert "tags" not in bank.load()[0]
    solver = FakeSolver()
    harness.solver = solver
    harness.answer(context="DOCUMENT B", question="Q")
    _, root_prompt = solver.calls[0]
    assert "<Memory_Block>" in root_prompt
