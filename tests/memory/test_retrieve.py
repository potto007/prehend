"""Tests for prehend retrieval (cosine ranking over the bank)."""
from __future__ import annotations

from prehend.memory.bank import Bank
from prehend.memory.retrieve import retrieve


class FakeBackend:
    """Maps known text to fixed vectors; unknown text -> zero vector."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table

    def embed(self, text: str) -> list[float]:
        return self.table.get(text, [0.0, 0.0])


def _entry(eid: str, embedding: list[float]) -> dict:
    return {
        "id": eid,
        "key_insight": f"insight {eid}",
        "polarity": "positive",
        "embedding": embedding,
        "stats": {"use_count": 0, "hit_count": 0},
    }


def test_empty_bank_returns_no_memory(tmp_path):
    bank = Bank(tmp_path / "mem")
    backend = FakeBackend({"q": [1.0, 0.0]})
    result = retrieve("q", bank, backend)
    assert result.mode == "no-memory"
    assert result.entries == []


def test_returns_most_similar_entry_first(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("aligned", [1.0, 0.0]))
    bank.append(_entry("orthogonal", [0.0, 1.0]))
    backend = FakeBackend({"q": [1.0, 0.0]})
    result = retrieve("q", bank, backend, min_cosine=0.5)
    assert result.mode == "with-memory"
    assert result.entries[0]["id"] == "aligned"


def test_drops_entries_below_min_cosine(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("orthogonal", [0.0, 1.0]))
    backend = FakeBackend({"q": [1.0, 0.0]})
    result = retrieve("q", bank, backend, min_cosine=0.65)
    assert result.mode == "no-memory"
    assert result.entries == []


def test_respects_k_max(tmp_path):
    bank = Bank(tmp_path / "mem")
    for i in range(5):
        bank.append(_entry(f"e{i}", [1.0, 0.01 * i]))
    backend = FakeBackend({"q": [1.0, 0.0]})
    result = retrieve("q", bank, backend, k_max=2, min_cosine=0.5)
    assert len(result.entries) == 2


def test_dedups_by_id(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("dup", [1.0, 0.0]))
    bank.append(_entry("dup", [1.0, 0.0]))
    backend = FakeBackend({"q": [1.0, 0.0]})
    result = retrieve("q", bank, backend, k_max=5, min_cosine=0.5)
    assert [e["id"] for e in result.entries] == ["dup"]


def _tagged_entry(eid, embedding, tags):
    e = _entry(eid, embedding)
    e["tags"] = tags
    return e


def test_query_tags_none_does_not_filter(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_tagged_entry("a", [1.0, 0.0], {"kind": "numeric"}))
    backend = FakeBackend({"q": [1.0, 0.0]})
    result = retrieve("q", bank, backend, min_cosine=0.5, query_tags=None)
    assert [e["id"] for e in result.entries] == ["a"]


def test_conflicting_tag_excludes_entry(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_tagged_entry("numeric", [1.0, 0.0], {"kind": "numeric"}))
    backend = FakeBackend({"q": [1.0, 0.0]})
    result = retrieve("q", bank, backend, min_cosine=0.5, query_tags={"kind": "entity"})
    assert result.mode == "no-memory"


def test_missing_tag_key_is_permissive(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("untagged", [1.0, 0.0]))  # no tags at all
    backend = FakeBackend({"q": [1.0, 0.0]})
    result = retrieve("q", bank, backend, min_cosine=0.5, query_tags={"kind": "numeric"})
    assert [e["id"] for e in result.entries] == ["untagged"]


def test_matching_tag_is_kept(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_tagged_entry("a", [1.0, 0.0], {"kind": "numeric"}))
    backend = FakeBackend({"q": [1.0, 0.0]})
    result = retrieve("q", bank, backend, min_cosine=0.5, query_tags={"kind": "numeric"})
    assert [e["id"] for e in result.entries] == ["a"]


def test_scores_align_with_entries(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("aligned", [1.0, 0.0]))
    backend = FakeBackend({"q": [1.0, 0.0]})
    result = retrieve("q", bank, backend, min_cosine=0.5)
    assert len(result.scores) == len(result.entries)
    assert result.scores[0] == 1.0
