"""Tests for the prehend memory bank (meta.json store)."""
from __future__ import annotations

from prehend.memory.bank import Bank


def _entry(eid: str, use: int = 0, hit: int = 0) -> dict:
    return {
        "id": eid,
        "key_insight": f"insight {eid}",
        "polarity": "positive",
        "stats": {"use_count": use, "hit_count": hit},
    }


def test_load_missing_bank_returns_empty(tmp_path):
    bank = Bank(tmp_path / "mem")
    assert bank.load() == []


def test_append_then_load_roundtrips(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a"))
    bank.append(_entry("b"))
    loaded = bank.load()
    assert [e["id"] for e in loaded] == ["a", "b"]


def test_append_creates_meta_file_on_disk(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a"))
    assert bank.meta_path.exists()
    # A fresh Bank instance sees the persisted data.
    assert [e["id"] for e in Bank(tmp_path / "mem").load()] == ["a"]


def test_append_refuses_to_shrink_existing_bank(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a"))
    bank.append(_entry("b"))
    # Directly attempting to save fewer entries than on disk is rejected.
    assert bank.save([_entry("a")]) is False
    assert [e["id"] for e in bank.load()] == ["a", "b"]


def test_bump_stats_increments_use_and_hit(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a"))
    bank.bump_stats("a", use_delta=1, hit_delta=0)
    bank.bump_stats("a", use_delta=1, hit_delta=1)
    entry = bank.load()[0]
    assert entry["stats"] == {"use_count": 2, "hit_count": 1}


def test_prune_archives_cold_entries_past_threshold(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("cold", use=0))
    bank.append(_entry("warm", use=5))
    removed = bank.prune(total_queries_seen=500, cold_query_threshold=500)
    assert removed == 1
    assert [e["id"] for e in bank.load()] == ["warm"]


def test_prune_keeps_cold_entries_before_threshold(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("cold", use=0))
    removed = bank.prune(total_queries_seen=10, cold_query_threshold=500)
    assert removed == 0
    assert [e["id"] for e in bank.load()] == ["cold"]
