"""prehend memory bank: a directory-backed experience store.

Persists experience entries to ``<bank_dir>/meta.json``. Generic and
domain-agnostic: an entry is a plain dict carrying at minimum an ``id``,
a ``key_insight``, a ``polarity``, and a ``stats`` block. Writes are atomic
(tmp -> rename) and guarded against shrinking a valid file to corruption.

Ported and genericized from FinAcumen's ``finacumen/fm/bank.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

BANK_VERSION = 1
COLD_QUERY_THRESHOLD = 500


class Bank:
    """A directory-backed store of experience entries (meta.json)."""

    def __init__(self, bank_dir: Path | str) -> None:
        self.bank_dir = Path(bank_dir)

    @property
    def meta_path(self) -> Path:
        return self.bank_dir / "meta.json"

    def load(self) -> list[dict]:
        """Return the list of entries, or ``[]`` if the bank does not exist."""
        if not self.meta_path.exists():
            return []
        with open(self.meta_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("entries", [])

    def save(self, entries: list[dict], allow_shrink: bool = False) -> bool:
        """Atomically write ``entries`` to meta.json.

        Refuses to write an empty list, or to shrink an existing valid file
        (corruption guard) unless ``allow_shrink`` is set (intentional pruning).
        Returns True if written, False if refused.
        """
        if not entries:
            return False
        if self.meta_path.exists() and not allow_shrink:
            try:
                old = self.load()
                if len(old) > len(entries):
                    return False
            except Exception:
                pass  # old file corrupted; proceed with overwrite
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.meta_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": BANK_VERSION, "entries": entries},
                      f, ensure_ascii=False, indent=2)
        tmp.replace(self.meta_path)
        return True

    def append(self, entry: dict) -> None:
        """Append one entry to the bank."""
        entries = self.load()
        entries.append(entry)
        self.save(entries)

    def bump_stats(self, entry_id: str, use_delta: int = 0, hit_delta: int = 0) -> None:
        """Increment the use/hit counters for the entry with ``id == entry_id``."""
        entries = self.load()
        for entry in entries:
            if entry.get("id") == entry_id:
                st = entry.setdefault("stats", {})
                st["use_count"] = st.get("use_count", 0) + use_delta
                st["hit_count"] = st.get("hit_count", 0) + hit_delta
                break
        else:
            return
        self.save(entries)

    def prune(self, total_queries_seen: int,
              cold_query_threshold: int = COLD_QUERY_THRESHOLD) -> int:
        """Drop entries never retrieved once the bank has seen enough queries.

        Returns the number of entries removed.
        """
        entries = self.load()
        kept = [
            e for e in entries
            if not (e.get("stats", {}).get("use_count", 0) == 0
                    and total_queries_seen >= cold_query_threshold)
        ]
        removed = len(entries) - len(kept)
        if removed > 0:
            self.save(kept, allow_shrink=True)
        return removed
