"""Tests for prehend memory-block rendering."""
from __future__ import annotations

from prehend.memory.inject import render_memory_block


def _entry(eid, polarity="positive", key_insight="do the thing", **extra):
    e = {"id": eid, "polarity": polarity, "key_insight": key_insight}
    e.update(extra)
    return e


def test_empty_entries_render_empty_string():
    assert render_memory_block([]) == ""


def test_block_wraps_and_includes_optout():
    block = render_memory_block([_entry("a")])
    assert block.startswith("<Memory_Block>")
    assert block.rstrip().endswith("</Memory_Block>")
    assert "<OptOut>" in block


def test_entry_renders_insight_and_polarity():
    block = render_memory_block([_entry("a", polarity="positive", key_insight="use ratios")])
    assert "use ratios" in block
    assert 'polarity="positive"' in block


def test_negative_polarity_is_marked():
    block = render_memory_block([_entry("a", polarity="negative", key_insight="watch units")])
    assert 'polarity="negative"' in block
    assert "watch units" in block


def test_xml_special_chars_are_escaped():
    block = render_memory_block([_entry("a", key_insight="x < y & z > 0")])
    assert "x &lt; y &amp; z &gt; 0" in block
    assert "x < y & z > 0" not in block


def test_findings_and_cautions_lists_are_rendered():
    block = render_memory_block([
        _entry("a", findings=["f1", "f2"], cautions=["c1"]),
    ])
    assert "f1" in block and "f2" in block
    assert "c1" in block
