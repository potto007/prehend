"""Tests for the trace-based experience distiller."""
from __future__ import annotations

import json
from types import SimpleNamespace

from lm_repl.memory.distill import TraceDistiller


class FakeBackend:
    def embed(self, text):
        return [0.5, 0.5]


def _reflect(payload):
    """Build a reflect_fn that returns a fixed JSON payload, recording prompts."""
    calls = []

    def fn(prompt):
        calls.append(prompt)
        return json.dumps(payload) if not isinstance(payload, str) else payload

    fn.calls = calls
    return fn


def _result(response="42", metadata=None):
    return SimpleNamespace(response=response, metadata=metadata or {})


def test_produces_wellformed_entry():
    reflect = _reflect({
        "polarity": "positive",
        "key_insight": "decompose into sub-queries first",
        "findings": ["chunk the context"],
        "cautions": [],
    })
    d = TraceDistiller(reflect, FakeBackend())
    entry = d("What is X?", "ctx", _result())
    assert entry["polarity"] == "positive"
    assert entry["key_insight"] == "decompose into sub-queries first"
    assert entry["findings"] == ["chunk the context"]
    assert entry["embedding"] == [0.5, 0.5]
    assert entry["stats"] == {"use_count": 0, "hit_count": 0}
    assert "id" in entry


def test_reflect_prompt_includes_question_and_answer():
    reflect = _reflect({"key_insight": "k"})
    d = TraceDistiller(reflect, FakeBackend())
    d("How many widgets?", "ctx", _result(response="seven"))
    prompt = reflect.calls[0]
    assert "How many widgets?" in prompt
    assert "seven" in prompt


def test_parses_json_wrapped_in_prose_and_fences():
    raw = 'Sure!\n```json\n{"key_insight": "use ratios", "polarity": "positive"}\n```\nDone.'
    d = TraceDistiller(_reflect(raw), FakeBackend())
    entry = d("q", "c", _result())
    assert entry["key_insight"] == "use ratios"


def test_anti_give_up_findings_are_filtered():
    reflect = _reflect({
        "key_insight": "verify before answering",
        "findings": ["data not available", "chunk the table"],
        "cautions": ["cannot determine the value"],
    })
    d = TraceDistiller(reflect, FakeBackend())
    entry = d("q", "c", _result())
    assert entry["findings"] == ["chunk the table"]
    assert entry["cautions"] == []


def test_returns_none_when_reflect_unparseable():
    d = TraceDistiller(_reflect("no json here at all"), FakeBackend())
    assert d("q", "c", _result()) is None


def test_returns_none_when_nothing_useful_survives():
    reflect = _reflect({
        "key_insight": "data not available",
        "findings": ["insufficient data"],
        "cautions": [],
    })
    d = TraceDistiller(reflect, FakeBackend())
    assert d("q", "c", _result()) is None


def test_id_is_deterministic_from_question():
    reflect = _reflect({"key_insight": "k"})
    d = TraceDistiller(reflect, FakeBackend())
    e1 = d("same question", "c", _result())
    e2 = d("same question", "c", _result())
    e3 = d("different question", "c", _result())
    assert e1["id"] == e2["id"]
    assert e1["id"] != e3["id"]


def test_invalid_polarity_defaults_to_positive():
    reflect = _reflect({"key_insight": "k", "polarity": "sideways"})
    d = TraceDistiller(reflect, FakeBackend())
    assert d("q", "c", _result())["polarity"] == "positive"
