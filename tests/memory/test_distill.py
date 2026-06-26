"""Tests for the trace-based experience distiller."""
from __future__ import annotations

import json
from types import SimpleNamespace

from prehend.memory.distill import TraceDistiller


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


def test_embeds_question_only_so_key_matches_retrieval():
    # Retrieval embeds the bare query (harness calls retrieve(question)), so the
    # stored key must also be the bare question - mixing the offloaded context
    # into the stored embedding dilutes it and breaks paraphrase retrieval.
    captured = []

    class RecordingBackend:
        def embed(self, text):
            captured.append(text)
            return [1.0, 0.0]

    d = TraceDistiller(_reflect({"key_insight": "k"}), RecordingBackend())
    d("What is 6 times 7?", "Basic arithmetic facts. Long offloaded context.", _result())
    assert captured == ["What is 6 times 7?"]


def test_invalid_polarity_defaults_to_positive():
    reflect = _reflect({"key_insight": "k", "polarity": "sideways"})
    d = TraceDistiller(reflect, FakeBackend())
    assert d("q", "c", _result())["polarity"] == "positive"


# --- Contrastive failure channel (ADR-0010 / 2026-06-22 spec) ---

def test_failed_true_uses_failure_prompt_and_differs_from_success():
    reflect = _reflect({"key_insight": "When chunks conflict, cross-check all before concluding"})
    d = TraceDistiller(reflect, FakeBackend())
    d("Q?", "ctx", _result(), failed=True)
    fprompt = reflect.calls[0]
    # the failure prompt frames the trace as incorrect and asks for a corrective guard
    assert "INCORRECT" in fprompt or "incorrect" in fprompt
    # and it differs from the success prompt
    succ = _reflect({"key_insight": "k"})
    TraceDistiller(succ, FakeBackend())("Q?", "ctx", _result(), failed=False)
    assert fprompt != succ.calls[0]


def test_failed_true_forces_negative_polarity():
    # even when the model returns positive, a failure entry is negative
    reflect = _reflect({"polarity": "positive",
                        "key_insight": "When the join is wrong, verify each sub-result"})
    d = TraceDistiller(reflect, FakeBackend())
    entry = d("Q?", "ctx", _result(), failed=True)
    assert entry["polarity"] == "negative"


def test_failed_false_keeps_model_polarity():
    reflect = _reflect({"polarity": "positive", "key_insight": "decompose first"})
    d = TraceDistiller(reflect, FakeBackend())
    entry = d("Q?", "ctx", _result(), failed=False)
    assert entry["polarity"] == "positive"


def test_failure_entry_has_derived_from_failure():
    reflect = _reflect({"key_insight": "When the join is wrong, verify each sub-result"})
    d = TraceDistiller(reflect, FakeBackend())
    entry = d("Q?", "ctx", _result(), failed=True)
    assert entry["derived_from"] == "failure"


def test_success_entry_has_derived_from_success():
    reflect = _reflect({"key_insight": "decompose first"})
    d = TraceDistiller(reflect, FakeBackend())
    entry = d("Q?", "ctx", _result(), failed=False)
    assert entry["derived_from"] == "success"


def test_failure_and_success_for_same_question_get_distinct_ids():
    # ADR-0011 amendment: a failure guard and a success recipe for the SAME
    # question must NOT share an experience id. With question-only ids they
    # collided, so the harness "success shadows failure" guard dropped every
    # negative and the contrastive failure channel never fired on context-
    # varying task sets (plain multihop: 60 tasks share only 5 questions).
    # Keying the id on (question, derived_from) lets the recipe and the guard
    # coexist so both can be retrieved/injected.
    reflect = _reflect({"key_insight": "k"})
    d = TraceDistiller(reflect, FakeBackend())
    succ = d("What does Alice own?", "ctx-A", _result(), failed=False)
    fail = d("What does Alice own?", "ctx-B", _result(), failed=True)
    assert succ["id"] != fail["id"]
    # but (question, provenance) is still stable -> same-provenance entries dedupe
    again = d("What does Alice own?", "ctx-C", _result(), failed=True)
    assert again["id"] == fail["id"]


def test_failure_capitulation_still_filtered_to_none():
    # a failure whose only content is capitulation yields no usable entry
    reflect = _reflect({"key_insight": "the information is missing from the context",
                        "findings": [], "cautions": []})
    d = TraceDistiller(reflect, FakeBackend())
    assert d("Q?", "ctx", _result(), failed=True) is None


def test_failure_premature_stop_filtered_to_none():
    reflect = _reflect({"key_insight": "When chunks conflict, prefer the first and stop searching",
                        "findings": [], "cautions": []})
    d = TraceDistiller(reflect, FakeBackend())
    assert d("Q?", "ctx", _result(), failed=True) is None


def test_failure_constructive_guard_survives():
    reflect = _reflect({"key_insight": "When chunks conflict, re-read and cross-check before concluding",
                        "findings": [], "cautions": []})
    d = TraceDistiller(reflect, FakeBackend())
    entry = d("Q?", "ctx", _result(), failed=True)
    assert entry is not None
    assert entry["polarity"] == "negative"


def test_success_path_not_filtered_by_premature_stop():
    # a positive recipe that mentions "prefer the first" must NOT be dropped on
    # the success path (is_premature_stop is failure-only).
    reflect = _reflect({"polarity": "positive",
                        "key_insight": "index once, then prefer the first matching entity"})
    d = TraceDistiller(reflect, FakeBackend())
    entry = d("Q?", "ctx", _result(), failed=False)
    assert entry is not None
