"""Tests for build_memory_harness and the full closed memory loop."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from prehend.memory.factory import build_memory_harness
from prehend.memory.harness import MemoryHarness


class FakeSolver:
    def __init__(self, answer="42"):
        self.answer = answer
        self.calls = []

    def completion(self, prompt, root_prompt=None):
        self.calls.append((prompt, root_prompt))
        return SimpleNamespace(response=self.answer, metadata={"iterations": []})


class ConstBackend:
    """Returns the same vector for any text, so a question always matches itself."""

    def embed(self, text):
        return [1.0, 0.0]


def _reflect_fn(payload):
    return lambda prompt: json.dumps(payload)


def test_returns_memory_harness_wired_to_bank(tmp_path):
    harness = build_memory_harness(
        FakeSolver(), tmp_path / "mem",
        embed_backend=ConstBackend(),
        reflect_fn=_reflect_fn({"key_insight": "k"}),
    )
    assert isinstance(harness, MemoryHarness)
    assert harness.bank.bank_dir == tmp_path / "mem"
    assert harness.distiller is not None


def test_requires_an_embedding_backend_or_client(tmp_path):
    with pytest.raises(ValueError):
        build_memory_harness(
            FakeSolver(), tmp_path / "mem",
            reflect_fn=_reflect_fn({"key_insight": "k"}),
        )


def test_requires_a_reflect_fn_or_client(tmp_path):
    with pytest.raises(ValueError):
        build_memory_harness(
            FakeSolver(), tmp_path / "mem",
            embed_backend=ConstBackend(),
        )


def test_full_loop_learns_then_retrieves(tmp_path):
    solver = FakeSolver(answer="seven")
    harness = build_memory_harness(
        solver, tmp_path / "mem",
        embed_backend=ConstBackend(),
        reflect_fn=_reflect_fn({
            "polarity": "positive",
            "key_insight": "count by chunking the table",
            "findings": ["sum the rows"],
            "cautions": [],
        }),
        min_cosine=0.5,
    )

    # Call 1: empty bank -> no memory injected, but collect writes an experience.
    harness.answer(context="ctx", question="How many widgets?")
    _, root_prompt_1 = solver.calls[0]
    assert root_prompt_1 == "How many widgets?"  # no-memory path, byte-identical
    assert len(harness.bank.load()) == 1

    # Call 2: same question -> the learned experience is retrieved and injected.
    harness.answer(context="ctx", question="How many widgets?")
    _, root_prompt_2 = solver.calls[1]
    assert "<Memory_Block>" in root_prompt_2
    assert "count by chunking the table" in root_prompt_2
    # Retrieval bumped the entry's use_count.
    assert harness.bank.load()[0]["stats"]["use_count"] == 1


class FakeTagger:
    def tag(self, query):
        return {"kind": "numeric"}


def test_tagger_is_passed_through(tmp_path):
    tagger = FakeTagger()
    harness = build_memory_harness(
        FakeSolver(), tmp_path / "mem",
        embed_backend=ConstBackend(),
        reflect_fn=_reflect_fn({"key_insight": "k"}),
        tagger=tagger,
    )
    assert harness.tagger is tagger


def test_full_loop_dedups_same_question(tmp_path):
    harness = build_memory_harness(
        FakeSolver(), tmp_path / "mem",
        embed_backend=ConstBackend(),
        reflect_fn=_reflect_fn({"key_insight": "k", "findings": ["f"]}),
        min_cosine=0.5,
    )
    harness.answer(context="c", question="Q")
    harness.answer(context="c", question="Q")
    # Same question -> same deterministic id -> still one entry.
    assert len(harness.bank.load()) == 1


# --- build_memory_harness_from_config: separate embed endpoint ----------------

def _patch_backends(monkeypatch, seen):
    from prehend.memory import factory as fac

    def fake_embed(*, base_url, model, api_key="EMPTY"):
        seen["embed"] = {"base_url": base_url, "model": model, "api_key": api_key}
        return ConstBackend()

    def fake_reflect(*, base_url, model, api_key="EMPTY", **kw):
        seen["reflect"] = {"base_url": base_url, "model": model, "api_key": api_key, "kw": kw}
        return _reflect_fn({"key_insight": "k"})

    monkeypatch.setattr(fac.OpenAIEmbeddingBackend, "from_config", fake_embed)
    monkeypatch.setattr(fac.OpenAIReflectFn, "from_config", fake_reflect)


def test_from_config_routes_embed_to_separate_endpoint(monkeypatch, tmp_path):
    from prehend.memory.factory import build_memory_harness_from_config
    seen = {}
    _patch_backends(monkeypatch, seen)
    harness = build_memory_harness_from_config(
        FakeSolver(), tmp_path / "mem",
        base_url="http://localhost:8080/v1",
        embed_model="bge-m3", reflect_model="gemma",
        embed_base_url="http://localhost:8081/v1",
        embed_api_key="embed-key",
    )
    assert isinstance(harness, MemoryHarness)
    assert seen["embed"]["base_url"] == "http://localhost:8081/v1"
    assert seen["embed"]["model"] == "bge-m3"
    assert seen["embed"]["api_key"] == "embed-key"
    assert seen["reflect"]["base_url"] == "http://localhost:8080/v1"
    assert seen["reflect"]["model"] == "gemma"


def test_from_config_disables_reflect_thinking_and_caps_tokens_by_default(monkeypatch, tmp_path):
    # Distillation is mechanical JSON extraction, not reasoning. On a thinking
    # reflect_model (e.g. a gemma sft-kb with CoT on) an unbounded call degenerates
    # into a huge thought trace per solve, dominating latency and saturating the
    # GPU. The factory must default thinking OFF and bound output.
    from prehend.memory.factory import build_memory_harness_from_config
    seen = {}
    _patch_backends(monkeypatch, seen)
    build_memory_harness_from_config(
        FakeSolver(), tmp_path / "mem",
        base_url="http://localhost:8080/v1",
        embed_model="bge-m3", reflect_model="gemma",
    )
    kw = seen["reflect"]["kw"]
    assert kw["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert isinstance(kw["max_tokens"], int) and kw["max_tokens"] > 0


def test_from_config_reflect_budget_is_overridable(monkeypatch, tmp_path):
    from prehend.memory.factory import build_memory_harness_from_config
    seen = {}
    _patch_backends(monkeypatch, seen)
    build_memory_harness_from_config(
        FakeSolver(), tmp_path / "mem",
        base_url="http://localhost:8080/v1",
        embed_model="bge-m3", reflect_model="gemma",
        reflect_enable_thinking=True, reflect_max_tokens=2048,
    )
    kw = seen["reflect"]["kw"]
    assert kw["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert kw["max_tokens"] == 2048


def test_from_config_embed_endpoint_defaults_to_base_url(monkeypatch, tmp_path):
    from prehend.memory.factory import build_memory_harness_from_config
    seen = {}
    _patch_backends(monkeypatch, seen)
    build_memory_harness_from_config(
        FakeSolver(), tmp_path / "mem",
        base_url="http://localhost:8080/v1",
        embed_model="bge-m3", reflect_model="gemma",
        api_key="shared-key",
    )
    # No embed_base_url/embed_api_key -> embed reuses the single endpoint + key.
    assert seen["embed"]["base_url"] == "http://localhost:8080/v1"
    assert seen["embed"]["api_key"] == "shared-key"
