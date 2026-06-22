"""Tests for the prehend MemoryHarness (retrieve -> inject -> solve -> collect)."""
from __future__ import annotations

from types import SimpleNamespace

from prehend.memory.bank import Bank
from prehend.memory.harness import MemoryHarness


class FakeSolver:
    """Records the (prompt, root_prompt) it was called with."""

    def __init__(self, answer="42"):
        self.answer = answer
        self.calls: list[tuple] = []

    def completion(self, prompt, root_prompt=None):
        self.calls.append((prompt, root_prompt))
        return SimpleNamespace(final_answer=self.answer)


class FakeBackend:
    def __init__(self, table=None, raises=False):
        self.table = table or {}
        self.raises = raises

    def embed(self, text):
        if self.raises:
            raise RuntimeError("embedding service down")
        return self.table.get(text, [0.0, 0.0])


def _entry(eid, embedding):
    return {
        "id": eid,
        "polarity": "positive",
        "key_insight": f"insight {eid}",
        "embedding": embedding,
        "stats": {"use_count": 0, "hit_count": 0},
    }


def test_no_memory_path_keeps_root_prompt_byte_identical(tmp_path):
    bank = Bank(tmp_path / "mem")  # empty
    solver = FakeSolver()
    harness = MemoryHarness(solver, bank, FakeBackend())
    harness.answer(context="ctx", question="What is 6*7?")
    prompt, root_prompt = solver.calls[0]
    assert prompt == "ctx"
    assert root_prompt == "What is 6*7?"  # no memory tokens leaked in


def test_with_memory_injects_block_into_root_prompt(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a", [1.0, 0.0]))
    solver = FakeSolver()
    harness = MemoryHarness(solver, bank, FakeBackend({"Q": [1.0, 0.0]}), min_cosine=0.5)
    harness.answer(context="ctx", question="Q")
    _, root_prompt = solver.calls[0]
    assert "<Memory_Block>" in root_prompt
    assert "insight a" in root_prompt
    assert root_prompt.rstrip().endswith("Q")


def test_returns_solver_result(tmp_path):
    bank = Bank(tmp_path / "mem")
    solver = FakeSolver(answer="hello")
    harness = MemoryHarness(solver, bank, FakeBackend())
    result = harness.answer(context="ctx", question="q")
    assert result.final_answer == "hello"


def test_retrieval_bumps_use_count(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a", [1.0, 0.0]))
    harness = MemoryHarness(FakeSolver(), bank, FakeBackend({"Q": [1.0, 0.0]}), min_cosine=0.5)
    harness.answer(context="ctx", question="Q")
    assert bank.load()[0]["stats"]["use_count"] == 1


def test_embedding_failure_degrades_to_no_memory(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a", [1.0, 0.0]))
    solver = FakeSolver()
    harness = MemoryHarness(solver, bank, FakeBackend(raises=True), min_cosine=0.5)
    # Must not raise; solve still happens with a clean root_prompt.
    harness.answer(context="ctx", question="Q")
    _, root_prompt = solver.calls[0]
    assert root_prompt == "Q"


def test_collect_appends_distilled_entry(tmp_path):
    bank = Bank(tmp_path / "mem")  # empty

    def distiller(question, context, result):
        return _entry("learned", [0.5, 0.5])

    harness = MemoryHarness(FakeSolver(), bank, FakeBackend(), distiller=distiller)
    harness.answer(context="ctx", question="q")
    assert [e["id"] for e in bank.load()] == ["learned"]


def test_collect_none_writes_nothing(tmp_path):
    bank = Bank(tmp_path / "mem")
    harness = MemoryHarness(FakeSolver(), bank, FakeBackend(),
                            distiller=lambda q, c, r: None)
    harness.answer(context="ctx", question="q")
    assert bank.load() == []


def _learn_distiller(question, context, result):
    return _entry("learned", [0.5, 0.5])


def test_defer_collect_does_not_distill_in_answer(tmp_path):
    # With defer_collect, answer() solves but does NOT write an experience -
    # the caller decides later (once it knows correctness) via collect_pending.
    bank = Bank(tmp_path / "mem")
    harness = MemoryHarness(FakeSolver(), bank, FakeBackend(),
                            distiller=_learn_distiller, defer_collect=True)
    harness.answer(context="ctx", question="q")
    assert bank.load() == []  # nothing distilled yet


def test_collect_pending_correct_distills(tmp_path):
    bank = Bank(tmp_path / "mem")
    harness = MemoryHarness(FakeSolver(), bank, FakeBackend(),
                            distiller=_learn_distiller, defer_collect=True)
    harness.answer(context="ctx", question="q")
    harness.collect_pending(correct=True)
    assert [e["id"] for e in bank.load()] == ["learned"]


def test_collect_pending_wrong_skips_distillation(tmp_path):
    # The whole point of #1: do NOT learn from a wrong solve.
    bank = Bank(tmp_path / "mem")
    harness = MemoryHarness(FakeSolver(), bank, FakeBackend(),
                            distiller=_learn_distiller, defer_collect=True)
    harness.answer(context="ctx", question="q")
    harness.collect_pending(correct=False)
    assert bank.load() == []


def test_collect_pending_unknown_correctness_distills(tmp_path):
    # correct=None (no expected answer / unscored) -> conservatively keep it.
    bank = Bank(tmp_path / "mem")
    harness = MemoryHarness(FakeSolver(), bank, FakeBackend(),
                            distiller=_learn_distiller, defer_collect=True)
    harness.answer(context="ctx", question="q")
    harness.collect_pending(correct=None)
    assert [e["id"] for e in bank.load()] == ["learned"]


def test_collect_pending_is_noop_without_pending(tmp_path):
    bank = Bank(tmp_path / "mem")
    harness = MemoryHarness(FakeSolver(), bank, FakeBackend(),
                            distiller=_learn_distiller, defer_collect=True)
    harness.collect_pending(correct=True)  # nothing solved yet
    assert bank.load() == []


def test_collect_skips_entry_whose_id_already_exists(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("dup", [0.1, 0.2]))  # id 'dup' already present

    harness = MemoryHarness(FakeSolver(), bank, FakeBackend(),
                            distiller=lambda q, c, r: _entry("dup", [0.3, 0.4]))
    harness.answer(context="ctx", question="q")
    # Still exactly one 'dup' entry; collect must not append a duplicate id.
    assert [e["id"] for e in bank.load()] == ["dup"]


class FakeTagger:
    def __init__(self, tags):
        self.tags = tags

    def tag(self, query):
        return dict(self.tags)


def test_tagger_gates_retrieval_by_conflicting_tag(tmp_path):
    bank = Bank(tmp_path / "mem")
    e = _entry("a", [1.0, 0.0])
    e["tags"] = {"kind": "numeric"}
    bank.append(e)
    solver = FakeSolver()
    harness = MemoryHarness(
        solver, bank, FakeBackend({"Q": [1.0, 0.0]}), min_cosine=0.5,
        tagger=FakeTagger({"kind": "entity"}),
    )
    harness.answer(context="ctx", question="Q")
    _, root_prompt = solver.calls[0]
    assert root_prompt == "Q"  # conflicting tag gated the entry out -> no memory


def test_collect_tags_entry_using_tagger(tmp_path):
    bank = Bank(tmp_path / "mem")
    harness = MemoryHarness(
        FakeSolver(), bank, FakeBackend(),
        tagger=FakeTagger({"kind": "numeric"}),
        distiller=lambda q, c, r: _entry("learned", [0.1, 0.2]),
    )
    harness.answer(context="ctx", question="q")
    assert bank.load()[0]["tags"] == {"kind": "numeric"}


def test_collect_failure_never_breaks_answer(tmp_path):
    bank = Bank(tmp_path / "mem")

    def boom(question, context, result):
        raise RuntimeError("distiller blew up")

    solver = FakeSolver(answer="ok")
    harness = MemoryHarness(solver, bank, FakeBackend(), distiller=boom)
    result = harness.answer(context="ctx", question="q")
    assert result.final_answer == "ok"


# --- completion() as a transparent Solver adapter (step 1) -------------------

def test_completion_is_drop_in_for_answer(tmp_path):
    # A memory-wrapped solver invoked via .completion(context, query) must drive
    # the inner solver identically to .answer(context, query).
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a", [1.0, 0.0]))
    solver = FakeSolver()
    harness = MemoryHarness(solver, bank, FakeBackend({"Q": [1.0, 0.0]}), min_cosine=0.5)
    harness.completion("ctx", "Q")
    prompt, root_prompt = solver.calls[0]
    assert prompt == "ctx"
    assert "<Memory_Block>" in root_prompt
    assert "insight a" in root_prompt
    assert root_prompt.rstrip().endswith("Q")


def test_completion_no_memory_path_keeps_root_prompt_byte_identical(tmp_path):
    # No-memory invariant holds through the completion() seam too.
    bank = Bank(tmp_path / "mem")  # empty
    solver = FakeSolver()
    harness = MemoryHarness(solver, bank, FakeBackend())
    harness.completion("ctx", "What is 6*7?")
    prompt, root_prompt = solver.calls[0]
    assert prompt == "ctx"
    assert root_prompt == "What is 6*7?"


def test_completion_returns_solver_result(tmp_path):
    bank = Bank(tmp_path / "mem")
    solver = FakeSolver(answer="hello")
    harness = MemoryHarness(solver, bank, FakeBackend())
    result = harness.completion("ctx", "q")
    assert result.final_answer == "hello"


def test_completion_defaults_root_prompt_to_prompt(tmp_path):
    # Single-arg completion(prompt): prompt doubles as the question.
    bank = Bank(tmp_path / "mem")  # empty
    solver = FakeSolver()
    harness = MemoryHarness(solver, bank, FakeBackend())
    harness.completion("only the prompt")
    prompt, root_prompt = solver.calls[0]
    assert prompt == "only the prompt"
    assert root_prompt == "only the prompt"


# --- observer telemetry seam ------------------------------------------------

class RecordingObserver:
    """Captures the kwargs of every on_retrieve/on_collect event."""

    def __init__(self):
        self.retrieves: list[dict] = []
        self.collects: list[dict] = []

    def on_retrieve(self, **kw):
        self.retrieves.append(kw)

    def on_collect(self, **kw):
        self.collects.append(kw)


def test_observer_records_hit(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a", [1.0, 0.0]))
    obs = RecordingObserver()
    harness = MemoryHarness(
        FakeSolver(), bank, FakeBackend({"Q": [1.0, 0.0]}), min_cosine=0.5,
        observer=obs,
    )
    harness.answer(context="ctx", question="Q")
    (ev,) = obs.retrieves
    assert ev["entries"] == 1 and ev["error"] is False
    assert ev["top_score"] is not None and ev["block_chars"] > 0
    assert ev["seconds"] >= 0.0


def test_observer_records_miss(tmp_path):
    bank = Bank(tmp_path / "mem")  # empty
    obs = RecordingObserver()
    harness = MemoryHarness(FakeSolver(), bank, FakeBackend(), observer=obs)
    harness.answer(context="ctx", question="q")
    (ev,) = obs.retrieves
    assert ev["entries"] == 0 and ev["error"] is False
    assert ev["top_score"] is None and ev["block_chars"] == 0


def test_observer_records_retrieval_error(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a", [1.0, 0.0]))
    obs = RecordingObserver()
    harness = MemoryHarness(
        FakeSolver(), bank, FakeBackend(raises=True), min_cosine=0.5, observer=obs,
    )
    harness.answer(context="ctx", question="Q")  # must not raise
    (ev,) = obs.retrieves
    assert ev["error"] is True and ev["entries"] == 0


def test_observer_records_written_collect(tmp_path):
    bank = Bank(tmp_path / "mem")
    obs = RecordingObserver()
    harness = MemoryHarness(
        FakeSolver(), bank, FakeBackend(),
        distiller=lambda q, c, r: _entry("learned", [0.5, 0.5]), observer=obs,
    )
    harness.answer(context="ctx", question="q")
    (ev,) = obs.collects
    assert ev["outcome"] == "written" and ev["bank_size"] == 1


def test_observer_records_empty_collect(tmp_path):
    bank = Bank(tmp_path / "mem")
    obs = RecordingObserver()
    harness = MemoryHarness(
        FakeSolver(), bank, FakeBackend(),
        distiller=lambda q, c, r: None, observer=obs,
    )
    harness.answer(context="ctx", question="q")
    (ev,) = obs.collects
    assert ev["outcome"] == "empty" and ev["bank_size"] is None


def test_observer_records_duplicate_collect(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("dup", [0.1, 0.2]))
    obs = RecordingObserver()
    harness = MemoryHarness(
        FakeSolver(), bank, FakeBackend(),
        distiller=lambda q, c, r: _entry("dup", [0.3, 0.4]), observer=obs,
    )
    harness.answer(context="ctx", question="q")
    (ev,) = obs.collects
    assert ev["outcome"] == "duplicate"


def test_observer_records_deferred_then_dropped(tmp_path):
    bank = Bank(tmp_path / "mem")
    obs = RecordingObserver()
    harness = MemoryHarness(
        FakeSolver(), bank, FakeBackend(),
        distiller=_learn_distiller, defer_collect=True, observer=obs,
    )
    harness.answer(context="ctx", question="q")
    harness.collect_pending(correct=False)
    assert [e["outcome"] for e in obs.collects] == ["deferred", "dropped"]


def test_observer_records_deferred_then_written(tmp_path):
    bank = Bank(tmp_path / "mem")
    obs = RecordingObserver()
    harness = MemoryHarness(
        FakeSolver(), bank, FakeBackend(),
        distiller=_learn_distiller, defer_collect=True, observer=obs,
    )
    harness.answer(context="ctx", question="q")
    harness.collect_pending(correct=True)
    assert [e["outcome"] for e in obs.collects] == ["deferred", "written"]


def test_observer_exception_never_breaks_answer(tmp_path):
    class Boom:
        def on_retrieve(self, **kw):
            raise RuntimeError("observer down")

        def on_collect(self, **kw):
            raise RuntimeError("observer down")

    bank = Bank(tmp_path / "mem")
    solver = FakeSolver(answer="ok")
    harness = MemoryHarness(
        solver, bank, FakeBackend(),
        distiller=lambda q, c, r: _entry("x", [0.1, 0.2]), observer=Boom(),
    )
    result = harness.answer(context="ctx", question="q")
    assert result.final_answer == "ok"  # telemetry failure swallowed
