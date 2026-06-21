import dataclasses
from prehend.harness import Defaults, VETTED, Runtime, MemoryConfig, detect_runtime, Harness
from prehend.core.srlm import SRLM
from prehend.memory.harness import MemoryHarness


class TestSupportingTypes:
    def test_vetted_defaults(self):
        assert VETTED.max_concurrent_subcalls == 4
        assert VETTED.max_retries == 0
        assert VETTED.max_output_chars == 500

    def test_defaults_override_is_a_copy(self):
        tuned = dataclasses.replace(VETTED, max_output_chars=2000)
        assert tuned.max_output_chars == 2000
        assert VETTED.max_output_chars == 500  # original unchanged

    def test_runtime_and_memory_config(self):
        rt = Runtime(slots=4, ctx=98304)
        assert (rt.slots, rt.ctx) == (4, 98304)
        mc = MemoryConfig(bank_dir="/tmp/bank", embed_model="bge-m3", reflect_model="m")
        assert mc.embed_url is None and mc.k_max is None


def _h(**kw):
    # base_url unused at construction (backends connect lazily), like test_srlm.py
    return Harness(model="m", base_url="http://localhost:9999/v1",
                   runtime=Runtime(slots=4, ctx=98304), **kw)


class TestHarnessCore:
    def test_builds_srlm_with_vetted_and_runtime(self):
        h = _h()
        assert isinstance(h.srlm, SRLM)
        assert h.runtime == Runtime(slots=4, ctx=98304)
        assert h.srlm.max_concurrent_subcalls == 4          # from slots
        assert h.srlm.max_iterations == VETTED.max_iterations
        assert h.srlm.max_depth == VETTED.max_depth

    def test_auto_runtime_falls_back_when_probe_ambiguous(self):
        h = Harness(model="m", base_url="http://localhost:9999/v1",
                    runtime="auto",
                    # inject ambiguous probe via monkeypatch-free seam:
                    )
        # default probe will fail to connect -> fallback to defaults slot count
        assert h.runtime.slots == VETTED.max_concurrent_subcalls
        assert h.srlm.max_concurrent_subcalls == VETTED.max_concurrent_subcalls

    def test_hooks_reach_srlm(self):
        sentinel_tools = {"mytool": {"tool": lambda: 1, "description": "d"}}
        seen = {}
        h = _h(subcall_verifier="V", answer_verifier="A", max_answer_retries=5,
               custom_tools=sentinel_tools, system_addendum="EXTRA",
               observability=lambda srlm: seen.setdefault("srlm", srlm))
        assert h.srlm.subcall_verifier == "V"
        assert h.srlm.answer_verifier == "A"
        assert h.srlm.max_answer_retries == 5
        assert h.srlm.custom_tools == sentinel_tools
        assert seen["srlm"] is h.srlm            # observability hook ran with raw SRLM

    def test_completion_delegates_to_solver(self):
        h = _h()
        h.solver = type("S", (), {"completion": lambda self, c, q: f"{c}|{q}"})()
        assert h.completion("ctx", "qry") == "ctx|qry"


class TestHarnessMemory:
    def test_no_memory_solver_is_srlm(self):
        h = _h()
        assert h.solver is h.srlm

    def test_memory_wraps_solver(self, tmp_path):
        h = _h(memory=MemoryConfig(
            bank_dir=str(tmp_path / "bank"),
            embed_model="bge-m3", reflect_model="m",
            embed_url="http://localhost:8081/v1",
        ))
        assert isinstance(h.solver, MemoryHarness)
        assert h.solver is not h.srlm


class TestHarnessPassthroughs:
    def test_advanced_knobs_reach_srlm(self):
        h = _h(direct_threshold=30000, n_candidates=4, candidate_temperature=0.7,
               candidate_parallel=2, confidence_elicitation=True,
               scheduler_max_concurrent=4)
        assert h.srlm.direct_threshold == 30000
        assert h.srlm.n_candidates == 4
        assert h.srlm.candidate_temperature == 0.7
        assert h.srlm.candidate_parallel == 2
        assert h.srlm.confidence_elicitation is True

    def test_unset_knobs_use_srlm_defaults(self):
        h = _h()
        assert h.srlm.direct_threshold == 0    # SRLM default (always-rlm)
        assert h.srlm.n_candidates == 1        # SRLM default (single trajectory)


class TestDetectRuntime:
    def test_clean_probe_returns_runtime(self):
        rt = detect_runtime("http://x/v1", probe=lambda b, k: Runtime(slots=4, ctx=98304))
        assert rt == Runtime(slots=4, ctx=98304)

    def test_ambiguous_probe_returns_none(self):
        # router-mode: probe yields slots<=0 -> treat as ambiguous
        rt = detect_runtime("http://x/v1", probe=lambda b, k: Runtime(slots=0, ctx=None))
        assert rt is None

    def test_probe_exception_returns_none_not_raises(self):
        def boom(b, k):
            raise RuntimeError("connection refused")
        assert detect_runtime("http://x/v1", probe=boom) is None
