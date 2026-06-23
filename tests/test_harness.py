import dataclasses
from prehend.harness import Defaults, VETTED, Runtime, MemoryConfig, detect_runtime, Harness
from prehend.core.srlm import SRLM
from prehend.memory.harness import MemoryHarness
from prehend.utils.prompts import RLM_SYSTEM_PROMPT


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

    def test_subcall_limit_is_shared_pool_divided_by_slots(self):
        # ctx=98304 is the SHARED kv-unified pool; with slots=4 concurrent
        # sub-calls the per-call guard budget must be 98304//4 so their sum
        # cannot exhaust the shared cache.
        h = _h()  # slots=4, ctx=98304, no explicit subcall_context_limit
        assert h.srlm.subcall_context_limit == 24576

    def test_explicit_subcall_limit_is_also_split_across_slots(self):
        # An operator passing the server n_ctx as the limit means the whole
        # pool; it is still per-call divided by the concurrent-sub-call count.
        h = _h(subcall_context_limit=98304)
        assert h.srlm.subcall_context_limit == 24576

    def test_subcall_base_url_routes_only_the_subcall_backend(self):
        h = Harness(model="m", base_url="http://localhost:8080/v1",
                    subcall_base_url="http://localhost:8081/v1",
                    runtime=Runtime(slots=1, ctx=32768),
                    subcall_runtime=Runtime(slots=4, ctx=65536))
        assert h.srlm.backend_kwargs["base_url"] == "http://localhost:8080/v1"
        assert h.srlm.other_backend_kwargs[0]["base_url"] == "http://localhost:8081/v1"

    def test_subcall_base_url_none_keeps_single_endpoint(self):
        h = _h()  # no subcall_base_url
        assert h.srlm.backend_kwargs["base_url"] == "http://localhost:9999/v1"
        assert h.srlm.other_backend_kwargs[0]["base_url"] == "http://localhost:9999/v1"

    def test_subcall_budget_and_fanout_use_worker_runtime(self):
        # orchestrator: 1 big slot; worker: 4 slots over a dedicated 65536 pool.
        h = Harness(model="m", base_url="http://localhost:8080/v1",
                    subcall_base_url="http://localhost:8081/v1",
                    runtime=Runtime(slots=1, ctx=32768),
                    subcall_runtime=Runtime(slots=4, ctx=65536))
        assert h.srlm.max_concurrent_subcalls == 4          # worker slots, not orchestrator's 1
        assert h.srlm.subcall_context_limit == 65536 // 4   # worker pool / worker slots

    def test_single_endpoint_budget_unchanged(self):
        # regression: subcall_base_url=None keeps the shared-pool division.
        h = _h()  # slots=4, ctx=98304
        assert h.srlm.max_concurrent_subcalls == 4
        assert h.srlm.subcall_context_limit == 24576

    def test_worker_runtime_auto_falls_back_when_unreachable(self):
        # subcall_base_url given but no subcall_runtime: probe fails (port closed)
        # -> fall back to default slots, ctx None -> guard off for sub-calls.
        h = Harness(model="m", base_url="http://localhost:8080/v1",
                    subcall_base_url="http://localhost:9998/v1",
                    runtime=Runtime(slots=1, ctx=32768))
        assert h.subcall_runtime.slots == VETTED.max_concurrent_subcalls
        assert h.srlm.max_concurrent_subcalls == VETTED.max_concurrent_subcalls

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
        # system_addendum appends to (not replaces) the base RLM system prompt
        assert "EXTRA" in h.srlm.system_prompt
        assert RLM_SYSTEM_PROMPT[:40] in h.srlm.system_prompt

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

    def test_memory_observer_flows_to_harness(self, tmp_path):
        # The eval reaches the memory layer only via Harness(memory=MemoryConfig);
        # the observer must thread through to MemoryHarness so the host process
        # can attach prehend.metrics.memory_observer() and emit the
        # localai_prehend_memory_* series.
        sentinel = object()
        h = _h(memory=MemoryConfig(
            bank_dir=str(tmp_path / "bank"),
            embed_model="bge-m3", reflect_model="m",
            embed_url="http://localhost:8081/v1",
            observer=sentinel,
        ))
        assert h.solver.observer is sentinel

    def test_memory_observer_defaults_to_none_field(self, tmp_path):
        # Default config carries no observer; MemoryHarness then installs its own
        # NullObserver (no-op), so the no-metrics path is unaffected.
        h = _h(memory=MemoryConfig(
            bank_dir=str(tmp_path / "bank"),
            embed_model="bge-m3", reflect_model="m",
            embed_url="http://localhost:8081/v1",
        ))
        from prehend.memory.harness import NullObserver
        assert isinstance(h.solver.observer, NullObserver)


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
