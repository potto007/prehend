import dataclasses
from prehend.harness import Defaults, VETTED, Runtime, MemoryConfig, detect_runtime


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
