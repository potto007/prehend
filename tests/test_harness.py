import dataclasses
from prehend.harness import Defaults, VETTED, Runtime, MemoryConfig


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
