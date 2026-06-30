"""Dynamic-KV-pool engines (sglang) bypass the per-slot sub-call division.

ADR-0015 migrates the served inference server from the llama.cpp dual-context fork to
SGLang: ONE engine serves both orchestrator and worker roles via continuous
batching + RadixAttention. SGLang's KV is a single PAGED pool the radix tree
draws from per-token and LRU-evicts under pressure - it does NOT 500 on
contention the way llama.cpp --kv-unified did. So the ADR-0012
`per_call_subcall_budget(pool, slots)` division (which existed only to stop
`slots` concurrent sub-calls from exhausting the ONE shared kv-unified cache)
is unnecessary and needlessly conservative here. With `dynamic_kv_pool=True`
each sub-call may be budgeted against the FULL resolved pool (the per-request
context-length cap), not pool // slots.
"""

from prehend.harness import Defaults, Harness, Runtime
from prehend.utils.token_utils import get_context_limit, per_call_subcall_budget

MODEL = "gemma-4-12b-it-sft-kb-v13-sft"


def _h(**kw):
    return Harness(model=MODEL, base_url="http://localhost:9999/v1",
                   runtime=Runtime(slots=4, ctx=98304), **kw)


def test_dynamic_pool_skips_slot_division():
    # runtime ctx 98304, slots 4 -> full pool, NOT 98304 // 4
    h = _h(dynamic_kv_pool=True)
    assert h.srlm.subcall_context_limit == 98304


def test_dynamic_pool_with_explicit_limit():
    h = _h(dynamic_kv_pool=True, subcall_context_limit=32768)
    assert h.srlm.subcall_context_limit == 32768


def test_dynamic_pool_falls_back_to_model_limit_when_ctx_unknown():
    # ctx None + dynamic -> full get_context_limit(model), undivided
    h = Harness(model=MODEL, base_url="http://localhost:9999/v1",
                runtime=Runtime(slots=4, ctx=None), dynamic_kv_pool=True)
    assert h.srlm.subcall_context_limit == get_context_limit(MODEL)


def test_dynamic_pool_via_defaults_field():
    d = Defaults(dynamic_kv_pool=True)
    h = Harness(model=MODEL, base_url="http://localhost:9999/v1",
                runtime=Runtime(slots=4, ctx=98304), defaults=d)
    assert h.srlm.subcall_context_limit == 98304


def test_param_overrides_defaults_field():
    # explicit Harness(dynamic_kv_pool=False) beats Defaults(dynamic_kv_pool=True)
    d = Defaults(dynamic_kv_pool=True)
    h = Harness(model=MODEL, base_url="http://localhost:9999/v1",
                runtime=Runtime(slots=4, ctx=98304), defaults=d, dynamic_kv_pool=False)
    assert h.srlm.subcall_context_limit == per_call_subcall_budget(98304, 4)


def test_static_pool_default_still_divides():
    # default (llama.cpp path) is unchanged: pool // slots
    h = _h()
    assert h.srlm.subcall_context_limit == per_call_subcall_budget(98304, 4)
