# Design: high-level `Harness` API in prehend

- **Date:** 2026-06-21
- **Status:** approved (brainstorming), pending spec review
- **Branch:** `harness-api`
- **ADR:** to be recorded as ADR-0008 (supersede-note on ADR-0005)

## Problem

prehend's public API is effectively just `RLM` and `SRLM`. `SRLM.__init__`
exposes ~20 low-level strategy knobs (dual backends, `environment_kwargs`,
`max_output_chars`, `max_iterations`/`max_depth`, `direct_threshold`, scheduler
coordination, `soft_timeout_pct`, reliability guards, ...), plus an out-of-band
`MAPREDUCE_CONCURRENCY` env var read in `mapreduce.py`. To get a correct,
reliable solve a client must know *the right approach and strategies* and
assemble all of this itself.

The two real clients prove the cost:

- **rlm-trainer `benchmark.py`** hand-builds the ~20-arg `SRLM(...)` and a
  bespoke `_maybe_wrap_memory(solver, params)` to wire ADR-0005 memory.
- **kb-librarian `ask.py`** builds an *even larger* `SRLM(...)` and additionally
  carries general llama-server reliability knobs (`max_retries=0`, `stream=True`,
  `repeat_guard_threshold`, `repair_doubled_calls`, `repair_unfilled_placeholders`,
  `max_subcalls`, `soft_timeout_pct`) that benchmark lacks.

The leakage is not just "too low-level" -- it is **divergent**: each client
independently discovered a *different subset* of the strategy, so the two have
drifted. Benchmark is missing reliability fixes the librarian learned the hard
way. The orchestration discipline itself (process context by reference, one
sub-prompt per slice, never loop `llm_query`) is even encoded as a copy-able
prose block in librarian's system prompt.

A concrete instance bit us during the memory eval: the client had to set
`MAPREDUCE_CONCURRENCY` to match the server's slot count, and nobody did, so
sub-RLM concurrency did not match the 4-slot v13 server. The harness should own
that decision, not the client.

## Goal

Add a high-level `Harness` object to prehend that **owns the orchestration
strategy and runtime decisions**, exposes **optional hooks** for legitimate
domain extension, composes the memory layer as an option, and keeps `SRLM` as
the unchanged low-level escape hatch. Migrate both clients onto it.

## Non-goals

- No named profiles (`"fast"`/`"thorough"`). YAGNI: ship one vetted default set
  of strategy params plus explicit override; add named profiles only when a
  second profile actually exists.
- No removal or behavior change of `SRLM` or `MemoryHarness` internals.
- No change to kb-librarian's domain modules (`corpus`, `retrieval`,
  `citations`). Those stay in kb-librarian.

## The core model: three tiers of params

Every argument the clients pass to `SRLM` falls into one of three buckets. The
design is just "assign each bucket an owner."

| Tier | Examples | Owner |
|------|----------|-------|
| **A. General strategy / reliability** | `max_output_chars` guard, `max_retries=0`, `stream`, `repeat_guard_*`, `repair_*`, `max_subcalls`, `soft_timeout_pct`, scheduler wiring, backend/other-backend assembly, the by-reference sub-prompt discipline | **Harness defaults** |
| **B. Runtime-derived** | `MAPREDUCE_CONCURRENCY` <-> slot count; direct-vs-mapreduce routing <-> ctx size | **Harness, via hybrid detection** |
| **C. Domain extension** | `subcall_verifier`, `answer_verifier`, `custom_tools`, system-prompt addendum, observability bind/scope, citation enforcement, per-client `max_output_chars` (500 vs 2000) | **Optional hooks the client supplies** |

Today Tier A+B live in the clients (copied, drifted). The fix: prehend owns A+B;
exposes C as hooks.

## API

```python
from prehend import Harness
from prehend.harness import Runtime, MemoryConfig, Defaults  # supporting types

h = Harness(
    model="gemma-4-12b-it-sft-kb-v13-sft",
    base_url="http://localhost:8080/v1",
    api_key="not-needed",

    # Tier B: hybrid detect + explicit override + safe fallback.
    runtime="auto",                       # or Runtime(slots=4, ctx=98304)

    # Tier A overrides (rare): a Defaults dataclass; omit to use vetted defaults.
    defaults=None,

    # Optional memory (ADR-0005); when set, Harness wraps the solver internally.
    memory=None,                          # or MemoryConfig(bank_dir=..., embed_url=...,
                                          #     reflect_model=..., k_max=..., min_cosine=...)

    # Tier C hooks (all optional; benchmark passes none):
    system_addendum=None,                 # str appended to the RLM system prompt
    subcall_verifier=None,
    answer_verifier=None,
    max_answer_retries=None,
    custom_tools=None,
    observability=None,                   # hook invoked with the constructed SRLM
    logger=None,
)

answer = h.completion(context, query)     # simple path; benchmark uses exactly this
```

### Tier A: defaults

A frozen `Defaults` dataclass holds the vetted union of the reliability/strategy
knobs both clients learned (`max_retries=0`, `stream=True`, `repair_*`,
`max_subcalls`, `soft_timeout_pct`, `max_output_chars`, `max_iterations`,
`max_depth`, `max_errors`, dual-backend assembly, ...). `Harness(...)` with
`defaults=None` uses the vetted instance. A client may pass a modified copy
(`dataclasses.replace`) for the rare override. No named profiles.

### Tier B: hybrid runtime detection

`runtime="auto"` (default):
1. Probe the endpoint for slots and ctx (e.g. `/props`, `/models`, and/or a
   calibration request). Router mode is known-fragile: `/props` on the proxy
   port returned `n_ctx 0` / `model none` because the model runs on an internal
   subprocess port.
2. If the probe is ambiguous/fails -> **safe fallback**: `slots=1`, conservative
   ctx, and log a one-line notice that detection fell back.
3. `runtime=Runtime(slots=, ctx=)` skips probing entirely (explicit override).

From the resolved `Runtime`, the Harness sets `MAPREDUCE_CONCURRENCY` (via the
mechanism `mapreduce.py` reads) and the routing threshold. **The client never
touches `MAPREDUCE_CONCURRENCY` again.** (Implementation note: prefer passing the
concurrency to the solver explicitly over mutating a process env var; if the env
var is the only seam today, the Harness sets it in a contained, documented way.)

### Memory composition

When `memory=MemoryConfig(...)` is set, the Harness wraps the constructed solver
via the existing `build_memory_harness_from_config(...)` -- the same wiring
`_maybe_wrap_memory` does today, moved inside the Harness. `_maybe_wrap_memory`
is deleted from benchmark. `MemoryHarness` keeps its name and ADR-0005 behavior
but is demoted to an internal building block (no longer hand-wired by clients).
The `MemoryHarness.completion` Solver adapter (commit fee96ae) is what makes this
composition clean.

### Tier C: hooks

All optional. Each maps to the corresponding `SRLM` arg. The Harness holds the
constructed `SRLM`, so the `observability` hook resolves the prior
`observability.bind(srlm)` / `call_scope(srlm)` raw-SRLM caveat *inside* the
Harness, where the raw SRLM is available -- the client passes a hook, not a raw
solver.

### Escape hatch

`SRLM` and `RLM` stay exported and unchanged. Anything the Harness does not cover
is still reachable by constructing `SRLM` directly.

## Client migration

### benchmark.py
Replace the ~20-line `SRLM(...)` block and `_maybe_wrap_memory` with:
```python
h = Harness(model=..., base_url=..., runtime="auto", memory=MemoryConfig(...) or None)
response = h.completion(task["context"], task["query"])
```
Net deletion. Benchmark also *gains* the reliability defaults it was missing.
Its existing args (`memory_k_max`, `memory_min_cosine`, `embed_url`,
`reflect_model`) map to `MemoryConfig` fields.

### kb-librarian ask.py
Replace its `SRLM(...)` with:
```python
h = Harness(
    model=settings.model, base_url=settings.base_url,
    runtime="auto",
    system_addendum=<the by-ref + citation prompt block>,
    subcall_verifier=verifier,
    answer_verifier=citation_verifier,
    max_answer_retries=settings.max_citation_retries,
    custom_tools=custom_tools,
    observability=<hook that binds + opens call_scope>,
    defaults=dataclasses.replace(VETTED, max_output_chars=2000, ...),  # KB tuning
)
result = h.completion(context, question)
```
KB modules (`corpus`, `retrieval`, `citations`) untouched. The handoff caveat
(observability needs the raw SRLM) is satisfied by the hook running inside the
Harness.

## Testing

- **Harness unit tests, fake backend, no live server:**
  - vetted defaults are applied to the SRLM; a `Defaults` override propagates.
  - hybrid runtime: clean probe -> detected slots/ctx; router-ambiguous ->
    fallback `slots=1` + notice; explicit `Runtime` -> no probe.
  - `memory=None` -> byte-identical to bare SRLM path; `memory=MemoryConfig(...)`
    -> solver is wrapped (assert MemoryHarness in the chain).
  - each Tier-C hook reaches the SRLM (system_addendum, verifiers, custom_tools,
    observability hook invoked with the SRLM).
- **Regression:** both clients' suites stay green post-migration -- benchmark's
  benchmark/memory tests; kb-librarian's `test_librarian_ask.py` (fake solver).
- Follow the repo's existing test patterns (fake solver / no network), per the
  memory-layer test suite.

## Risks & mitigations

- **Runtime probe fragility (router mode).** Mitigated by the safe fallback to
  `slots=1` + explicit-override path; never hard-fail on an ambiguous probe.
- **Hidden coupling via the `MAPREDUCE_CONCURRENCY` env var.** If concurrency can
  only be injected via env today, the Harness setting it is a shared-process
  side effect; benchmark already runs each task in its own process (safe), but
  document it and prefer an explicit-arg seam if one exists.
- **kb-librarian behavior drift.** Its many tuned knobs must survive migration;
  the `Defaults`-override + Tier-C hooks must cover every arg `ask.py` sets
  today. Mitigation: enumerate `ask.py`'s current `SRLM(...)` args and map each
  to defaults/hook before deleting the old block; keep `test_librarian_ask.py`
  green.
- **Two "Harness" names.** `Harness` (public) vs `MemoryHarness` (internal).
  Documented in ADR-0008; acceptable since MemoryHarness is no longer a
  client-facing wiring step.

## ADR

Record ADR-0008 "high-level Harness API; clients stop hand-wiring SRLM". Add a
supersede-note to ADR-0005 pointing at it (the memory layer is now composed via
`Harness(memory=...)`, not hand-wrapped).
