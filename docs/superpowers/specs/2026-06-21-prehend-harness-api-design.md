# Design: high-level `Harness` API in prehend

- **Date:** 2026-06-21
- **Status:** approved (brainstorming), pending spec review
- **Branch:** `harness-api`
- **ADR:** to be recorded as ADR-0008 (supersede-note on ADR-0005)

## Problem

prehend's public API is effectively just `RLM` and `SRLM`. `SRLM.__init__` (plus
the `RLM.__init__` it forwards `**kwargs` to) exposes ~20 low-level strategy
knobs (dual backends, `environment_kwargs`/`max_output_chars`,
`max_iterations`/`max_depth`, `direct_threshold` routing, `max_concurrent_subcalls`,
scheduler coordination, `soft_timeout_pct`, reliability guards, ...). To get a
correct, reliable solve a client must know *the right approach and strategies*
and assemble all of this itself.

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

The runtime-dependent knobs make this fragile. Solve-path sub-call concurrency
is `max_concurrent_subcalls` (RLM, default 4) and `scheduler_max_concurrent`;
context routing is `direct_threshold` (SRLM). These *should* track the server's
slot count and ctx, but the clients leave them at hard-coded defaults. During the
memory eval the default `max_concurrent_subcalls=4` happened to match v13's 4
slots -- it worked by luck, not because anyone derived it. A 2-slot or 8-slot
server would over- or under-subscribe silently. The harness should *derive* these
from the server, not leave them to a lucky default in the client.

(Note: `MAPREDUCE_CONCURRENCY` / `rlm-trainer/mapreduce.py` is a *separate*
subsystem -- typed map-reduce tools for `generate.py`'s trajectory-gen pipeline,
not the benchmark/SRLM solve path -- and is out of scope here.)

## Goal

Add a high-level `Harness` object to prehend that **owns the orchestration
strategy and runtime decisions**, exposes **optional hooks** for legitimate
domain extension, composes the memory layer as an option, and keeps `SRLM` as
the unchanged low-level escape hatch. Migrate `benchmark.py` onto it in this
effort; design the Tier-C hooks so kb-librarian *can* adopt it next, but migrate
it as a fast-follow (separate plan), not here.

## Scope (de-risked)

- **In scope (this plan):** the `Harness` API (Tier A defaults, Tier B hybrid
  runtime, memory option, all Tier-C hooks), `benchmark.py` migration, ADR-0008.
- **Fast-follow (separate plan):** kb-librarian `ask.py` migration. The Harness
  ships with every hook librarian needs (`system_addendum`, `subcall_verifier`,
  `answer_verifier`, `custom_tools`, `observability`, `defaults` override), so
  the API is proven design-complete for it, but the riskier client (many tuned
  knobs + the observability raw-SRLM caveat) is migrated separately to keep this
  plan landable and green.

## Non-goals

- No named profiles (`"fast"`/`"thorough"`). YAGNI: ship one vetted default set
  of strategy params plus explicit override; add named profiles only when a
  second profile actually exists.
- No removal or behavior change of `SRLM` or `MemoryHarness` internals.
- No change to kb-librarian's domain modules (`corpus`, `retrieval`,
  `citations`). Those stay in kb-librarian.
- No kb-librarian migration in this plan (fast-follow, see Scope).

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

The seam is **explicit SRLM/RLM constructor args** -- no env var, no process
mutation. The resolved `Runtime(slots, ctx)` maps to:
- `max_concurrent_subcalls = slots` (RLM): the map-reduce fan-out matches the
  server's slot count instead of the hard-coded 4.
- `scheduler_max_concurrent` derived from `slots` (the per-process in-flight cap).
- `direct_threshold`: kept at the vetted default (0 = always decompose). Deriving
  it from `ctx` is a possible future refinement; YAGNI for v1.

`runtime="auto"` (default):
1. Probe the endpoint for slots/ctx (`/props`, `/models`, and/or a calibration
   request). Router mode is known-fragile: `/props` on the proxy port returned
   `n_ctx 0` / `model none` because the model runs on an internal subprocess port.
2. If the probe is ambiguous/fails -> **safe fallback**: keep the vetted-default
   concurrency (today's behavior, `max_concurrent_subcalls=4`) and log a one-line
   notice that detection fell back. (Fallback preserves current behavior rather
   than regressing throughput; a client wanting strict-conservative passes an
   explicit `Runtime`.)
3. `runtime=Runtime(slots=, ctx=)` skips probing entirely (explicit override).

So the Harness *derives* the slot-dependent concurrency the client used to leave
to a lucky default.

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

### kb-librarian ask.py (FAST-FOLLOW, not this plan)
Deferred to a separate plan to keep this one landable. Sketch of the target so
the hooks are designed correctly now -- replace its `SRLM(...)` with:
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
- **Regression (this plan):** benchmark's benchmark/memory tests stay green
  post-migration. (kb-librarian's `test_librarian_ask.py` is the fast-follow's
  gate, not this plan's.)
- The Harness unit tests cover *all* Tier-C hooks even though only benchmark
  migrates now, so the fast-follow librarian migration has no unproven surface.
- Follow the repo's existing test patterns (fake solver / no network), per the
  memory-layer test suite.

## Risks & mitigations

- **Runtime probe fragility (router mode).** Mitigated by the safe fallback to
  `slots=1` + explicit-override path; never hard-fail on an ambiguous probe.
- **Concurrency seam (RESOLVED during planning).** The solve-path concurrency is
  the explicit `max_concurrent_subcalls` / `scheduler_max_concurrent` SRLM/RLM
  args -- no env var, no process mutation. (`MAPREDUCE_CONCURRENCY` belongs to a
  different subsystem, `generate.py` trajectory-gen, not the solve path.) Tier B
  is therefore a clean arg-derivation, not an env hack.
- **kb-librarian behavior drift (deferred to fast-follow).** Its many tuned knobs
  must survive its eventual migration; the `Defaults`-override + Tier-C hooks
  must cover every arg `ask.py` sets today. This plan de-risks by NOT migrating
  librarian yet -- but it must ship every hook librarian needs. Mitigation: when
  designing the hooks, enumerate `ask.py`'s current `SRLM(...)` args and confirm
  each maps to a default or a Tier-C hook; the actual swap + `test_librarian_ask.py`
  gate happen in the fast-follow plan.
- **Two "Harness" names.** `Harness` (public) vs `MemoryHarness` (internal).
  Documented in ADR-0008; acceptable since MemoryHarness is no longer a
  client-facing wiring step.

## ADR

Record ADR-0008 "high-level Harness API; clients stop hand-wiring SRLM". Add a
supersede-note to ADR-0005 pointing at it (the memory layer is now composed via
`Harness(memory=...)`, not hand-wrapped).
