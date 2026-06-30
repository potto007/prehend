---
status: "accepted"
date: "2026-06-21"
deciders: "potto"
---

# Add a high-level `Harness` API; clients stop hand-wiring SRLM

## Context and Problem Statement

prehend's public API is effectively just `RLM` and `SRLM`. `SRLM.__init__` (plus
the `RLM.__init__` it forwards `**kwargs` to) exposes approximately 20 low-level
strategy knobs: dual backends, `environment_kwargs`/`max_output_chars`,
`max_iterations`/`max_depth`, `direct_threshold` context routing,
`max_concurrent_subcalls`, scheduler coordination, `soft_timeout_pct`, reliability
guards, and more. To obtain a correct, reliable solve a client must know the right
approach and assemble all of this itself.

The two real clients demonstrate the cost:

- **rlm-trainer `benchmark.py`** hand-builds the ~20-arg `SRLM(...)` and a
  bespoke `_maybe_wrap_memory(inference_client, params)` helper to wire the ADR-0005 memory
  layer on top.
- **kb-librarian `ask.py`** builds an even larger `SRLM(...)` and carries
  reliability knobs (`max_retries`, `stream`, `repeat_guard_threshold`,
  `repair_doubled_calls`, `repair_unfilled_placeholders`, `max_subcalls`,
  `soft_timeout_pct`) that benchmark lacks.

The leakage is not just "too low-level" - it is **divergent**: each client
independently discovered a different subset of the strategy, so the two have
drifted. Benchmark is missing reliability fixes the librarian learned the hard way.

The runtime-dependent knobs compound the fragility. Solve-path concurrency is
controlled by explicit SRLM/RLM constructor args - `max_concurrent_subcalls` (RLM,
default 4) and `scheduler_max_concurrent` - and should track the server's slot
count; but clients leave them at hard-coded defaults, which worked by coincidence
during the memory eval (4 slots happened to match) and would silently
over-subscribe or under-subscribe a different server.

Note: `MAPREDUCE_CONCURRENCY` is a separate subsystem (`generate.py`
trajectory-gen pipeline) and is **not** the solve-path concurrency addressed here.
The solve-path seam is purely explicit SRLM/RLM constructor args.

Every argument the clients pass to `SRLM` falls into one of three buckets:

| Tier | Examples | Owner today |
|------|----------|-------------|
| **A. General strategy / reliability** | `max_output_chars`, `max_retries`, `stream`, `repair_*`, `max_subcalls`, `soft_timeout_pct`, backend assembly, scheduler wiring | Client (copied, drifted) |
| **B. Runtime-derived** | `max_concurrent_subcalls` <- server slot count | Client (hard-coded defaults) |
| **C. Domain extension** | `subcall_verifier`, `answer_verifier`, `custom_tools`, system-prompt addendum, observability bind, per-client `max_output_chars` | Client (legitimate) |

## Decision Drivers

- The orchestration strategy should live in prehend, not be copy-pasted into every
  client.
- Concurrency should be derived from the server's actual slot count, not left to a
  lucky default.
- The memory layer (ADR-0005) should compose cleanly without each client
  re-implementing `_maybe_wrap_memory`.
- `SRLM`/`RLM` must remain available, unchanged, as the low-level escape hatch.
- kb-librarian migration is a known fast-follow; the API must be designed-complete
  for it now even though the migration itself happens separately.

## Considered Options

- **Add a high-level `Harness` that owns Tier A + B and exposes Tier C as optional
  hooks; keep `SRLM`/`RLM` unchanged as the escape hatch.**
- Add a thin factory function (no object).
- Add named profiles (`"fast"` / `"thorough"`).
- Leave the hand-assembly in clients, add shared helper modules.

## Decision Outcome

Chosen option: **high-level `Harness` class in `prehend/harness.py`**, exported
from `prehend.__init__`. The Harness owns Tier A and B, exposes Tier C as optional
hooks, composes the memory layer internally, and leaves `SRLM`/`RLM` fully
accessible for anything it does not cover.

### Tier A: vetted defaults

A frozen `Defaults` dataclass (`VETTED = Defaults()`) holds the union of
reliability/strategy knobs both clients learned, including `max_retries=0`,
`stream=False`, `subcall_enable_thinking=False`, `max_output_chars=500`,
`max_iterations`, `max_depth`, `max_errors`, `soft_timeout_pct`, and
`max_concurrent_subcalls` (the fallback). The Harness applies `VETTED` unless the
caller passes a modified copy (via `dataclasses.replace`). No named profiles -
YAGNI: ship one vetted set and add a second only when it genuinely exists.

### Tier B: hybrid runtime detection

`detect_runtime(base_url)` probes `/props` on the llama-server root (stripping
any `/v1` suffix). Router mode is known-fragile - `/props` on the proxy port
returns `n_ctx=0` / `model=none` because the model runs on an internal subprocess
port - so the probe is always best-effort:

1. Probe -> `Runtime(slots, ctx)`: map `max_concurrent_subcalls = slots`.
2. Probe ambiguous / fails -> **safe fallback**: use `d.max_concurrent_subcalls`
   (the `VETTED` default) and log a one-line notice. Preserves today's behavior
   rather than regressing.
3. Caller passes an explicit `Runtime(slots=N, ctx=M)` -> skip probing entirely.

The seam is purely explicit SRLM/RLM args. No env-var mutation, no process-global
side effects. `direct_threshold` is kept at the `SRLM` default (`0 = always
decompose`) for v1; deriving it from `ctx` is a possible future refinement (YAGNI).

### Memory composition

When `memory=MemoryConfig(...)` is set, the Harness calls
`build_memory_harness_from_config(self.srlm, ...)` internally after constructing
the SRLM. `MemoryHarness` keeps its ADR-0005 behavior unchanged but is demoted to
an **internal building block** - clients no longer hand-wire it or call
`_maybe_wrap_memory`. `_maybe_wrap_memory` is deleted from `benchmark.py`.
The `MemoryHarness.completion` InferenceClient adapter (commit `fee96ae`) is what makes
this composition clean.

### Tier C: optional hooks

All optional: `system_addendum` (str appended to the RLM system prompt),
`subcall_verifier`, `answer_verifier`, `max_answer_retries`, `custom_tools`,
`observability` (callable invoked with the constructed SRLM). The `observability`
hook receives the raw SRLM inside the Harness, resolving the `observability.bind`
/ `call_scope` caveat that previously required the client to hold the raw inference client.

Advanced passthroughs forwarded verbatim when not `None`: `direct_threshold`,
`n_candidates`, `candidate_temperature`, `candidate_parallel`,
`confidence_elicitation`, `scheduler_max_concurrent`,
`scheduler_coordination_dir`.

### Client migration

`benchmark.py` migrates in this effort: the ~20-arg `SRLM(...)` block and
`_maybe_wrap_memory` are replaced by:

```python
h = Harness(model=..., base_url=..., runtime="auto",
            memory=MemoryConfig(...) if memory_on else None)
response = h.completion(task["context"], task["query"])
```

`benchmark.py` also gains the reliability defaults it was previously missing.
kb-librarian `ask.py` migration is a documented fast-follow (separate plan); the
Tier-C hooks are designed to cover every arg `ask.py` sets today.

### Consequences

- Good, because clients no longer hand-assemble a ~20-arg `SRLM` and the two
  assemblies can no longer drift from each other.
- Good, because `max_concurrent_subcalls` is now derived from the server's actual
  slot count rather than left to a lucky hard-coded default.
- Good, because memory composition is a single `memory=MemoryConfig(...)` kwarg;
  the `_maybe_wrap_memory` pattern is retired.
- Good, because `MemoryHarness` is demoted to an internal building block; clients
  no longer need to know it exists.
- Good, because `SRLM` and `RLM` remain exported and unchanged as the low-level
  escape hatch.
- Neutral, because two names now both contain "Harness" (`Harness` public,
  `MemoryHarness` internal); this is acceptable since `MemoryHarness` is no longer
  a client-facing wiring step.
- Bad, because kb-librarian's many tuned knobs must survive its eventual migration;
  this is mitigated by designing the Tier-C hooks against `ask.py`'s current arg
  list now, before the migration happens.

## More Information

- Implementation: `prehend/harness.py` - `Defaults`, `VETTED`, `Runtime`,
  `MemoryConfig`, `detect_runtime`, `Harness`.
- Design spec: `docs/superpowers/specs/2026-06-21-prehend-harness-api-design.md`.
- Supersedes the client-wiring aspect of ADR-0005 (memory layer now composed via
  `Harness(memory=...)`; see supersede-note on ADR-0005).
- Builds on ADR-0001 (RLM/SRLM fork rationale) and ADR-0005 (memory layer).
