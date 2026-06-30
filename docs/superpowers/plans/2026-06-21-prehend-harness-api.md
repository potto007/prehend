# Prehend Harness API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a high-level `Harness` object to prehend that owns orchestration strategy + runtime detection + memory composition, exposing optional hooks, and migrate `benchmark.py` onto it.

**Architecture:** `Harness` is a thin object that assembles an `SRLM` from a vetted `Defaults` dataclass (Tier A), a resolved `Runtime` (Tier B: hybrid detect/override/fallback that sets `max_concurrent_subcalls` from slot count), and optional Tier-C hooks. When `memory=MemoryConfig(...)` is given, it wraps the SRLM via the existing `build_memory_harness_from_config`. `SRLM`/`RLM` stay unchanged as the escape hatch.

**Tech Stack:** Python 3, `uv` (use `~/.local/bin/uv`), pytest. Spec: `docs/superpowers/specs/2026-06-21-prehend-harness-api-design.md`.

## Global Constraints

- NEVER use the em dash; use a regular dash. (project + user rule)
- This venv uses `uv`; there is NO `pip` in `.venv/bin`. Run tests with `~/.local/bin/uv run python -m pytest`.
- TDD throughout: failing test first, minimal code, green, commit. (handoff user directive)
- Do not modify `SRLM`/`RLM`/`MemoryHarness` internals; only add `prehend/harness.py` and exports.
- New public types live in `prehend/harness.py` and are re-exported from `prehend/__init__.py`.
- Tests use the fake-backend pattern (`tests/mock_lm.py` `MockLM`); no live server, no network.
- prehend is consumed by rlm-trainer via an editable install; Task 6 runs in `~/src/rlm-trainer`.
- Conventional-commit messages; never mention tooling or co-authorship in commits.

## File Structure

- Create `prehend/harness.py` -- `Defaults`/`VETTED`, `Runtime`, `MemoryConfig`, `detect_runtime()`, `Harness`. One focused module (YAGNI: no package split).
- Modify `prehend/__init__.py` -- export `Harness`, `Runtime`, `MemoryConfig`, `Defaults`; extend `__all__`.
- Create `tests/test_harness.py` -- all Harness unit tests (fake backend).
- Modify `tests/test_imports.py` -- assert the new public names import.
- Modify `prehend/harness.py` again (Task 6) -- optional advanced passthrough kwargs.
- Modify `~/src/rlm-trainer/benchmark.py` (Task 7) -- replace SRLM block + `_maybe_wrap_memory` with `Harness`, forwarding the advanced knobs.
- Modify `~/src/rlm-trainer/tests/test_benchmark_direct_routing.py` (Task 7) -- the two construction tests patch `Harness` instead of `SRLM`; flag/help tests unchanged.
- Create `docs/decisions/0008-high-level-harness-api.md`; modify `docs/decisions/0005-mnemex-experience-memory-layer.md` (supersede-note) (Task 8).

---

### Task 1: Supporting types (`Defaults`, `Runtime`, `MemoryConfig`)

**Files:**
- Create: `prehend/harness.py`
- Test: `tests/test_harness.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Defaults` with fields: `max_output_chars:int=500`, `max_iterations:int=10`, `max_depth:int=2`, `max_errors:int=3`, `max_retries:int=0`, `stream:bool=False`, `subcall_enable_thinking:bool=False`, `max_concurrent_subcalls:int=4`, `soft_timeout_pct:float|None=None`.
  - `VETTED = Defaults()` module constant.
  - `@dataclass(frozen=True) Runtime` with `slots:int`, `ctx:int|None=None`.
  - `@dataclass(frozen=True) MemoryConfig` with `bank_dir:str`, `embed_model:str`, `reflect_model:str`, `embed_url:str|None=None`, `embed_api_key:str|None=None`, `k_max:int|None=None`, `min_cosine:float|None=None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/bin/uv run python -m pytest tests/test_harness.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'prehend.harness'`

- [ ] **Step 3: Write minimal implementation**

```python
# prehend/harness.py
"""High-level Harness API: owns orchestration strategy, runtime detection, and
memory composition so clients do not hand-assemble SRLM. See
docs/superpowers/specs/2026-06-21-prehend-harness-api-design.md and ADR-0008."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Defaults:
    """Vetted Tier-A strategy/reliability defaults the Harness applies to SRLM."""
    max_output_chars: int = 500
    max_iterations: int = 10
    max_depth: int = 2
    max_errors: int = 3
    max_retries: int = 0
    stream: bool = False
    subcall_enable_thinking: bool = False
    max_concurrent_subcalls: int = 4
    soft_timeout_pct: float | None = None


VETTED = Defaults()


@dataclass(frozen=True)
class Runtime:
    """Resolved server facts (Tier B). slots drives map-reduce fan-out."""
    slots: int
    ctx: int | None = None


@dataclass(frozen=True)
class MemoryConfig:
    """ADR-0005 memory wiring, mapped to build_memory_harness_from_config."""
    bank_dir: str
    embed_model: str
    reflect_model: str
    embed_url: str | None = None
    embed_api_key: str | None = None
    k_max: int | None = None
    min_cosine: float | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/bin/uv run python -m pytest tests/test_harness.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add prehend/harness.py tests/test_harness.py
git commit -m "feat(harness): supporting types Defaults/Runtime/MemoryConfig"
```

---

### Task 2: `detect_runtime()` -- hybrid probe with safe fallback

**Files:**
- Modify: `prehend/harness.py`
- Test: `tests/test_harness.py`

**Interfaces:**
- Consumes: `Runtime` (Task 1).
- Produces: `detect_runtime(base_url: str, *, api_key: str = "not-needed", probe: Callable[[str, str], Runtime | None] | None = None) -> Runtime | None`. Returns a `Runtime` on a clean probe, or `None` when ambiguous/failed (caller applies fallback). `probe` is an injectable seam so tests need no network; default probe hits `/props` + `/models`.

**Notes:** Router mode is fragile -- `/props` on the proxy port returns `n_ctx 0` / `model none`. `detect_runtime` must treat `slots<=0` or a raised exception as ambiguous and return `None`, never raise.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_harness.py
from prehend.harness import detect_runtime, Runtime


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/bin/uv run python -m pytest tests/test_harness.py::TestDetectRuntime -q`
Expected: FAIL with `ImportError: cannot import name 'detect_runtime'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to prehend/harness.py
from collections.abc import Callable
import json
import urllib.request


def _default_probe(base_url: str, api_key: str) -> Runtime | None:
    """Best-effort llama-server probe. Returns None if facts are unavailable."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        with urllib.request.urlopen(f"{root}/props", timeout=5) as r:
            props = json.loads(r.read())
        gen = props.get("default_generation_settings", {}) or {}
        ctx = gen.get("n_ctx") or None
        slots = props.get("total_slots") or gen.get("n_parallel") or 0
        if not slots or slots <= 0:
            return None
        return Runtime(slots=int(slots), ctx=int(ctx) if ctx else None)
    except Exception:
        return None


def detect_runtime(
    base_url: str,
    *,
    api_key: str = "not-needed",
    probe: Callable[[str, str], Runtime | None] | None = None,
) -> Runtime | None:
    """Hybrid Tier-B detection. None means 'ambiguous, caller should fall back'."""
    p = probe or _default_probe
    try:
        rt = p(base_url, api_key)
    except Exception:
        return None
    if rt is None or rt.slots <= 0:
        return None
    return rt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/bin/uv run python -m pytest tests/test_harness.py::TestDetectRuntime -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add prehend/harness.py tests/test_harness.py
git commit -m "feat(harness): hybrid detect_runtime with safe-fallback (None on ambiguous)"
```

---

### Task 3: `Harness` core -- build SRLM from defaults + runtime + hooks, `.completion()`

**Files:**
- Modify: `prehend/harness.py`
- Test: `tests/test_harness.py`

**Interfaces:**
- Consumes: `Defaults`/`VETTED`, `Runtime`, `detect_runtime` (Tasks 1-2); `prehend.core.srlm.SRLM`.
- Produces:
  - `Harness(model: str, base_url: str, *, api_key: str = "not-needed", timeout: float | None = None, runtime: Runtime | str = "auto", defaults: Defaults | None = None, system_addendum: str | None = None, subcall_verifier=None, answer_verifier=None, max_answer_retries: int | None = None, custom_tools: dict | None = None, observability: Callable[[object], None] | None = None, logger=None, memory: "MemoryConfig | None" = None)`. (Task 4 adds `memory` behavior; this task wires everything except memory.)
  - `Harness.completion(context: str, query: str) -> str` -- delegates to the inference_client.
  - Attributes for tests: `harness.srlm` (raw SRLM), `harness.runtime` (resolved `Runtime`), `harness.inference_client` (== srlm until Task 4 adds memory wrapping).

**Resolution rules:**
- `runtime="auto"` -> `detect_runtime(base_url)`; if `None`, fall back to `Runtime(slots=defaults.max_concurrent_subcalls)` and log one line via `prehend.logger`.
- `runtime=Runtime(...)` -> use as-is (no probe).
- `max_concurrent_subcalls = resolved_runtime.slots`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_harness.py
from prehend.harness import Harness, Runtime, VETTED
from prehend.core.srlm import SRLM


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

    def test_completion_delegates_to_inference_client(self):
        h = _h()
        h.inference_client = type("S", (), {"completion": lambda self, c, q: f"{c}|{q}"})()
        assert h.completion("ctx", "qry") == "ctx|qry"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/bin/uv run python -m pytest tests/test_harness.py::TestHarnessCore -q`
Expected: FAIL with `ImportError: cannot import name 'Harness'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to prehend/harness.py
import logging

from prehend.core.srlm import SRLM

_log = logging.getLogger("prehend.harness")  # prehend/logger has no generic factory


class Harness:
    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str = "not-needed",
        timeout: float | None = None,
        runtime: "Runtime | str" = "auto",
        defaults: Defaults | None = None,
        system_addendum: str | None = None,
        subcall_verifier=None,
        answer_verifier=None,
        max_answer_retries: int | None = None,
        custom_tools: dict | None = None,
        observability: Callable[[object], None] | None = None,
        logger=None,
        memory=None,            # behavior added in Task 4
    ):
        d = defaults or VETTED
        self.runtime = self._resolve_runtime(runtime, base_url, api_key, d)

        backend_kwargs = {
            "model_name": model, "base_url": base_url, "api_key": api_key,
            "max_retries": d.max_retries, "stream": d.stream,
        }
        subcall_kwargs = dict(backend_kwargs)
        subcall_kwargs["default_extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": d.subcall_enable_thinking}
        }
        srlm_kwargs = dict(
            backend="openai",
            backend_kwargs=backend_kwargs,
            other_backends=["openai"],
            other_backend_kwargs=[subcall_kwargs],
            environment="local",
            environment_kwargs={"max_output_chars": d.max_output_chars},
            max_iterations=d.max_iterations,
            max_depth=d.max_depth,
            max_errors=d.max_errors,
            max_timeout=timeout,
            max_concurrent_subcalls=self.runtime.slots,
            soft_timeout_pct=d.soft_timeout_pct,
            logger=logger,
            verbose=False,
        )
        if system_addendum is not None:
            srlm_kwargs["custom_system_prompt"] = system_addendum
        if subcall_verifier is not None:
            srlm_kwargs["subcall_verifier"] = subcall_verifier
        if answer_verifier is not None:
            srlm_kwargs["answer_verifier"] = answer_verifier
        if max_answer_retries is not None:
            srlm_kwargs["max_answer_retries"] = max_answer_retries
        if custom_tools is not None:
            srlm_kwargs["custom_tools"] = custom_tools

        self.srlm = SRLM(**srlm_kwargs)
        if observability is not None:
            observability(self.srlm)
        self.inference_client = self.srlm     # Task 4 may wrap this

    def _resolve_runtime(self, runtime, base_url, api_key, d: Defaults) -> Runtime:
        if isinstance(runtime, Runtime):
            return runtime
        detected = detect_runtime(base_url, api_key=api_key)
        if detected is not None:
            return detected
        _log.info("harness: runtime probe ambiguous; falling back to slots=%d",
                  d.max_concurrent_subcalls)
        return Runtime(slots=d.max_concurrent_subcalls)

    def completion(self, context: str, query: str) -> str:
        return self.inference_client.completion(context, query)
```

NOTE for implementer: `SRLM` accepts `custom_system_prompt` (it forwards
`**kwargs` to `RLM`,
which defines `custom_system_prompt`). If `enable_thinking` should follow
benchmark's `RLM_SUBCALL_THINKING` env, that is benchmark's concern (Task 6), not
a Harness default.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/bin/uv run python -m pytest tests/test_harness.py::TestHarnessCore -q`
Expected: PASS (4 passed). If `test_auto_runtime_falls_back...` is slow due to the
5s probe timeout against the dead port, that is acceptable; the fallback still
returns. (Optional: pass `runtime` an injected ambiguous probe via a thin wrapper
if you prefer a fast test -- but the seam is `detect_runtime`'s `probe` param,
exercised in Task 2.)

- [ ] **Step 5: Commit**

```bash
git add prehend/harness.py tests/test_harness.py
git commit -m "feat(harness): Harness core - build SRLM from defaults+runtime+hooks"
```

---

### Task 4: Memory composition (`memory=MemoryConfig(...)`)

**Files:**
- Modify: `prehend/harness.py`
- Test: `tests/test_harness.py`

**Interfaces:**
- Consumes: `MemoryConfig` (Task 1); `Harness` (Task 3); `prehend.memory.factory.build_memory_harness_from_config`; `prehend.memory.harness.MemoryHarness`.
- Produces: when `memory` is set, `harness.inference_client` is a `MemoryHarness` wrapping the SRLM; when `None`, `harness.inference_client is harness.srlm` (unchanged).

**Mapping:** `MemoryConfig` -> `build_memory_harness_from_config(self.srlm, bank_dir=mc.bank_dir, base_url=base_url, embed_model=mc.embed_model, reflect_model=mc.reflect_model, api_key=api_key, embed_base_url=mc.embed_url, embed_api_key=mc.embed_api_key, **{k: v for k, v in [("k_max", mc.k_max), ("min_cosine", mc.min_cosine)] if v is not None})`. (Omitting `k_max`/`min_cosine` when `None` preserves prehend's defaults, matching benchmark's `_maybe_wrap_memory`.)

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_harness.py
from prehend.harness import MemoryConfig
from prehend.memory.harness import MemoryHarness


class TestHarnessMemory:
    def test_no_memory_inference_client_is_srlm(self):
        h = _h()
        assert h.inference_client is h.srlm

    def test_memory_wraps_inference_client(self, tmp_path):
        h = _h(memory=MemoryConfig(
            bank_dir=str(tmp_path / "bank"),
            embed_model="bge-m3", reflect_model="m",
            embed_url="http://localhost:8084/v1",
        ))
        assert isinstance(h.inference_client, MemoryHarness)
        assert h.inference_client is not h.srlm
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/bin/uv run python -m pytest tests/test_harness.py::TestHarnessMemory -q`
Expected: FAIL on `test_memory_wraps_inference_client` (inference_client is the SRLM, not a MemoryHarness)

- [ ] **Step 3: Write minimal implementation**

In `Harness.__init__`, store `self._base_url = base_url` and `self._api_key = api_key`, then replace `self.inference_client = self.srlm` with:

```python
        self.inference_client = self.srlm
        if memory is not None:
            from prehend.memory.factory import build_memory_harness_from_config
            tight = {k: v for k, v in (("k_max", memory.k_max),
                                       ("min_cosine", memory.min_cosine)) if v is not None}
            self.inference_client = build_memory_harness_from_config(
                self.srlm,
                bank_dir=memory.bank_dir,
                base_url=base_url,
                embed_model=memory.embed_model,
                reflect_model=memory.reflect_model,
                api_key=api_key,
                embed_base_url=memory.embed_url,
                embed_api_key=memory.embed_api_key,
                **tight,
            )
```

(Place this AFTER the `observability(self.srlm)` call so the hook always binds the
raw SRLM, never the wrapper.)

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/bin/uv run python -m pytest tests/test_harness.py::TestHarnessMemory -q`
Expected: PASS (2 passed). If `build_memory_harness_from_config` attempts network
at construction, it should not (per benchmark's note: embed/reflect are lazy); if
it does, mark the test to inject a fake factory -- but verify laziness first.

- [ ] **Step 5: Commit**

```bash
git add prehend/harness.py tests/test_harness.py
git commit -m "feat(harness): compose memory via MemoryConfig (wraps SRLM in MemoryHarness)"
```

---

### Task 5: Public exports + imports test

**Files:**
- Modify: `prehend/__init__.py`
- Modify: `tests/test_imports.py`

**Interfaces:**
- Consumes: all of `prehend.harness` (Tasks 1-4).
- Produces: `from prehend import Harness, Runtime, MemoryConfig, Defaults` works; names in `prehend.__all__`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_imports.py
def test_harness_public_api():
    import prehend
    from prehend import Harness, Runtime, MemoryConfig, Defaults
    for name in ("Harness", "Runtime", "MemoryConfig", "Defaults"):
        assert name in prehend.__all__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/bin/uv run python -m pytest tests/test_imports.py::test_harness_public_api -q`
Expected: FAIL with `ImportError: cannot import name 'Harness' from 'prehend'`

- [ ] **Step 3: Write minimal implementation**

```python
# prehend/__init__.py -- add after the existing RLM/SRLM imports
from prehend.harness import Harness, Runtime, MemoryConfig, Defaults
```
and extend `__all__` with `"Harness", "Runtime", "MemoryConfig", "Defaults"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/bin/uv run python -m pytest tests/test_imports.py -q`
Expected: PASS

- [ ] **Step 5: Run the FULL prehend suite (no regressions)**

Run: `~/.local/bin/uv run python -m pytest -q`
Expected: previous green count + the new tests; 0 failures.

- [ ] **Step 6: Commit**

```bash
git add prehend/__init__.py tests/test_imports.py
git commit -m "feat(harness): export Harness/Runtime/MemoryConfig/Defaults"
```

---

### Task 6: Harness advanced passthroughs (direct routing + multi-trajectory)

**Files:**
- Modify: `prehend/harness.py` -- add optional passthrough kwargs to `Harness.__init__`.
- Test: `tests/test_harness.py`

**Why:** `benchmark.py` exposes (and `tests/test_benchmark_direct_routing.py` tests) SRLM's context-routing + multi-trajectory knobs. Task 7 migrates benchmark onto `Harness`, so `Harness` must forward these or that capability is lost. All are real `SRLM`/`RLM` params; the five candidate/routing ones are stored as SRLM attributes (`prehend/core/srlm.py:121-125`).

**Interfaces:**
- Consumes: `Harness` (Task 3).
- Produces: `Harness.__init__` gains optional kwargs, all defaulting to `None` and forwarded to SRLM ONLY when not `None` (so SRLM's own defaults apply otherwise): `direct_threshold: int | None = None`, `n_candidates: int | None = None`, `candidate_temperature: float | None = None`, `candidate_parallel: int | None = None`, `confidence_elicitation: bool | None = None`, `scheduler_max_concurrent: int | None = None`, `scheduler_coordination_dir: str | None = None`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_harness.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/bin/uv run python -m pytest tests/test_harness.py::TestHarnessPassthroughs -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'direct_threshold'`

- [ ] **Step 3: Write minimal implementation**

Add the seven params to `Harness.__init__`'s signature (each `... | None = None`),
then, after the existing optional-hook gating and BEFORE `self.srlm = SRLM(**srlm_kwargs)`:

```python
        for _name, _val in (
            ("direct_threshold", direct_threshold),
            ("n_candidates", n_candidates),
            ("candidate_temperature", candidate_temperature),
            ("candidate_parallel", candidate_parallel),
            ("confidence_elicitation", confidence_elicitation),
            ("scheduler_max_concurrent", scheduler_max_concurrent),
            ("scheduler_coordination_dir", scheduler_coordination_dir),
        ):
            if _val is not None:
                srlm_kwargs[_name] = _val
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/bin/uv run python -m pytest tests/test_harness.py -q`
Expected: PASS (whole file green, including Tasks 1-5).

- [ ] **Step 5: Commit**

```bash
git add prehend/harness.py tests/test_harness.py
git commit -m "feat(harness): optional passthroughs for direct routing + multi-trajectory knobs"
```

---

### Task 7: Migrate `benchmark.py` onto `Harness` (rlm-trainer)

**Files:**
- Modify: `~/src/rlm-trainer/benchmark.py` -- `_run_one_task` SRLM block + `_maybe_wrap_memory`.
- Modify: `~/src/rlm-trainer/tests/test_benchmark_direct_routing.py` -- the two SRLM-construction tests now assert against `Harness` (the `--help` flag tests are unchanged; the flags STAY).
- Test (must stay green): `~/src/rlm-trainer/tests/test_benchmark_memory.py` AND `~/src/rlm-trainer/tests/test_benchmark_direct_routing.py`.

**Interfaces:**
- Consumes: `from prehend import Harness, MemoryConfig`.
- Produces: identical solve behavior, zero capability loss; `_maybe_wrap_memory` deleted; the advanced CLI flags + `run_benchmark` params are KEPT (forwarded through Harness passthroughs).

**Working dir:** `cd ~/src/rlm-trainer`. prehend is an editable install here; if missing run `~/.local/bin/uv pip install -e ~/src/prehend` (handoff gotcha). This is a DIFFERENT git repo from prehend; commit here.

- [ ] **Step 1: Capture the green baseline**

Run: `cd ~/src/rlm-trainer && .venv/bin/python -m pytest tests/test_benchmark_memory.py tests/test_benchmark_direct_routing.py -q`
Expected: PASS (record the counts).

- [ ] **Step 2: Replace the SRLM construction + memory wrap with Harness (forwarding the advanced knobs)**

In `_run_one_task`, replace the whole `srlm = SRLM(...)` block and
`inference_client = _maybe_wrap_memory(srlm, params)` with:

```python
    mem = None
    if params.get("memory_bank"):
        tight = {}
        if params.get("memory_k_max") is not None:
            tight["k_max"] = params["memory_k_max"]
        if params.get("memory_min_cosine") is not None:
            tight["min_cosine"] = params["memory_min_cosine"]
        mem = MemoryConfig(
            bank_dir=str(params["memory_bank"]),
            embed_model=params["embed_model"],
            reflect_model=params["reflect_model"],
            embed_url=params.get("embed_url"),
            **tight,
        )
    harness = Harness(
        model=params["model_name"],
        base_url=params["base_url"],
        timeout=params["timeout"],
        runtime="auto",
        memory=mem,
        logger=logger,
        # advanced knobs preserved (kept on the CLI / run_benchmark); forwarded
        # to SRLM by the Harness only when set:
        direct_threshold=params.get("direct_threshold", 0),
        n_candidates=params.get("n_candidates", 1),
        candidate_temperature=params.get("candidate_temperature"),
        candidate_parallel=params.get("candidate_parallel", 1),
        confidence_elicitation=params.get("confidence_elicitation", False),
        scheduler_max_concurrent=params.get("scheduler_max_concurrent"),
        scheduler_coordination_dir=params.get("scheduler_coordination_dir"),
    )
    inference_client = harness
    start = time.time()
    try:
        response = inference_client.completion(task["context"], task["query"])
```

Delete the `_maybe_wrap_memory` function and the now-unused `SRLM` import if no
other reference remains (grep first: `grep -n "SRLM\|_maybe_wrap_memory" benchmark.py`).
Add `from prehend import Harness, MemoryConfig` at the top. DO NOT remove the
argparse flags or the `run_benchmark` params for the advanced knobs -- they are
preserved. (The `RLM_SUBCALL_THINKING` subcall-thinking env toggle and the
`soft_timeout_pct` param are NOT forwarded by this migration; both default to the
Harness vetted default of off/None which matches benchmark's prior default, so
behavior is unchanged unless someone set them. If you find a run that sets either,
stop and report it rather than silently dropping it.)

- [ ] **Step 3: Update `tests/test_benchmark_direct_routing.py` construction tests**

The two tests that do `@patch("benchmark.SRLM")` and assert `_run_one_task` built
an SRLM with `direct_threshold`/`n_candidates` must now patch the Harness instead.
Change `@patch("benchmark.SRLM")` to `@patch("benchmark.Harness")` in
`test_creates_srlm_with_threshold` and `test_creates_srlm_with_candidates`, and
assert on the Harness call kwargs:

```python
    @patch("benchmark.Harness")
    @patch("benchmark.RLMLogger")
    def test_creates_srlm_with_threshold(self, mock_logger_cls, mock_harness_cls):
        benchmark._run_one_task(self._base_params(50000, direct_threshold=30000))
        call_kwargs = mock_harness_cls.call_args.kwargs
        assert call_kwargs["direct_threshold"] == 30000
```
(Apply the analogous change to `test_creates_srlm_with_candidates` for
`n_candidates == 4`. Confirm the exact logger-patch target/name from the existing
test; keep the `--help` flag tests unchanged since the flags are kept.)

- [ ] **Step 4: Run both benchmark test files**

Run: `cd ~/src/rlm-trainer && .venv/bin/python -m pytest tests/test_benchmark_memory.py tests/test_benchmark_direct_routing.py -q`
Expected: PASS, same counts as Step 1.

- [ ] **Step 5: Smoke a single real solve (no-memory + memory), 1 task each**

Run (server must be up; v13 loaded):
```bash
cd ~/src/rlm-trainer
.venv/bin/python scripts/memory_cold_warm.py off  --model gemma-4-12b-it-sft-kb-v13-sft --tasks-dir tasks/subset --tasks kb --max-tasks 1 --out /tmp/harness-smoke
.venv/bin/python scripts/memory_cold_warm.py cold --model gemma-4-12b-it-sft-kb-v13-sft --tasks-dir tasks/subset --tasks kb --max-tasks 1 --out /tmp/harness-smoke
```
Expected: both `[1/1] ... CORRECT` (or a genuine solve), 0 infra-fail from wiring.

- [ ] **Step 6: Commit (in rlm-trainer)**

```bash
cd ~/src/rlm-trainer
git add benchmark.py tests/test_benchmark_direct_routing.py
git commit -m "refactor(benchmark): solve via prehend Harness; drop hand-wired SRLM + _maybe_wrap_memory"
```

---

### Task 8: ADR-0008 + supersede-note on ADR-0005

**Files:**
- Create: `docs/decisions/0008-high-level-harness-api.md`
- Modify: `docs/decisions/0005-mnemex-experience-memory-layer.md`

- [ ] **Step 1: Write ADR-0008 (MADR format, matching the repo's existing ADRs)**

Content: context (SRLM low-level surface; clients hand-assemble + diverge;
concurrency seam is explicit args, MAPREDUCE_CONCURRENCY is a different
subsystem); decision (add `Harness` owning Tier A defaults + Tier B hybrid
runtime + memory composition, with Tier-C hooks; SRLM stays the escape hatch;
benchmark migrated, kb-librarian a fast-follow); consequences (clients stop
diverging; MemoryHarness demoted to internal building block; YAGNI: no named
profiles). Mirror the header/section style of `0007-rename-mnemex-to-prehend.md`.

- [ ] **Step 2: Add a supersede-note to ADR-0005**

At the top of `0005-mnemex-experience-memory-layer.md`, add one line: the memory
layer is now composed via `Harness(memory=...)`; clients no longer hand-wire
`MemoryHarness` / `_maybe_wrap_memory`. See ADR-0008.

- [ ] **Step 3: Commit**

```bash
cd ~/src/prehend
git add docs/decisions/0008-high-level-harness-api.md docs/decisions/0005-mnemex-experience-memory-layer.md
git commit -m "docs(adr): ADR-0008 high-level Harness API; supersede-note on 0005"
```

---

## Self-Review

**Spec coverage:**
- Tier A defaults -> Task 1 (`Defaults`/`VETTED`) + Task 3 (applied to SRLM). ✓
- Tier B hybrid runtime -> Task 2 (`detect_runtime`) + Task 3 (resolve/fallback). ✓
- Memory composition -> Task 4. ✓
- Tier-C hooks -> Task 3 (`system_addendum`, verifiers, `custom_tools`, `observability`). ✓
- Advanced passthroughs (direct routing + multi-trajectory) -> Task 6 (added after review found `test_benchmark_direct_routing.py` covers them). ✓
- SRLM escape hatch unchanged -> no task modifies SRLM/RLM. ✓
- Public API -> Task 5. ✓
- benchmark migration (in scope) -> Task 7 (forwards the advanced knobs; updates `test_benchmark_direct_routing.py`). ✓
- kb-librarian (fast-follow, NOT this plan) -> intentionally absent; hooks shipped in Task 3. ✓
- ADR-0008 + 0005 note -> Task 8. ✓

**Placeholder scan:** No "TBD/implement later". The one `# TODO(harness):` in Task 6
is conditional and explicitly gated on "only if a real run needs it"; the default
path omits it.

**Type consistency:** `Harness`, `Runtime`, `MemoryConfig`, `Defaults`, `VETTED`,
`detect_runtime`, `Harness.srlm/.inference_client/.runtime/.completion` are used identically
across Tasks 1-6. `build_memory_harness_from_config` kwargs match its real
signature (verified). `max_concurrent_subcalls`, `subcall_verifier`,
`answer_verifier`, `max_answer_retries`, `custom_tools`, `custom_system_prompt`
match `RLM.__init__`.

**Open implementation confirmations:**
- RESOLVED: `prehend/logger` has no generic factory; Harness uses stdlib
  `logging.getLogger("prehend.harness")`.
- RESOLVED: `build_memory_harness_from_config` builds `openai.OpenAI` clients,
  which construct lazily (no network at build) -> Task 4's construction test is safe.
- OPEN (Task 6 Step 2): benchmark's genuinely-used SRLM knobs
  (`scheduler_max_concurrent`/`direct_threshold`/`n_candidates`) -- decide by
  grepping how `run_benchmark` is actually invoked; do not silently drop one.
