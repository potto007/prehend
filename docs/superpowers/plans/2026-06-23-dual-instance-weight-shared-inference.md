# Dual-instance Weight-shared Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the v13 Gnosis model as two persistent llama-server processes (orchestrator master :8080, sub-call worker :8081) that share one VRAM weights copy via CUDA IPC, giving each role a private KV pool and prefix cache; teach the prehend Harness to route sub-calls to the worker endpoint and budget per-instance.

**Architecture:** `cuda-llm-weight-share.so` (LD_PRELOAD) intercepts the weights `cudaMalloc` so the worker maps the master's copy; KV stays private per process. Prehend's `Harness` gains a `subcall_base_url` param: the orchestrator backend keeps `base_url`, the sub-call backend uses `subcall_base_url`, and the sub-call budget divides the *worker's* dedicated pool by the *worker's* slots (ADR-0012 made per-instance).

**Tech Stack:** Python 3 (prehend, pytest), bash + systemd user units (local-ai), C LD_PRELOAD lib (cuda-llm-weight-share, prebuilt).

## Global Constraints

- NO em dashes anywhere (CLAUDE.md). Use a regular dash.
- Validate serving config with a SUSTAINED run, not a burst (CLAUDE.md).
- `prehend/CLAUDE.md` must NOT be modified.
- Backward compat: `subcall_base_url=None` MUST produce behavior byte-identical to today (single-server / OpenRouter / vLLM callers unaffected).
- venv uses `uv` (no `pip`): run tests via `~/.local/bin/uv run pytest`.
- llama-server relaunch needs `LD_LIBRARY_PATH=/usr/local/cuda-13/lib64`.
- NEVER `pkill -f llama-server` (self-match). Kill by explicit PID.
- Record the decision as an ADR extending ADR-0012 (budget now per-instance).

---

## Part A: Prehend Harness (TDD, pure Python)

### Task A1: `subcall_base_url` routes the sub-call backend

**Files:**
- Modify: `prehend/harness.py` (`Harness.__init__` signature + backend wiring, around lines 129-185)
- Test: `tests/test_harness.py`

**Interfaces:**
- Consumes: `SRLM.backend_kwargs: dict`, `SRLM.other_backend_kwargs: list[dict]` (already populated from `backend_kwargs` / `subcall_kwargs`).
- Produces: `Harness(..., subcall_base_url: str | None = None)`. When set, `srlm.other_backend_kwargs[0]["base_url"] == subcall_base_url` while `srlm.backend_kwargs["base_url"] == base_url`. When None, both equal `base_url` (unchanged).

- [ ] **Step 1: Write the failing test**

```python
# in tests/test_harness.py, class TestHarnessCore
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/bin/uv run pytest tests/test_harness.py::TestHarnessCore::test_subcall_base_url_routes_only_the_subcall_backend -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'subcall_base_url'`

- [ ] **Step 3: Write minimal implementation**

In `Harness.__init__` signature add after `base_url`:
```python
        subcall_base_url: str | None = None,
        subcall_runtime: "Runtime | str | None" = None,
```
After `self.runtime = self._resolve_runtime(...)` (line 156) add:
```python
        eff_subcall_url = subcall_base_url or base_url
        if subcall_base_url is None:
            self.subcall_runtime = self.runtime
        else:
            sc_arg = subcall_runtime if subcall_runtime is not None else "auto"
            self.subcall_runtime = self._resolve_runtime(sc_arg, eff_subcall_url, api_key, d)
        self.subcall_base_url = eff_subcall_url
```
Change `subcall_kwargs` construction (line 182) to point at the worker:
```python
        subcall_kwargs = dict(backend_kwargs)
        subcall_kwargs["base_url"] = eff_subcall_url
        subcall_kwargs["default_extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": d.subcall_enable_thinking}
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.local/bin/uv run pytest tests/test_harness.py -v`
Expected: PASS (both new tests + all existing).

- [ ] **Step 5: Commit**

```bash
git add prehend/harness.py tests/test_harness.py
git commit -m "feat(harness): route sub-calls to subcall_base_url (dual-instance inference)"
```

### Task A2: Sub-call budget + fan-out come from the worker runtime

**Files:**
- Modify: `prehend/harness.py` (lines 168-176 shared_pool/eff_subcall_limit, line 202 max_concurrent_subcalls)
- Test: `tests/test_harness.py`

**Interfaces:**
- Consumes: `per_call_subcall_budget(pool, slots)`, `resolve_subcall_limit(model, explicit, runtime_ctx)` (unchanged), `self.subcall_runtime` from A1.
- Produces: `srlm.subcall_context_limit == subcall_runtime.ctx // subcall_runtime.slots`; `srlm.max_concurrent_subcalls == subcall_runtime.slots`.

- [ ] **Step 1: Write the failing test**

```python
# in tests/test_harness.py, class TestHarnessCore
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/bin/uv run pytest tests/test_harness.py::TestHarnessCore::test_subcall_budget_and_fanout_use_worker_runtime -v`
Expected: FAIL (assert 1 == 4 / wrong budget; still computed from `self.runtime`).

- [ ] **Step 3: Write minimal implementation**

Change the budget block (lines 173-176) to use the sub-call runtime:
```python
        shared_pool = resolve_subcall_limit(
            model, explicit=explicit_limit, runtime_ctx=self.subcall_runtime.ctx
        )
        eff_subcall_limit = per_call_subcall_budget(shared_pool, self.subcall_runtime.slots)
```
Change `max_concurrent_subcalls=self.runtime.slots` (line 202) to:
```python
            max_concurrent_subcalls=self.subcall_runtime.slots,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.local/bin/uv run pytest tests/test_harness.py -v`
Expected: PASS (new tests + `test_subcall_limit_is_shared_pool_divided_by_slots` still 24576 since single-endpoint subcall_runtime == self.runtime).

- [ ] **Step 5: Commit**

```bash
git add prehend/harness.py tests/test_harness.py
git commit -m "feat(harness): budget+fanout from worker runtime (ADR-0012 per-instance)"
```

### Task A3: Auto-probe the worker endpoint when `subcall_runtime` omitted

**Files:**
- Modify: `prehend/harness.py` (none beyond A1 if `_resolve_runtime("auto", ...)` already probes; verify)
- Test: `tests/test_harness.py`

**Interfaces:**
- Consumes: `_resolve_runtime(runtime, base_url, api_key, d)` with `runtime="auto"` -> `detect_runtime(base_url)`; falls back to `Runtime(slots=d.max_concurrent_subcalls)` when probe ambiguous.
- Produces: with `subcall_base_url` set and `subcall_runtime` omitted, `self.subcall_runtime` is the resolution of the worker URL (falls back to `slots=4, ctx=None` when unreachable).

- [ ] **Step 1: Write the failing test**

```python
# in tests/test_harness.py, class TestHarnessCore
def test_worker_runtime_auto_falls_back_when_unreachable(self):
    # subcall_base_url given but no subcall_runtime: probe fails (port closed)
    # -> fall back to default slots, ctx None -> guard off for sub-calls.
    h = Harness(model="m", base_url="http://localhost:8080/v1",
                subcall_base_url="http://localhost:9998/v1",
                runtime=Runtime(slots=1, ctx=32768))
    assert h.subcall_runtime.slots == VETTED.max_concurrent_subcalls
    assert h.srlm.max_concurrent_subcalls == VETTED.max_concurrent_subcalls
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `~/.local/bin/uv run pytest tests/test_harness.py::TestHarnessCore::test_worker_runtime_auto_falls_back_when_unreachable -v`
Expected: PASS if A1's `sc_arg="auto"` path already routes through `_resolve_runtime` (likely). If FAIL, the auto branch is not wired - fix by ensuring the `else` branch in A1 passes `"auto"` (not None) to `_resolve_runtime`.

- [ ] **Step 3: (only if Step 2 failed) wire the auto branch** - already specified in A1 Step 3 (`sc_arg = subcall_runtime if subcall_runtime is not None else "auto"`). No new code if A1 correct.

- [ ] **Step 4: Run the full prehend suite**

Run: `~/.local/bin/uv run pytest tests/test_harness.py tests/test_token_utils.py tests/test_subcall_context_limit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_harness.py
git commit -m "test(harness): worker-runtime auto-probe fallback"
```

### Task A4: ADR + docstring

**Files:**
- Create: `docs/decisions/0013-dual-instance-weight-shared-inference.md`
- Modify: `prehend/harness.py` (class docstring note on the new param)

- [ ] **Step 1: Write the ADR** (MADR format, matching `docs/decisions/0012-*.md`): context = kv-unified shared-pool contention + prefix eviction; decision = two weight-shared instances, Harness `subcall_base_url`, per-instance budget; consequences = no idle-unload on the pair, master-first ordering, N x prefix KV; supersedes-extends ADR-0012. NO em dashes.

- [ ] **Step 2: Reference the ADR in the harness docstring** - add one line to the `Harness` class docstring: "Sub-calls may target a separate weight-shared worker via subcall_base_url (ADR-0013)."

- [ ] **Step 3: Commit**

```bash
git add docs/decisions/0013-dual-instance-weight-shared-inference.md prehend/harness.py
git commit -m "docs(adr): 0013 dual-instance weight-shared inference (extends 0012)"
```

---

## Part B: Serving / ops (local-ai, empirical validation)

> Validated by the commands below on the GPU host, not pytest. Run on branch in `~/src/local-ai`.

### Task B1: Reconnaissance - capture the v13 weights allocation size

**Files:** none (one-off measurement; record the number in B2's units)

- [ ] **Step 1: Build the lib if absent**

```bash
cd ~/src/cuda-llm-weight-share
gcc -shared -fPIC -O2 -g -Wall -Wextra -I/usr/local/cuda/include \
  cuda-llm-weight-share.c -o cuda-llm-weight-share.so -ldl
nm -D ./cuda-llm-weight-share.so | grep -E ' cudaMalloc| cudaFree'
```
Expected: `T cudaMalloc` and `T cudaFree`.

- [ ] **Step 2: Recon run (no MODEL_SIZE)**

Ensure no orphan clients (`ps -eo cmd | grep '[.]venv/bin/python'` empty) and port 8080 free first. Then:
```bash
LD_LIBRARY_PATH=/usr/local/cuda-13/lib64 \
LD_PRELOAD=~/src/cuda-llm-weight-share/cuda-llm-weight-share.so \
<llama-server-bin> -m ~/src/local-ai/models/gemma-4-12B-it-sft-kb-v13-sft.Q4_0.gguf --port 8080 2>&1 | grep VRAM_HOOK
```
Expected: a `cudaMalloc normal: ~6650 MB (<bytes>)` line near the model size. Record `<bytes>` as `MODEL_SIZE`. Stop the server (explicit PID).

### Task B2: Two systemd user units + `start-pair`/`stop-pair`

**Files:**
- Create: `~/src/local-ai/scripts/localai-llama-orch.service`
- Create: `~/src/local-ai/scripts/localai-llama-worker.service`
- Modify: `~/src/local-ai/scripts/llama-server.sh` (add `start-pair`, `stop-pair`, `status-pair`)

Both units set: `LD_PRELOAD`, `LD_LIBRARY_PATH=/usr/local/cuda-13/lib64`, `CUDA_VISIBLE_DEVICES=0`, `MODEL_SIZE=<B1 bytes>`, `CUDA_VRAM_IPC_NAME=/cuda_vram_ipc_v13_gpu0`, `CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC=3`. Launch plain `--model .../v13-sft.Q4_0.gguf` (NOT `--models-preset`) with shared knobs `--flash-attn on --cache-type-k q4_0 --cache-type-v q4_0 --swa-full true --cache-reuse 256 --cache-ram 4096 --jinja --temp 0 --n-gpu-layers 99 --batch-size 4096 --ubatch-size 1024`. NO `--sleep-idle-seconds` on either unit (idle-unload would free the master's shared weights). Per-role:
- orch: `--port 8080 --ctx-size 32768 --parallel 1`
- worker: `--port 8081 --ctx-size 65536 --parallel 4`

- [ ] **Step 1: Write the two unit files** (model `ExecStart` mirroring `localai-llama-server.service`, with the env block above).

- [ ] **Step 2: Add `start-pair` to `llama-server.sh`** - asserts no orphans, asserts ports 8080+8081 free, `rm -f /dev/shm/cuda_vram_ipc_v13_gpu0`, starts orch unit, waits for `MASTER role` in its log AND port 8080 up, THEN starts worker unit, waits for `WORKER: IPC handle mapped` AND port 8081 up. `stop-pair` stops worker first, then orch; confirms both ports free + VRAM back to idle. `status-pair` prints listeners + `/models` health for both.

- [ ] **Step 3: Start the pair**

```bash
~/src/local-ai/scripts/llama-server.sh start-pair
```
Expected: orch log `MASTER: allocated ... published`; worker log `WORKER: IPC handle mapped ... No duplicate weights allocation`.

- [ ] **Step 4: Verify single shared weights copy**

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```
Expected: TWO llama-server pids; combined VRAM well under 2x weights (only one ~6.5 GB weights copy + two KV pools), not ~13 GB of duplicated weights.

### Task B3: Endpoint + no-contention validation

- [ ] **Step 1: Both endpoints distinct**

```bash
curl -s localhost:8080/props | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["default_generation_settings"]["n_ctx"])'
curl -s localhost:8081/props | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["default_generation_settings"]["n_ctx"])'
```
Expected: 32768 and 65536.

- [ ] **Step 2: Sustained no-contention run** - run the multihop eval that previously hit the overflow, pointing prehend at the pair:

```python
Harness(model="gemma-4-12b-it-sft-kb-v13-sft",
        base_url="http://localhost:8080/v1",
        subcall_base_url="http://localhost:8081/v1")
```
Run the sustained multihop benchmark (not a single burst). Expected in the server logs: NO `failed to find free space in the KV cache`, NO `Context size has been exceeded`. Tail both logs during the run.

- [ ] **Step 3: Prefix-cache reuse fires on both** - grep both server logs for prefix reuse (e.g. `prompt processing progress` / cache-reuse hit counts); confirm the orchestrator prefix is not re-prefilled after sub-call bursts.

- [ ] **Step 4: Ramp worker pool (optional)** - if VRAM headroom remains (`nvidia-smi`), raise worker `--ctx-size` toward 98304 in B2's worker unit, restart pair, re-run Step 2 under SUSTAINED load until spill, then back off one step. Record the chosen size in the unit + ADR-0013.

- [ ] **Step 5: Commit local-ai changes**

```bash
cd ~/src/local-ai && git add scripts/ && \
git commit -m "feat(serving): dual-instance weight-shared v13 inference pair (prehend ADR-0013)"
```

### Task B4: Lifecycle guard check

- [ ] **Step 1: Double-master prevention** - with the pair down, start ONLY the worker; confirm `start-pair`'s ordering would have prevented it (worker alone self-elects master and allocates its own weights - this is why ordering matters). Then `stop-pair`.
- [ ] **Step 2: Master-exit behavior** - per README, kill the master while the worker runs; confirm the worker keeps its mapping (or document the observed behavior). Restore via `stop-pair` + `start-pair`.

---

## Self-Review

- **Spec coverage:** Architecture -> B2; serving/ops 3 consequences -> B2 (no idle-unload Step), B2 Step 2 (master-first ordering), B2 Step 2 (stale shm rm). Harness API -> A1/A2/A3. Sizing -> B2 + B3 ramp. Test/validation 1-6 -> B2 Step 4 (#1), B3 Step 1 (#2), B3 Step 2 (#3), B3 Step 3 (#4), A1/A2 regression tests (#5), B4 (#6). ADR -> A4. All spec sections mapped.
- **Placeholder scan:** none (`<bytes>`, `<llama-server-bin>` are explicit measured/path values, not deferred work).
- **Type consistency:** `subcall_base_url` / `subcall_runtime` / `self.subcall_runtime` consistent across A1-A3; `per_call_subcall_budget(pool, slots)` and `resolve_subcall_limit(model, explicit, runtime_ctx)` match `token_utils.py`.
