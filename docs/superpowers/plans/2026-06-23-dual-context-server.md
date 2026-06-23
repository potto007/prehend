# Dual-context server Implementation Plan

> **For agentic workers:** implement task-by-task; each task ends with an independently checkable deliverable. GPU smoke tests are GATED - do not run anything on the GPU; stop and hand back when the binary builds.

**Goal:** A single-process, OpenAI-compatible server that loads the v13 weights ONCE and serves two roles - orchestrator (`:8080`, one big KV slot, CoT on) and sub-call worker (`:8081`, many small slots, CoT off) - each `server_context` owning a PRIVATE KV cache + prompt cache. Replaces the WSL2-impossible two-process CUDA-IPC weight-share (ADR-0013) with intra-process multi-context sharing.

**Architecture:** In fork `potto007/llama.cpp` (branch off `diffusion-http-server`). One `llama_model_load_from_file`; two `server_context` instances constructed against that shared model (each via the refactored `load_model(params, shared_model)`); two `server_http_context` listeners; both `start_loop()` on threads. Each `server_context` already manages slots + KV + prompt-cache (prefix reuse) per model, so no kv-unified shared pool exists.

**Tech Stack:** C++17, llama.cpp server internals (`tools/server/server-context.*`, `server-http.*`), cpp-httplib, nlohmann/json, CMake. Prehend harness (Python) unchanged. local-ai systemd unit. Verified prereq: `~/src/cuda-llm-weight-share/wsl-experiments/multi_ctx_proof.c` already proved one model backs two private KV caches with no weight duplication (model load +7.1GB, ctxA KV +6.3GB, ctxB KV +1.96GB).

## Global Constraints

- **GPU runs are GATED.** Compile only (CPU). Do NOT launch the binary or any llama-server on the GPU; stop and report when it builds. Smoke tests are the user's step.
- Fork work branches from `diffusion-http-server`; do NOT alter the diffusion server target.
- NO em dashes. Use a regular dash.
- GPU hygiene (when the user later runs): kill by PID, never `pkill -f llama-server`, confirm VRAM idle before/after.
- The prehand harness (`subcall_base_url`) is DONE and unchanged: orchestrator->:8080, sub-calls->:8081.
- ADR-0013 is accepted/immutable: SUPERSEDE it with a new ADR-0014, do not edit it.

---

## Task 1: Refactor `server_context` to accept a shared (borrowed) model

**Files (fork):**
- Modify: `tools/server/server-context.h` (`load_model` signature)
- Modify: `tools/server/server-context.cpp` (`server_context_impl` model load + teardown)

**Interfaces:**
- Produces: `bool server_context::load_model(common_params & params, llama_model * shared_model = nullptr);` - when `shared_model != nullptr`, skip `llama_model_load_from_file`, use the borrowed pointer, and mark it NOT-owned so teardown never frees it. Default `nullptr` preserves today's owning behavior (zero change for `llama-server`).

- [ ] **Step 1:** Read `server-context.cpp` to find where the model is loaded (`llama_model_load_from_file` / `common_init_from_params`) and freed (dtor / `llama_model_free`). Identify the owned-model member and its free path.
- [ ] **Step 2:** Add a failing build-level check: change the header signature to the new overload and add an `owns_model` bool to `server_context_impl`; leave impl unchanged so it FAILS to compile (proves the seam is wired).
- [ ] **Step 3:** Implement: in load, `if (shared_model) { model = shared_model; owns_model = false; }` else load as before with `owns_model = true`. In teardown, free model only if `owns_model`. The context (`llama_init_from_model`) is ALWAYS created locally and always freed (KV is per-instance).
- [ ] **Step 4:** Build `llama-server` (existing target) to prove no regression: `cmake --build build -j --target llama-server` compiles clean.
- [ ] **Step 5:** Commit (fork): `feat(server): server_context can borrow a shared llama_model (dual-context)`.

## Task 2: `llama-server-objs` reusable object/static lib (if needed for linking)

**Files (fork):**
- Modify: `tools/server/CMakeLists.txt`

**Interfaces:**
- Produces: the `server-*.cpp` translation units (context, http, queue, task, common, chat, tools) available to link into a second executable, without duplicating sources. Prefer an `OBJECT` or `STATIC` library target (e.g. `llama-server-core`) that both `llama-server` and the new target link.

- [ ] **Step 1:** Read `tools/server/CMakeLists.txt` - see how `llama-server` is assembled and which `server-*.cpp` it compiles.
- [ ] **Step 2:** Extract the server sources (excluding `main.cpp`) into a `STATIC`/`OBJECT` lib target; relink `llama-server` against it. Build `llama-server`; expect clean (refactor only).
- [ ] **Step 3:** Commit (fork): `build(server): extract server core into a linkable lib`.

(If `llama-server` already builds a reusable lib, skip this task and link it directly in Task 3.)

## Task 3: New target `examples/dual-context-server/`

**Files (fork):**
- Create: `examples/dual-context-server/CMakeLists.txt` (target `llama-dual-context-server`, link the server core lib + llama + cpp-httplib), modeled on `examples/diffusion-gemma-server/CMakeLists.txt`
- Create: `examples/dual-context-server/dual-context-server.cpp`
- Modify: `examples/CMakeLists.txt` if it must register the subdir

**Interfaces:**
- Consumes: `server_context::load_model(params, shared_model)` (Task 1), `server_http_context` / `server_routes` (server-http.h), `common_params`.
- Produces: a binary that takes `--model`, plus per-role knobs (orchestrator `--orch-ctx N --orch-parallel N`, worker `--worker-ctx N --worker-parallel N --worker-port P`, default ports 8080/8081), loads the model once, runs both roles.

- [ ] **Step 1:** Read `tools/server/main.cpp` to see how ONE `server_context` + `server_http_context` + `server_routes` are wired and how `start_loop()` runs. This is the single-role template to double.
- [ ] **Step 2:** Write `dual-context-server.cpp`: parse args into two `common_params` (role differences: ctx, parallel, port; CoT handled per-request by the client, not the server, matching prehend's `subcall_enable_thinking`); `llama_model * model = llama_model_load_from_file(...)` once; `server_context orch, worker; orch.load_model(p_orch, model); worker.load_model(p_worker, model);`; build a `server_http_context` + `server_routes` for each on its port; run `orch.start_loop()` and `worker.start_loop()` on two threads; join + clean shutdown (terminate both, then `llama_model_free(model)` once).
- [ ] **Step 3:** Register + build: `cmake --build build -j --target llama-dual-context-server`. Expect clean compile + link.
- [ ] **Step 4:** Static sanity (NO GPU): `./build/bin/llama-dual-context-server --help` (or `--version`) prints and exits 0; confirms arg wiring without loading the model.
- [ ] **Step 5:** Commit (fork): `feat(dual-context-server): single-process two-context OpenAI server`.

## Task 4: local-ai - single unit launching the dual server

**Files (local-ai, branch `feat/dual-instance-weight-shared-solver`):**
- Create: `scripts/localai-llama-dual-context.service`
- Modify: `scripts/llama-server.sh` (replace `recon`/`start-pair`/`stop-pair`/`status-pair` with `start-dual`/`stop-dual`/`status-dual`; drop the LD_PRELOAD/MODEL_SIZE/IPC machinery)
- Delete: `scripts/localai-llama-solver-orch.service`, `scripts/localai-llama-solver-worker.service` (the IPC two-process units)

- [ ] **Step 1:** Write the unit: `ExecStart=.../llama-dual-context-server --model .../v13-sft.Q4_0.gguf --orch-ctx 98304 --orch-parallel 4 --worker-ctx 65536 --worker-parallel 4` plus shared knobs (flash-attn, q4_0 KV, swa-full, cache-reuse, cache-ram, jinja, temp 0); `LD_LIBRARY_PATH=/usr/local/cuda-13/lib64`; NO LD_PRELOAD; one log file.
- [ ] **Step 2:** Rewrite the script commands: `start-dual` (assert orphans/ports/VRAM, start unit, wait for BOTH ports), `stop-dual`, `status-dual` (both ports + VRAM + one weights copy). Remove the IPC/recon helpers.
- [ ] **Step 3:** `bash -n scripts/llama-server.sh` clean. Commit (local-ai).

## Task 5: ADR-0014 superseding ADR-0013

**Files (prehend):** Create `docs/decisions/0014-single-process-dual-context-solver.md`

- [ ] **Step 1:** MADR format: context = ADR-0013's two-process CUDA-IPC weight-share is inoperable on WSL2 (`cudaIpcOpenMemHandle` -> 400; VMM-fd works but unneeded); decision = single-process multi-context (one `llama_model`, two `server_context`, private KV/prefix each) in a custom fork binary; consequences = one weights copy, no kv-unified contention, prefix reuse per role, custom binary tracks llama.cpp server internals; supersedes ADR-0013. Mark ADR-0013 status `superseded by 0014` in 0014's text only (do not edit 0013's body). NO em dashes.
- [ ] **Step 2:** Commit (prehend).

## Self-Review

- Coverage: shared-model seam (T1), linkability (T2), the dual-role binary (T3), ops (T4), decision record (T5). Harness unchanged (prior work). GPU smoke test = user-gated, intentionally not a task here.
- Risk: T1 (model lifetime in the pimpl) and T2 (CMake linkability) are the unknowns; both are compile-verifiable without the GPU.
