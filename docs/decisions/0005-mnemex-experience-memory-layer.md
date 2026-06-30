---
status: "accepted"
date: "2026-06-20"
deciders: "potto"
---

> **Note (2026-06-21):** The memory layer is now composed via `Harness(memory=MemoryConfig(...))`
> in `prehend/harness.py`; clients no longer hand-wire `MemoryHarness` or
> `_maybe_wrap_memory` directly. See ADR-0008.

# Adopt FinAcumen's FM as lm-repl's experience-memory layer (mnemex)

## Context and Problem Statement

lm-repl (per ADR-0001, a patched RLM/SRLM fork) solves the *intra-task* problem:
context too large to attend to is offloaded into a REPL variable and queried
programmatically, with SRLM searching K candidate trajectories. But every
`completion()` is **amnesiac** - it re-derives the same reasoning on every
problem and learns nothing across tasks. FinAcumen (anonymized MIT paper code,
cloned to `~/src/FinAcumen`) ships exactly the missing *inter-task* axis: a
self-evolving experience memory (its **FM** subsystem) with the loop
`retrieve -> inject -> solve -> collect -> cross-verify -> write`. How much of
FinAcumen do we adopt, in what shape, and what do we call the result?

## Decision Drivers

- lm-repl owns the spatial axis; FM owns the temporal axis. They are orthogonal
  and composable, not competing.
- FM's FT runtime is finance-coupled (FinancialDataLookup, OCR, 88KB planning,
  77KB finance-extraction); lm-repl's RLM/SRLM engine already *is* the inference client.
- The shipped FM code diverges from its own ARCHITECTURE.md; we port what runs,
  not what is documented (verified by reading `finacumen/fm/*`):
  1. `retrieve.py` is a single-stage cosine matmul + id-dedup + top-k, NOT the
     docs' 3-stage tagger -> hard-gate -> LLM-rerank (that pipeline is absent;
     only optional `relevance.annotate` is wired).
  2. Eval embeddings are pre-baked `datasets/*_emb.npy` keyed by `target.id`
     (zero API calls); a real live `embed_text` facade exists but is unused on
     the benchmark path. A general harness must embed arbitrary queries live.
  3. `collect`/`cross_verify` does NOT distill from the live solve trace - it
     re-solves with K=8 parallel finance agents (heavy, domain-coupled). But
     lm-repl's SRLM *already* does K-candidate generation + uncertainty
     selection, which is a better-built version of the same idea.
- The donor format (`MEMORY.md` manifest + per-entry frontmatter + embedding)
  is isomorphic to the Claude Code auto-memory store - git-friendly, portable.

## Considered Options

- **Port FM, replace FT with the existing SRLM engine, genericize the one
  finance coupling (the taxonomy) behind a pluggable Tagger.**
- Port FinAcumen wholesale (FT + FM) and strip finance later.
- Build a bespoke memory layer from scratch.
- Bolt an off-the-shelf vector DB / RAG store onto SRLM.

## Decision Outcome

Chosen option: **port FM, drop FT, delegate solving to SRLM**. A new
`lm_repl/memory/` package wraps any inference client exposing the lm-repl
`completion(prompt, root_prompt)` interface, mirroring FinAcumen's
`MemoryAgentVariant` (Stage A retrieve / Stage B delegate solve / Stage C
collect). `prompt` is the offloaded REPL context; `root_prompt` is the question
the orchestrator attends to directly, and is where the retrieved
`<Memory_Block>` is injected (NOT the offloaded context var, or the guidance is
buried where only generated code can reach it). The single finance coupling -
FinAcumen's `question_class`/`question_type`/`tool_tags` taxonomy - becomes a
pluggable `Tagger` protocol whose default is embedding-only (which is what the
shipped code actually does). The merged capability is rebranded **mnemex**
(from "mnemonic"): the name describes "a harness that learns", not the REPL
mechanism.

Four FinAcumen design invariants are carried over and enforced by tests:
no-memory baseline integrity (empty retrieval -> `root_prompt` byte-identical to
the bare question), collect off-critical-path / best-effort, graceful no-memory
fallback on any retrieval failure, and a uniform `solve`-style interface so the
memory wrapper is transparent.

### Consequences

- Good, because lm-repl gains cross-task learning without inheriting any
  finance/benchmark baggage; the RLM engine stays the sole inference client.
- Good, because FM's cross-verified positive (guiding-path) / negative
  (guard-rule) pairs are DPO-grade training data, feeding rlm-trainer, not just
  inference (see `project_lm-repl-downstream-consumers`).
- Good, because the live `EmbeddingBackend` reuses the OpenAI-compatible server
  lm-repl already drives - no new infra.
- Bad, because we diverge from the donor's documented (but unimplemented)
  retrieval sophistication; tagger/rerank/cross-verify are deferred to later
  phases and may need to be rebuilt rather than lifted.
- Bad, because storing embeddings inline in `meta.json` (MVP) does not scale to
  large banks; a sidecar/manifest format is a known follow-up.

## Pros and Cons of the Options

### Port FM, replace FT with SRLM, genericize the taxonomy

- Good, because it adopts the genuinely novel, near-domain-agnostic half and
  discards the part lm-repl already supersedes.
- Good, because the integration seam (`MemoryAgentVariant` -> `SRLM.completion`)
  is a near drop-in; the MVP proved it with an injected fake inference client.
- Bad, because the heaviest quality piece (distillation) must be redesigned, not
  copied.

### Port FinAcumen wholesale

- Bad, because it drags in finance tools, OCR, and benchmark eval that lm-repl
  has no use for, and two competing inference runtimes.

### Build from scratch

- Bad, because FM's lifecycle, polarity schema, and pruning/anti-give-up rules
  are hard-won and already validated; reinventing them wastes the donor.

### Off-the-shelf vector DB / RAG

- Bad, because plain RAG retrieves *documents*, not distilled, polarity-tagged,
  verified *experience*; it has no collect/verify/prune lifecycle and no
  positive/negative guard-rule structure.

## More Information

- MVP commit `f785c10` (`feat(memory): add mnemex experience-memory MVP`):
  `bank.py`, `embed.py`, `retrieve.py`, `inject.py`, `harness.py` + 31 tests,
  all built test-first. Full suite 561 passed / 9 skipped.
- Donor: `~/src/FinAcumen`, `finacumen/fm/` (FM) and `docs/ARCHITECTURE.md`
  (treat as aspirational - verify against source).
- Deferred phases: live `EmbeddingBackend` against the local server; trace-based
  `Distiller` (or reuse SRLM `n_candidates`); pluggable `Tagger`; cross-verify;
  `MEMORY.md`-manifest storage; DPO export to rlm-trainer.
- Builds on lm-repl ADR-0001 (fork rationale). Related auto-memory:
  `project_finacumen-experience-memory`.
