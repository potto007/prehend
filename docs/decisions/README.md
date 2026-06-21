# Architecture Decision Records

This directory holds [MADR](https://adr.github.io/madr/)-format Architecture
Decision Records for **prehend** (the patched RLM/SRLM orchestrator with an
experience-memory layer; renamed `lm-repl` -> `mnemex` per ADR-0006, then
`mnemex` -> `prehend` per ADR-0007 - older ADR titles keep the name they were
accepted under). Each ADR captures one
hard-won, non-obvious choice so it is found once and reversed only with evidence. New decisions get the next number; accepted ADRs are immutable (to
change course, write a new ADR and mark the old one `superseded by ADR-NNNN`).

Cross-repo references are written `<repo> ADR-NNNN`. The cross-repo master index
lives in [rlm-trainer `docs/decisions/README.md`](https://github.com/ClearBridgeRIP/rlm-trainer/tree/main/docs/decisions).

| # | Title | Status |
|---|-------|--------|
| [0001](0001-lm-repl-as-patched-fork-of-rlms.md) | lm-repl as a patched fork of `rlms` | accepted |
| [0002](0002-hard-per-generation-decode-token-ceiling.md) | Hard per-generation decode-token ceiling (`max_decode_tokens`) | accepted |
| [0003](0003-runaway-generation-guards.md) | Runaway-generation guards (soft-budget, subcall caps, contention-retry) | accepted |
| [0004](0004-reasoning-loop-repeat-guard.md) | Reasoning-loop repeat-guard + escalation (4th runaway-generation guard) | accepted |
| [0005](0005-mnemex-experience-memory-layer.md) | Adopt FinAcumen's FM as lm-repl's experience-memory layer (mnemex) | accepted |
| [0006](0006-rename-lm-repl-to-mnemex.md) | Full rename `lm-repl` -> `mnemex` (package, PyPI, repo) | superseded by [0007](0007-rename-mnemex-to-prehend.md) |
| [0007](0007-rename-mnemex-to-prehend.md) | Full rename `mnemex` -> `prehend` (package, PyPI, repo) | accepted |

`0000-template.md` is the MADR template for new records.
