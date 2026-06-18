---
status: "accepted"
date: "2026-06-09"
deciders: "potto"
---

# lm-repl as a patched fork of `rlms`

## Context and Problem Statement

The RLM training harness needs an orchestrator that (a) gives sub-RLM children a
*different* system prompt than the orchestrator, and (b) supports the Apple SRLM
behaviours (context-length routing, multi-trajectory generation, uncertainty-guided
selection). Upstream MIT `rlms` has neither. Should we fork, vendor, or upstream?

## Considered Options

- **Fork** `rlms` as `lm-repl` (package import `lm_repl`), installed editable.
- Monkeypatch `rlms` at runtime from the trainer.
- Upstream the features to MIT `rlms` and depend on the release.

## Decision Outcome

Chosen option: **maintain a fork** at `~/src/lm-repl` (package `lm-repl`, branch
`main`), installed **editable** into the trainer venv
(`uv pip install -e ~/src/lm-repl --no-deps`). The fork adds:

- `child_system_prompt` on `RLM.__init__` (distinct orchestrator vs child prompts).
- an `SRLM` subclass with context-length routing, multi-trajectory generation, and
  uncertainty-guided selection (per the Apple SRLM paper).

### Consequences

- Good, because the harness gets first-class child-prompt + SRLM support that
  upstream lacks, and editable install means changes are live without rebuild.
- Bad, because a stale/non-editable install silently reverts behaviour: the symptom
  is `generate.py` dying at task 1 with `TypeError: RLM.__init__() got an unexpected
  keyword argument 'child_system_prompt'` (venv has upstream `rlms`, not the fork).
  Fix the environment (reinstall editable); do **not** patch `generate.py` to drop
  the kwarg (children would inherit the orchestrator prompt).
- Bad, because we carry the maintenance cost of tracking upstream `rlms`.

## More Information

rlm-trainer `CLAUDE.md` (the lm-repl-fork section + verification one-liner).
Verify: `child_system_prompt` is in `inspect.signature(RLM.__init__).parameters`
and `inspect.getsourcefile(RLM)` points under `~/src/lm-repl`.
