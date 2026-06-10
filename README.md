# LM-REPL

**Recursive Language Models with self-reflective program search.**

[![PyPI](https://img.shields.io/pypi/v/lm-repl)](https://pypi.org/project/lm-repl/)
[![Python](https://img.shields.io/pypi/pyversions/lm-repl)](https://pypi.org/project/lm-repl/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`lm-repl` (package import: `lm_repl`) is a fork of [`rlms`](https://github.com/alexzhang13/rlm), the MIT OASYS lab's inference engine for [Recursive Language Models](https://arxiv.org/abs/2512.24601) (RLMs). An RLM replaces the canonical `llm.completion(prompt)` call with `rlm.completion(prompt)`: the context is offloaded into a variable inside a REPL environment, and the model writes programs that slice, search, and recursively query that context instead of attending over it directly.

This fork keeps the upstream engine and layers two things on top:

1. **Map-reduce style orchestration.** Patches that harden the orchestrator-plus-workers pattern: long contexts are chunked and fanned out to parallel batched sub-calls (the map), and the orchestrator aggregates the partial answers (the reduce). The fork adds distinct system prompts for the orchestrator and its workers, per-child iteration budgets, and client fixes needed to drive local OpenAI-compatible servers reliably.
2. **Self-reflective program search (SRLM).** An `SRLM` subclass implementing uncertainty-guided trajectory selection per Apple's [SRLM paper](https://arxiv.org/abs/2603.15653): generate K candidate context-interaction trajectories, then select using the model's own uncertainty signals (self-consistency, verbalized confidence, reasoning trace length) instead of trusting a single rollout. The same paper motivates context-length routing, since recursive decomposition often hurts when the context already fits the model's window.

## Lineage

| Stage | What it contributed |
|-------|---------------------|
| [`rlms` 0.1.1](https://github.com/alexzhang13/rlm) (Zhang, Kraska, Khattab) | The RLM paradigm and engine: REPL environments, recursive sub-calls, parallel `rlm_query_batched`, clients, logging, visualizer |
| Local `rlms` patches | Map-reduce orchestration support: `child_system_prompt` (workers get a different system prompt than the orchestrator), `child_max_iterations`, `max_output_chars` stdout truncation, `default_extra_body` on the OpenAI client, consecutive same-role message merging (required by llama-server), `response_format` pass-through |
| `lm-repl` fork | The `SRLM` subclass: context-length routing, multi-trajectory generation with parallel candidates, and joint uncertainty-guided selection |

## SRLM: uncertainty-guided trajectory selection

The quality of an RLM answer depends heavily on which program trajectory the model happens to sample. `SRLM` subclasses `RLM` and replaces single-rollout inference with search over K candidates:

```python
from lm_repl import SRLM

srlm = SRLM(
    backend="openai",
    backend_kwargs={"model_name": "my-model", "base_url": "http://localhost:8080/v1"},
    direct_threshold=30_000,      # contexts under 30K chars skip the REPL entirely
    n_candidates=4,               # K candidate trajectories
    candidate_parallel=2,         # candidates in flight at once (match server slots)
    candidate_temperature=0.7,    # sampling diversity across candidates
    confidence_elicitation=True,  # elicit per-step {"confidence": N} and use it in selection
)

result = srlm.completion(long_context, "What changed between Q3 and Q4?")
```

How a winner is chosen, per the SRLM paper:

1. **Self-consistency.** Final answers are clustered semantically (normalization plus word-boundary containment, so "42" and "The answer is 42" vote together) and the plurality cluster survives. Tied clusters pool their candidates rather than favoring whichever answer appeared first.
2. **Joint uncertainty score.** Within the surviving set, each trajectory gets `VC(p) * Len(p)`, where `VC` is the sum of log per-step verbalized confidences (steps that skip reporting are imputed with the trajectory mean, so under-reporting cannot inflate the score) and `Len` is the trace length in output tokens. The candidate closest to zero wins. Without `confidence_elicitation`, selection falls back to the shortest trace.

Implementation notes:

- Each candidate runs on a fresh `RLM` instance with its own logger and config copy, so parallel candidates share no mutable state. A crashing candidate is dropped; only if every candidate fails does the call raise.
- `confidence_elicitation=True` appends the reporting instruction to the system prompt automatically; spawned candidates inherit it.
- `direct_threshold` routes short contexts to a plain LLM call. The SRLM paper finds recursive decomposition frequently underperforms the base model within its native window, so set this to roughly the served context size.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `direct_threshold` | `0` (off) | Context length in chars below which the REPL is bypassed |
| `n_candidates` | `1` | Candidate trajectories per completion |
| `candidate_parallel` | `1` | Candidates run concurrently (thread pool) |
| `candidate_temperature` | `None` | Temperature injected into candidate backends |
| `confidence_elicitation` | `False` | Elicit per-step confidence and use VC*Len selection |

All `RLM` constructor arguments pass through unchanged, including `child_system_prompt`.

## Install

Requires **Python 3.11+**. Available on [PyPI](https://pypi.org/project/lm-repl/); note that `pip install rlms` installs the upstream package, not this fork.

```bash
pip install lm-repl
```

For development, install editable from a checkout:

```bash
uv pip install -e /path/to/lm-repl --no-deps
```

Verify you got the fork and not a stale upstream build:

```bash
python -c "import inspect; from lm_repl import RLM, SRLM; print('child_system_prompt' in inspect.signature(RLM.__init__).parameters)"
```

## Quick start

```python
from lm_repl import RLM

rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5-nano"},
    verbose=True,
)

print(rlm.completion("Print me the first 100 powers of two, each on a newline.").response)
```

For the orchestrator/worker split used in map-reduce style runs:

```python
rlm = RLM(
    backend="openai",
    backend_kwargs={...},
    custom_system_prompt=ORCHESTRATOR_PROMPT,   # the root model plans and reduces
    child_system_prompt=WORKER_PROMPT,          # sub-call workers map over chunks
    child_max_iterations=5,
    max_concurrent_subcalls=4,
)
```

## REPL environments

Non-isolated environments run code on the host (fine for benchmarking, not for untrusted prompts); isolated environments run in cloud sandboxes. Natively supported: `local` (default), `ipython`, `docker`, `modal`, `prime`, `daytona`, `e2b`.

```python
rlm = RLM(
    environment="local",
    environment_kwargs={"max_output_chars": 500},
)
```

- **`local`**: in-process `exec` with namespaced globals. `max_output_chars` truncates REPL stdout fed back to the model.
- **`ipython`** (`pip install 'lm-repl[ipython]'`): real IPython session, in-process or in an `ipykernel` subprocess with hard cell timeouts.
- **`docker`**: REPL inside a container (`python:3.11-slim` by default).
- **`modal` / `prime` / `daytona` / `e2b`**: fully isolated cloud sandboxes; sub-calls are proxied back to the host.

## Model providers

OpenAI, Anthropic, OpenRouter, and Portkey clients are included. Local models work through any OpenAI-compatible server (vLLM, llama-server); the fork's `default_extra_body` and same-role message merging exist specifically to make local serving smooth. See `lm_repl/clients/` to add providers.

## Trajectory metadata and logging

`RLMChatCompletion.metadata` holds the full trajectory (run config plus every iteration and sub-call) when a logger is attached. SRLM relies on this for confidence scoring, and spawns per-candidate loggers automatically.

```python
from lm_repl import RLM
from lm_repl.logger import RLMLogger

logger = RLMLogger(log_dir="./logs")   # omit log_dir for in-memory only
rlm = RLM(..., logger=logger)
```

JSONL logs feed the bundled visualizer:

```bash
cd visualizer/
npm run dev   # default localhost:3001
```

## Citations

This fork builds directly on two papers. The engine:

```bibtex
@misc{zhang2026recursivelanguagemodels,
      title={Recursive Language Models},
      author={Alex L. Zhang and Tim Kraska and Omar Khattab},
      year={2026},
      eprint={2512.24601},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24601},
}
```

The selection strategy:

```bibtex
@misc{alizadeh2026srlm,
      title={Recursive Language Models Meet Uncertainty: The Surprising Effectiveness of Self-Reflective Program Search for Long Context},
      author={Keivan Alizadeh and Parshin Shojaee and Minsik Cho and Mehrdad Farajtabar},
      year={2026},
      eprint={2603.15653},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.15653},
}
```

Upstream documentation, blogpost, and minimal implementation: [docs](https://alexzhang13.github.io/rlm/) | [blogpost](https://alexzhang13.github.io/blog/2025/rlm/) | [rlm-minimal](https://github.com/alexzhang13/rlm-minimal).
