"""SRLM - Self-Reflective Language Model.

Extends RLM with context-length routing, multi-trajectory generation,
and uncertainty-guided selection per the Apple SRLM paper
(arxiv.org/abs/2603.15653).
"""
from __future__ import annotations

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from lm_repl.clients import get_client
from lm_repl.core.rlm import RLM
from lm_repl.core.types import RLMChatCompletion
from lm_repl.logger import RLMLogger

try:  # Optional Prometheus instrumentation; no-op if metrics module unavailable.
    from lm_repl import metrics as _metrics
except ImportError:
    _metrics = None


def _emit_route(mode: str) -> None:
    if _metrics is None:
        return
    try:
        _metrics.srlm_route_total.labels(route="direct" if mode == "direct" else "repl").inc()
    except Exception:
        try:
            _metrics.callback_failures_total.inc()
        except Exception:
            pass


def _emit_candidates_in_flight(n: int) -> None:
    if _metrics is None:
        return
    try:
        _metrics.srlm_candidates_in_flight.set(n)
    except Exception:
        try:
            _metrics.callback_failures_total.inc()
        except Exception:
            pass


def _emit_candidate_outcome(outcome: str) -> None:
    if _metrics is None:
        return
    try:
        _metrics.srlm_candidates_used_total.labels(outcome=outcome).inc()
    except Exception:
        try:
            _metrics.callback_failures_total.inc()
        except Exception:
            pass


def _emit_selection_seconds(seconds: float) -> None:
    if _metrics is None:
        return
    try:
        _metrics.srlm_selection_seconds.observe(seconds)
    except Exception:
        try:
            _metrics.callback_failures_total.inc()
        except Exception:
            pass


# Appended to the orchestrator system prompt when confidence_elicitation is
# on. Wording matches the rlm-trainer teacher suffix (generate.py) that the
# student is trained on. The JSON example is brace-escaped because
# build_rlm_system_prompt .format()s the whole system prompt.
CONFIDENCE_ELICITATION_SUFFIX = (
    "\n\nCONFIDENCE REPORTING (required at end of EVERY response):\n"
    "After your code block, on a NEW line, report your confidence in this "
    "step's correctness:\n"
    '{{"confidence": N}}\n'
    "where N is 0-100. Be precise: 100 = certain, 50 = guessing, "
    "0 = no idea. This line MUST appear after every response you give."
)


def _choose_mode(context_len: int, direct_threshold: int | None) -> str:
    if not direct_threshold or direct_threshold <= 0:
        return "rlm"
    return "direct" if context_len < direct_threshold else "rlm"


def _build_direct_messages(context: str, query: str) -> list[dict]:
    return [
        {"role": "system", "content": "Answer the question using only the provided context. Be concise."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]


class SRLM(RLM):
    """Self-Reflective Language Model.

    Subclasses RLM to add:
    - Context-length routing (direct LLM call for short contexts)
    - Multi-trajectory generation (K candidates)
    - Uncertainty-guided selection (self-consistency + trace length)
    """

    def __init__(
        self,
        *,
        direct_threshold: int = 0,
        n_candidates: int = 1,
        candidate_temperature: float | None = None,
        candidate_parallel: int = 1,
        confidence_elicitation: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.direct_threshold = direct_threshold
        self.n_candidates = n_candidates
        self.candidate_temperature = candidate_temperature
        self.candidate_parallel = candidate_parallel
        self.confidence_elicitation = confidence_elicitation
        if confidence_elicitation:
            self.system_prompt = self.system_prompt + CONFIDENCE_ELICITATION_SUFFIX

    def _direct_completion(self, prompt: str | dict[str, Any]) -> RLMChatCompletion:
        """Bypass REPL - direct LLM chat completion for short contexts."""
        context = prompt if isinstance(prompt, str) else str(prompt)
        root_prompt = ""
        if isinstance(prompt, dict):
            context = prompt.get("context", str(prompt))
            root_prompt = prompt.get("query", "")

        backend_name = (
            self.other_backends[0] if self.other_backends else self.backend
        )
        backend_kw = (
            self.other_backend_kwargs[0] if self.other_backend_kwargs else self.backend_kwargs
        ) or {}
        client = get_client(backend_name, backend_kw)
        messages = _build_direct_messages(context, root_prompt)

        start = time.perf_counter()
        response_text = client.completion(messages)
        elapsed = time.perf_counter() - start

        model_name = backend_kw.get("model_name", "unknown")
        return RLMChatCompletion(
            root_model=model_name,
            prompt=prompt,
            response=response_text,
            usage_summary=client.get_usage_summary(),
            execution_time=elapsed,
            metadata={"mode": "direct"},
        )

    def completion(
        self, prompt: str | dict[str, Any], root_prompt: str | None = None
    ) -> RLMChatCompletion:
        context_str = prompt if isinstance(prompt, str) else str(prompt)
        mode = _choose_mode(len(context_str), self.direct_threshold)
        _emit_route(mode)

        if mode == "direct":
            return self._direct_completion(prompt)

        if self.n_candidates <= 1:
            result = super().completion(prompt, root_prompt)
            if result.metadata is None:
                result.metadata = {}
            if isinstance(result.metadata, dict):
                result.metadata["mode"] = "rlm"
                result.metadata["n_candidates"] = 1
            return result

        return self._multi_trajectory_completion(prompt, root_prompt)

    def _spawn_candidate_rlm(self, index: int) -> RLM:
        """Fresh RLM for one candidate trajectory.

        Each candidate gets its own logger (so trajectory metadata, and thus
        VC scoring, is always captured) and its own copy of backend_kwargs
        with candidate_temperature injected - no shared state is mutated, so
        candidates are safe to run in parallel threads.
        """
        backend_kw = dict(self.backend_kwargs) if self.backend_kwargs else {}
        extra = dict(backend_kw.get("default_extra_body") or {})
        if self.candidate_temperature is not None:
            extra["temperature"] = self.candidate_temperature
        if extra:
            backend_kw["default_extra_body"] = extra

        if self.logger is not None and self.logger.log_dir:
            logger = RLMLogger(log_dir=self.logger.log_dir, file_name=f"candidate_c{index}")
        else:
            logger = RLMLogger()

        return RLM(
            backend=self.backend,
            backend_kwargs=backend_kw,
            environment=self.environment_type,
            environment_kwargs=dict(self.environment_kwargs),
            depth=self.depth,
            max_depth=self.max_depth,
            max_iterations=self.max_iterations,
            max_budget=self.max_budget,
            max_timeout=self.max_timeout,
            max_tokens=self.max_tokens,
            max_decode_tokens=self.max_decode_tokens,
            max_errors=self.max_errors,
            # A candidate is a full root-orchestrator clone of this SRLM (it
            # replaces the K=1 super().completion path), so EVERY caller-set
            # RLM-level guard must follow it. Omitting any of these let a K>1
            # trajectory search run looser than the same orchestrator at K=1
            # (rlm-trainer #9). Keep this list in sync with the recursion-child
            # construction in rlm.py RLM._subcall (the other hand-maintained
            # forwarding site); tests/test_subcall_guards.py
            # test_srlm_candidate_inherits_all_caller_guards is the regression
            # guard if a new guard is added and forgotten here.
            root_max_tokens=self.root_max_tokens,
            subcall_max_tokens=self.subcall_max_tokens,
            subcall_max_timeout=self.subcall_max_timeout,
            subcall_extra_body=self.subcall_extra_body,
            # Shared instances, not copies: resubmission memory / veto telemetry
            # and answer-acceptance behavior must match the K=1 orchestrator.
            subcall_verifier=self.subcall_verifier,
            answer_verifier=self.answer_verifier,
            clean_retry_on_error=self.clean_retry_on_error,
            max_answer_retries=self.max_answer_retries,
            soft_timeout_pct=self.soft_timeout_pct,
            soft_timeout_message=self.soft_timeout_message,
            scheduler_max_concurrent=self.scheduler_max_concurrent,
            scheduler_aging_interval=self.scheduler_aging_interval,
            scheduler_coordination_dir=self.scheduler_coordination_dir,
            custom_system_prompt=self.system_prompt,
            other_backends=self.other_backends,
            other_backend_kwargs=self.other_backend_kwargs,
            logger=logger,
            verbose=self.verbose.enabled,
            custom_tools=self.custom_tools,
            custom_sub_tools=self.custom_sub_tools,
            compaction=self.compaction,
            compaction_threshold_pct=self.compaction_threshold_pct,
            max_concurrent_subcalls=self.max_concurrent_subcalls,
            on_subcall_start=self.on_subcall_start,
            on_subcall_complete=self.on_subcall_complete,
            on_iteration_start=self.on_iteration_start,
            on_iteration_complete=self.on_iteration_complete,
            child_max_iterations=self.child_max_iterations,
            child_system_prompt=self.child_system_prompt,
        )

    def _multi_trajectory_completion(
        self, prompt: str | dict[str, Any], root_prompt: str | None = None
    ) -> RLMChatCompletion:
        """Generate K candidates and select the best by uncertainty signals.

        Candidates run on fresh per-candidate RLM instances, in parallel when
        candidate_parallel > 1. A failing candidate is dropped; only if every
        candidate fails is the last error raised.
        """
        spawned = [self._spawn_candidate_rlm(i) for i in range(self.n_candidates)]
        _emit_candidates_in_flight(self.n_candidates)

        def _run(cand: RLM) -> RLMChatCompletion | Exception:
            try:
                return cand.completion(prompt, root_prompt)
            except Exception as exc:  # noqa: BLE001 - candidate isolation
                return exc

        try:
            if self.candidate_parallel > 1:
                workers = min(self.candidate_parallel, self.n_candidates)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    outcomes = list(pool.map(_run, spawned))
            else:
                outcomes = [_run(cand) for cand in spawned]
        finally:
            _emit_candidates_in_flight(0)

        for o in outcomes:
            _emit_candidate_outcome("success" if isinstance(o, RLMChatCompletion) else "error")

        candidates = [o for o in outcomes if isinstance(o, RLMChatCompletion)]
        if not candidates:
            errors = [o for o in outcomes if isinstance(o, Exception)]
            raise errors[-1] if errors else RuntimeError("no candidate produced a completion")

        _sel_t0 = time.perf_counter()
        best = _select_best(candidates, use_confidence=self.confidence_elicitation)
        _emit_selection_seconds(time.perf_counter() - _sel_t0)
        if best.metadata is None:
            best.metadata = {}
        if isinstance(best.metadata, dict):
            best.metadata["mode"] = "rlm"
            best.metadata["n_candidates"] = self.n_candidates
            best.metadata["n_completed"] = len(candidates)
        return best


def _parse_confidence_scores(text: str) -> list[float]:
    """Extract {"confidence": N} values from model response text."""
    pattern = r'\{\s*"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*\}'
    matches = re.findall(pattern, text)
    return [min(float(m), 100.0) for m in matches]


def _extract_step_texts(metadata: dict | None) -> list[str]:
    """Per-step response texts from RLMChatCompletion.metadata.

    RLM attaches logger.get_trajectory(): {"run_metadata": ..., "iterations":
    [{"response": ..., ...}, ...]}. Also accepts a legacy {"trajectory_text":
    str} blob as a single step.
    """
    if not isinstance(metadata, dict):
        return []
    iterations = metadata.get("iterations")
    if isinstance(iterations, list) and iterations:
        return [str(it.get("response", "") or "") for it in iterations]
    legacy = metadata.get("trajectory_text")
    if legacy:
        return [str(legacy)]
    return []


def _compute_vc_score(step_texts: list[str]) -> float:
    """Verbalized confidence score: sum over steps of log(confidence/100).

    Uses the last reported confidence in each step. Steps with no report are
    imputed with the mean of the reported steps (per the SRLM paper), so a
    trajectory cannot inflate its score by skipping reports.

    Returns a value <= 0; closer to 0 = higher confidence. Returns -inf when
    no step reports any confidence, or a reported confidence is 0.
    """
    per_step: list[float | None] = []
    for text in step_texts:
        scores = _parse_confidence_scores(text)
        per_step.append(scores[-1] if scores else None)

    reported = [v for v in per_step if v is not None]
    if not reported:
        return float('-inf')
    mean = sum(reported) / len(reported)

    total = 0.0
    for v in per_step:
        value = v if v is not None else mean
        if value <= 0:
            return float('-inf')
        total += math.log(value / 100.0)
    return total


_NUMERIC_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d*)\.?\d+")


def _normalize_answer(s: str) -> str:
    """Canonical form of a final answer for self-consistency comparison."""
    s = re.sub(r"\s+", " ", s.strip().lower())
    s = s.strip("\"'“”‘’")
    s = s.rstrip(".!?,;: ").strip()
    if _NUMERIC_RE.fullmatch(s):
        try:
            return f"{float(s.replace(',', '')):g}"
        except ValueError:
            pass
    return s


def _answers_equivalent(a: str, b: str) -> bool:
    """Whether two normalized answers express the same final answer.

    Equal strings, or the shorter answer appearing in the longer one on word
    boundaries (so "42" matches "the answer is 42" but not "417").
    """
    if a == b:
        return True
    if not a or not b:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return re.search(rf"(?<!\w){re.escape(shorter)}(?!\w)", longer) is not None


def _cluster_answers(answers: list[str]) -> list[list[int]]:
    """Greedy clustering of normalized answers into equivalence groups.

    Each answer joins the first cluster whose representative (first member)
    it is equivalent to, else starts a new cluster. Returns index groups.
    """
    clusters: list[tuple[str, list[int]]] = []
    for i, a in enumerate(answers):
        for rep, members in clusters:
            if _answers_equivalent(rep, a):
                members.append(i)
                break
        else:
            clusters.append((a, [i]))
    return [members for _, members in clusters]


def _trace_len(c: RLMChatCompletion) -> float:
    """Len(p): trace length in output tokens, per the paper.

    Falls back to execution_time when the backend reported no token usage.
    Tokens are preferred because wall clock is confounded by prefix-cache
    hits and server slot contention.
    """
    tokens = c.usage_summary.total_output_tokens if c.usage_summary else 0
    return float(tokens) if tokens and tokens > 0 else c.execution_time


def _select_best(
    candidates: list[RLMChatCompletion], use_confidence: bool = False
) -> RLMChatCompletion:
    """Select the best candidate using self-consistency + uncertainty signals.

    1. Plurality vote on the final answer (self-consistency), with answers
       compared semantically (normalization + word-boundary containment)
       rather than by exact string match. Tied clusters (including the
       all-unique case) pool their candidates for the scoring stage instead
       of arbitrarily preferring the first-seen answer.
    2. Among the consistent set:
       - If use_confidence: joint score VC(p) * Len(p), pick argmax (closest to 0)
       - Otherwise: pick shortest trace (output tokens, else execution time)
    """
    if len(candidates) == 1:
        return candidates[0]

    answers = [_normalize_answer(c.response) for c in candidates]
    clusters = _cluster_answers(answers)
    max_size = max(len(members) for members in clusters)
    consistent = [
        candidates[i]
        for members in clusters
        if len(members) == max_size
        for i in members
    ]

    if not use_confidence:
        return min(consistent, key=_trace_len)

    def _joint_score(c: RLMChatCompletion) -> float:
        vc = _compute_vc_score(_extract_step_texts(c.metadata))
        if vc == float('-inf'):
            return float('-inf')
        return vc * _trace_len(c)

    scored = [(c, _joint_score(c)) for c in consistent]
    has_scores = any(s != float('-inf') for _, s in scored)

    if not has_scores:
        return min(consistent, key=_trace_len)

    return max(scored, key=lambda x: x[1])[0]
