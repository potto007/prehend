"""Trace-based experience distillation.

Turns a completed solve (an prehend ``RLMChatCompletion``) into a bank entry by
reflecting on the actual trajectory -- the cheap, honest distiller from the
prehend plan (ADR-0005), distinct from FinAcumen's heavy re-solve cross-verify.

A reflect function (prompt -> raw LLM text) and an embedding backend are
injected, so the distiller stays decoupled from any specific client and is
unit-testable without network access.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from prehend.memory.embed import EmbeddingBackend
from prehend.memory.pruning_rules import is_anti_give_up, is_premature_stop

ReflectFn = Callable[[str], str]

_VALID_POLARITY = {"positive", "negative"}

REFLECT_PROMPT = """\
You are distilling a reusable lesson from a solved problem so a future agent can
reuse it. Read the question, the final answer, and the solving trajectory, then
return ONLY a JSON object with these keys:
  "polarity": "positive" (a guiding-path recipe that worked) or "negative"
              (a guard rule from a mistake, phrased "When <condition>, <action>").
  "key_insight": one distilled, reusable sentence (50-70 words max).
  "findings": list of short reusable strategy strings (may be empty).
  "cautions": list of short guard-rule strings (may be empty).
Do not restate the specific numbers; generalize to the problem shape.

<Question>
{question}
</Question>
<FinalAnswer>
{answer}
</FinalAnswer>
<Trajectory>
{trajectory}
</Trajectory>"""


FAILURE_REFLECT_PROMPT = """\
This solve attempt was INCORRECT. Distill the single most useful CORRECTIVE lesson
a future agent should apply to AVOID this failure, so it does better next time.
Return ONLY a JSON object with these keys:
  "key_insight": one corrective guard rule phrased "When <condition>, <action>"
                 (50-70 words max).
  "findings": list of short corrective strategy strings (may be empty).
  "cautions": list of short guard-rule strings (may be empty).
The corrective <action> MUST be something to do ADDITIONALLY or DIFFERENTLY -- e.g.
re-read, cross-check across all chunks, decompose differently, verify each
sub-result, widen the search, increase overlap. It must NOT be to stop early,
accept a partial / first / best-available / most-frequent answer, prefer the first
match, or narrow scope to save time: those reproduce the shallow-search-then-give-up
failure this lesson exists to prevent. Do NOT produce a strategy to imitate. Do NOT
conclude the data or information is missing, absent, unavailable, garbled, or
unreadable -- that is capitulation, not a lesson. Generalize to the problem shape;
do not restate specific numbers.

<Question>
{question}
</Question>
<FinalAnswer>
{answer}
</FinalAnswer>
<Trajectory>
{trajectory}
</Trajectory>"""


def _blocked(text: str, failed: bool) -> bool:
    """Write-time content guard. Capitulation is blocked on both channels; the
    behavioral premature-stop guard is blocked ONLY on the failure channel (it
    would drop legitimate positive recipes on the success path)."""
    if is_anti_give_up(text):
        return True
    if failed and is_premature_stop(text):
        return True
    return False


class TraceDistiller:
    """Callable ``(question, context, result) -> entry dict | None``."""

    def __init__(
        self,
        reflect_fn: ReflectFn,
        backend: EmbeddingBackend,
        *,
        source: str = "prehend",
        trajectory_cap: int = 6000,
    ) -> None:
        self.reflect_fn = reflect_fn
        self.backend = backend
        self.source = source
        self.trajectory_cap = trajectory_cap

    def __call__(
        self, question: str, context: str, result: Any, *, failed: bool = False
    ) -> dict | None:
        answer = self._final_answer(result)
        trajectory = self._trajectory(result)[: self.trajectory_cap]

        template = FAILURE_REFLECT_PROMPT if failed else REFLECT_PROMPT
        prompt = template.format(
            question=question, answer=answer, trajectory=trajectory
        )
        raw = self.reflect_fn(prompt) or ""
        parsed = _parse_json(raw)
        if not parsed:
            return None

        key_insight = str(parsed.get("key_insight", "")).strip()
        findings = [str(f) for f in (parsed.get("findings") or []) if not _blocked(str(f), failed)]
        cautions = [str(c) for c in (parsed.get("cautions") or []) if not _blocked(str(c), failed)]

        insight_ok = bool(key_insight) and not _blocked(key_insight, failed)
        if not insight_ok and not findings and not cautions:
            return None
        if not insight_ok:
            key_insight = ""

        # A failure lesson is ALWAYS a negative guard rule (never an imitable
        # recipe); the success path keeps the model's polarity.
        if failed:
            polarity = "negative"
        else:
            polarity = str(parsed.get("polarity", "")).strip().lower()
            if polarity not in _VALID_POLARITY:
                polarity = "positive"

        return {
            "id": _entry_id(question, "failure" if failed else "success"),
            "polarity": polarity,
            "key_insight": key_insight,
            "findings": findings,
            "cautions": cautions,
            "derived_from": "failure" if failed else "success",
            # Embed the bare question: retrieval embeds the bare query
            # (harness.retrieve(question)), so the stored key MUST match it.
            # Folding the offloaded context in here dilutes the key and breaks
            # paraphrase retrieval as context grows.
            "embedding": list(self.backend.embed(question)),
            "stats": {"use_count": 0, "hit_count": 0},
            "source": self.source,
            "question": question,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }

    @staticmethod
    def _final_answer(result: Any) -> str:
        for attr in ("response", "final_answer"):
            val = getattr(result, attr, None)
            if val:
                return str(val)
        return str(result)

    @staticmethod
    def _trajectory(result: Any) -> str:
        meta = getattr(result, "metadata", None)
        if meta:
            try:
                return json.dumps(meta, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(meta)
        return ""


def _entry_id(question: str, derived_from: str = "success") -> str:
    # Key on (question, provenance) so a failure guard and a success recipe for
    # the SAME question get distinct ids and COEXIST (ADR-0011 amendment). With a
    # question-only id they collided and the harness "success shadows failure"
    # rule dropped every negative, so the contrastive channel never fired on
    # context-varying task sets (plain multihop: 60 tasks, only 5 questions).
    # The NUL separator can't occur in normal question text, so distinct
    # (question, derived_from) pairs never alias.
    digest = hashlib.sha1(f"{question}\x00{derived_from}".encode("utf-8")).hexdigest()[:12]
    return f"exp_{digest}"


def _parse_json(raw: str) -> dict | None:
    """Extract the first JSON object from possibly-fenced, prose-wrapped text."""
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
