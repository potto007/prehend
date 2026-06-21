"""Tests for the strategy verifier: the optimizer layer that reviews
llm_query / rlm_query calls before they execute (spec:
docs/superpowers/specs/2026-06-11-strategy-verifier-design.md)."""

from unittest.mock import Mock, patch

import prehend.core.rlm as rlm_module
from prehend import RLM
from prehend.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from prehend.core.lm_handler import LMHandler
from prehend.core.verifier import (
    LMVerifier,
    RuleVerifier,
    SubcallReview,
    TieredVerifier,
    Verdict,
)
from tests.mock_lm import MockLM
from tests.test_subcall import create_mock_lm, final

ROOT = (
    "A Colorado behavioral-health provider organization discovered it billed "
    "Cigna and Colorado Medicaid for outpatient counseling services rendered "
    "by a counselor before that counselor received their license. Please "
    "answer two things grounded in source documents: remediation steps and "
    "consequences the provider faces, citing specific corpus documents."
)


def review(prompt, kind="rlm_query", root=ROOT, depth=0):
    return SubcallReview(kind=kind, prompt=prompt, root_prompt=root, depth=depth)


# ---------------------------------------------------------------------------
# RuleVerifier: whole-task delegation
# ---------------------------------------------------------------------------


class TestRuleVerifier:
    def test_vetoes_prompt_embedding_entire_root_task(self):
        prompt = f"Analyze the context and answer the following:\n{ROOT}\nBe thorough."
        verdict = RuleVerifier().review(review(prompt))
        assert not verdict.approved
        assert "task" in verdict.reason.lower()

    def test_vetoes_lightly_rephrased_whole_task(self):
        # Same content, different framing and whitespace: shingle overlap
        # must catch what exact containment misses.
        rephrased = ROOT.replace("Please answer two things", "Answer these two items")
        prompt = f"Deep analysis required. {rephrased} Use rigorous reasoning."
        verdict = RuleVerifier().review(review(prompt))
        assert not verdict.approved

    def test_approves_decomposed_subtask(self):
        prompt = (
            "Summarize the refund-obligation rules in this excerpt of doc 014: "
            "<2000 chars of document text>. One paragraph."
        )
        assert RuleVerifier().review(review(prompt)).approved

    def test_approves_when_root_unknown(self):
        verdict = RuleVerifier().review(review("anything at all", root=None))
        assert verdict.approved

    def test_short_root_never_triggers_containment(self):
        # A terse root question legitimately appears inside slice prompts.
        verdict = RuleVerifier().review(
            review("Given this doc slice <...>, answer: why are claims denied?",
                   root="why are claims denied?")
        )
        assert verdict.approved

    def test_llm_query_quoting_whole_root_is_approved(self):
        """Manifest triage legitimately quotes the full question next to the
        catalog ('pick relevant doc ids for: <question>'). llm_query is
        already bounded by subcall_max_tokens; only rlm_query (an unbounded
        child re-running the task) gets the whole-task veto."""
        prompt = f"Pick the most relevant doc ids from this manifest for:\n{ROOT}\n<manifest>"
        verdict = RuleVerifier().review(review(prompt, kind="llm_query"))
        assert verdict.approved

    def test_rlm_query_after_vetoed_same_prompt_as_llm_query_is_fresh(self):
        """Downgrading a vetoed rlm_query to a capped llm_query is compliance,
        not resubmission: veto memory is scoped per call kind."""
        tiered = TieredVerifier(rules=RuleVerifier())
        prompt = f"Analyze: {ROOT}"
        assert not tiered.review(review(prompt, kind="rlm_query")).approved
        assert tiered.review(review(prompt, kind="llm_query")).approved


# ---------------------------------------------------------------------------
# LMVerifier
# ---------------------------------------------------------------------------


class TestLMVerifier:
    def _verifier(self, lm_text):
        client = Mock()
        client.completion.return_value = lm_text
        v = LMVerifier(backend="openai", backend_kwargs={"model_name": "m"})
        v._client = client  # bypass lazy client creation
        return v, client

    def test_parses_approval(self):
        v, _ = self._verifier('{"approve": true, "reason": "well scoped"}')
        assert v.review(review("subtask")).approved

    def test_parses_rejection_with_reason(self):
        v, _ = self._verifier('{"approve": false, "reason": "re-asks the root task"}')
        verdict = v.review(review("subtask"))
        assert not verdict.approved
        assert "re-asks the root task" in verdict.reason

    def test_fails_open_on_unparseable_output(self):
        v, _ = self._verifier("I think this is probably fine, hard to say.")
        assert v.review(review("subtask")).approved

    def test_fails_open_on_lm_error(self):
        v, client = self._verifier("")
        client.completion.side_effect = RuntimeError("server down")
        assert v.review(review("subtask")).approved

    def test_verifier_output_is_token_capped(self):
        v, client = self._verifier('{"approve": true, "reason": "ok"}')
        v.review(review("subtask"))
        assert client.completion.call_args.kwargs.get("max_tokens") == 256


# ---------------------------------------------------------------------------
# TieredVerifier: composition, resubmission escalation, telemetry
# ---------------------------------------------------------------------------


def _approving_lm():
    lm = Mock(spec=["review"])
    lm.review.return_value = Verdict(approved=True)
    return lm


class TestTieredVerifier:
    def test_rules_run_before_lm(self):
        lm = _approving_lm()
        tiered = TieredVerifier(rules=RuleVerifier(), lm=lm)
        prompt = f"Do everything: {ROOT}"
        assert not tiered.review(review(prompt)).approved
        lm.review.assert_not_called()

    def test_lm_reviews_rlm_query_only(self):
        lm = _approving_lm()
        tiered = TieredVerifier(rules=RuleVerifier(), lm=lm)
        tiered.review(review("summarize this slice", kind="llm_query"))
        lm.review.assert_not_called()
        tiered.review(review("solve this decomposed subproblem", kind="rlm_query"))
        lm.review.assert_called_once()

    def test_resubmission_of_vetoed_prompt_escalates(self):
        tiered = TieredVerifier(rules=RuleVerifier())
        prompt = f"Analyze: {ROOT}"
        first = tiered.review(review(prompt))
        second = tiered.review(review(prompt))
        assert not first.approved and not second.approved
        assert first.reason != second.reason
        assert "must" in second.reason.lower()

    def test_resubmission_check_covers_lm_vetoes_without_rereview(self):
        lm = Mock(spec=["review"])
        lm.review.return_value = Verdict(approved=False, reason="too vague")
        tiered = TieredVerifier(rules=RuleVerifier(), lm=lm)
        tiered.review(review("do the thing"))
        tiered.review(review("do the thing"))
        assert lm.review.call_count == 1  # second attempt short-circuits

    def test_whitespace_variation_still_counts_as_resubmission(self):
        tiered = TieredVerifier(rules=RuleVerifier())
        tiered.review(review(f"Analyze:  {ROOT}"))
        verdict = tiered.review(review(f"analyze: {ROOT} "))
        assert "must" in verdict.reason.lower()

    def test_vetoes_are_recorded_for_telemetry(self):
        tiered = TieredVerifier(rules=RuleVerifier())
        prompt = f"Handle it all: {ROOT}"
        tiered.review(review(prompt))
        tiered.review(review(prompt))
        assert len(tiered.vetoes) == 2
        assert tiered.vetoes[0]["kind"] == "rlm_query"
        assert tiered.vetoes[1]["attempt"] == 2
        assert all("reason" in v and "prompt_preview" in v for v in tiered.vetoes)

    def test_approval_passes_through_and_records_nothing(self):
        tiered = TieredVerifier(rules=RuleVerifier(), lm=_approving_lm())
        assert tiered.review(review("summarize doc 003 section 2")).approved
        assert tiered.vetoes == []


# ---------------------------------------------------------------------------
# RLM wiring: rlm_query path (_subcall)
# ---------------------------------------------------------------------------


class TestRLMWiring:
    def _parent(self, mock_get_client, **kwargs):
        mock_get_client.return_value = create_mock_lm([final("answer")])
        return RLM(
            backend="openai",
            backend_kwargs={"model_name": "parent-model"},
            max_depth=3,
            **kwargs,
        )

    def test_subcall_veto_blocks_child_spawn(self):
        captured = {}

        class CapturingRLM(rlm_module.RLM):
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            parent = self._parent(
                mock_get_client, subcall_verifier=TieredVerifier(rules=RuleVerifier())
            )
            parent._verifier_root = ROOT
            with patch.object(rlm_module, "RLM", CapturingRLM):
                result = parent._subcall(f"Analyze thoroughly: {ROOT}")
            assert "Strategy verifier rejected" in result.response
            assert captured == {}  # the child was never constructed
            parent.close()

    def test_subcall_propagates_verifier_instance_to_child(self):
        captured = {}

        class CapturingRLM(rlm_module.RLM):
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                super().__init__(*args, **kwargs)

        verifier = TieredVerifier(rules=RuleVerifier())
        with patch.object(rlm_module, "get_client") as mock_get_client:
            parent = self._parent(mock_get_client, subcall_verifier=verifier)
            parent._verifier_root = ROOT
            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("summarize the refund rules excerpt from doc 014")
            assert captured.get("subcall_verifier") is verifier
            parent.close()

    def test_completion_records_root_prompt_for_review(self):
        with patch.object(rlm_module, "get_client") as mock_get_client:
            parent = self._parent(mock_get_client)
            parent.completion("a long context string", root_prompt="the question")
            assert parent._verifier_root == "the question"
            parent.close()

    def test_completion_falls_back_to_prompt_as_root(self):
        with patch.object(rlm_module, "get_client") as mock_get_client:
            parent = self._parent(mock_get_client)
            parent.completion("just one combined prompt")
            assert parent._verifier_root == "just one combined prompt"
            parent.close()


# ---------------------------------------------------------------------------
# LMHandler wiring: llm_query path
# ---------------------------------------------------------------------------


class _VetoMarked:
    """Stub verifier: vetoes any prompt containing 'VETO-ME'. Wiring tests
    use this so they exercise plumbing, not rule policy."""

    def review(self, call):
        if "VETO-ME" in call.prompt:
            return Verdict(approved=False, reason="marked for veto")
        return Verdict(approved=True)


class TestHandlerWiring:
    def test_vetoed_llm_query_never_reaches_the_model(self):
        mock = MockLM(responses=["should never be used"])
        with LMHandler(client=mock, verifier=_VetoMarked(), verifier_root=ROOT) as handler:
            resp = send_lm_request(
                handler.address, LMRequest(prompt="VETO-ME: do everything")
            )
        assert resp.success
        assert "Strategy verifier rejected" in resp.chat_completion.response
        assert mock.seen_max_tokens == []  # model never called

    def test_approved_llm_query_executes_normally(self):
        mock = MockLM(responses=["the summary"])
        with LMHandler(client=mock, verifier=_VetoMarked(), verifier_root=ROOT) as handler:
            resp = send_lm_request(
                handler.address, LMRequest(prompt="summarize this doc slice: <text>")
            )
        assert resp.chat_completion.response == "the summary"

    def test_batched_review_is_per_prompt(self):
        mock = MockLM(responses=["fine"])
        with LMHandler(client=mock, verifier=_VetoMarked(), verifier_root=ROOT) as handler:
            resp = send_lm_request_batched(
                handler.address,
                ["VETO-ME: handle the whole thing", "summarize slice 2 of doc 014"],
            )
        responses = [r.chat_completion.response for r in resp]
        assert "Strategy verifier rejected" in responses[0]
        assert responses[1] == "fine"

    def test_no_verifier_means_no_review(self):
        mock = MockLM(responses=["ok"])
        with LMHandler(client=mock) as handler:
            resp = send_lm_request(
                handler.address, LMRequest(prompt=f"Answer everything: {ROOT}")
            )
        assert resp.chat_completion.response == "ok"
