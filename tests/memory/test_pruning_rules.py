"""Tests for the prehend anti-give-up write-time filter."""
from __future__ import annotations

import pytest

from prehend.memory.pruning_rules import is_anti_give_up


@pytest.mark.parametrize("text", [
    "data not available, return unknown",
    "Insufficient data to answer.",
    "Cannot determine the result.",
    "no data available for this query",
])
def test_capitulation_text_is_flagged(text):
    assert is_anti_give_up(text) is True


@pytest.mark.parametrize("text", [
    "first compute the ratio, then divide by the base",
    "When the value is missing, retry with a wider window before concluding.",
    "Do not give up; re-read the context from the start.",
])
def test_useful_experience_is_not_flagged(text):
    assert is_anti_give_up(text) is False


def test_protective_override_beats_capitulation_phrase():
    # Mentions "data not available" but instructs to retry first -> keep it.
    text = "If data not available, retry with a different parameter before concluding."
    assert is_anti_give_up(text) is False
