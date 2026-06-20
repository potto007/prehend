"""Write-time content guards for the mnemex bank.

``is_anti_give_up`` blocks experiences that codify capitulation ("data not
available", "cannot determine") from being learned, UNLESS the text is actually
a protective guard rule (e.g. "retry before concluding"), which is exactly the
kind of negative-polarity lesson worth keeping.

Generalized from FinAcumen's ``finacumen/fm/pruning_rules.py`` (finance-specific
protective patterns about tickers/lookups dropped).
"""
from __future__ import annotations

import re

ANTI_GIVE_UP_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bdata\s+not\s+available\b",
        r"\bdata\s+not\s+provided\b",
        r"\bdata\s+is\s+missing\b",
        r"\bdata\s+unavailable\b",
        r"\binsufficient\s+data\b",
        r"\bcannot\s+determine\b",
        r"\bno\s+data\s+available\b",
        r"\bunable\s+to\s+find\b",
        r"\breturn\s+.*\b(unavailable|unknown)\b",
        r"\bconclude\s+.*\b(unavailable|unknown|missing)\b",
        r"\bstate\s+.*\b(unavailable|unknown)\b",
    ]
]

PROTECTIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bretry\b.*\bbefore\b.*\bconcluding\b",
        r"\bretry\b.*\bwith\b.*\b(different|alternate|wider|relaxed)\b",
        r"\bverify\b.*\bdata\b.*\b(before|first)\b",
        r"\bdo\s+not\b.*\b(give\s*up|conclude|assume)\b",
        r"\bnever\b.*\b(give\s*up|conclude|assume)\b",
        r"\bre.?read\b",
        r"\bwiden\b.*\b(range|window)\b",
        r"\bproxy\b.*\bmetric\b",
    ]
]


def is_anti_give_up(text: str) -> bool:
    """True if ``text`` encodes a capitulation directive worth blocking.

    A protective directive (retry/verify/re-read before concluding) overrides
    the capitulation match, so genuine guard rules are kept.
    """
    for p in PROTECTIVE_PATTERNS:
        if p.search(text):
            return False
    for p in ANTI_GIVE_UP_PATTERNS:
        if p.search(text):
            return True
    return False
