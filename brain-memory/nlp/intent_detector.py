"""
Intent Cue Detector — recognise recall-intent signals in user text.

Detects phrases like "remember when …", "last time …", "what was my …",
"you told me …" that signal the user is trying to *retrieve* from memory
rather than state new information.  Returns both a confidence flag and any
noun-phrase targets extracted from the cue (e.g. "my API key" from
"what was my API key?").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence


# ── compiled patterns ────────────────────────────────────────────────
# Each pattern has a named group ``target`` that captures the noun phrase
# the user is recalling.  Not all patterns yield a target — that's fine;
# the presence of the cue itself is the primary signal.

_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    # "remember when …" / "do you remember …"
    (re.compile(
        r"\b(?:do\s+you\s+)?remember\s+(?:when|that|how)\s+(?P<target>.+?)(?:\?|$)",
        re.IGNORECASE,
    ), 1.0),

    # "last time …" / "the other day …" / "earlier …"
    (re.compile(
        r"\b(?:last\s+time|the\s+other\s+day|earlier|previously|before)\b.*?(?:(?:we|I|you)\s+(?P<target>.+?))?(?:[.?!]|$)",
        re.IGNORECASE,
    ), 0.9),

    # "what was my …" / "what is my …" / "what's my …"
    (re.compile(
        r"\bwhat(?:'s|\s+is|\s+was)\s+my\s+(?P<target>.+?)(?:\?|$)",
        re.IGNORECASE,
    ), 1.0),

    # "what did I …" / "what did we …"
    (re.compile(
        r"\bwhat\s+did\s+(?:I|we)\s+(?P<target>.+?)(?:\?|$)",
        re.IGNORECASE,
    ), 0.9),

    # "you told me …" / "you mentioned …" / "you said …"
    (re.compile(
        r"\byou\s+(?:told\s+me|mentioned|said|recommended|suggested)\s+(?P<target>.+?)(?:[.?!]|$)",
        re.IGNORECASE,
    ), 0.95),

    # "my … was …" — possessive recall
    (re.compile(
        r"\bmy\s+(?P<target>\w[\w\s]{0,40}?)(?:\s+was|\s+is|\s+were)\b",
        re.IGNORECASE,
    ), 0.7),

    # "didn't I …" / "haven't we …"
    (re.compile(
        r"\b(?:didn't|haven't|don't)\s+(?:I|we)\s+(?P<target>.+?)(?:\?|$)",
        re.IGNORECASE,
    ), 0.85),

    # "can you recall …" / "can you look up …"
    (re.compile(
        r"\bcan\s+you\s+(?:recall|look\s+up|find|check)\s+(?P<target>.+?)(?:\?|$)",
        re.IGNORECASE,
    ), 0.9),

    # "go back to …" / "return to …"
    (re.compile(
        r"\b(?:go\s+back\s+to|return\s+to|revisit)\s+(?P<target>.+?)(?:[.?!]|$)",
        re.IGNORECASE,
    ), 0.8),

    # "remind me …"
    (re.compile(
        r"\bremind\s+me\s+(?:about|of|what)\s+(?P<target>.+?)(?:[.?!]|$)",
        re.IGNORECASE,
    ), 1.0),
]


# ── result dataclass ────────────────────────────────────────────────


@dataclass
class IntentCueResult:
    """Result of intent-cue detection."""

    is_recall: bool = False
    """Whether the text contains a recall-intent cue."""

    confidence: float = 0.0
    """Highest pattern confidence that matched (0.0–1.0)."""

    targets: list[str] = field(default_factory=list)
    """Noun-phrase targets extracted from matched cue patterns.

    E.g. for "what was my API key?" → ``["API key"]``.
    """

    matched_patterns: list[str] = field(default_factory=list)
    """The regex pattern strings that fired (for debugging)."""


# ── public API ───────────────────────────────────────────────────────


def detect_intent_cues(text: str) -> IntentCueResult:
    """Scan *text* for recall-intent cues.

    Returns an ``IntentCueResult`` with flags and extracted targets.
    """
    result = IntentCueResult()
    seen_targets: set[str] = set()

    for pattern, confidence in _PATTERNS:
        m = pattern.search(text)
        if m is None:
            continue

        result.is_recall = True
        result.confidence = max(result.confidence, confidence)
        result.matched_patterns.append(pattern.pattern[:60])

        target = m.groupdict().get("target", "")
        if target:
            target = target.strip().rstrip("?.!,")
            if target and target.lower() not in seen_targets:
                seen_targets.add(target.lower())
                result.targets.append(target)

    return result
