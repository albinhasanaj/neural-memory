"""
Topic Tagger — keyword-based topic classification.

Assigns zero or more topic tags to a piece of text using a keyword
dictionary.  Designed to be swapped out for a learned classifier later.
"""

from __future__ import annotations

import re
from typing import Sequence

# ── topic keyword dictionary ─────────────────────────────────────────
# Each key is a topic label; the value is a set of trigger keywords /
# phrases (lowercased).  A topic is assigned if ANY keyword matches.

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "programming": [
        "code", "coding", "program", "function", "class", "variable",
        "compile", "debug", "refactor", "algorithm", "api", "sdk",
        "library", "framework", "git", "repository",
    ],
    "python": [
        "python", "pip", "pypi", "django", "flask", "fastapi", "pytorch",
        "numpy", "pandas", "pydantic", "venv", "conda",
    ],
    "javascript": [
        "javascript", "typescript", "node", "npm", "react", "vue",
        "angular", "next.js", "deno", "bun",
    ],
    "machine_learning": [
        "machine learning", "deep learning", "neural network", "transformer",
        "llm", "gpt", "embedding", "training", "model", "inference",
        "fine-tune", "dataset", "tensor",
    ],
    "web_development": [
        "html", "css", "frontend", "backend", "full-stack", "rest",
        "graphql", "endpoint", "middleware", "router", "server",
    ],
    "databases": [
        "database", "sql", "nosql", "postgres", "mysql", "sqlite",
        "mongodb", "redis", "query", "schema", "migration",
    ],
    "devops": [
        "docker", "kubernetes", "ci/cd", "pipeline", "deploy",
        "container", "aws", "azure", "gcp", "terraform", "ansible",
    ],
    "personal": [
        "prefer", "like", "dislike", "hate", "love", "favourite",
        "favorite", "hobby", "birthday", "family", "friend", "opinion",
    ],
    "work": [
        "project", "deadline", "meeting", "team", "sprint", "task",
        "client", "stakeholder", "roadmap", "retrospective",
    ],
    "learning": [
        "learn", "study", "course", "tutorial", "book", "documentation",
        "understand", "concept", "explain", "teach",
    ],
}

# Pre-compile regex patterns for each topic
_TOPIC_PATTERNS: dict[str, re.Pattern[str]] = {}
for topic, keywords in TOPIC_KEYWORDS.items():
    escaped = [re.escape(k) for k in keywords]
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    _TOPIC_PATTERNS[topic] = re.compile(pattern, re.IGNORECASE)


def extract_topics(text: str, max_topics: int = 5) -> list[str]:
    """Assign topic tags to *text* based on keyword matching.

    Parameters
    ----------
    text:
        The input text to tag.
    max_topics:
        Maximum number of topic tags to return. Topics are returned in
        order of decreasing keyword hit count.

    Returns
    -------
    list[str] — selected topic labels (e.g. ``["python", "machine_learning"]``).
    """
    hits: list[tuple[str, int]] = []
    for topic, pattern in _TOPIC_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            hits.append((topic, len(matches)))

    # Sort by number of keyword hits (desc) and return top N
    hits.sort(key=lambda x: x[1], reverse=True)
    return [topic for topic, _ in hits[:max_topics]]
