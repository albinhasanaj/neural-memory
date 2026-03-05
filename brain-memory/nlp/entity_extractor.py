"""
Named Entity Extraction — regex fallback with optional spaCy upgrade.

Provides a simple regex-based NER that catches common patterns (proper
nouns, programming languages, tools, etc.) and can be transparently
replaced with a spaCy pipeline when the ``nlp`` extra is installed.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

logger = logging.getLogger(__name__)

# ── attempt to load spaCy ────────────────────────────────────────────

_nlp = None
try:
    import spacy

    try:
        _nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy NER loaded (en_core_web_sm).")
    except OSError:
        logger.info("spaCy model 'en_core_web_sm' not found; using regex fallback.")
except ImportError:
    logger.info("spaCy not installed; using regex fallback NER.")


# ── regex fallback patterns ──────────────────────────────────────────

# Capitalised multi-word names (e.g. "New York", "John Smith")
_PROPER_NOUN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

# Single capitalised words that are likely proper nouns (not at sentence start)
_SINGLE_PROPER = re.compile(r"(?<=[.!?]\s)([A-Z][a-z]{2,})\b|(?<=\s)([A-Z][a-z]{2,})\b")

# Tech / programming terms often mentioned as entities
_TECH_TOKENS = re.compile(
    r"\b(Python|JavaScript|TypeScript|Rust|Go|Java|C\+\+|Ruby|PHP|Swift|Kotlin|"
    r"React|Vue|Angular|Django|Flask|FastAPI|Node\.?js|Next\.?js|"
    r"Docker|Kubernetes|AWS|Azure|GCP|GitHub|GitLab|Linux|macOS|Windows|"
    r"PostgreSQL|MySQL|SQLite|MongoDB|Redis|Elasticsearch|"
    r"PyTorch|TensorFlow|Hugging\s?Face|OpenAI|Anthropic|"
    r"WordPress|Shopify|Vercel|Netlify|Supabase|Firebase)\b",
    re.IGNORECASE,
)

# Email-like patterns
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")

# URL-like patterns
_URL = re.compile(r"https?://[^\s,)]+")


def _regex_extract(text: str) -> list[str]:
    """Best-effort entity extraction using regex heuristics."""
    entities: list[str] = []

    for m in _PROPER_NOUN.finditer(text):
        entities.append(m.group(0))

    for m in _TECH_TOKENS.finditer(text):
        entities.append(m.group(0))

    for m in _EMAIL.finditer(text):
        entities.append(m.group(0))

    for m in _URL.finditer(text):
        entities.append(m.group(0))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for e in entities:
        key = e.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(e.strip())
    return unique


def _spacy_extract(text: str) -> list[str]:
    """spaCy-based NER."""
    assert _nlp is not None
    doc = _nlp(text)
    seen: set[str] = set()
    entities: list[str] = []
    for ent in doc.ents:
        key = ent.text.lower()
        if key not in seen:
            seen.add(key)
            entities.append(ent.text)
    return entities


# ── public API ───────────────────────────────────────────────────────


def extract_entities(text: str) -> list[str]:
    """Extract named entities from *text*.

    Uses spaCy when available, else falls back to regex heuristics.

    Returns
    -------
    list[str] — deduplicated entity strings.
    """
    if _nlp is not None:
        return _spacy_extract(text)
    return _regex_extract(text)
