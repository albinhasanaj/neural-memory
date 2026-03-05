"""
Shared pytest fixtures for brain-memory tests.

Provides sample episodes, a pre-built semantic graph, and a mock
embedding encoder that returns deterministic vectors without loading
a real model.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Generator

import pytest
import torch
from torch import Tensor

from config.settings import settings
from memory.episodic import EpisodicEntry, EpisodicStore
from memory.semantic import SemanticEdge, SemanticGraph, SemanticNode


# ────────────────────────────────────────────────────────────────────
# Mock encoder
# ────────────────────────────────────────────────────────────────────


class MockEncoder:
    """Deterministic mock for ``EmbeddingEncoder``.

    Produces a fixed-dim vector whose first element is the hash of
    the input text (modulo some range), so identical texts always
    produce the same embedding.
    """

    def __init__(self, dim: int = settings.embedding_dim) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> Tensor:
        torch.manual_seed(hash(text) % (2**31))
        return torch.randn(self._dim)

    def encode_batch(self, texts: list[str]) -> Tensor:
        return torch.stack([self.encode(t) for t in texts])


@pytest.fixture
def mock_encoder() -> MockEncoder:
    """Return a deterministic mock encoder."""
    return MockEncoder()


# ────────────────────────────────────────────────────────────────────
# Sample episodes
# ────────────────────────────────────────────────────────────────────


def _make_episode(
    text: str,
    speaker: str = "user",
    salience: float = 0.5,
    entities: list[str] | None = None,
    topics: list[str] | None = None,
) -> EpisodicEntry:
    enc = MockEncoder()
    return EpisodicEntry(
        speaker=speaker,
        raw_text=text,
        embedding=enc.encode(text).tolist(),
        entities=entities or [],
        topics=topics or [],
        salience=salience,
        recall_times=[datetime.now(timezone.utc).timestamp()],
    )


@pytest.fixture
def sample_episodes() -> list[EpisodicEntry]:
    """Return a list of 5 sample episodes with varying salience."""
    return [
        _make_episode("I really dislike WordPress.", salience=0.7, entities=["WordPress"], topics=["web_development"]),
        _make_episode("Python is my favourite language.", salience=0.8, entities=["Python"], topics=["python", "programming"]),
        _make_episode("Can you help me with a FastAPI project?", salience=0.4, entities=["FastAPI"], topics=["python", "web_development"]),
        _make_episode("The weather is nice today.", salience=0.1),
        _make_episode("Remember that I prefer dark mode in all editors.", salience=0.9, topics=["personal"]),
    ]


@pytest.fixture
def episodic_store(sample_episodes: list[EpisodicEntry]) -> EpisodicStore:
    """Pre-populated in-memory episodic store."""
    store = EpisodicStore()
    for ep in sample_episodes:
        store.add(ep)
    return store


# ────────────────────────────────────────────────────────────────────
# Sample graph
# ────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_graph(mock_encoder: MockEncoder) -> SemanticGraph:
    """Pre-built graph with a few nodes and edges."""
    g = SemanticGraph()

    nodes = [
        SemanticNode(id="python", label="Python", node_type="entity", embedding=mock_encoder.encode("Python").tolist()),
        SemanticNode(id="javascript", label="JavaScript", node_type="entity", embedding=mock_encoder.encode("JavaScript").tolist()),
        SemanticNode(id="wordpress", label="WordPress", node_type="entity", embedding=mock_encoder.encode("WordPress").tolist()),
        SemanticNode(id="fastapi", label="FastAPI", node_type="entity", embedding=mock_encoder.encode("FastAPI").tolist()),
        SemanticNode(id="user", label="User", node_type="entity", embedding=mock_encoder.encode("User").tolist()),
    ]
    for n in nodes:
        g.upsert_node(n)

    edges = [
        SemanticEdge(source="user", target="python", relation="prefers", weight=0.9, confidence=0.72, evidence=["ep1", "ep2"]),
        SemanticEdge(source="user", target="wordpress", relation="dislikes", weight=0.85, confidence=0.85, evidence=["ep3", "ep4", "ep5"]),
        SemanticEdge(source="user", target="fastapi", relation="uses", weight=0.6, confidence=0.5, evidence=["ep6"]),
        SemanticEdge(source="python", target="fastapi", relation="has_framework", weight=0.8, confidence=0.9, evidence=["ep7"]),
    ]
    for e in edges:
        g.upsert_edge(e)

    return g
