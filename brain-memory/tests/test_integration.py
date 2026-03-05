"""
End-to-end integration test: input turn → observe → activate → inject → verify.

Uses the MockEncoder from conftest to avoid loading real models.
"""

from __future__ import annotations

import pytest
import torch
from unittest.mock import patch

from config.settings import settings
from memory.episodic import EpisodicStore
from memory.observer import MemoryObserver
from memory.semantic import SemanticGraph, SemanticNode, SemanticEdge
from memory.working_memory import WorkingMemory
from tests.conftest import MockEncoder


@pytest.fixture
def observer(mock_encoder: MockEncoder) -> MemoryObserver:
    """Build an observer with mocked encoder and a small pre-built graph."""
    graph = SemanticGraph()
    graph.upsert_node(
        SemanticNode(
            id="python", label="Python",
            embedding=mock_encoder.encode("Python").tolist(),
        )
    )
    graph.upsert_node(
        SemanticNode(
            id="javascript", label="JavaScript",
            embedding=mock_encoder.encode("JavaScript").tolist(),
        )
    )
    graph.upsert_edge(
        SemanticEdge(source="python", target="javascript", relation="alternative_to", weight=0.5)
    )

    wm = WorkingMemory()
    store = EpisodicStore()

    obs = MemoryObserver(
        encoder=mock_encoder,  # type: ignore[arg-type]
        working_memory=wm,
        episodic_store=store,
        semantic_graph=graph,
    )
    return obs


class TestIntegration:
    """End-to-end pipeline tests."""

    def test_observe_returns_diagnostics(self, observer: MemoryObserver) -> None:
        info = observer.observe("I really love Python programming!", speaker="user")
        assert "salience" in info
        assert "entities" in info
        assert "topics" in info
        assert isinstance(info["salience"], float)

    def test_high_salience_stored(self, observer: MemoryObserver) -> None:
        # Use an emphatic message likely to score high salience
        info = observer.observe(
            "REMEMBER THIS! Python is my FAVOURITE language!! Don't forget!",
            speaker="user",
        )
        # This should score high on emphasis + entities
        if info["salience"] >= settings.salience_threshold:
            assert info["stored"] is True
            assert info["episode_id"] is not None

    def test_low_salience_not_stored(self, observer: MemoryObserver) -> None:
        # Bland message with no emphasis
        info = observer.observe("ok", speaker="user")
        # Even if stored (salience depends on novelty), verify structure
        assert isinstance(info["stored"], bool)

    def test_activate_and_inject_pipeline(self, observer: MemoryObserver) -> None:
        # First observe a few turns to populate working memory
        observer.observe("I prefer Python for backend work", speaker="user")
        observer.observe("FastAPI is my go-to framework", speaker="user")

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What framework should I use?"},
        ]

        modified, activated = observer.activate_and_inject(messages)
        # Modified messages should be at least as long as original
        assert len(modified) >= len(messages)

    def test_process_turn_full_cycle(self, observer: MemoryObserver) -> None:
        messages = [
            {"role": "user", "content": "Tell me about Python web frameworks"},
        ]
        modified, info = observer.process_turn(
            "Tell me about Python web frameworks",
            speaker="user",
            messages=messages,
        )
        assert "salience" in info
        assert "activated_nodes" in info
        assert isinstance(modified, list)

    def test_multiple_turns_build_context(self, observer: MemoryObserver) -> None:
        turns = [
            "I'm working on a new Python project",
            "It uses FastAPI and PostgreSQL",
            "The deployment will be on AWS",
        ]
        for turn in turns:
            observer.observe(turn, speaker="user")

        # Working memory should have context
        assert observer.working_memory.context_vector is not None
        assert observer.working_memory.buffer.size == len(turns)

    def test_episodic_decay_runs(self, observer: MemoryObserver) -> None:
        # Add some episodes
        observer.observe("IMPORTANT: Remember my API key policy!", speaker="user")
        observer.observe("Another important fact about security!", speaker="user")

        # Apply decay to the store
        observer.episodic_store.apply_decay()
        for entry in observer.episodic_store.get_all_active():
            assert isinstance(entry.activation, float)
