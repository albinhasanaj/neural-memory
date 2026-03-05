"""Tests for memory context injector: formatting and injection."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memory.episodic import EpisodicEntry
from memory.injector import (
    build_memory_context,
    format_episodic_memories,
    format_semantic_memories,
    inject,
)
from memory.semantic import SemanticGraph


class TestFormatSemanticMemories:
    """Semantic fact formatting."""

    def test_with_activated_nodes(self, sample_graph: SemanticGraph) -> None:
        activated = [("user", 0.9), ("python", 0.7)]
        text = format_semantic_memories(sample_graph, activated)
        assert "prefers" in text or "dislikes" in text
        assert "confidence" in text

    def test_empty_activations(self, sample_graph: SemanticGraph) -> None:
        text = format_semantic_memories(sample_graph, [])
        assert text == ""

    def test_max_facts_limit(self, sample_graph: SemanticGraph) -> None:
        activated = [("user", 0.9)]
        text = format_semantic_memories(sample_graph, activated, max_facts=1)
        lines = [l for l in text.strip().split("\n") if l.strip().startswith("-")]
        assert len(lines) <= 1


class TestFormatEpisodicMemories:
    """Episodic memory formatting."""

    def test_with_episodes(self, sample_episodes: list[EpisodicEntry]) -> None:
        text = format_episodic_memories(sample_episodes[:2])
        assert "[" in text  # date brackets
        assert "user:" in text.lower() or "User:" in text

    def test_max_episodes_limit(self, sample_episodes: list[EpisodicEntry]) -> None:
        text = format_episodic_memories(sample_episodes, max_episodes=2)
        lines = [l for l in text.strip().split("\n") if l.strip().startswith("-")]
        assert len(lines) <= 2

    def test_empty_list(self) -> None:
        assert format_episodic_memories([]) == ""


class TestBuildMemoryContext:
    """Full memory context block assembly."""

    def test_includes_header_and_footer(self, sample_graph: SemanticGraph, sample_episodes: list[EpisodicEntry]) -> None:
        ctx = build_memory_context(sample_graph, [("user", 0.9)], sample_episodes[:2])
        assert "[MEMORY CONTEXT" in ctx
        assert "[END MEMORY CONTEXT]" in ctx

    def test_empty_when_nothing_activated(self) -> None:
        g = SemanticGraph()
        ctx = build_memory_context(g, [], [])
        assert ctx == ""


class TestInject:
    """Message list injection."""

    def test_prepends_memory_after_system(self, sample_graph: SemanticGraph, sample_episodes: list[EpisodicEntry]) -> None:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = inject(messages, sample_graph, [("user", 0.9)], sample_episodes[:2])
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert "[MEMORY CONTEXT" in result[1]["content"]
        assert result[2]["role"] == "user"

    def test_does_not_mutate_original(self, sample_graph: SemanticGraph, sample_episodes: list[EpisodicEntry]) -> None:
        messages = [{"role": "user", "content": "Hi"}]
        original_len = len(messages)
        inject(messages, sample_graph, [("user", 0.9)], sample_episodes[:1])
        assert len(messages) == original_len

    def test_no_injection_when_empty(self) -> None:
        messages = [{"role": "user", "content": "Hi"}]
        g = SemanticGraph()
        result = inject(messages, g, [], [])
        assert len(result) == 1
