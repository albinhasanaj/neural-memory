"""Tests for memory context injector: formatting and injection."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memory.episodic import EpisodicEntry
from memory.injector import (
    build_memory_context,
    format_episodic_memories,
    format_hopfield_memories,
    format_semantic_memories,
    inject,
    inject_hopfield,
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


class TestFormatHopfieldMemories:
    """Hopfield-decoded memory formatting."""

    def test_formats_decoded_dicts(self) -> None:
        memories = [
            {"text": "I like Python", "speaker": "user", "timestamp": 1709683200.0, "entities": ["Python"], "score": 0.9},
            {"text": "Dark mode preferred", "speaker": "user", "timestamp": 1709769600.0, "entities": [], "score": 0.7},
        ]
        text = format_hopfield_memories(memories)
        assert "user:" in text.lower() or "user:" in text
        assert "Python" in text or "Dark mode" in text

    def test_empty_list(self) -> None:
        assert format_hopfield_memories([]) == ""

    def test_truncates_long_text(self) -> None:
        memories = [{"text": "x" * 300, "speaker": "user", "timestamp": 1709683200.0, "entities": [], "score": 0.5}]
        text = format_hopfield_memories(memories)
        assert "..." in text

    def test_max_memories_limit(self) -> None:
        memories = [
            {"text": f"Memory {i}", "speaker": "user", "timestamp": 1709683200.0, "entities": [], "score": 0.5}
            for i in range(10)
        ]
        text = format_hopfield_memories(memories, max_memories=2)
        lines = [l for l in text.strip().split("\n") if l.strip().startswith("-")]
        assert len(lines) <= 2


class TestInjectHopfield:
    """Hopfield injection into message list."""

    def test_injects_hopfield_memories(self, sample_graph: SemanticGraph) -> None:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        memories = [
            {"text": "I like Python", "speaker": "user", "timestamp": 1709683200.0, "entities": ["Python"], "score": 0.9},
        ]
        result = inject_hopfield(messages, sample_graph, [("user", 0.9)], memories)
        assert len(result) == 3
        assert "[MEMORY CONTEXT" in result[1]["content"]

    def test_no_injection_when_empty(self) -> None:
        messages = [{"role": "user", "content": "Hi"}]
        g = SemanticGraph()
        result = inject_hopfield(messages, g, [], [])
        assert len(result) == 1
