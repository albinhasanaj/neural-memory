"""Tests for data loaders and offline training pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from data.conversation_loader import (
    Conversation,
    Turn,
    load_local_json,
    stream_conversations,
)


# ── Turn / Conversation dataclasses ────────────────────────────────


class TestConversation:
    def test_turn_creation(self) -> None:
        t = Turn(role="user", content="hello")
        assert t.role == "user"
        assert t.content == "hello"

    def test_conversation_num_turns(self) -> None:
        conv = Conversation(
            id="test",
            turns=[
                Turn(role="user", content="hi"),
                Turn(role="assistant", content="hello"),
                Turn(role="user", content="bye"),
            ],
        )
        assert conv.num_turns == 3
        assert len(conv.user_turns()) == 2
        assert len(conv.assistant_turns()) == 1


# ── Local JSON loader ──────────────────────────────────────────────


class TestLocalJsonLoader:
    def _make_json_file(self, data: list[dict]) -> Path:
        """Create a temp JSON file with conversation data."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        )
        json.dump(data, tmp)
        tmp.close()
        return Path(tmp.name)

    def _make_jsonl_file(self, data: list[dict]) -> Path:
        """Create a temp JSONL file."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
        )
        for obj in data:
            tmp.write(json.dumps(obj) + "\n")
        tmp.close()
        return Path(tmp.name)

    def test_load_json_basic(self) -> None:
        data = [
            {
                "id": "conv_001",
                "messages": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "A programming language."},
                    {"role": "user", "content": "Tell me more."},
                    {"role": "assistant", "content": "It was created by Guido."},
                ],
            },
        ]
        path = self._make_json_file(data)
        convos = load_local_json(path)
        assert len(convos) == 1
        assert convos[0].num_turns == 4
        assert convos[0].turns[0].role == "user"
        Path(path).unlink()

    def test_load_json_min_turns_filter(self) -> None:
        data = [
            {
                "id": "short",
                "messages": [
                    {"role": "user", "content": "Hi"},
                ],
            },
            {
                "id": "long",
                "messages": [
                    {"role": "user", "content": "Tell me about AI"},
                    {"role": "assistant", "content": "AI is..."},
                    {"role": "user", "content": "More?"},
                    {"role": "assistant", "content": "Sure..."},
                ],
            },
        ]
        path = self._make_json_file(data)
        convos = load_local_json(path, min_turns=3)
        assert len(convos) == 1
        assert convos[0].id == "long"
        Path(path).unlink()

    def test_load_jsonl(self) -> None:
        data = [
            {
                "id": f"conv_{i}",
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer {i}"},
                ],
            }
            for i in range(5)
        ]
        path = self._make_jsonl_file(data)
        convos = load_local_json(path)
        assert len(convos) == 5
        Path(path).unlink()

    def test_max_conversations(self) -> None:
        data = [
            {
                "id": f"conv_{i}",
                "messages": [
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"A{i}"},
                ],
            }
            for i in range(20)
        ]
        path = self._make_json_file(data)
        convos = load_local_json(path, max_conversations=5)
        assert len(convos) == 5
        Path(path).unlink()


# ── Streaming helper ───────────────────────────────────────────────


class TestStreamConversations:
    def test_single_epoch(self) -> None:
        convos = [
            Conversation(id=str(i), turns=[
                Turn(role="user", content=f"msg {i}"),
                Turn(role="assistant", content=f"reply {i}"),
            ])
            for i in range(5)
        ]
        streamed = list(stream_conversations(convos, repeat=1, shuffle=False))
        assert len(streamed) == 5

    def test_multiple_epochs(self) -> None:
        convos = [
            Conversation(id="0", turns=[
                Turn(role="user", content="hi"),
                Turn(role="assistant", content="hello"),
            ])
        ]
        streamed = list(stream_conversations(convos, repeat=3, shuffle=False))
        assert len(streamed) == 3

    def test_shuffle_changes_order(self) -> None:
        convos = [
            Conversation(id=str(i), turns=[
                Turn(role="user", content=f"msg {i}"),
                Turn(role="assistant", content=f"reply {i}"),
            ])
            for i in range(20)
        ]
        order1 = [c.id for c in stream_conversations(convos, repeat=1, shuffle=True, seed=1)]
        order2 = [c.id for c in stream_conversations(convos, repeat=1, shuffle=True, seed=2)]
        # Different seeds should produce different orders (very likely for 20 items)
        assert order1 != order2
