"""Tests for memory.hopfield_memory — Modern Hopfield Network."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from config.settings import settings

_device = settings.resolved_device


class TestHippocampalMemory:
    """Test the Hopfield associative store."""

    def _make_memory(self, max_patterns: int = 100):
        from memory.hopfield_memory import HippocampalMemory
        return HippocampalMemory(
            pattern_dim=settings.embedding_dim,
            max_patterns=max_patterns,
        )

    def test_empty_store_retrieval(self) -> None:
        mem = self._make_memory()
        query = torch.randn(settings.embedding_dim, device=_device)
        retrieved, attn, indices = mem(query)
        assert retrieved.shape == (settings.embedding_dim,)
        assert attn.numel() == 0

    def test_store_and_retrieve(self) -> None:
        mem = self._make_memory()
        emb = torch.randn(settings.embedding_dim, device=_device)
        mem.store(emb, episode_id="ep1")
        assert mem.num_patterns == 1

        retrieved, attn, indices = mem(emb)
        assert retrieved.shape == (settings.embedding_dim,)
        assert attn.shape == (1,)
        # Attention should be 1.0 (only one pattern)
        assert pytest.approx(attn.item(), abs=1e-4) == 1.0

    def test_multiple_patterns(self) -> None:
        mem = self._make_memory()
        for i in range(5):
            torch.manual_seed(i)
            emb = torch.randn(settings.embedding_dim, device=_device)
            mem.store(emb, episode_id=f"ep{i}")

        assert mem.num_patterns == 5

        # Retrieve with the first pattern's embedding
        torch.manual_seed(0)
        query = torch.randn(settings.embedding_dim, device=_device)
        retrieved, attn, indices = mem(query, top_k=3)
        assert retrieved.shape == (settings.embedding_dim,)
        assert attn.shape == (5,)
        assert len(indices) == 3

    def test_consolidation_evicts_least_accessed(self) -> None:
        mem = self._make_memory(max_patterns=3)

        # Store 4 patterns — should trigger consolidation
        for i in range(4):
            torch.manual_seed(i + 42)
            emb = torch.randn(settings.embedding_dim, device=_device)
            mem.store(emb, episode_id=f"ep{i}")

        # Should have been trimmed to max_patterns
        assert mem.num_patterns <= 3

    def test_retrieve_episode_ids(self) -> None:
        mem = self._make_memory()
        for i in range(3):
            torch.manual_seed(i + 100)
            emb = torch.randn(settings.embedding_dim, device=_device)
            mem.store(emb, episode_id=f"ep_{i}")

        torch.manual_seed(100)
        query = torch.randn(settings.embedding_dim, device=_device)
        results = mem.retrieve_episode_ids(query, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r[0], str) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    def test_clear(self) -> None:
        mem = self._make_memory()
        mem.store(torch.randn(settings.embedding_dim, device=_device), episode_id="x")
        assert mem.num_patterns == 1
        mem.clear()
        assert mem.num_patterns == 0

    def test_beta_is_positive(self) -> None:
        mem = self._make_memory()
        assert mem.beta.item() > 0

    # ── Decode index tests ──────────────────────────────────────────

    def test_store_with_metadata_populates_decode_index(self) -> None:
        mem = self._make_memory()
        emb = torch.randn(settings.embedding_dim, device=_device)
        mem.store(emb, episode_id="ep1", metadata={
            "text": "Hello world",
            "speaker": "user",
            "timestamp": 1000.0,
            "entities": ["world"],
            "topics": ["greeting"],
            "salience": 0.8,
        })
        assert 0 in mem._decode_index
        assert mem._decode_index[0]["text"] == "Hello world"
        assert mem._decode_index[0]["episode_id"] == "ep1"

    def test_store_without_metadata_no_decode_entry(self) -> None:
        mem = self._make_memory()
        emb = torch.randn(settings.embedding_dim, device=_device)
        mem.store(emb, episode_id="ep1")
        assert 0 not in mem._decode_index

    def test_retrieve_decoded_empty(self) -> None:
        mem = self._make_memory()
        query = torch.randn(settings.embedding_dim, device=_device)
        results = mem.retrieve_decoded(query, top_k=5)
        assert results == []

    def test_retrieve_decoded_returns_metadata(self) -> None:
        mem = self._make_memory()
        for i in range(3):
            torch.manual_seed(i + 200)
            emb = torch.randn(settings.embedding_dim, device=_device)
            mem.store(emb, episode_id=f"ep{i}", metadata={
                "text": f"Memory number {i}",
                "speaker": "user",
                "timestamp": 1000.0 + i,
                "entities": [f"entity_{i}"],
                "topics": [],
                "salience": 0.5,
            })

        torch.manual_seed(200)
        query = torch.randn(settings.embedding_dim, device=_device)
        results = mem.retrieve_decoded(query, top_k=2)
        assert len(results) == 2
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)
        assert all("slot_index" in r for r in results)
        assert all("episode_id" in r for r in results)
        # Sorted by score descending
        assert results[0]["score"] >= results[1]["score"]

    def test_decode_method(self) -> None:
        mem = self._make_memory()
        emb = torch.randn(settings.embedding_dim, device=_device)
        mem.store(emb, episode_id="ep_x", metadata={
            "text": "Test decode",
            "speaker": "assistant",
            "timestamp": 2000.0,
            "entities": [],
            "topics": [],
            "salience": 0.3,
        })
        decoded = mem.decode([0])
        assert len(decoded) == 1
        assert decoded[0]["text"] == "Test decode"

    def test_consolidation_keeps_decode_index_in_sync(self) -> None:
        mem = self._make_memory(max_patterns=3)
        for i in range(4):
            torch.manual_seed(i + 300)
            emb = torch.randn(settings.embedding_dim, device=_device)
            mem.store(emb, episode_id=f"ep{i}", metadata={
                "text": f"Text {i}",
                "speaker": "user",
                "timestamp": 1000.0 + i,
                "entities": [],
                "topics": [],
                "salience": 0.5,
            })
        # After consolidation, decode index should match num_patterns
        assert len(mem._decode_index) == mem.num_patterns
        # All indices should be contiguous 0..num_patterns-1
        assert set(mem._decode_index.keys()) == set(range(mem.num_patterns))

    def test_clear_resets_decode_index(self) -> None:
        mem = self._make_memory()
        emb = torch.randn(settings.embedding_dim, device=_device)
        mem.store(emb, episode_id="x", metadata={"text": "t", "speaker": "u", "timestamp": 0.0, "entities": [], "topics": [], "salience": 0.0})
        mem.clear()
        assert mem._decode_index == {}

    def test_save_and_load_decode_index(self) -> None:
        mem = self._make_memory()
        for i in range(3):
            torch.manual_seed(i + 400)
            emb = torch.randn(settings.embedding_dim, device=_device)
            mem.store(emb, episode_id=f"ep{i}", metadata={
                "text": f"Saved text {i}",
                "speaker": "user",
                "timestamp": 3000.0 + i,
                "entities": [f"ent_{i}"],
                "topics": [],
                "salience": 0.7,
            })

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "decode.json"
            mem.save_decode_index(save_path)
            assert save_path.exists()

            # Load into a fresh memory
            mem2 = self._make_memory()
            mem2.load_decode_index(save_path)
            assert len(mem2._decode_index) == 3
            assert mem2._decode_index[0]["text"] == "Saved text 0"
            assert mem2._decode_index[2]["entities"] == ["ent_2"]
