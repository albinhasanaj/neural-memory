"""Tests for memory.hopfield_memory — Modern Hopfield Network."""

from __future__ import annotations

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
