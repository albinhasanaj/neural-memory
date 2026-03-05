"""Tests for Phase 2 — Modular Hopfield with Learned Routing."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from config.settings import settings

_device = settings.resolved_device


# ════════════════════════════════════════════════════════════════════
# HopfieldModule
# ════════════════════════════════════════════════════════════════════


class TestHopfieldModule:
    """Tests for the individual Hopfield sub-module."""

    def _make_module(self, max_patterns: int = 100):
        from memory.hopfield_memory import HopfieldModule
        return HopfieldModule(
            pattern_dim=settings.embedding_dim,
            max_patterns=max_patterns,
        )

    def test_empty_retrieval(self) -> None:
        mod = self._make_module()
        query = torch.randn(settings.embedding_dim, device=_device)
        retrieved, attn, indices = mod(query)
        assert retrieved.shape == (settings.embedding_dim,)
        assert attn.numel() == 0

    def test_store_and_retrieve(self) -> None:
        mod = self._make_module()
        for i in range(5):
            torch.manual_seed(i)
            emb = torch.randn(settings.embedding_dim, device=_device)
            mod.store(emb, episode_id=f"ep{i}")
        assert mod.num_patterns == 5

        torch.manual_seed(0)
        query = torch.randn(settings.embedding_dim, device=_device)
        retrieved, attn, indices = mod(query, top_k=3)
        assert retrieved.shape == (settings.embedding_dim,)
        assert attn.shape == (5,)
        assert len(indices) == 3

    def test_occupancy(self) -> None:
        mod = self._make_module(max_patterns=10)
        assert mod.occupancy() == 0.0
        for i in range(5):
            mod.store(torch.randn(settings.embedding_dim, device=_device), episode_id=f"ep{i}")
        assert mod.occupancy() == pytest.approx(0.5, abs=0.01)

    def test_consolidation(self) -> None:
        mod = self._make_module(max_patterns=3)
        for i in range(4):
            torch.manual_seed(i + 42)
            emb = torch.randn(settings.embedding_dim, device=_device)
            mod.store(emb, episode_id=f"ep{i}", metadata={
                "text": f"Text {i}", "speaker": "user", "timestamp": 1000.0 + i,
                "entities": [], "topics": [], "salience": 0.5,
            })
        assert mod.num_patterns <= 3
        assert len(mod._decode_index) == mod.num_patterns

    def test_retrieve_decoded(self) -> None:
        mod = self._make_module()
        for i in range(3):
            torch.manual_seed(i + 200)
            emb = torch.randn(settings.embedding_dim, device=_device)
            mod.store(emb, episode_id=f"ep{i}", metadata={
                "text": f"Memory {i}", "speaker": "user",
                "timestamp": 1000.0 + i, "entities": [f"ent_{i}"],
                "topics": [], "salience": 0.5,
            })
        torch.manual_seed(200)
        query = torch.randn(settings.embedding_dim, device=_device)
        results = mod.retrieve_decoded(query, top_k=2)
        assert len(results) == 2
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)

    def test_clear(self) -> None:
        mod = self._make_module()
        mod.store(torch.randn(settings.embedding_dim, device=_device), episode_id="x")
        assert mod.num_patterns == 1
        mod.clear()
        assert mod.num_patterns == 0
        assert mod._decode_index == {}

    def test_beta_is_positive(self) -> None:
        mod = self._make_module()
        assert mod.beta.item() > 0

    def test_load_state_dict_rebuilds_metadata(self) -> None:
        mod = self._make_module()
        mod.store(torch.randn(settings.embedding_dim, device=_device), episode_id="ep0")
        state = mod.state_dict()

        mod2 = self._make_module()
        mod2.load_state_dict(state)
        assert mod2.num_patterns == 1
        assert 0 in mod2._decode_index


# ════════════════════════════════════════════════════════════════════
# MemoryRouter
# ════════════════════════════════════════════════════════════════════


class TestMemoryRouter:
    """Tests for the learned memory router."""

    def _make_router(self, num_modules: int = 8):
        from memory.hopfield_memory import MemoryRouter
        return MemoryRouter(
            num_modules=num_modules,
            embedding_dim=settings.embedding_dim,
        )

    def test_route_write_returns_correct_count(self) -> None:
        router = self._make_router()
        emb = torch.randn(settings.embedding_dim, device=_device)
        targets = router.route_write(emb, top_k=2)
        assert len(targets) == 2
        for idx, score in targets:
            assert 0 <= idx < 8
            assert 0.0 <= score <= 1.0

    def test_route_read_returns_correct_count(self) -> None:
        router = self._make_router()
        query = torch.randn(settings.embedding_dim, device=_device)
        targets = router.route_read(query, top_k=3)
        assert len(targets) == 3
        for idx, score in targets:
            assert 0 <= idx < 8

    def test_scores_sum_to_one(self) -> None:
        router = self._make_router()
        emb = torch.randn(settings.embedding_dim, device=_device)
        scores = router.scores_for(emb)
        assert scores.shape == (8,)
        assert scores.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_temperature_is_positive(self) -> None:
        router = self._make_router()
        assert router.temperature.item() > 0

    def test_route_write_top_k_clamped(self) -> None:
        """top_k > num_modules should return num_modules entries."""
        router = self._make_router(num_modules=4)
        emb = torch.randn(settings.embedding_dim, device=_device)
        targets = router.route_write(emb, top_k=10)
        assert len(targets) == 4

    def test_different_embeddings_route_differently(self) -> None:
        """Two very different embeddings should (usually) pick different top modules."""
        router = self._make_router(num_modules=16)
        torch.manual_seed(0)
        emb_a = torch.randn(settings.embedding_dim, device=_device)
        torch.manual_seed(999)
        emb_b = torch.randn(settings.embedding_dim, device=_device)
        targets_a = {idx for idx, _ in router.route_write(emb_a, top_k=2)}
        targets_b = {idx for idx, _ in router.route_write(emb_b, top_k=2)}
        # Not guaranteed different, but very likely with 16 modules
        # At minimum, the routing should succeed without error
        assert len(targets_a) == 2
        assert len(targets_b) == 2


# ════════════════════════════════════════════════════════════════════
# ModularHippocampalMemory
# ════════════════════════════════════════════════════════════════════


class TestModularHippocampalMemory:
    """Tests for the orchestrator that wraps M HopfieldModules + a MemoryRouter."""

    def _make_modular(self, num_modules: int = 4, patterns_per_module: int = 50):
        from memory.hopfield_memory import ModularHippocampalMemory
        return ModularHippocampalMemory(
            num_modules=num_modules,
            pattern_dim=settings.embedding_dim,
            patterns_per_module=patterns_per_module,
        )

    def test_store_and_retrieve(self) -> None:
        mem = self._make_modular()
        for i in range(5):
            torch.manual_seed(i + 500)
            emb = torch.randn(settings.embedding_dim, device=_device)
            mem.store(emb, episode_id=f"ep{i}", metadata={
                "text": f"Memory {i}", "speaker": "user",
                "timestamp": 1000.0 + i, "entities": [f"ent_{i}"],
                "topics": [f"topic_{i}"], "salience": 0.5,
            })
        assert mem.num_patterns > 0

        torch.manual_seed(500)
        query = torch.randn(settings.embedding_dim, device=_device)
        results = mem.retrieve_decoded(query, top_k=3)
        assert len(results) <= 3
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)

    def test_deduplication_by_episode_id(self) -> None:
        """Same episode_id stored in multiple modules should be deduped on retrieval."""
        mem = self._make_modular(num_modules=4)
        emb = torch.randn(settings.embedding_dim, device=_device)
        # Store same episode in all 4 modules manually
        for m in mem.modules_list:
            m.store(emb, episode_id="shared_ep", metadata={
                "text": "Shared", "speaker": "user", "timestamp": 0.0,
                "entities": [], "topics": [], "salience": 0.5,
            })
        results = mem.retrieve_decoded(emb, top_k=10)
        ep_ids = [r["episode_id"] for r in results]
        assert ep_ids.count("shared_ep") == 1  # deduped

    def test_clear(self) -> None:
        mem = self._make_modular()
        mem.store(torch.randn(settings.embedding_dim, device=_device), episode_id="x")
        assert mem.num_patterns > 0
        mem.clear()
        assert mem.num_patterns == 0

    def test_forward_returns_correct_shape(self) -> None:
        mem = self._make_modular()
        for i in range(3):
            mem.store(torch.randn(settings.embedding_dim, device=_device), episode_id=f"ep{i}")
        query = torch.randn(settings.embedding_dim, device=_device)
        retrieved, attn, indices = mem(query)
        assert retrieved.shape == (settings.embedding_dim,)

    def test_save_and_load_decode_index(self) -> None:
        mem = self._make_modular(num_modules=2)
        for i in range(4):
            torch.manual_seed(i + 600)
            emb = torch.randn(settings.embedding_dim, device=_device)
            mem.store(emb, episode_id=f"ep{i}", metadata={
                "text": f"Saved text {i}", "speaker": "user",
                "timestamp": 3000.0 + i, "entities": [f"ent_{i}"],
                "topics": [], "salience": 0.7,
            })

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "modular_decode.json"
            mem.save_decode_index(save_path)
            assert save_path.exists()

            # Load into a fresh modular memory
            mem2 = self._make_modular(num_modules=2)
            mem2.load_decode_index(save_path)
            # At least some decode entries should be loaded
            total_decode = sum(len(m._decode_index) for m in mem2.modules_list)
            assert total_decode > 0

    def test_retrieve_episode_ids(self) -> None:
        mem = self._make_modular()
        for i in range(3):
            torch.manual_seed(i + 700)
            emb = torch.randn(settings.embedding_dim, device=_device)
            mem.store(emb, episode_id=f"ep_{i}", metadata={
                "text": f"Memory {i}", "speaker": "user",
                "timestamp": 0.0, "entities": [], "topics": [], "salience": 0.5,
            })
        torch.manual_seed(700)
        query = torch.randn(settings.embedding_dim, device=_device)
        results = mem.retrieve_episode_ids(query, top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r[0], str) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    def test_module_summary(self) -> None:
        mem = self._make_modular(num_modules=2)
        for i in range(3):
            emb = torch.randn(settings.embedding_dim, device=_device)
            mem.store(emb, episode_id=f"ep{i}", metadata={
                "text": f"Text {i}", "speaker": "user",
                "timestamp": 0.0, "entities": [f"entity_{i}"],
                "topics": [f"topic_{i}"], "salience": 0.5,
            })
        summaries = mem.module_summary()
        assert len(summaries) == 2
        for s in summaries:
            assert "module_index" in s
            assert "num_patterns" in s
            assert "occupancy" in s
            assert "top_entities" in s
            assert "top_topics" in s

    def test_empty_retrieval(self) -> None:
        mem = self._make_modular()
        query = torch.randn(settings.embedding_dim, device=_device)
        results = mem.retrieve_decoded(query, top_k=5)
        assert results == []

    def test_backward_compat_properties(self) -> None:
        """ModularHippocampalMemory exposes separator/query_proj/log_beta for trainer."""
        mem = self._make_modular()
        assert mem.separator is not None
        assert mem.query_proj is not None
        assert mem.log_beta is not None
        assert mem.beta.item() > 0


# ════════════════════════════════════════════════════════════════════
# RouterReplayBuffer + train_router_step
# ════════════════════════════════════════════════════════════════════


class TestRouterReplayBuffer:
    """Tests for the router replay buffer."""

    def test_push_and_sample(self) -> None:
        from memory.hopfield_memory import RouterReplayBuffer
        buf = RouterReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(torch.randn(settings.embedding_dim), module_idx=i % 4, reward=0.5)
        assert len(buf) == 10
        batch = buf.sample(5)
        assert batch is not None
        assert len(batch) == 5

    def test_sample_insufficient(self) -> None:
        from memory.hopfield_memory import RouterReplayBuffer
        buf = RouterReplayBuffer(capacity=100)
        buf.push(torch.randn(settings.embedding_dim), module_idx=0, reward=1.0)
        assert buf.sample(5) is None


class TestTrainRouterStep:
    """Tests for the router training function."""

    def test_train_step_runs(self) -> None:
        from memory.hopfield_memory import MemoryRouter, RouterReplayBuffer, train_router_step

        router = MemoryRouter(num_modules=4, embedding_dim=settings.embedding_dim)
        optimizer = torch.optim.Adam(router.parameters(), lr=1e-3)
        buf = RouterReplayBuffer(capacity=100)

        for i in range(16):
            buf.push(torch.randn(settings.embedding_dim), module_idx=i % 4, reward=0.5 + 0.1 * (i % 3))

        result = train_router_step(router, optimizer, buf, batch_size=8)
        assert result is not None
        assert "loss" in result

    def test_train_step_insufficient_data(self) -> None:
        from memory.hopfield_memory import MemoryRouter, RouterReplayBuffer, train_router_step

        router = MemoryRouter(num_modules=4, embedding_dim=settings.embedding_dim)
        optimizer = torch.optim.Adam(router.parameters(), lr=1e-3)
        buf = RouterReplayBuffer(capacity=100)

        result = train_router_step(router, optimizer, buf, batch_size=8)
        assert result is None


# ════════════════════════════════════════════════════════════════════
# Backward compatibility
# ════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """Ensure the legacy alias still works."""

    def test_hippocampal_memory_alias(self) -> None:
        from memory.hopfield_memory import HippocampalMemory, LegacyHippocampalMemory
        assert HippocampalMemory is LegacyHippocampalMemory

    def test_legacy_import_still_works(self) -> None:
        from memory.hopfield_memory import HippocampalMemory
        mem = HippocampalMemory(
            pattern_dim=settings.embedding_dim,
            max_patterns=100,
        )
        emb = torch.randn(settings.embedding_dim, device=_device)
        mem.store(emb, episode_id="test")
        assert mem.num_patterns == 1
