"""Tests for Phase 3 — Continuous Weight Memory (Fast Weights)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from config.settings import settings

_device = settings.resolved_device


# ════════════════════════════════════════════════════════════════════
# FastWeightModule
# ════════════════════════════════════════════════════════════════════


class TestFastWeightModule:
    """Tests for the individual fast-weight memory unit."""

    def _make_module(self):
        from memory.hopfield_memory import FastWeightModule
        return FastWeightModule(
            embedding_dim=settings.embedding_dim,
            hidden_dim=settings.fast_weight_hidden_dim,
            write_lr=settings.fast_weight_write_lr,
        )

    def test_write_changes_weights(self) -> None:
        mod = self._make_module()
        w_key_before = mod.W_key.clone()
        w_value_before = mod.W_value.clone()

        emb = torch.randn(settings.embedding_dim, device=_device)
        mod.write(emb, episode_id="ep1", metadata={"text": "hello"})

        assert not torch.equal(mod.W_key, w_key_before)
        assert not torch.equal(mod.W_value, w_value_before)
        assert mod.write_count.item() == 1

    def test_store_alias_works(self) -> None:
        mod = self._make_module()
        emb = torch.randn(settings.embedding_dim, device=_device)
        mod.store(emb, episode_id="ep1", metadata={"text": "hello via store"})
        assert mod.write_count.item() == 1

    def test_retrieve_returns_output(self) -> None:
        mod = self._make_module()
        emb = torch.randn(settings.embedding_dim, device=_device)
        mod.write(emb, episode_id="ep1", metadata={"text": "test"})

        output, energy = mod.retrieve(emb)
        assert output.shape == (settings.embedding_dim,)
        assert energy.ndim == 0  # scalar

    def test_retrieve_decoded_finds_written_memory(self) -> None:
        mod = self._make_module()
        torch.manual_seed(42)
        emb = torch.randn(settings.embedding_dim, device=_device)
        mod.write(emb, episode_id="ep1", metadata={
            "text": "I like pizza",
            "speaker": "user",
            "timestamp": 1000.0,
            "entities": ["pizza"],
            "topics": ["food"],
            "salience": 0.8,
        })

        results = mod.retrieve_decoded(emb, top_k=5)
        assert len(results) >= 1
        assert results[0]["text"] == "I like pizza"
        assert results[0]["episode_id"] == "ep1"

    def test_retrieve_decoded_empty_returns_empty(self) -> None:
        mod = self._make_module()
        query = torch.randn(settings.embedding_dim, device=_device)
        results = mod.retrieve_decoded(query, top_k=5)
        assert results == []

    def test_homeostatic_decay(self) -> None:
        mod = self._make_module()
        # Write enough to trigger decay
        interval = settings.fast_weight_decay_interval
        for i in range(interval):
            emb = torch.randn(settings.embedding_dim, device=_device)
            mod.write(emb, episode_id=f"ep{i}", metadata={"text": f"turn {i}"})

        # After exactly `interval` writes, weights should have been decayed
        # Check that write_count matches
        assert mod.write_count.item() == interval

    def test_interference_protection(self) -> None:
        mod = self._make_module()
        emb = torch.randn(settings.embedding_dim, device=_device)
        mod.write(emb, episode_id="ep1", metadata={"text": "important"})

        # Reinforce to increase omega (importance)
        mod.reinforce_retrieval(emb)
        assert mod.omega_key.abs().sum().item() > 0
        assert mod.omega_value.abs().sum().item() > 0

    def test_clear_resets_everything(self) -> None:
        mod = self._make_module()
        for i in range(3):
            emb = torch.randn(settings.embedding_dim, device=_device)
            mod.write(emb, episode_id=f"ep{i}", metadata={"text": f"turn {i}"})

        mod.reinforce_retrieval(torch.randn(settings.embedding_dim, device=_device))

        mod.clear()
        assert mod.write_count.item() == 0
        assert mod.total_energy.item() == 0.0
        assert torch.all(mod.W_key == 0)
        assert torch.all(mod.W_value == 0)
        assert torch.all(mod.omega_key == 0)
        assert torch.all(mod.omega_value == 0)
        assert len(mod._decode_index) == 0
        assert len(mod._write_history) == 0

    def test_occupancy_increases_with_writes(self) -> None:
        mod = self._make_module()
        assert mod.occupancy() == 0.0

        for i in range(5):
            emb = torch.randn(settings.embedding_dim, device=_device)
            mod.write(emb, episode_id=f"ep{i}", metadata={"text": f"msg {i}"})

        occ = mod.occupancy()
        assert occ > 0.0

    def test_decode_index_save_load(self) -> None:
        mod = self._make_module()
        emb = torch.randn(settings.embedding_dim, device=_device)
        mod.write(emb, episode_id="ep1", metadata={
            "text": "save this",
            "speaker": "user",
            "timestamp": 123.0,
            "entities": [],
            "topics": [],
            "salience": 0.5,
        })

        data = mod.save_decode_index_data()
        assert "decode_index" in data
        assert len(data["decode_index"]) == 1

        mod2 = self._make_module()
        mod2.load_decode_index_data(data)
        assert len(mod2._decode_index) == 1

    def test_prune_decode_index(self) -> None:
        mod = self._make_module()
        for i in range(10):
            torch.manual_seed(i + 100)
            emb = torch.randn(settings.embedding_dim, device=_device)
            mod.write(emb, episode_id=f"ep{i}", metadata={
                "text": f"turn {i}",
                "speaker": "user",
                "timestamp": float(i),
                "entities": [],
                "topics": [],
                "salience": 0.5,
            })

        mod.prune_decode_index(max_entries=3)
        assert len(mod._decode_index) <= 3


# ════════════════════════════════════════════════════════════════════
# ModularFastWeightMemory
# ════════════════════════════════════════════════════════════════════


class TestModularFastWeightMemory:
    """Tests for the multi-module fast-weight orchestrator."""

    def _make_memory(self, num_modules: int = 4):
        from memory.hopfield_memory import ModularFastWeightMemory
        return ModularFastWeightMemory(
            embedding_dim=settings.embedding_dim,
            num_modules=num_modules,
            hidden_dim=settings.fast_weight_hidden_dim,
            write_lr=settings.fast_weight_write_lr,
        )

    def test_store_and_retrieve_roundtrip(self) -> None:
        mem = self._make_memory()
        torch.manual_seed(42)
        emb = torch.randn(settings.embedding_dim, device=_device)
        mem.store(
            emb,
            episode_id="ep1",
            metadata={
                "text": "I love dogs",
                "speaker": "user",
                "timestamp": 1000.0,
                "entities": ["dogs"],
                "topics": ["pets"],
                "salience": 0.9,
            },
        )

        results = mem.retrieve_decoded(emb, top_k=5)
        assert len(results) >= 1
        found_texts = [r["text"] for r in results]
        assert "I love dogs" in found_texts

    def test_deduplication(self) -> None:
        mem = self._make_memory()
        torch.manual_seed(42)
        emb = torch.randn(settings.embedding_dim, device=_device)
        # Store same episode — routes to multiple modules, but retrieve should deduplicate
        mem.store(
            emb,
            episode_id="ep_dup",
            metadata={
                "text": "duplicate test",
                "speaker": "user",
                "timestamp": 1.0,
                "entities": [],
                "topics": [],
                "salience": 0.5,
            },
        )
        results = mem.retrieve_decoded(emb, top_k=10)
        episode_ids = [r["episode_id"] for r in results if r.get("episode_id") == "ep_dup"]
        assert len(episode_ids) <= 1  # deduplicated

    def test_module_summary(self) -> None:
        mem = self._make_memory()
        for i in range(8):
            torch.manual_seed(i)
            emb = torch.randn(settings.embedding_dim, device=_device)
            mem.store(
                emb,
                episode_id=f"ep{i}",
                metadata={
                    "text": f"turn {i}",
                    "speaker": "user",
                    "timestamp": float(i),
                    "entities": [f"entity_{i}"],
                    "topics": [f"topic_{i % 3}"],
                    "salience": 0.5,
                },
            )

        summary = mem.module_summary()
        assert len(summary) == 4
        total_writes = sum(s["write_count"] for s in summary)
        assert total_writes > 0
        assert all("w_key_norm" in s for s in summary)

    def test_forward_compatibility(self) -> None:
        mem = self._make_memory()
        emb = torch.randn(settings.embedding_dim, device=_device)
        mem.store(emb, episode_id="ep1", metadata={"text": "test"})

        output, attn, indices = mem.forward(emb)
        assert output.shape == (settings.embedding_dim,)

    def test_clear_resets_all_modules(self) -> None:
        mem = self._make_memory()
        for i in range(5):
            emb = torch.randn(settings.embedding_dim, device=_device)
            mem.store(emb, episode_id=f"ep{i}", metadata={"text": f"t{i}"})

        assert mem.total_writes() > 0
        mem.clear()
        assert mem.total_writes() == 0

    def test_module_occupancies(self) -> None:
        mem = self._make_memory()
        occs = mem.module_occupancies()
        assert len(occs) == 4
        assert all(o == 0.0 for o in occs)

        for i in range(10):
            emb = torch.randn(settings.embedding_dim, device=_device)
            mem.store(emb, episode_id=f"ep{i}", metadata={"text": f"t{i}"})

        occs = mem.module_occupancies()
        assert any(o > 0.0 for o in occs)

    def test_save_and_load_decode_index(self) -> None:
        mem = self._make_memory()
        for i in range(5):
            torch.manual_seed(i)
            emb = torch.randn(settings.embedding_dim, device=_device)
            mem.store(
                emb,
                episode_id=f"ep{i}",
                metadata={
                    "text": f"turn {i}",
                    "speaker": "user",
                    "timestamp": float(i),
                    "entities": [],
                    "topics": [],
                    "salience": 0.5,
                },
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "fast_weight_decode.json"
            mem.save_decode_index(idx_path)
            assert idx_path.exists()

            mem2 = self._make_memory()
            mem2.load_decode_index(idx_path)
            # At least some modules should have loaded data
            total_loaded = sum(
                len(m._decode_index) for m in mem2.modules_list
            )
            assert total_loaded > 0

    def test_retrieve_episode_ids(self) -> None:
        mem = self._make_memory()
        torch.manual_seed(42)
        emb = torch.randn(settings.embedding_dim, device=_device)
        mem.store(
            emb,
            episode_id="ep42",
            metadata={
                "text": "find me",
                "speaker": "user",
                "timestamp": 42.0,
                "entities": [],
                "topics": [],
                "salience": 0.9,
            },
        )
        pairs = mem.retrieve_episode_ids(emb, top_k=5)
        eids = [eid for eid, _score in pairs]
        assert "ep42" in eids


# ════════════════════════════════════════════════════════════════════
# FastWeightReplayBuffer & train_fast_weight_step
# ════════════════════════════════════════════════════════════════════


class TestFastWeightTraining:
    """Tests for the fast-weight training infrastructure."""

    def test_replay_buffer_push_and_sample(self) -> None:
        from memory.hopfield_memory import FastWeightReplayBuffer
        buf = FastWeightReplayBuffer(capacity=100)
        for i in range(20):
            q = torch.randn(settings.embedding_dim)
            t = torch.randn(settings.embedding_dim)
            buf.push(q, t)
        assert len(buf) == 20

        batch = buf.sample(8)
        assert batch is not None
        queries, targets = batch
        assert queries.shape == (8, settings.embedding_dim)
        assert targets.shape == (8, settings.embedding_dim)

    def test_train_step_runs(self) -> None:
        from memory.hopfield_memory import FastWeightModule, FastWeightReplayBuffer, train_fast_weight_step

        mod = FastWeightModule()
        optimizer = torch.optim.Adam(mod.parameters(), lr=1e-3)
        buf = FastWeightReplayBuffer(capacity=100)

        # Write some memories first so retrieval is meaningful
        for i in range(5):
            emb = torch.randn(settings.embedding_dim, device=_device)
            mod.write(emb, episode_id=f"ep{i}", metadata={"text": f"t{i}"})

        # Fill buffer
        for i in range(20):
            q = torch.randn(settings.embedding_dim)
            t = torch.randn(settings.embedding_dim)
            buf.push(q, t)

        result = train_fast_weight_step(mod, optimizer, buf, batch_size=8)
        assert result is not None
        assert "loss" in result
        assert result["loss"] >= 0.0

    def test_train_step_returns_none_when_buffer_too_small(self) -> None:
        from memory.hopfield_memory import FastWeightModule, FastWeightReplayBuffer, train_fast_weight_step

        mod = FastWeightModule()
        optimizer = torch.optim.Adam(mod.parameters(), lr=1e-3)
        buf = FastWeightReplayBuffer(capacity=100)

        result = train_fast_weight_step(mod, optimizer, buf, batch_size=8)
        assert result is None


# ════════════════════════════════════════════════════════════════════
# Config flags
# ════════════════════════════════════════════════════════════════════


class TestFastWeightConfig:
    """Verify Phase 3 config flags exist and have correct defaults."""

    def test_defaults(self) -> None:
        assert hasattr(settings, "use_fast_weight_memory")
        assert settings.use_fast_weight_memory is False
        assert hasattr(settings, "fast_weight_write_lr")
        assert settings.fast_weight_write_lr == 0.1
        assert hasattr(settings, "fast_weight_hidden_dim")
        assert settings.fast_weight_hidden_dim == 128
        assert hasattr(settings, "fast_weight_decay_factor")
        assert settings.fast_weight_decay_factor == 0.995
        assert hasattr(settings, "fast_weight_decay_interval")
        assert settings.fast_weight_decay_interval == 10
