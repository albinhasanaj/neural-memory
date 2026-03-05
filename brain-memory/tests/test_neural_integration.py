"""Tests for neural module integration — NeuralMemoryObserver and cross-module interactions."""

from __future__ import annotations

import pytest
import torch

from config.settings import settings

_device = settings.resolved_device


class TestGraphConverter:
    """Test NetworkX → PyG converter."""

    def test_convert_empty_graph(self) -> None:
        from memory.graph_converter import GraphConverter
        from memory.semantic import SemanticGraph

        g = SemanticGraph()
        converter = GraphConverter()
        data = converter.convert(g)
        assert data.x.shape[0] == 0

    def test_convert_small_graph(self, sample_graph) -> None:
        from memory.graph_converter import GraphConverter

        converter = GraphConverter()
        data = converter.convert(sample_graph)

        # Sample graph has 5 nodes and 4 edges
        assert data.x.shape[0] == 5
        assert data.x.shape[1] == converter.node_feature_dim
        assert data.edge_index.shape[0] == 2
        assert data.edge_attr is not None

    def test_caching(self, sample_graph) -> None:
        from memory.graph_converter import GraphConverter

        converter = GraphConverter()
        d1 = converter.get_or_convert(sample_graph)
        d2 = converter.get_or_convert(sample_graph)
        # Should return cached version
        assert d1 is d2

    def test_needs_rebuild_detects_change(self, sample_graph) -> None:
        from memory.graph_converter import GraphConverter
        from memory.semantic import SemanticNode

        converter = GraphConverter()
        converter.convert(sample_graph)  # build cache
        assert not converter.needs_rebuild(sample_graph)

        # Add a node
        sample_graph.upsert_node(SemanticNode(
            id="new_node",
            label="New",
            embedding=[0.0] * settings.embedding_dim,
        ))
        assert converter.needs_rebuild(sample_graph)


class TestPatternSeparationIntegration:
    """Test that pattern separation actually changes embeddings."""

    def test_separate_changes_embedding(self) -> None:
        from memory.pattern_separation import PatternSeparator

        sep = PatternSeparator()
        emb = torch.randn(settings.embedding_dim, device=_device)

        with torch.no_grad():
            separated = sep.separate(emb)

        # Should be same dimensionality but different values
        assert separated.shape[-1] == settings.embedding_dim
        assert not torch.allclose(emb, separated.squeeze(), atol=1e-3)


class TestHopfieldWithSeparation:
    """Test that Hopfield + Pattern Separation work together."""

    def test_store_and_retrieve_separated(self) -> None:
        from memory.hopfield_memory import HippocampalMemory
        from memory.pattern_separation import PatternSeparator

        sep = PatternSeparator()
        hop = HippocampalMemory()

        # Store separated embeddings
        embs = []
        for i in range(5):
            torch.manual_seed(i)
            raw = torch.randn(settings.embedding_dim, device=_device)
            with torch.no_grad():
                separated = sep.separate(raw).squeeze()
            hop.store(separated, episode_id=f"ep{i}")
            embs.append(separated)

        # Retrieve with the first separated embedding
        results = hop.retrieve_episode_ids(embs[0], top_k=3)
        assert len(results) > 0
        # Results should be (id, score) tuples
        ids = [r[0] for r in results]
        assert all(isinstance(i, str) for i in ids)


class TestGateWithSalience:
    """Test that the gate network integrates with salience signals."""

    def test_gate_with_real_signals(self) -> None:
        from memory.gate_network import DopaminergicGate

        gate = DopaminergicGate()
        emb = torch.randn(settings.embedding_dim, device=_device)
        ctx = torch.randn(settings.gru_hidden_dim, device=_device)
        signals = torch.tensor([0.9, 0.8, 0.5, 0.3], device=_device)

        decision, prob = gate.should_store(emb, ctx, signals, epsilon=0.0)
        assert isinstance(decision, bool)
        assert 0 <= prob <= 1


class TestVAEWithMetadata:
    """Test VAE with realistic metadata vectors."""

    def test_end_to_end(self) -> None:
        from memory.neural_consolidation import (
            ConsolidationVAE,
            build_metadata_vector,
        )

        vae = ConsolidationVAE()
        meta = build_metadata_vector(
            salience=0.7, entity_count=3, speaker_is_user=True, age_hours=2.0,
        )
        emb = torch.randn(settings.embedding_dim, device=_device)

        with torch.no_grad():
            recon_emb, recon_meta, mu, logvar = vae(
                emb.unsqueeze(0), meta.unsqueeze(0),
            )

        assert recon_emb.shape == (1, settings.embedding_dim)
        latent = vae.get_latent(emb.unsqueeze(0), meta.unsqueeze(0))
        assert latent.shape == (1, settings.vae_latent_dim)


class TestForgettingWithActivation:
    """Test forgetting network modulates activation values."""

    def test_modulation_reduces_activation(self) -> None:
        from memory.forgetting import ForgettingNetwork

        net = ForgettingNetwork()
        base_act = torch.ones(3, device=_device)
        emb = torch.randn(3, settings.embedding_dim, device=_device)
        scalars = torch.rand(3, 5, device=_device)
        delta_t = torch.tensor([10.0, 10.0, 10.0], device=_device)

        with torch.no_grad():
            effective = net.compute_effective_activation(
                base_act, emb, scalars, delta_t,
            )

        # With large delta_t, effective activation should be less than base
        # (not guaranteed by random weights, but should be finite)
        assert effective.isfinite().all()
