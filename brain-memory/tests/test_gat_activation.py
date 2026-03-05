"""Tests for memory.neural_activation — GAT spreading activation."""

from __future__ import annotations

import pytest
import torch

from config.settings import settings

_device = settings.resolved_device


# ── ManualGATv2Layer ────────────────────────────────────────────────


class TestManualGATv2Layer:
    """Tests for the pure-PyTorch GATv2 fallback layer."""

    def test_output_shape(self) -> None:
        from memory.neural_activation import ManualGATv2Layer

        layer = ManualGATv2Layer(in_dim=32, out_dim=16, num_heads=2, edge_dim=4).to(_device)
        N = 10
        x = torch.randn(N, 32, device=_device)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long, device=_device
        )
        edge_attr = torch.randn(5, 4, device=_device)

        out = layer(x, edge_index, edge_attr)
        # Multi-head output concatenated: 2 heads × 16 channels = 32
        assert out.shape == (N, 2 * 16)

    def test_no_edges(self) -> None:
        from memory.neural_activation import ManualGATv2Layer

        layer = ManualGATv2Layer(in_dim=16, out_dim=8, num_heads=1, edge_dim=3).to(_device)
        x = torch.randn(5, 16, device=_device)
        edge_index = torch.empty(2, 0, dtype=torch.long, device=_device)
        edge_attr = torch.empty(0, 3, device=_device)

        out = layer(x, edge_index, edge_attr)
        assert out.shape == (5, 8)


# ── MemoryGAT ──────────────────────────────────────────────────────


class TestMemoryGAT:
    """Tests for the full GAT model."""

    def test_forward_shape(self) -> None:
        from memory.neural_activation import MemoryGAT
        from memory.graph_converter import PyGData

        node_feat_dim = 392  # matches graph_converter output
        edge_feat_dim = 13
        gat = MemoryGAT(
            node_feature_dim=node_feat_dim,
            edge_dim=edge_feat_dim,
            hidden_dims=[32, 16, 8],
            num_heads=2,
            context_dim=settings.embedding_dim,
        )

        N = 6
        data = PyGData(
            x=torch.randn(N, node_feat_dim, device=_device),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long, device=_device),
            edge_attr=torch.randn(3, edge_feat_dim, device=_device),
        )
        context = torch.randn(settings.embedding_dim, device=_device)

        scores = gat(data, context)
        assert scores.shape == (N,)
        # Scores should be probabilities (after sigmoid)
        assert (scores >= 0).all()
        assert (scores <= 1).all()


# ── ActivationReplayBuffer & Training ──────────────────────────────


class TestActivationTraining:
    """Test the replay buffer and one training step."""

    def test_replay_push_and_sample(self) -> None:
        from memory.neural_activation import ActivationReplayBuffer

        buf = ActivationReplayBuffer(capacity=10)
        # Push synthetic data
        from memory.graph_converter import PyGData

        for _ in range(5):
            data = PyGData(
                x=torch.randn(3, 392),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                edge_attr=torch.randn(2, 13),
            )
            ctx = torch.randn(settings.embedding_dim)
            labels = torch.rand(3)
            buf.push(data, ctx, labels)

        assert len(buf) == 5

    def test_train_step_runs(self) -> None:
        from memory.neural_activation import (
            ActivationReplayBuffer,
            MemoryGAT,
            train_gat_step,
        )
        from memory.graph_converter import PyGData

        gat = MemoryGAT(
            node_feature_dim=392,
            edge_dim=13,
            hidden_dims=[16, 8, 4],
            num_heads=1,
        )
        optimizer = torch.optim.Adam(gat.parameters(), lr=1e-3)
        buf = ActivationReplayBuffer(capacity=50)

        # Fill buffer
        for _ in range(10):
            data = PyGData(
                x=torch.randn(4, 392),
                edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
                edge_attr=torch.randn(3, 13),
            )
            buf.push(data, torch.randn(settings.embedding_dim), torch.rand(4))

        loss = train_gat_step(gat, optimizer, buf.sample(4))
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0
