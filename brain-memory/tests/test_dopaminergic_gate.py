"""Tests for memory.gate_network — Dopaminergic Gate."""

from __future__ import annotations

import pytest
import torch

from config.settings import settings

_device = settings.resolved_device


class TestDopaminergicGate:
    """Tests for the gating network."""

    def _make_gate(self):
        from memory.gate_network import DopaminergicGate
        return DopaminergicGate()

    def test_output_shape_single(self) -> None:
        gate = self._make_gate()
        emb = torch.randn(settings.embedding_dim, device=_device)
        ctx = torch.randn(settings.gru_hidden_dim, device=_device)
        sig = torch.tensor([0.5, 0.3, 0.2, 0.1], device=_device)

        prob = gate(emb, ctx, sig)
        assert prob.shape == ()  # scalar
        assert 0.0 <= prob.item() <= 1.0

    def test_output_shape_batch(self) -> None:
        gate = self._make_gate()
        B = 4
        emb = torch.randn(B, settings.embedding_dim, device=_device)
        ctx = torch.randn(B, settings.gru_hidden_dim, device=_device)
        sig = torch.rand(B, 4, device=_device)

        probs = gate(emb, ctx, sig)
        assert probs.shape == (B,)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_should_store_returns_tuple(self) -> None:
        gate = self._make_gate()
        emb = torch.randn(settings.embedding_dim, device=_device)
        ctx = torch.randn(settings.gru_hidden_dim, device=_device)
        sig = torch.tensor([0.5, 0.3, 0.2, 0.1], device=_device)

        decision, prob = gate.should_store(emb, ctx, sig, epsilon=0.0)
        assert isinstance(decision, bool)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_epsilon_exploration(self) -> None:
        """With epsilon=1.0, decisions should be random."""
        gate = self._make_gate()
        emb = torch.randn(settings.embedding_dim, device=_device)
        ctx = torch.randn(settings.gru_hidden_dim, device=_device)
        sig = torch.tensor([0.5, 0.3, 0.2, 0.1], device=_device)

        decisions = set()
        for _ in range(50):
            d, _ = gate.should_store(emb, ctx, sig, epsilon=1.0)
            decisions.add(d)

        # With 50 tries and full exploration, we should see both True and False
        assert len(decisions) == 2


class TestResidualBlock:
    def test_residual_preserves_shape(self) -> None:
        from memory.gate_network import ResidualBlock
        block = ResidualBlock(dim=64).to(_device)
        x = torch.randn(3, 64, device=_device)
        out = block(x)
        assert out.shape == (3, 64)


class TestGateTraining:
    """Test replay buffer and training."""

    def test_train_step(self) -> None:
        from memory.gate_network import (
            DopaminergicGate,
            GateReplayBuffer,
            train_gate_step,
        )

        gate = DopaminergicGate()
        opt = torch.optim.Adam(gate.parameters(), lr=1e-3)
        buf = GateReplayBuffer(capacity=50)

        for _ in range(20):
            buf.push(
                torch.randn(settings.embedding_dim, device=_device),
                torch.randn(settings.gru_hidden_dim, device=_device),
                torch.rand(4, device=_device),
                reward=float(torch.rand(1).item()),
            )

        diag = train_gate_step(gate, opt, buf, batch_size=8)
        assert diag is not None
        assert "gate_loss" in diag
        assert diag["gate_loss"] >= 0

    def test_training_reduces_loss(self) -> None:
        from memory.gate_network import (
            DopaminergicGate,
            GateReplayBuffer,
            train_gate_step,
        )

        gate = DopaminergicGate()
        opt = torch.optim.Adam(gate.parameters(), lr=1e-2)
        buf = GateReplayBuffer(capacity=100)

        # Create consistent data — reward=1 for high-salience inputs
        for _ in range(40):
            buf.push(
                torch.randn(settings.embedding_dim, device=_device),
                torch.randn(settings.gru_hidden_dim, device=_device),
                torch.tensor([0.9, 0.8, 0.7, 0.6], device=_device),
                reward=1.0,
            )

        losses = []
        for _ in range(10):
            diag = train_gate_step(gate, opt, buf, batch_size=16)
            if diag:
                losses.append(diag["gate_loss"])

        # Loss should generally decrease
        assert len(losses) >= 5
        assert losses[-1] <= losses[0] + 0.1  # allow small noise
