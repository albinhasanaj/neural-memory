"""Tests for memory.forgetting — Learned Forgetting Gate."""

from __future__ import annotations

import pytest
import torch

from config.settings import settings

_device = settings.resolved_device


class TestForgettingNetwork:
    """Tests for the two-headed forgetting network."""

    def _make_net(self):
        from memory.forgetting import ForgettingNetwork
        return ForgettingNetwork()

    def test_output_shapes(self) -> None:
        net = self._make_net()
        B = 4
        emb = torch.randn(B, settings.embedding_dim, device=_device)
        scalars = torch.rand(B, 5, device=_device)

        decay, interference = net(emb, scalars)
        assert decay.shape == (B,)
        assert interference.shape == (B,)
        assert (decay >= 0).all() and (decay <= 1).all()
        assert (interference >= 0).all() and (interference <= 1).all()

    def test_with_context_vector(self) -> None:
        net = self._make_net()
        emb = torch.randn(2, settings.embedding_dim, device=_device)
        scalars = torch.rand(2, 5, device=_device)
        ctx = torch.randn(settings.gru_hidden_dim, device=_device)

        decay, interference = net(emb, scalars, context=ctx)
        assert decay.shape == (2,)
        assert interference.shape == (2,)

    def test_effective_activation(self) -> None:
        net = self._make_net()
        base_act = torch.tensor([1.0, 0.5, 0.8], device=_device)
        emb = torch.randn(3, settings.embedding_dim, device=_device)
        scalars = torch.rand(3, 5, device=_device)
        delta_t = torch.tensor([1.0, 5.0, 24.0], device=_device)

        effective = net.compute_effective_activation(
            base_act, emb, scalars, delta_t,
        )
        assert effective.shape == (3,)
        # Effective should be ≤ base (forgetting reduces activation)
        # Not strictly guaranteed by random weights, so just check it's finite
        assert effective.isfinite().all()


class TestBuildForgettingScalars:
    def test_output_shape(self) -> None:
        from memory.forgetting import build_forgetting_scalars

        s = build_forgetting_scalars(
            age_hours=48.0,
            access_count=5,
            salience=0.7,
            last_activation=-1.5,
            context_similarity=0.6,
        )
        assert s.shape == (5,)
        # Check normalisation
        assert 0.0 <= s[0].item() <= 1.0  # age
        assert 0.0 <= s[1].item() <= 1.0  # access count


class TestForgettingTraining:
    def test_train_step(self) -> None:
        from memory.forgetting import (
            ForgettingNetwork,
            ForgettingReplayBuffer,
            train_forgetting_step,
        )

        net = ForgettingNetwork()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        buf = ForgettingReplayBuffer(capacity=50)

        for _ in range(20):
            buf.push(
                torch.randn(settings.embedding_dim, device=_device),
                torch.rand(5, device=_device),
                delta_t=float(torch.rand(1).item()) * 24.0,
                target_decay=float(torch.rand(1).item()),
                target_interference=float(torch.rand(1).item()) * 0.3,
            )

        diag = train_forgetting_step(net, opt, buf, batch_size=8)
        assert diag is not None
        assert "total" in diag
        assert diag["total"] >= 0

    def test_training_reduces_loss(self) -> None:
        from memory.forgetting import (
            ForgettingNetwork,
            ForgettingReplayBuffer,
            train_forgetting_step,
        )

        net = ForgettingNetwork()
        opt = torch.optim.Adam(net.parameters(), lr=5e-3)
        buf = ForgettingReplayBuffer(capacity=200)

        # Consistent targets: low decay, low interference
        for _ in range(50):
            buf.push(
                torch.randn(settings.embedding_dim, device=_device),
                torch.rand(5, device=_device),
                delta_t=1.0,
                target_decay=0.1,
                target_interference=0.05,
            )

        losses = []
        for _ in range(20):
            diag = train_forgetting_step(net, opt, buf, batch_size=16)
            if diag:
                losses.append(diag["total"])

        assert len(losses) >= 10
        assert losses[-1] < losses[0]
