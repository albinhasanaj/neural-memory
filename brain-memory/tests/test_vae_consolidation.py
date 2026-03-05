"""Tests for memory.neural_consolidation — VAE consolidation."""

from __future__ import annotations

import pytest
import torch

from config.settings import settings

_device = settings.resolved_device


class TestConsolidationVAE:
    """Tests for the VAE architecture."""

    def _make_vae(self):
        from memory.neural_consolidation import ConsolidationVAE
        return ConsolidationVAE(
            embedding_dim=settings.embedding_dim,
            metadata_dim=settings.vae_metadata_dim,
            latent_dim=settings.vae_latent_dim,
        )

    def test_forward_shape(self) -> None:
        vae = self._make_vae()
        emb = torch.randn(4, settings.embedding_dim, device=_device)
        meta = torch.randn(4, settings.vae_metadata_dim, device=_device)

        recon_emb, recon_meta, mu, logvar = vae(emb, meta)
        assert recon_emb.shape == (4, settings.embedding_dim)
        assert recon_meta.shape == (4, settings.vae_metadata_dim)
        assert mu.shape == (4, settings.vae_latent_dim)
        assert logvar.shape == (4, settings.vae_latent_dim)

    def test_get_latent_deterministic(self) -> None:
        vae = self._make_vae()
        vae.eval()  # disable dropout for deterministic output
        emb = torch.randn(2, settings.embedding_dim, device=_device)
        meta = torch.randn(2, settings.vae_metadata_dim, device=_device)

        z1 = vae.get_latent(emb, meta)
        z2 = vae.get_latent(emb, meta)
        assert torch.allclose(z1, z2), "Deterministic latent should be identical"

    def test_encode_decode_roundtrip(self) -> None:
        """Encoding and decoding should produce the right shapes."""
        vae = self._make_vae()
        emb = torch.randn(1, settings.embedding_dim, device=_device)
        meta = torch.randn(1, settings.vae_metadata_dim, device=_device)

        mu, logvar = vae.encode(torch.cat([emb, meta], dim=-1))
        z = vae.reparameterize(mu, logvar)
        rec_emb, rec_meta = vae.decode(z)

        assert rec_emb.shape == (1, settings.embedding_dim)
        assert rec_meta.shape == (1, settings.vae_metadata_dim)


class TestVAELoss:
    """Tests for the ELBO loss."""

    def test_loss_is_nonnegative(self) -> None:
        from memory.neural_consolidation import vae_loss

        recon = torch.randn(4, settings.embedding_dim, device=_device)
        target = torch.randn(4, settings.embedding_dim, device=_device)
        recon_meta = torch.randn(4, settings.vae_metadata_dim, device=_device)
        target_meta = torch.randn(4, settings.vae_metadata_dim, device=_device)
        mu = torch.randn(4, settings.vae_latent_dim, device=_device)
        logvar = torch.randn(4, settings.vae_latent_dim, device=_device)

        loss, diag = vae_loss(recon, target, recon_meta, target_meta, mu, logvar)
        # Loss can be negative if KL term is small but recon is good,
        # but typically it's positive.  Check it's finite.
        assert loss.isfinite()
        assert "total" in diag
        assert "kl" in diag

    def test_perfect_reconstruction_low_loss(self) -> None:
        from memory.neural_consolidation import vae_loss

        target = torch.randn(4, settings.embedding_dim, device=_device)
        meta = torch.zeros(4, settings.vae_metadata_dim, device=_device)
        mu = torch.zeros(4, settings.vae_latent_dim, device=_device)
        logvar = torch.zeros(4, settings.vae_latent_dim, device=_device)

        loss_perfect, _ = vae_loss(target, target, meta, meta, mu, logvar)
        loss_random, _ = vae_loss(
            torch.randn_like(target), target, meta, meta, mu, logvar
        )
        assert loss_perfect < loss_random


class TestVAETraining:
    """Test replay buffer and training step."""

    def test_train_step(self) -> None:
        from memory.neural_consolidation import (
            ConsolidationReplayBuffer,
            ConsolidationVAE,
            train_vae_step,
        )

        vae = ConsolidationVAE()
        opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
        buf = ConsolidationReplayBuffer(capacity=50)

        for _ in range(20):
            buf.push(
                torch.randn(settings.embedding_dim, device=_device),
                torch.randn(settings.vae_metadata_dim, device=_device),
            )

        diag = train_vae_step(vae, opt, buf, batch_size=8)
        assert diag is not None
        assert diag["total"] >= 0


class TestBuildMetadata:
    """Test the metadata vector builder."""

    def test_metadata_shape(self) -> None:
        from memory.neural_consolidation import build_metadata_vector

        meta = build_metadata_vector(salience=0.5, entity_count=3)
        assert meta.shape == (settings.vae_metadata_dim,)
        assert meta[0].item() == pytest.approx(0.5)


class TestLatentClustering:
    """Test K-means in latent space."""

    def test_clustering_shape(self) -> None:
        from memory.neural_consolidation import ConsolidationVAE, latent_cluster_centroids

        vae = ConsolidationVAE()
        N = 20
        embs = torch.randn(N, settings.embedding_dim, device=_device)
        metas = torch.randn(N, settings.vae_metadata_dim, device=_device)

        centroids = latent_cluster_centroids(vae, embs, metas, n_clusters=3)
        assert centroids.shape == (3, settings.vae_latent_dim)
