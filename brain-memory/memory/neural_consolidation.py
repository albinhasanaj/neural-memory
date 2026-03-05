"""
VAE-based Memory Consolidation — hippocampus → neocortex transfer.

Instead of the LLM-based fact extraction used in
:pymod:`memory.consolidation`, this module learns a Variational
Autoencoder over episodic memory embeddings.  The VAE's latent space
*is* the semantic memory: nearby latent codes represent related
information.

Key design ideas
~~~~~~~~~~~~~~~~
* **Encoder** — compresses an episodic embedding + metadata into a
  low-dimensional latent vector.
* **Decoder** — reconstructs the embedding (confirms information is
  preserved) and predicts a *cluster-assignment logit* (soft categories).
* **Training** — ELBO loss with β-annealing.  Trains online via a
  replay buffer so we can interleave consolidation with conversation.
* **Consolidation** — encode a batch of episodes, cluster them in latent
  space (K-means on z), and form semantic graph nodes from cluster
  centroids.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config.settings import settings

logger = logging.getLogger(__name__)
_device = settings.resolved_device


# ────────────────────────────────────────────────────────────────────
# Replay buffer
# ────────────────────────────────────────────────────────────────────


class ConsolidationReplayBuffer:
    """Stores ``(embedding, metadata)`` tuples for VAE training."""

    def __init__(self, capacity: int = settings.training_replay_buffer_size) -> None:
        self._buffer: deque[tuple[Tensor, Tensor]] = deque(maxlen=capacity)

    def push(self, embedding: Tensor, metadata: Tensor) -> None:
        self._buffer.append((embedding.detach().to(_device), metadata.detach().to(_device)))

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Return a random mini-batch ``(embeddings [B,D], metadata [B,M])``."""
        import random
        indices = random.sample(range(len(self._buffer)), min(batch_size, len(self._buffer)))
        embs = torch.stack([self._buffer[i][0] for i in indices])
        metas = torch.stack([self._buffer[i][1] for i in indices])
        return embs, metas

    def __len__(self) -> int:
        return len(self._buffer)


# ────────────────────────────────────────────────────────────────────
# VAE architecture
# ────────────────────────────────────────────────────────────────────


class ConsolidationVAE(nn.Module):
    """Variational Autoencoder for episodic consolidation.

    Input
    -----
    ``concat(embedding, metadata)`` of dimension
    ``embedding_dim + metadata_dim``.

    Latent
    ------
    ``z ∈ R^{latent_dim}`` — this IS the semantic representation.

    Output
    ------
    * Reconstructed embedding ``x̂ ∈ R^{embedding_dim}``
    * Metadata prediction ``m̂ ∈ R^{metadata_dim}``
    """

    def __init__(
        self,
        embedding_dim: int = settings.embedding_dim,
        metadata_dim: int = settings.vae_metadata_dim,
        latent_dim: int = settings.vae_latent_dim,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.metadata_dim = metadata_dim
        self.latent_dim = latent_dim

        in_dim = embedding_dim + metadata_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.fc_recon_emb = nn.Linear(256, embedding_dim)
        self.fc_recon_meta = nn.Linear(256, metadata_dim)

        self.to(_device)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(mu, log_var)``."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(reconstructed_embedding, reconstructed_metadata)``."""
        h = self.decoder(z)
        return self.fc_recon_emb(h), self.fc_recon_meta(h)

    def forward(
        self, embedding: Tensor, metadata: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full forward pass.

        Returns ``(recon_emb, recon_meta, mu, logvar)``.
        """
        x = torch.cat([embedding, metadata], dim=-1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_emb, recon_meta = self.decode(z)
        return recon_emb, recon_meta, mu, logvar

    def get_latent(self, embedding: Tensor, metadata: Tensor) -> Tensor:
        """Deterministic latent (μ only — no noise)."""
        x = torch.cat([embedding, metadata], dim=-1)
        mu, _ = self.encode(x)
        return mu


# ────────────────────────────────────────────────────────────────────
# Loss
# ────────────────────────────────────────────────────────────────────


def vae_loss(
    recon_emb: Tensor,
    target_emb: Tensor,
    recon_meta: Tensor,
    target_meta: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """ELBO loss = reconstruction + β × KL divergence.

    Returns ``(total_loss, diagnostics_dict)``.
    """
    # Reconstruction
    recon_loss_emb = F.mse_loss(recon_emb, target_emb, reduction="mean")
    recon_loss_meta = F.mse_loss(recon_meta, target_meta, reduction="mean")
    recon_loss = recon_loss_emb + recon_loss_meta

    # KL divergence:  -0.5 * Σ(1 + log(σ²) − μ² − σ²)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_loss + beta * kl
    diag = {
        "recon_emb": recon_loss_emb.item(),
        "recon_meta": recon_loss_meta.item(),
        "kl": kl.item(),
        "total": total.item(),
    }
    return total, diag


# ────────────────────────────────────────────────────────────────────
# Training step
# ────────────────────────────────────────────────────────────────────


def train_vae_step(
    vae: ConsolidationVAE,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ConsolidationReplayBuffer,
    batch_size: int = settings.vae_replay_batch,
    beta: float = 1.0,
    noise_std: float = settings.vae_replay_noise,
) -> dict[str, float] | None:
    """One training step from the replay buffer.

    Returns loss diagnostics, or *None* if the buffer is too small.
    """
    if len(replay_buffer) < batch_size:
        return None

    embs, metas = replay_buffer.sample(batch_size)

    # Optional noise augmentation
    if noise_std > 0:
        embs = embs + torch.randn_like(embs) * noise_std

    vae.train()
    recon_emb, recon_meta, mu, logvar = vae(embs, metas)
    loss, diag = vae_loss(recon_emb, embs, recon_meta, metas, mu, logvar, beta=beta)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(vae.parameters(), settings.training_grad_clip)
    optimizer.step()

    return diag


# ────────────────────────────────────────────────────────────────────
# Consolidation helpers
# ────────────────────────────────────────────────────────────────────


def build_metadata_vector(
    salience: float = 0.0,
    entity_count: int = 0,
    speaker_is_user: bool = True,
    age_hours: float = 0.0,
) -> Tensor:
    """Build a fixed-dimension metadata vector from episode attributes.

    Currently ``metadata_dim`` features; unused slots are zero-padded.
    """
    meta = torch.zeros(settings.vae_metadata_dim, device=_device)
    meta[0] = salience
    meta[1] = float(entity_count) / 10.0  # normalise
    meta[2] = 1.0 if speaker_is_user else 0.0
    meta[3] = min(age_hours / 168.0, 1.0)  # normalise to ~1 week
    return meta


def latent_cluster_centroids(
    vae: ConsolidationVAE,
    embeddings: Tensor,
    metadata: Tensor,
    n_clusters: int = 8,
) -> Tensor:
    """Compute K-means cluster centroids in the VAE latent space.

    Parameters
    ----------
    vae :
        Trained ConsolidationVAE.
    embeddings :
        ``Tensor[N, D]`` — episodic embeddings.
    metadata :
        ``Tensor[N, M]`` — metadata vectors.
    n_clusters :
        Number of clusters.

    Returns
    -------
    centroids :
        ``Tensor[K, latent_dim]`` — cluster centroids.
    """
    with torch.no_grad():
        latent = vae.get_latent(embeddings, metadata)  # [N, Z]

    # Simple K-means (Lloyd's algorithm) in latent space
    N = latent.shape[0]
    k = min(n_clusters, N)
    if k == 0:
        return torch.empty(0, vae.latent_dim, device=_device)

    # Initialise with K-means++: first centroid is random
    indices = [torch.randint(N, (1,)).item()]
    for _ in range(1, k):
        dists = torch.cdist(latent, latent[indices])  # [N, curr_k]
        min_dists = dists.min(dim=1).values  # [N]
        probs = min_dists / (min_dists.sum() + 1e-9)
        idx = torch.multinomial(probs, 1).item()
        indices.append(idx)

    centroids = latent[indices].clone()  # [K, Z]

    for _ in range(20):  # fixed iterations
        dists = torch.cdist(latent, centroids)  # [N, K]
        assignments = dists.argmin(dim=1)  # [N]
        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            mask = (assignments == j)
            if mask.any():
                new_centroids[j] = latent[mask].mean(dim=0)
            else:
                new_centroids[j] = centroids[j]
        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return centroids


def latent_cluster_assignments(
    vae: ConsolidationVAE,
    embeddings: Tensor,
    metadata: Tensor,
    n_clusters: int = 8,
) -> list[int]:
    """Compute K-means cluster assignments in the VAE latent space.

    Returns a list of cluster indices (one per episode), suitable for
    grouping episodes into clusters for LLM-based fact extraction.
    """
    with torch.no_grad():
        latent = vae.get_latent(embeddings, metadata)  # [N, Z]

    N = latent.shape[0]
    k = min(n_clusters, N)
    if k == 0:
        return []

    # K-means++ initialization
    indices = [torch.randint(N, (1,)).item()]
    for _ in range(1, k):
        dists = torch.cdist(latent, latent[indices])
        min_dists = dists.min(dim=1).values
        probs = min_dists / (min_dists.sum() + 1e-9)
        idx = torch.multinomial(probs, 1).item()
        indices.append(idx)

    centroids = latent[indices].clone()

    assignments = torch.zeros(N, dtype=torch.long)
    for _ in range(20):
        dists = torch.cdist(latent, centroids)
        assignments = dists.argmin(dim=1)
        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            mask = (assignments == j)
            if mask.any():
                new_centroids[j] = latent[mask].mean(dim=0)
            else:
                new_centroids[j] = centroids[j]
        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return assignments.tolist()
