"""
Pattern Separation — sparse autoencoder (dentate gyrus analogue).

In the brain, the dentate gyrus receives ~384-dimensional cortical
representations and projects them into a *much* higher-dimensional
sparse code (granule cells).  This orthogonalises overlapping inputs
so they can be stored distinctly in CA3 (the Hopfield network).

Architecture
~~~~~~~~~~~~
``384 → 2048 (expansion) → ReLU → top-K sparsity → 384 (bottleneck)``

The loss combines:
  * **Reconstruction** — the decoded vector should match the input.
  * **Sparsity penalty** — L1 on the hidden activations to enforce
    only *top_k* active units.

The separated (sparse) representation is what actually gets stored in
EpisodicStore and the Hopfield network, making similar memories more
distinguishable.
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
# Sparse Autoencoder
# ────────────────────────────────────────────────────────────────────


class PatternSeparator(nn.Module):
    """Sparse autoencoder for pattern separation.

    Parameters
    ----------
    input_dim :
        Embedding dimensionality   (384).
    expansion_dim :
        Size of the sparse hidden layer  (2048).
    top_k :
        Number of active (non-zero) units in the sparse code  (50).
    """

    def __init__(
        self,
        input_dim: int = settings.embedding_dim,
        expansion_dim: int = settings.pattern_sep_expansion,
        top_k: int = settings.pattern_sep_top_k,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.expansion_dim = expansion_dim
        self.top_k = top_k

        # Encoder: input → expansion
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, expansion_dim),
            nn.LayerNorm(expansion_dim),
            nn.ReLU(),
        )

        # Decoder: expansion → reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(expansion_dim, input_dim),
        )

        # Bottleneck projector: sparse code → compact separated embedding
        self.bottleneck = nn.Sequential(
            nn.Linear(expansion_dim, input_dim),
            nn.LayerNorm(input_dim),
        )

        self.to(_device)

    def _top_k_mask(self, h: Tensor) -> Tensor:
        """Zero out all but the top-K activations per sample."""
        if h.dim() == 1:
            h = h.unsqueeze(0)
        vals, idx = h.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(1, idx, 1.0)
        return h * mask

    def encode_sparse(self, x: Tensor) -> Tensor:
        """Return the sparse hidden code — ``Tensor[..., expansion_dim]``.

        Only the top-K activations are non-zero.
        """
        h = self.encoder(x.to(_device))
        return self._top_k_mask(h)

    def separate(self, x: Tensor) -> Tensor:
        """Separate an embedding: input → sparse code → bottleneck.

        Returns a 384-dim vector that is more orthogonal to other
        separated embeddings than the raw input would be.
        """
        was_1d = x.dim() == 1
        sparse = self.encode_sparse(x)
        out = self.bottleneck(sparse)
        if was_1d:
            out = out.squeeze(0)
        return out

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Full forward pass.

        Returns ``(reconstruction, sparse_code, separated_embedding)``.
        """
        was_1d = x.dim() == 1
        x = x.to(_device)
        sparse = self.encode_sparse(x)
        recon = self.decoder(sparse)
        separated = self.bottleneck(sparse)
        if was_1d:
            recon = recon.squeeze(0)
            sparse = sparse.squeeze(0)
            separated = separated.squeeze(0)
        return recon, sparse, separated


# ────────────────────────────────────────────────────────────────────
# Loss
# ────────────────────────────────────────────────────────────────────


def pattern_sep_loss(
    recon: Tensor,
    target: Tensor,
    sparse_code: Tensor,
    l1_weight: float = 1e-4,
) -> tuple[Tensor, dict[str, float]]:
    """Reconstruction loss + L1 sparsity penalty.

    Parameters
    ----------
    recon : Tensor[B, D]
    target : Tensor[B, D]
    sparse_code : Tensor[B, E]
    l1_weight : float
        Strength of the sparsity penalty.

    Returns
    -------
    (total_loss, diagnostics)
    """
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    l1_loss = sparse_code.abs().mean()
    total = recon_loss + l1_weight * l1_loss
    return total, {
        "recon": recon_loss.item(),
        "l1_sparsity": l1_loss.item(),
        "total": total.item(),
    }


# ────────────────────────────────────────────────────────────────────
# Replay buffer & training
# ────────────────────────────────────────────────────────────────────


class SeparationReplayBuffer:
    """Stores raw embeddings for autoencoder training."""

    def __init__(self, capacity: int = settings.training_replay_buffer_size) -> None:
        self._buffer: deque[Tensor] = deque(maxlen=capacity)

    def push(self, embedding: Tensor) -> None:
        self._buffer.append(embedding.detach().to(_device))

    def sample(self, batch_size: int) -> Tensor:
        import random
        indices = random.sample(range(len(self._buffer)), min(batch_size, len(self._buffer)))
        return torch.stack([self._buffer[i] for i in indices])

    def __len__(self) -> int:
        return len(self._buffer)


def train_separator_step(
    separator: PatternSeparator,
    optimizer: torch.optim.Optimizer,
    replay_buffer: SeparationReplayBuffer,
    batch_size: int = 16,
    l1_weight: float = 1e-4,
) -> dict[str, float] | None:
    """One training step for the pattern separator."""
    if len(replay_buffer) < batch_size:
        return None

    batch = replay_buffer.sample(batch_size)
    separator.train()

    recon, sparse, _ = separator(batch)
    loss, diag = pattern_sep_loss(recon, batch, sparse, l1_weight=l1_weight)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(separator.parameters(), settings.training_grad_clip)
    optimizer.step()

    return diag


# ────────────────────────────────────────────────────────────────────
# Quality metrics
# ────────────────────────────────────────────────────────────────────


def separation_quality(
    separator: PatternSeparator,
    embeddings: Tensor,
) -> dict[str, float]:
    """Compute metrics on how well the separator orthogonalises inputs.

    Returns
    -------
    dict with keys:
      * ``mean_cosine_raw`` — avg pairwise cosine sim of raw inputs
      * ``mean_cosine_separated`` — avg pairwise cosine sim after separation
      * ``separation_gain`` — improvement ratio
    """
    separator.eval()
    with torch.no_grad():
        separated = separator.separate(embeddings)

    def _mean_cosine(x: Tensor) -> float:
        x = F.normalize(x, dim=-1)
        sim = x @ x.t()
        n = sim.shape[0]
        if n < 2:
            return 0.0
        # Exclude diagonal
        mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
        return sim[mask].mean().item()

    raw_sim = _mean_cosine(embeddings.to(_device))
    sep_sim = _mean_cosine(separated)

    gain = (raw_sim - sep_sim) / (abs(raw_sim) + 1e-9)
    return {
        "mean_cosine_raw": raw_sim,
        "mean_cosine_separated": sep_sim,
        "separation_gain": gain,
    }
