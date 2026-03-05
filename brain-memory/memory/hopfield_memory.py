"""
Modern Hopfield Network — hippocampal associative memory.

Implements the Modern Hopfield Network (Ramsauer et al., 2020):

    retrieval = softmax(β · query @ stored_patterns.T) @ stored_values

Key brain-inspired properties:
  * **Pattern completion** — retrieve a full memory from a partial cue
  * **Pattern separation** — learned linear transform makes similar
    memories more distinct (mimics dentate gyrus)
  * **Exponential capacity** — unlike classical Hopfield nets
  * **Consolidation** — least-activated patterns are evicted when the
    store exceeds capacity (hippocampus → neocortex transfer)
"""

from __future__ import annotations

import logging
from typing import Any

import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config.settings import settings
from memory.episodic import EpisodicEntry

logger = logging.getLogger(__name__)
_device = settings.resolved_device


class HippocampalMemory(nn.Module):
    """Modern Hopfield Network acting as the hippocampal associative store.

    Parameters
    ----------
    pattern_dim :
        Dimensionality of stored patterns (embedding dim, default 384).
    value_dim :
        Dimensionality of stored values.  Defaults to
        ``pattern_dim + metadata_features`` for rich retrieval.
    beta_init :
        Initial inverse temperature for softmax attention.
    max_patterns :
        When the store exceeds this, consolidate least-activated patterns.
    """

    def __init__(
        self,
        pattern_dim: int = settings.embedding_dim,
        value_dim: int | None = None,
        beta_init: float = settings.hopfield_beta_init,
        max_patterns: int = settings.hopfield_max_patterns,
    ) -> None:
        super().__init__()
        self.pattern_dim = pattern_dim
        self.value_dim = value_dim or pattern_dim
        self.max_patterns = max_patterns

        # Learnable inverse temperature  (higher → sharper retrieval)
        self.log_beta = nn.Parameter(torch.tensor(float(beta_init)).log())

        # Pattern separation transform (mimics dentate gyrus)
        self.separator = nn.Sequential(
            nn.Linear(pattern_dim, pattern_dim),
            nn.LayerNorm(pattern_dim),
        )

        # Query projection — maps query into the stored-pattern space
        self.query_proj = nn.Linear(pattern_dim, pattern_dim)

        # Stored patterns and values as buffers (not parameters —
        # they grow dynamically, not trained via backprop)
        self.register_buffer("patterns", torch.empty(0, pattern_dim))
        self.register_buffer("values", torch.empty(0, self.value_dim))
        self.register_buffer("access_counts", torch.empty(0))

        # Episode ID index for mapping back to EpisodicStore
        self._episode_ids: list[str] = []

        self.to(_device)

    def load_state_dict(self, state_dict, strict=True, assign=False):  # type: ignore[override]
        """Populate ``_episode_ids`` to match loaded patterns size."""
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        n = self.patterns.shape[0]
        if len(self._episode_ids) != n:
            self._episode_ids = [f"__loaded_{i}" for i in range(n)]
        return result

    # ── Properties ──────────────────────────────────────────────────

    @property
    def beta(self) -> Tensor:
        return self.log_beta.exp()

    @property
    def num_patterns(self) -> int:
        return self.patterns.shape[0]  # type: ignore[union-attr]

    # ── Store ───────────────────────────────────────────────────────

    def store(
        self,
        embedding: Tensor,
        value: Tensor | None = None,
        episode_id: str = "",
    ) -> None:
        """Add a new pattern to the associative store.

        Parameters
        ----------
        embedding :
            ``Tensor[D]`` — the raw embedding to store. Pattern separation
            is applied internally.
        value :
            ``Tensor[V]`` — the value vector returned on retrieval.
            Defaults to the embedding itself.
        episode_id :
            ID linking back to the EpisodicStore entry.
        """
        with torch.no_grad():
            pattern = self.separator(embedding.to(_device).unsqueeze(0))  # [1, D]
        if value is None:
            value = embedding.to(_device).unsqueeze(0)
        else:
            value = value.to(_device).unsqueeze(0)

        self.patterns = torch.cat([self.patterns, pattern], dim=0)  # type: ignore
        self.values = torch.cat([self.values, value], dim=0)  # type: ignore
        self.access_counts = torch.cat([  # type: ignore
            self.access_counts,
            torch.zeros(1, device=_device),
        ])
        self._episode_ids.append(episode_id)

        # Capacity management
        if self.num_patterns > self.max_patterns:
            self._consolidate()

    # ── Retrieve ────────────────────────────────────────────────────

    def forward(
        self,
        query: Tensor,
        top_k: int | None = None,
    ) -> tuple[Tensor, Tensor, list[int]]:
        """Retrieve from associative memory via pattern completion.

        Parameters
        ----------
        query :
            ``Tensor[D]`` or ``Tensor[B, D]`` — retrieval cue.
        top_k :
            If given, return only the top-K matched patterns.

        Returns
        -------
        retrieved_values :
            ``Tensor[V]`` or ``Tensor[B, V]`` — pattern-completed values.
        attention_weights :
            ``Tensor[M]`` or ``Tensor[B, M]`` — attention over stored patterns.
        indices :
            Indices of top-K patterns (if top_k is set).
        """
        if self.num_patterns == 0:
            D = self.value_dim
            if query.dim() == 1:
                return torch.zeros(D, device=_device), torch.empty(0, device=_device), []
            return torch.zeros(query.shape[0], D, device=_device), torch.empty(0, device=_device), []

        was_1d = False
        if query.dim() == 1:
            query = query.unsqueeze(0)
            was_1d = True

        query = query.to(_device)
        q = self.query_proj(query)  # [B, D]

        # Modern Hopfield retrieval: softmax(β * Q @ K^T) @ V
        patterns = self.patterns  # [M, D]
        logits = self.beta * (q @ patterns.t())  # [B, M]
        attn = F.softmax(logits, dim=-1)  # [B, M]

        # Update access counts
        with torch.no_grad():
            self.access_counts += attn.sum(dim=0)  # type: ignore

        retrieved = attn @ self.values  # [B, V]

        # Top-K selection
        indices: list[int] = []
        if top_k is not None and top_k < self.num_patterns:
            _, topk_idx = attn.topk(top_k, dim=-1)
            indices = topk_idx.squeeze(0).tolist() if was_1d else topk_idx.tolist()
        else:
            indices = list(range(self.num_patterns))

        if was_1d:
            return retrieved.squeeze(0), attn.squeeze(0), indices

        return retrieved, attn, indices

    def retrieve_episode_ids(
        self,
        query: Tensor,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Convenience: return the top-K episode IDs with their attention weights."""
        _, attn, _ = self.forward(query, top_k=None)
        if attn.numel() == 0:
            return []
        weights, indices = attn.topk(min(top_k, self.num_patterns))
        results = []
        for w, idx in zip(weights.tolist(), indices.tolist()):
            if idx < len(self._episode_ids):
                results.append((self._episode_ids[idx], w))
        return results

    # ── Consolidation (evict least-accessed) ────────────────────────

    def _consolidate(self) -> None:
        """Remove the least-accessed patterns to stay under capacity."""
        keep = self.max_patterns
        if self.num_patterns <= keep:
            return

        # Keep the top-K by access count
        _, keep_idx = self.access_counts.topk(keep)
        keep_idx = keep_idx.sort().values

        self.patterns = self.patterns[keep_idx]  # type: ignore
        self.values = self.values[keep_idx]  # type: ignore
        self.access_counts = self.access_counts[keep_idx]  # type: ignore

        old_ids = self._episode_ids
        self._episode_ids = [old_ids[i] for i in keep_idx.tolist()]

        logger.info(
            "Hopfield consolidation: %d → %d patterns",
            len(old_ids), len(self._episode_ids),
        )

    # ── Utilities ───────────────────────────────────────────────────

    def clear(self) -> None:
        """Reset the store."""
        self.patterns = torch.empty(0, self.pattern_dim, device=_device)  # type: ignore
        self.values = torch.empty(0, self.value_dim, device=_device)  # type: ignore
        self.access_counts = torch.empty(0, device=_device)  # type: ignore
        self._episode_ids = []


# ────────────────────────────────────────────────────────────────────
# Replay buffer & training for Hopfield network
# ────────────────────────────────────────────────────────────────────


class HopfieldReplayBuffer:
    """Stores ``(query_embedding, positive_embedding)`` pairs.

    The training signal is retrieval accuracy: given a query, the
    Hopfield network's retrieval should be closer (in cosine space) to
    the positive (correct) embedding than to random negatives.

    This trains the learnable parameters: ``separator``, ``query_proj``,
    and ``log_beta``.
    """

    def __init__(self, capacity: int = settings.training_replay_buffer_size) -> None:
        self._buffer: deque[tuple[Tensor, Tensor]] = deque(maxlen=capacity)

    def push(self, query: Tensor, positive: Tensor) -> None:
        """Store a (query, positive_target) pair.

        Parameters
        ----------
        query :
            ``Tensor[D]`` — the retrieval cue (e.g., current embedding).
        positive :
            ``Tensor[D]`` — the embedding that *should* rank highest.
        """
        self._buffer.append((
            query.detach().cpu(),
            positive.detach().cpu(),
        ))

    def sample(self, batch_size: int) -> list[tuple[Tensor, Tensor]] | None:
        if len(self._buffer) < batch_size:
            return None
        indices = random.sample(range(len(self._buffer)), batch_size)
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


def train_hopfield_step(
    net: HippocampalMemory,
    optimizer: torch.optim.Optimizer,
    replay_buffer: HopfieldReplayBuffer,
    batch_size: int = 8,
) -> dict[str, float] | None:
    """One training step for the Hopfield network.

    Uses a contrastive retrieval objective: the query_proj and separator
    transforms are trained so that ``forward(query)`` retrieves the
    correct associated value with high attention weight.

    Loss: for each (query, positive) pair, compute the retrieval output
    and measure cosine distance to the positive target.
    """
    if net.num_patterns < 2:
        return None  # Need stored patterns to train meaningfully

    samples = replay_buffer.sample(batch_size)
    if samples is None:
        return None

    net.train()
    total_loss = torch.tensor(0.0, device=_device)
    n = 0

    for query, positive in samples:
        query = query.to(_device)
        positive = positive.to(_device)

        # Forward through trainable components (query_proj, separator, beta)
        q = net.query_proj(query.unsqueeze(0))  # [1, D]

        # Compute attention
        patterns = net.patterns.detach()  # [M, D] — detach stored patterns
        logits = net.beta * (q @ patterns.t())   # [1, M]
        attn = F.softmax(logits, dim=-1)         # [1, M]

        # Retrieve
        values = net.values.detach()             # [M, V]
        retrieved = (attn @ values).squeeze(0)   # [V]

        # Loss: 1 - cosine_similarity(retrieved, positive)
        cos_sim = F.cosine_similarity(
            retrieved.unsqueeze(0),
            positive.unsqueeze(0),
        )
        total_loss = total_loss + (1.0 - cos_sim.squeeze())
        n += 1

    loss = total_loss / max(n, 1)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), settings.training_grad_clip)
    optimizer.step()

    return {"loss": loss.item(), "total": loss.item()}
