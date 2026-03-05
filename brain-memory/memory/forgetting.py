"""
Learned Forgetting Gate — neural replacement for ACT-R power-law decay.

In the baseline system, episodic memories decay via the ACT-R formula:
``activation = ln(Σ(t − t_j)^{-0.5})``.  This is elegant but static —
it can't learn that certain *kinds* of memories should be forgotten
faster (e.g., temporary task context) while others should persist
(e.g., user preferences).

Architecture
~~~~~~~~~~~~
A two-headed network that takes a memory's features and outputs:

1. **Decay rate** ∈ (0, 1] — how quickly this memory should fade.
2. **Interference score** ∈ [0, 1] — how much this memory conflicts
   with more recent memories (proactive / retroactive interference).

A memory's effective activation becomes:

    activation = base_activation × (1 − decay_rate)^Δt × (1 − interference)

Training signal
~~~~~~~~~~~~~~~
* Memories that were recalled and useful → low decay, low interference.
* Memories that were recalled but led to contradictions → high interference.
* Memories never recalled → natural signal for higher decay.
"""

from __future__ import annotations

import logging
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config.settings import settings

logger = logging.getLogger(__name__)
_device = settings.resolved_device


# ────────────────────────────────────────────────────────────────────
# Forgetting Network
# ────────────────────────────────────────────────────────────────────


class ForgettingNetwork(nn.Module):
    """Two-headed network for learned memory decay.

    Input features per memory:
    * embedding (384)
    * age_hours (1)
    * access_count (1)
    * salience (1)
    * last_activation (1)
    * context_similarity (1)   — cosine sim to current context

    Total: ``embedding_dim + 5``

    Output:
    * decay_rate ∈ (0, 1]   — higher = forget faster
    * interference ∈ [0, 1] — higher = more conflicting
    """

    def __init__(
        self,
        embedding_dim: int = settings.embedding_dim,
        context_dim: int = settings.forgetting_context_dim,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        # feature_dim: embedding + 5 scalar features + optional context
        feature_dim = embedding_dim + 5
        if context_dim > 0:
            feature_dim += context_dim

        self.context_dim = context_dim
        self.use_context = context_dim > 0

        # Optional context projection
        if self.use_context:
            self.context_proj = nn.Linear(settings.gru_hidden_dim, context_dim)

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Decay head
        self.decay_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # ∈ (0, 1)
        )

        # Interference head
        self.interference_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # ∈ [0, 1]
        )

        self.to(_device)

    def forward(
        self,
        embedding: Tensor,
        scalars: Tensor,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Predict decay rate and interference score.

        Parameters
        ----------
        embedding :
            ``Tensor[B, D]`` — memory embeddings.
        scalars :
            ``Tensor[B, 5]`` — [age_hours, access_count, salience,
            last_activation, context_similarity].
        context :
            ``Tensor[H]`` — current context vector (optional).

        Returns
        -------
        (decay_rate, interference) — each ``Tensor[B]``
        """
        embedding = embedding.to(_device)
        scalars = scalars.to(_device)

        parts = [embedding, scalars]

        if self.use_context and context is not None:
            ctx = self.context_proj(context.to(_device))
            # Broadcast context to batch
            if ctx.dim() == 1:
                ctx = ctx.unsqueeze(0).expand(embedding.shape[0], -1)
            parts.append(ctx)
        elif self.use_context:
            parts.append(torch.zeros(embedding.shape[0], self.context_dim, device=_device))

        x = torch.cat(parts, dim=-1)
        h = self.trunk(x)

        decay = self.decay_head(h).squeeze(-1)            # [B]
        interference = self.interference_head(h).squeeze(-1)  # [B]

        return decay, interference

    def compute_effective_activation(
        self,
        base_activation: Tensor,
        embedding: Tensor,
        scalars: Tensor,
        delta_t: Tensor,
        context: Tensor | None = None,
    ) -> Tensor:
        """Apply learned forgetting to base activations.

        effective = base × (1 − decay)^Δt × (1 − interference)
        """
        decay, interference = self.forward(embedding, scalars, context)
        modulated = base_activation * (1 - decay).pow(delta_t) * (1 - interference)
        return modulated


# ────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ────────────────────────────────────────────────────────────────────


def build_forgetting_scalars(
    age_hours: float,
    access_count: int,
    salience: float,
    last_activation: float,
    context_similarity: float,
) -> Tensor:
    """Build the 5-dim scalar feature vector for a single memory.

    Values are lightly normalised so the network sees roughly [0, 1] inputs.
    """
    return torch.tensor([
        min(age_hours / 168.0, 1.0),  # normalise to ~1 week
        min(access_count / 20.0, 1.0),
        salience,
        (last_activation + 5.0) / 10.0,  # ACT-R activations are in [-∞, +∞]
        context_similarity,
    ], dtype=torch.float32, device=_device)


# ────────────────────────────────────────────────────────────────────
# Replay buffer & training
# ────────────────────────────────────────────────────────────────────


class ForgettingReplayBuffer:
    """Stores ``(embedding, scalars, delta_t, target_decay, target_interference)``."""

    def __init__(self, capacity: int = settings.training_replay_buffer_size) -> None:
        self._buffer: deque[tuple[Tensor, Tensor, float, float, float]] = deque(maxlen=capacity)

    def push(
        self,
        embedding: Tensor,
        scalars: Tensor,
        delta_t: float,
        target_decay: float,
        target_interference: float,
    ) -> None:
        self._buffer.append((
            embedding.detach().to(_device),
            scalars.detach().to(_device),
            delta_t,
            target_decay,
            target_interference,
        ))

    def sample(
        self, batch_size: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        indices = random.sample(range(len(self._buffer)), min(batch_size, len(self._buffer)))
        embs = torch.stack([self._buffer[i][0] for i in indices])
        scls = torch.stack([self._buffer[i][1] for i in indices])
        dts = torch.tensor([self._buffer[i][2] for i in indices], device=_device)
        decays = torch.tensor([self._buffer[i][3] for i in indices], device=_device)
        interfs = torch.tensor([self._buffer[i][4] for i in indices], device=_device)
        return embs, scls, dts, decays, interfs

    def __len__(self) -> int:
        return len(self._buffer)


def train_forgetting_step(
    net: ForgettingNetwork,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ForgettingReplayBuffer,
    batch_size: int = 16,
) -> dict[str, float] | None:
    """One training step for the forgetting network."""
    if len(replay_buffer) < batch_size:
        return None

    embs, scls, _dts, target_decays, target_interfs = replay_buffer.sample(batch_size)

    net.train()
    pred_decay, pred_interf = net(embs, scls)

    loss_decay = F.mse_loss(pred_decay, target_decays)
    loss_interf = F.mse_loss(pred_interf, target_interfs)
    loss = loss_decay + loss_interf

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), settings.training_grad_clip)
    optimizer.step()

    return {
        "decay_loss": loss_decay.item(),
        "interference_loss": loss_interf.item(),
        "total": loss.item(),
    }
