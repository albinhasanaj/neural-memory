"""
Dopaminergic Gate Network — learned salience gating.

Replaces the hand-tuned weighted-sum salience scorer
(:pymod:`memory.salience`) with a trainable neural network that learns
which turns are worth storing.

Brain analogy
~~~~~~~~~~~~~
Dopamine neurons in the VTA/SNc don't just signal *reward*; they signal
*prediction error* — "this is surprising and worth remembering."  The
gate network takes the same four salience signals, plus the context
vector and raw embedding, and learns a store/don't-store decision.

Architecture
~~~~~~~~~~~~
``[embedding(384) + context(256) + salience_signals(4)] → 2 residual
blocks → sigmoid → store probability``

The gate is trained with **delayed reward**: after consolidation, we
check whether stored episodes actually contributed to useful semantic
facts.  Episodes that led to facts get reward +1; those that didn't
get 0; explicitly bad memories get -1.
"""

from __future__ import annotations

import logging
import random
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
# Residual block
# ────────────────────────────────────────────────────────────────────


class ResidualBlock(nn.Module):
    """Pre-activation residual block: LN → GELU → Linear → LN → GELU → Linear + skip."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


# ────────────────────────────────────────────────────────────────────
# Gate Network
# ────────────────────────────────────────────────────────────────────


class DopaminergicGate(nn.Module):
    r"""Learned salience gate.

    Input: ``concat(embedding, context_vector, salience_signals)``
    where:
    * embedding ∈ R^{embedding_dim}  (384)
    * context_vector ∈ R^{gru_hidden_dim}  (256)
    * salience_signals ∈ R^4  (novelty, pred_error, emphasis, entity_density)

    Output: ``p(store) ∈ [0, 1]``
    """

    def __init__(
        self,
        embedding_dim: int = settings.embedding_dim,
        context_dim: int = settings.gru_hidden_dim,
        signal_dim: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        in_dim = embedding_dim + context_dim + signal_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.to(_device)

    def forward(
        self,
        embedding: Tensor,
        context_vector: Tensor,
        salience_signals: Tensor,
    ) -> Tensor:
        """Return ``p(store)`` for each input.

        Parameters
        ----------
        embedding :
            ``Tensor[D]`` or ``Tensor[B, D]``
        context_vector :
            ``Tensor[H]`` or ``Tensor[B, H]``
        salience_signals :
            ``Tensor[4]`` or ``Tensor[B, 4]``

        Returns
        -------
        ``Tensor[]`` (scalar) or ``Tensor[B]``
        """
        # Ensure batch dim
        was_1d = embedding.dim() == 1
        if was_1d:
            embedding = embedding.unsqueeze(0)
        if context_vector.dim() == 1:
            context_vector = context_vector.unsqueeze(0)
        if salience_signals.dim() == 1:
            salience_signals = salience_signals.unsqueeze(0)

        x = torch.cat([
            embedding.to(_device),
            context_vector.to(_device),
            salience_signals.to(_device),
        ], dim=-1)

        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        out = self.head(x).squeeze(-1)
        if was_1d:
            out = out.squeeze(0)
        return out

    def should_store(
        self,
        embedding: Tensor,
        context_vector: Tensor,
        salience_signals: Tensor,
        epsilon: float = settings.gate_exploration_epsilon,
    ) -> tuple[bool, float]:
        r"""Epsilon-greedy store decision.

        With probability ``ε`` the gate decides randomly (exploration).

        Returns ``(decision, probability)``.
        """
        with torch.no_grad():
            prob = self.forward(embedding, context_vector, salience_signals).item()

        if random.random() < epsilon:
            decision = random.random() < 0.5
        else:
            decision = prob >= 0.3

        return decision, prob


# ────────────────────────────────────────────────────────────────────
# Replay buffer for delayed-reward training
# ────────────────────────────────────────────────────────────────────


class GateReplayBuffer:
    """Stores ``(embedding, context, signals, reward)`` transitions.

    Rewards start at 0.0 and are updated retroactively when memories are
    retrieved and used by the system (see ``update_rewards_for_embeddings``).
    """

    def __init__(self, capacity: int = settings.training_replay_buffer_size) -> None:
        # Store as lists so rewards are mutable
        self._embeddings: deque[Tensor] = deque(maxlen=capacity)
        self._contexts: deque[Tensor] = deque(maxlen=capacity)
        self._signals: deque[Tensor] = deque(maxlen=capacity)
        self._rewards: deque[float] = deque(maxlen=capacity)

    def push(
        self,
        embedding: Tensor,
        context_vector: Tensor,
        salience_signals: Tensor,
        reward: float,
    ) -> None:
        self._embeddings.append(embedding.detach().to(_device))
        self._contexts.append(context_vector.detach().to(_device))
        self._signals.append(salience_signals.detach().to(_device))
        self._rewards.append(reward)

    def update_rewards_for_embeddings(
        self,
        used_embeddings: list[Tensor],
        reward: float = 1.0,
        similarity_threshold: float = 0.85,
    ) -> int:
        """Retroactively assign reward to buffer entries whose embeddings
        are cosine-similar to the retrieved/used embeddings.

        Returns the number of entries updated.
        """
        if not used_embeddings or len(self._embeddings) == 0:
            return 0

        updated = 0
        for used_emb in used_embeddings:
            used_emb = used_emb.to(_device)
            for i in range(len(self._embeddings)):
                sim = F.cosine_similarity(
                    self._embeddings[i].unsqueeze(0),
                    used_emb.unsqueeze(0),
                ).item()
                if sim >= similarity_threshold and self._rewards[i] < reward:
                    self._rewards[i] = reward
                    updated += 1
        return updated

    def sample(
        self, batch_size: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``(embs, ctx, signals, rewards)`` mini-batch."""
        indices = random.sample(range(len(self._embeddings)), min(batch_size, len(self._embeddings)))
        embs = torch.stack([self._embeddings[i] for i in indices])
        ctxs = torch.stack([self._contexts[i] for i in indices])
        sigs = torch.stack([self._signals[i] for i in indices])
        rews = torch.tensor([self._rewards[i] for i in indices], device=_device)
        return embs, ctxs, sigs, rews

    def __len__(self) -> int:
        return len(self._embeddings)


# ────────────────────────────────────────────────────────────────────
# Training step
# ────────────────────────────────────────────────────────────────────


def train_gate_step(
    gate: DopaminergicGate,
    optimizer: torch.optim.Optimizer,
    replay_buffer: GateReplayBuffer,
    batch_size: int = 16,
) -> dict[str, float] | None:
    """One REINFORCE-style training step with delayed reward.

    The loss is ``-reward * log(p)`` when the gate said store,
    ``-reward * log(1 - p)`` otherwise.  For simplicity we use
    BCE with the reward as the target (reward ∈ {0, 1}).
    """
    if len(replay_buffer) < batch_size:
        return None

    embs, ctxs, sigs, rewards = replay_buffer.sample(batch_size)

    gate.train()
    probs = gate(embs, ctxs, sigs)  # [B]

    # Treat reward as target probability (0 = shouldn't store, 1 = should store)
    targets = rewards.clamp(0.0, 1.0)
    loss = F.binary_cross_entropy(probs, targets)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(gate.parameters(), settings.training_grad_clip)
    optimizer.step()

    return {"gate_loss": loss.item(), "mean_prob": probs.mean().item()}
