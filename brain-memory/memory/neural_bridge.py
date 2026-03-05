"""Neural bridge: projects brain memory activations into LLM hidden state space.

This replaces the text decode index as the link between memory and language.
Instead of: memory → text → LLM reads text
It's now:   memory → projection → LLM hidden states modified
"""
from __future__ import annotations

import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryProjection(nn.Module):
    """Projects brain memory activations into LLM hidden state space.

    Brain memory hidden dim (128) → LLM hidden dim (e.g., 3584 for Qwen 7B).

    This is a learned mapping — it needs training to produce vectors that
    are meaningful in the LLM's activation space. Without training, the
    injected vectors would be noise.

    Brain analogy: this is like the synaptic interface between the hippocampus
    (memory system) and the cortex (language system). The hippocampus stores
    memories in its own representation, and this projection translates those
    representations into a form the cortex can use.
    """

    def __init__(
        self,
        brain_dim: int = 128,
        llm_dim: int = 3584,
        injection_strength: float = 0.1,
    ) -> None:
        super().__init__()

        self.brain_dim = brain_dim
        self.llm_dim = llm_dim
        self.injection_strength = injection_strength

        # Two-layer projection with residual-like structure
        # brain_dim → intermediate → llm_dim
        intermediate = max(brain_dim * 2, llm_dim // 4)

        self.proj = nn.Sequential(
            nn.Linear(brain_dim, intermediate),
            nn.GELU(),
            nn.LayerNorm(intermediate),
            nn.Linear(intermediate, llm_dim),
        )

        # Learnable gate that controls injection strength based on retrieval energy
        # High energy (confident retrieval) → strong injection
        # Low energy (uncertain retrieval) → weak injection
        self.energy_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, memory_hidden: torch.Tensor, energy: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            memory_hidden: [brain_dim] activation from FastWeightModule.retrieve()
            energy: scalar — retrieval confidence from Hopfield attention

        Returns:
            [llm_dim] vector ready for injection into LLM hidden states
        """
        # Project to LLM space
        projected = self.proj(memory_hidden)  # [llm_dim]

        # Gate by energy — uncertain retrievals get suppressed
        gate = self.energy_gate(energy.unsqueeze(0))  # [1]

        # Scale by injection strength and gate
        output = projected * gate * self.injection_strength

        # Normalize direction but let the injection hook scale magnitude
        # relative to the LLM's actual hidden state norms
        output = F.normalize(output, dim=0)

        return output


class ProjectionReplayBuffer:
    """Stores (brain_hidden, energy, target_hidden_delta) for training the projection."""

    def __init__(self, maxlen: int = 500) -> None:
        self._buffer: deque = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(
        self,
        brain_hidden: torch.Tensor,
        energy: torch.Tensor,
        target_delta: torch.Tensor,
    ) -> None:
        """
        Args:
            brain_hidden: memory activation from brain
            energy: retrieval confidence
            target_delta: the difference in LLM hidden states between
                         "with memory text injected" and "without" — this is
                         the signal we want the projection to learn to produce
        """
        self._buffer.append((
            brain_hidden.detach().cpu(),
            energy.detach().cpu(),
            target_delta.detach().cpu(),
        ))

    def sample(
        self, batch_size: int = 16
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if len(self._buffer) < batch_size:
            return None
        batch = random.sample(list(self._buffer), batch_size)
        brain_hiddens = torch.stack([b[0] for b in batch])
        energies = torch.stack([b[1] for b in batch])
        target_deltas = torch.stack([b[2] for b in batch])
        return brain_hiddens, energies, target_deltas


def train_projection_step(
    projection: MemoryProjection,
    buffer: ProjectionReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 16,
) -> float | None:
    """Train the projection to produce vectors that shift hidden states
    in the same direction as text-injected memories would.

    Loss: cosine distance between projected memory and target delta.
    The target delta comes from comparing LLM hidden states with vs without
    the text memory injection — so the projection learns to produce the same
    effect as text injection, but directly.
    """
    batch = buffer.sample(batch_size)
    if batch is None:
        return None

    brain_hiddens, energies, target_deltas = batch
    device = next(projection.parameters()).device
    brain_hiddens = brain_hiddens.to(device)
    energies = energies.to(device)
    target_deltas = target_deltas.to(device)

    # Forward pass
    projected = torch.stack([
        projection(bh, e) for bh, e in zip(brain_hiddens, energies)
    ])

    # Cosine similarity loss — projected should point in same direction as target
    loss = 1.0 - F.cosine_similarity(projected, target_deltas, dim=1).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
    optimizer.step()

    return loss.item()
