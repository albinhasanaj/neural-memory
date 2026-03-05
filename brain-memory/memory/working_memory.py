"""
Working Memory — ring buffer + GRU context encoder.

Working memory maintains a fixed-capacity buffer of the most recent
conversation turns (as embedding vectors).  A GRU encoder compresses
the buffer into a single *context vector* that is used by the
spreading activation engine and salience scorer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from config.settings import settings

_device = settings.resolved_device


# ────────────────────────────────────────────────────────────────────
# Ring Buffer
# ────────────────────────────────────────────────────────────────────


@dataclass
class RingBuffer:
    """Fixed-capacity FIFO buffer of embedding tensors.

    When the buffer is full, the oldest item is silently evicted.

    Parameters
    ----------
    capacity:
        Maximum number of items the buffer can hold.
    """

    capacity: int = settings.working_memory_capacity
    _items: deque[Tensor] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        self._items = deque(maxlen=self.capacity)

    def append(self, item: Tensor) -> None:
        """Push an embedding tensor into the buffer."""
        self._items.append(item.detach().to(_device))

    def as_tensor(self) -> Tensor:
        """Stack all items into ``Tensor[T, D]`` (T = current length).

        Returns a zero-length tensor ``[0, D]`` when empty — callers
        should guard against that.
        """
        if len(self._items) == 0:
            return torch.zeros(0, settings.embedding_dim, device=_device)
        return torch.stack(list(self._items), dim=0)

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def is_full(self) -> bool:
        return len(self._items) == self.capacity

    def clear(self) -> None:
        self._items.clear()


# ────────────────────────────────────────────────────────────────────
# GRU Context Encoder
# ────────────────────────────────────────────────────────────────────


class GRUContextEncoder(nn.Module):
    """Single-layer GRU that encodes a sequence of turn embeddings into
    a context vector.

    Parameters
    ----------
    input_dim:
        Dimensionality of input embeddings (default: ``settings.embedding_dim``).
    hidden_dim:
        Hidden state dimensionality (default: ``settings.gru_hidden_dim``).
    """

    def __init__(
        self,
        input_dim: int = settings.embedding_dim,
        hidden_dim: int = settings.gru_hidden_dim,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        # Linear projector from hidden → embedding space (for prediction)
        self.predictor = nn.Linear(hidden_dim, input_dim)

    def forward(self, sequence: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a sequence of embeddings.

        Parameters
        ----------
        sequence:
            ``Tensor[1, T, D]`` — batch of 1, T timesteps, D features.

        Returns
        -------
        context_vector:
            ``Tensor[H]`` — the final hidden state.
        predicted_next:
            ``Tensor[D]`` — predicted next embedding.
        """
        # sequence: [1, T, D]
        output, h_n = self.gru(sequence)  # h_n: [1, 1, H]
        context_vector = h_n.squeeze(0).squeeze(0)  # [H]
        predicted_next = self.predictor(context_vector)  # [D]
        return context_vector, predicted_next


# ────────────────────────────────────────────────────────────────────
# Working Memory (combines buffer + encoder)
# ────────────────────────────────────────────────────────────────────


class WorkingMemory:
    """Manages the ring buffer and produces context vectors on demand.

    Usage::

        wm = WorkingMemory()
        ctx, pred = wm.update(embedding)  # append + encode
    """

    def __init__(
        self,
        capacity: int = settings.working_memory_capacity,
        embedding_dim: int = settings.embedding_dim,
        hidden_dim: int = settings.gru_hidden_dim,
    ) -> None:
        self.buffer = RingBuffer(capacity=capacity)
        self.encoder = GRUContextEncoder(input_dim=embedding_dim, hidden_dim=hidden_dim)
        self.encoder.to(_device)
        self._last_context: Tensor | None = None
        self._last_prediction: Tensor | None = None

    @torch.no_grad()
    def update(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        """Append a turn embedding and return the updated context.

        Parameters
        ----------
        embedding:
            ``Tensor[D]`` — the embedding of the new turn.

        Returns
        -------
        context_vector:
            ``Tensor[H]`` — compressed representation of working memory.
        predicted_next:
            ``Tensor[D]`` — predicted next-turn embedding
            (used for prediction-error based salience).
        """
        self.buffer.append(embedding)
        seq = self.buffer.as_tensor().unsqueeze(0).to(_device)  # [1, T, D]
        context_vector, predicted_next = self.encoder(seq)
        self._last_context = context_vector
        self._last_prediction = predicted_next
        return context_vector, predicted_next

    @property
    def context_vector(self) -> Tensor | None:
        """Most recently computed context vector, or *None*."""
        return self._last_context

    @property
    def predicted_next(self) -> Tensor | None:
        """Most recently predicted next embedding, or *None*."""
        return self._last_prediction

    def predict_next_embedding(self) -> Tensor:
        """Convenience: run the encoder on the current buffer and return
        just the predicted-next embedding.

        Raises ``RuntimeError`` if the buffer is empty.
        """
        if self.buffer.size == 0:
            raise RuntimeError("Cannot predict from an empty working memory buffer.")
        seq = self.buffer.as_tensor().unsqueeze(0)
        _, predicted = self.encoder(seq)
        return predicted

    def clear(self) -> None:
        """Reset the buffer and cached states."""
        self.buffer.clear()
        self._last_context = None
        self._last_prediction = None


# ────────────────────────────────────────────────────────────────────
# GRU Predictor Replay Buffer & Training
# ────────────────────────────────────────────────────────────────────


class GRUReplayBuffer:
    """Stores ``(sequence, target_next_embedding)`` pairs for GRU predictor training.

    Each sample is a snapshot of the working-memory buffer *before* a new
    embedding was appended, paired with that new embedding as the prediction target.
    """

    def __init__(self, capacity: int = settings.training_replay_buffer_size) -> None:
        from collections import deque
        self._buffer: deque[tuple[Tensor, Tensor]] = deque(maxlen=capacity)

    def push(self, sequence: Tensor, target_next: Tensor) -> None:
        """Store a (sequence_snapshot, actual_next) pair."""
        self._buffer.append((
            sequence.detach().to(_device),
            target_next.detach().to(_device),
        ))

    def sample(self, batch_size: int) -> list[tuple[Tensor, Tensor]]:
        import random
        k = min(batch_size, len(self._buffer))
        return random.sample(list(self._buffer), k)

    def __len__(self) -> int:
        return len(self._buffer)


def train_gru_step(
    encoder: GRUContextEncoder,
    optimizer: torch.optim.Optimizer,
    replay_buffer: GRUReplayBuffer,
    batch_size: int = 8,
    grad_clip: float = 1.0,
) -> dict[str, float] | None:
    """One training step for the GRU predictor head.

    Loss: ``1 - cosine_similarity(predicted_next, actual_next)``, same as
    the Transformer WM training objective.

    Returns loss diagnostics, or *None* if the buffer is too small.
    """
    if len(replay_buffer) < batch_size:
        return None

    samples = replay_buffer.sample(batch_size)
    encoder.train()
    total_loss = 0.0

    for seq, target in samples:
        # seq: [T, D] → [1, T, D]
        inp = seq.unsqueeze(0)
        _, predicted = encoder(inp)  # predicted: [D]
        loss = 1.0 - nn.functional.cosine_similarity(
            predicted.unsqueeze(0), target.unsqueeze(0), dim=1,
        ).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    encoder.eval()
    return {"loss": total_loss / len(samples)}
