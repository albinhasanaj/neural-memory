"""
Transformer Working Memory — central executive replacement.

The existing GRU-based working memory (:pymod:`memory.working_memory`)
compresses the ring buffer into a context vector sequentially.  This
module replaces it with a Transformer encoder that can attend over all
turns in parallel, producing a richer context representation.

Architecture
~~~~~~~~~~~~
* **Learned positional embeddings** — position within the ring buffer.
* **Multi-head self-attention** — 6 heads × 2 layers.
* **CLS token** — a learnable [CLS] is prepended; its output is the
  context vector.
* **Prediction head** — linear projection from the CLS representation
  to the predicted-next embedding.

The Transformer WM is a drop-in replacement: it reads the same
``RingBuffer`` and produces the same ``(context_vector, predicted_next)``
tuple.
"""

from __future__ import annotations

import logging
import math
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
# Transformer encoder
# ────────────────────────────────────────────────────────────────────


class TransformerContextEncoder(nn.Module):
    """Transformer-based context encoder for working memory.

    Parameters
    ----------
    input_dim :
        Embedding dimensionality  (384).
    hidden_dim :
        Internal transformer dimension (same as GRU hidden dim: 256).
    num_layers :
        Number of transformer encoder layers (default: 2).
    num_heads :
        Number of attention heads (default: 6).
    ff_dim :
        Feed-forward hidden dim (default: 512).
    max_len :
        Maximum buffer capacity (for positional embeddings).
    """

    def __init__(
        self,
        input_dim: int = settings.embedding_dim,
        hidden_dim: int = settings.gru_hidden_dim,
        num_layers: int = settings.wm_transformer_layers,
        num_heads: int = settings.wm_transformer_heads,
        ff_dim: int = settings.wm_transformer_ff_dim,
        max_len: int = settings.working_memory_capacity + 1,  # +1 for CLS
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Project input embeddings to transformer dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Learned positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        # Prediction head: CLS hidden → next embedding
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.to(_device)

    def forward(self, sequence: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a sequence of turn embeddings.

        Parameters
        ----------
        sequence :
            ``Tensor[1, T, D]`` — batch-of-1, T turns, D features.

        Returns
        -------
        context_vector :
            ``Tensor[H]`` — CLS representation.
        predicted_next :
            ``Tensor[D]`` — predicted next embedding.
        """
        B, T, _ = sequence.shape
        x = self.input_proj(sequence.to(_device))  # [B, T, H]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, H]
        x = torch.cat([cls, x], dim=1)  # [B, T+1, H]

        # Add positional embeddings
        x = x + self.pos_embedding[:, :T + 1, :]

        # Transformer
        x = self.transformer(x)  # [B, T+1, H]

        # CLS representation
        context_vector = x[:, 0, :]  # [B, H]
        context_vector = context_vector.squeeze(0)  # [H]

        predicted_next = self.predictor(context_vector)  # [D]

        return context_vector, predicted_next


# ────────────────────────────────────────────────────────────────────
# TransformerWorkingMemory — drop-in replacement for WorkingMemory
# ────────────────────────────────────────────────────────────────────


class TransformerWorkingMemory:
    """Transformer-based working memory.

    API-compatible with :class:`memory.working_memory.WorkingMemory`.
    Uses a Transformer encoder instead of a GRU.
    """

    def __init__(
        self,
        capacity: int = settings.working_memory_capacity,
        embedding_dim: int = settings.embedding_dim,
        hidden_dim: int = settings.gru_hidden_dim,
    ) -> None:
        from memory.working_memory import RingBuffer

        self.buffer = RingBuffer(capacity=capacity)
        self.encoder = TransformerContextEncoder(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
        self._last_context: Tensor | None = None
        self._last_prediction: Tensor | None = None

    @torch.no_grad()
    def update(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        """Append a turn and return ``(context_vector, predicted_next)``."""
        self.buffer.append(embedding)
        seq = self.buffer.as_tensor().unsqueeze(0).to(_device)  # [1, T, D]
        context_vector, predicted_next = self.encoder(seq)
        self._last_context = context_vector
        self._last_prediction = predicted_next
        return context_vector, predicted_next

    @property
    def context_vector(self) -> Tensor | None:
        return self._last_context

    @property
    def predicted_next(self) -> Tensor | None:
        return self._last_prediction

    def predict_next_embedding(self) -> Tensor:
        if self.buffer.size == 0:
            raise RuntimeError("Cannot predict from an empty working memory buffer.")
        seq = self.buffer.as_tensor().unsqueeze(0)
        _, predicted = self.encoder(seq)
        return predicted

    def clear(self) -> None:
        self.buffer.clear()
        self._last_context = None
        self._last_prediction = None


# ────────────────────────────────────────────────────────────────────
# Replay buffer & training for Transformer WM
# ────────────────────────────────────────────────────────────────────


class TransformerWMReplayBuffer:
    """Stores ``(context_sequence, target_next_embedding)`` pairs.

    Each entry is a snapshot of the ring buffer contents (a sequence of
    embeddings) paired with the *actual* next embedding that followed.
    The training signal is next-embedding prediction loss.
    """

    def __init__(self, capacity: int = settings.training_replay_buffer_size) -> None:
        self._buffer: deque[tuple[Tensor, Tensor]] = deque(maxlen=capacity)

    def push(self, sequence: Tensor, target_next: Tensor) -> None:
        """Store a (sequence, target_next) pair.

        Parameters
        ----------
        sequence :
            ``Tensor[T, D]`` — the ring buffer snapshot *before* the
            target turn was appended.
        target_next :
            ``Tensor[D]`` — the actual embedding that followed.
        """
        self._buffer.append((
            sequence.detach().cpu(),
            target_next.detach().cpu(),
        ))

    def sample(
        self, batch_size: int,
    ) -> list[tuple[Tensor, Tensor]] | None:
        if len(self._buffer) < batch_size:
            return None
        indices = random.sample(range(len(self._buffer)), batch_size)
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


def train_transformer_wm_step(
    encoder: TransformerContextEncoder,
    optimizer: torch.optim.Optimizer,
    replay_buffer: TransformerWMReplayBuffer,
    batch_size: int = 8,
) -> dict[str, float] | None:
    """One training step for the Transformer working memory encoder.

    Loss is cosine-similarity-based next-embedding prediction:
    ``loss = 1 − cos_sim(predicted, actual)``
    """
    samples = replay_buffer.sample(batch_size)
    if samples is None:
        return None

    encoder.train()

    total_loss = torch.tensor(0.0, device=_device)
    n = 0
    for seq, target in samples:
        seq = seq.unsqueeze(0).to(_device)       # [1, T, D]
        target = target.to(_device)               # [D]
        _ctx, predicted = encoder(seq)            # [D]
        # Cosine embedding loss: 1 - cos_sim
        cos_sim = F.cosine_similarity(predicted.unsqueeze(0), target.unsqueeze(0))
        total_loss = total_loss + (1.0 - cos_sim.squeeze())
        n += 1

    loss = total_loss / max(n, 1)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(encoder.parameters(), settings.training_grad_clip)
    optimizer.step()

    return {"loss": loss.item(), "total": loss.item()}


# ────────────────────────────────────────────────────────────────────
# Attention visualization utility
# ────────────────────────────────────────────────────────────────────


def get_attention_weights(
    encoder: TransformerContextEncoder,
    sequence: Tensor,
) -> list[Tensor]:
    """Extract per-layer attention weights for visualization.

    Returns a list of ``Tensor[H, T+1, T+1]`` — one per layer.
    """
    # We need to register hooks to capture attention weights
    weights: list[Tensor] = []

    hooks = []
    for layer in encoder.transformer.layers:  # type: ignore
        def _hook(module: nn.Module, inp: Any, out: Any, _w: list = weights) -> None:
            # TransformerEncoderLayer stores attention in self_attn
            if hasattr(module, "self_attn"):
                # Re-run the attention to capture weights
                # This is a diagnostic tool, not performance-critical
                pass
        hooks.append(layer.register_forward_hook(_hook))

    # Forward pass
    encoder.eval()
    with torch.no_grad():
        B, T, _ = sequence.shape
        x = encoder.input_proj(sequence.to(_device))
        cls = encoder.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + encoder.pos_embedding[:, :T + 1, :]

        for layer in encoder.transformer.layers:  # type: ignore
            # Manual forward to extract attention
            src = x
            src2, attn_w = layer.self_attn(
                layer.norm1(src), layer.norm1(src), layer.norm1(src),
                need_weights=True,
            )
            weights.append(attn_w.detach())
            # Complete the layer forward
            src = src + src2
            src = src + layer.linear2(
                layer.activation(layer.linear1(layer.norm2(src)))
            )
            x = src

    for h in hooks:
        h.remove()

    return weights
