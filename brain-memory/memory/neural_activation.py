"""
Graph Attention Network (GAT) for learned spreading activation.

Instead of fixed edge_weight × decay spreading, the GAT learns:
- Which neighbors are relevant given the current context (attention)
- How much activation to propagate across different edge types
- Multi-hop reasoning through multiple GAT layers

Falls back to a pure-PyTorch implementation when ``torch_geometric``
is not installed, using manual multi-head attention over adjacency lists.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config.settings import settings
from memory.graph_converter import GraphConverter, PyGData, NUM_RELATION_TYPES, NUM_NODE_TYPES

logger = logging.getLogger(__name__)
_device = settings.resolved_device

# ────────────────────────────────────────────────────────────────────
# Try loading PyG — fall back to manual implementation if absent
# ────────────────────────────────────────────────────────────────────

_HAS_PYG = False
try:
    from torch_geometric.nn import GATv2Conv
    _HAS_PYG = True
    logger.info("torch_geometric available — using GATv2Conv layers.")
except ImportError:
    logger.info("torch_geometric not installed — using manual GAT implementation.")


# ────────────────────────────────────────────────────────────────────
# Manual GAT layer (fallback when PyG is not installed)
# ────────────────────────────────────────────────────────────────────


class ManualGATv2Layer(nn.Module):
    """Single-layer GATv2 attention implemented without PyG.

    Computes multi-head attention scores from source → target using
    the GATv2 formulation: ``a^T LeakyReLU(W_l [x_i || x_j || e_ij])``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        edge_dim: int = 0,
        dropout: float = 0.1,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat = concat
        self.dropout = dropout

        self.W_src = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        self.W_tgt = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(num_heads, out_dim))
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))

        if edge_dim > 0:
            self.W_edge = nn.Linear(edge_dim, num_heads * out_dim, bias=False)
        else:
            self.W_edge = None  # type: ignore[assignment]

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        N = x.shape[0]
        H, D = self.num_heads, self.out_dim

        h_src = self.W_src(x).view(N, H, D)  # [N, H, D]
        h_tgt = self.W_tgt(x).view(N, H, D)

        if edge_index.shape[1] == 0:
            # No edges — return projected features
            return h_src.reshape(N, H * D) if self.concat else h_src.mean(dim=1)

        src_idx, tgt_idx = edge_index[0], edge_index[1]  # [E]

        # GATv2: apply attention AFTER non-linearity
        msg = h_src[src_idx] + h_tgt[tgt_idx]  # [E, H, D]

        if self.W_edge is not None and edge_attr is not None:
            e_proj = self.W_edge(edge_attr).view(-1, H, D)
            msg = msg + e_proj

        msg = self.leaky_relu(msg)
        alpha = (msg * self.attn.unsqueeze(0)).sum(dim=-1)  # [E, H]

        # Softmax per target node
        alpha_max = torch.zeros(N, H, device=x.device)
        alpha_max.scatter_reduce_(0, tgt_idx.unsqueeze(1).expand(-1, H), alpha, reduce="amax", include_self=True)
        alpha = torch.exp(alpha - alpha_max[tgt_idx])
        alpha_sum = torch.zeros(N, H, device=x.device)
        alpha_sum.scatter_add_(0, tgt_idx.unsqueeze(1).expand(-1, H), alpha)
        alpha = alpha / (alpha_sum[tgt_idx] + 1e-8)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Aggregate
        weighted = h_src[src_idx] * alpha.unsqueeze(-1)  # [E, H, D]
        out = torch.zeros(N, H, D, device=x.device)
        out.scatter_add_(0, tgt_idx.unsqueeze(1).unsqueeze(2).expand(-1, H, D), weighted)

        if self.concat:
            return out.reshape(N, H * D)
        return out.mean(dim=1)


# ────────────────────────────────────────────────────────────────────
# Unified GAT layer wrapper
# ────────────────────────────────────────────────────────────────────


def _make_gat_layer(
    in_dim: int,
    out_dim: int,
    heads: int,
    edge_dim: int,
    dropout: float,
    concat: bool = True,
) -> nn.Module:
    """Create a GAT layer using PyG if available, else manual fallback."""
    if _HAS_PYG:
        return GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=concat,
        )
    return ManualGATv2Layer(
        in_dim=in_dim,
        out_dim=out_dim,
        num_heads=heads,
        edge_dim=edge_dim,
        dropout=dropout,
        concat=concat,
    )


# ────────────────────────────────────────────────────────────────────
# MemoryGAT: the full model
# ────────────────────────────────────────────────────────────────────


class MemoryGAT(nn.Module):
    """Graph Attention Network that learns activation propagation patterns
    over the memory graph.

    Architecture:
        3 GATv2 layers with residual (skip) connections.
        Layer dims determined by ``gat_hidden_dims`` setting.
        Multi-head attention (default 4 heads, concatenated).
        Final linear projection → per-node activation score.
    """

    def __init__(
        self,
        node_feature_dim: int = settings.embedding_dim + NUM_NODE_TYPES + 3,
        context_dim: int = settings.embedding_dim,
        hidden_dims: list[int] | None = None,
        num_heads: int = settings.gat_num_heads,
        edge_dim: int = NUM_RELATION_TYPES + 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        dims = hidden_dims or settings.gat_hidden_dims  # [256, 128, 64]
        self.num_heads = num_heads

        # Input includes concatenated context vector
        full_input = node_feature_dim + context_dim

        # Input projection to first hidden dim
        self.input_proj = nn.Linear(full_input, dims[0])
        self.input_norm = nn.LayerNorm(dims[0])

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        in_dim = dims[0]
        for i, out_dim in enumerate(dims):
            concat = (i < len(dims) - 1)  # Last layer averages heads
            actual_out = out_dim * num_heads if concat else out_dim
            self.gat_layers.append(
                _make_gat_layer(in_dim, out_dim, num_heads, edge_dim, dropout, concat)
            )
            self.norms.append(nn.LayerNorm(actual_out))
            # Skip connection projection if dims don't match
            if in_dim != actual_out:
                self.skip_projs.append(nn.Linear(in_dim, actual_out))
            else:
                self.skip_projs.append(nn.Identity())
            in_dim = actual_out

        # Final activation scoring head
        self.score_head = nn.Sequential(
            nn.Linear(dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.to(_device)

    def forward(
        self,
        data: PyGData,
        context_vector: Tensor,
    ) -> Tensor:
        """Compute per-node activation scores conditioned on context.

        Parameters
        ----------
        data :
            Graph data with node features ``x [N, F]`` and ``edge_index [2, E]``.
        context_vector :
            ``Tensor[D]`` — current working-memory context projected to
            embedding space.

        Returns
        -------
        scores : ``Tensor[N]`` — activation score for every node.
        """
        x = data.x.to(_device)
        edge_index = data.edge_index.to(_device)
        edge_attr = data.edge_attr.to(_device) if data.edge_attr is not None else None

        N = x.shape[0]
        if N == 0:
            return torch.zeros(0, device=_device)

        # Concatenate context vector to every node feature
        ctx = context_vector.to(_device).unsqueeze(0).expand(N, -1)
        x = torch.cat([x, ctx], dim=-1)  # [N, F + D]

        # Input projection
        x = self.input_norm(F.gelu(self.input_proj(x)))

        # GAT layers with skip connections
        for gat, norm, skip in zip(self.gat_layers, self.norms, self.skip_projs):
            residual = skip(x)
            if _HAS_PYG:
                x = gat(x, edge_index, edge_attr=edge_attr)
            else:
                x = gat(x, edge_index, edge_attr=edge_attr)
            x = norm(x + residual)
            x = self.dropout(F.gelu(x))

        # Score each node (sigmoid → activation in [0, 1])
        scores = torch.sigmoid(self.score_head(x).squeeze(-1))  # [N]
        return scores


# ────────────────────────────────────────────────────────────────────
# Replay buffer for contrastive training
# ────────────────────────────────────────────────────────────────────


class ActivationReplayBuffer:
    """Stores (graph_snapshot, context, relevance_labels) for training."""

    def __init__(self, capacity: int = settings.training_replay_buffer_size) -> None:
        self._buffer: deque[tuple[PyGData, Tensor, Tensor]] = deque(maxlen=capacity)

    def push(
        self,
        data: PyGData,
        context: Tensor,
        labels: Tensor,
    ) -> None:
        """Store a training sample.

        labels: Tensor[N] — 1.0 for nodes whose content appeared in the
        LLM response, 0.0 otherwise.
        """
        self._buffer.append((data, context.detach().cpu(), labels.detach().cpu()))

    def sample(self, batch_size: int) -> list[tuple[PyGData, Tensor, Tensor]]:
        """Sample a mini-batch (list of individual graph samples)."""
        import random
        k = min(batch_size, len(self._buffer))
        return random.sample(list(self._buffer), k)

    def __len__(self) -> int:
        return len(self._buffer)


# ────────────────────────────────────────────────────────────────────
# Training helper
# ────────────────────────────────────────────────────────────────────


def train_gat_step(
    model: MemoryGAT,
    optimizer: torch.optim.Optimizer,
    batch: list[tuple[PyGData, Tensor, Tensor]],
    grad_clip: float = settings.training_grad_clip,
) -> float:
    """Run one contrastive training step on the GAT.

    Loss: Binary cross-entropy between predicted activation scores and
    relevance labels.

    Returns the mean loss.
    """
    model.train()
    total_loss = 0.0

    for data, ctx, labels in batch:
        data = data.to(_device)
        ctx = ctx.to(_device)
        labels = labels.to(_device)

        scores = model(data, ctx)
        loss = F.binary_cross_entropy(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(batch), 1)
