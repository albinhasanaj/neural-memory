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

import json
import logging
from pathlib import Path
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


class LegacyHippocampalMemory(nn.Module):
    """Modern Hopfield Network acting as the hippocampal associative store.

    .. deprecated::
        Renamed from ``HippocampalMemory`` in Phase 2.  Prefer
        ``ModularHippocampalMemory`` when ``use_modular_hopfield`` is set.

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

        # Decode index: slot index → metadata for text reconstruction
        self._decode_index: dict[int, dict] = {}

        self.to(_device)

    def load_state_dict(self, state_dict, strict=True, assign=False):  # type: ignore[override]
        """Populate ``_episode_ids`` and ``_decode_index`` to match loaded patterns size."""
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        n = self.patterns.shape[0]
        if len(self._episode_ids) != n:
            self._episode_ids = [f"__loaded_{i}" for i in range(n)]
        # Rebuild decode index placeholders for loaded patterns without decode data
        for i in range(n):
            if i not in self._decode_index:
                self._decode_index[i] = {
                    "text": "", "speaker": "", "timestamp": 0.0,
                    "entities": [], "topics": [], "salience": 0.0,
                    "episode_id": self._episode_ids[i],
                }
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
        metadata: dict | None = None,
        strength: float = 1.0,
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
        metadata :
            Optional dict with ``{text, speaker, timestamp, entities, topics, salience}``
            for the decode index. Enables ``retrieve_decoded()``.
        """
        with torch.no_grad():
            pattern = self.separator(embedding.to(_device).unsqueeze(0))  # [1, D]
        if value is None:
            value = embedding.to(_device).unsqueeze(0)
        else:
            value = value.to(_device).unsqueeze(0)

        slot_index = self.num_patterns  # index before append

        self.patterns = torch.cat([self.patterns, pattern], dim=0)  # type: ignore
        self.values = torch.cat([self.values, value], dim=0)  # type: ignore
        self.access_counts = torch.cat([  # type: ignore
            self.access_counts,
            torch.zeros(1, device=_device),
        ])
        self._episode_ids.append(episode_id)

        # Populate decode index
        if metadata is not None:
            self._decode_index[slot_index] = {
                "text": metadata.get("text", ""),
                "speaker": metadata.get("speaker", ""),
                "timestamp": metadata.get("timestamp", 0.0),
                "entities": metadata.get("entities", []),
                "topics": metadata.get("topics", []),
                "salience": metadata.get("salience", 0.0),
                "episode_id": episode_id,
            }

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

    def retrieve_decoded(
        self,
        query: Tensor,
        top_k: int = 5,
    ) -> list[dict]:
        """Retrieve top-K patterns and return decoded metadata.

        Returns a list of dicts with ``{text, speaker, timestamp, entities,
        score, slot_index, episode_id}`` sorted by score descending.
        """
        _, attn, _ = self.forward(query, top_k=None)
        if attn.numel() == 0:
            return []
        k = min(top_k, self.num_patterns)
        weights, indices = attn.topk(k)
        results = []
        for w, idx in zip(weights.tolist(), indices.tolist()):
            meta = self._decode_index.get(idx)
            if meta is not None:
                results.append({
                    "text": meta["text"],
                    "speaker": meta["speaker"],
                    "timestamp": meta["timestamp"],
                    "entities": meta.get("entities", []),
                    "topics": meta.get("topics", []),
                    "salience": meta.get("salience", 0.0),
                    "score": w,
                    "slot_index": idx,
                    "episode_id": meta.get("episode_id", ""),
                })
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def decode(self, slot_indices: list[int]) -> list[dict]:
        """Return the decode-index metadata for the given slot indices."""
        return [self._decode_index[i] for i in slot_indices if i in self._decode_index]

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
        keep_list = keep_idx.tolist()
        self._episode_ids = [old_ids[i] for i in keep_list]

        # Rebuild decode index with new contiguous slot indices
        old_decode = self._decode_index
        self._decode_index = {}
        for new_idx, old_idx in enumerate(keep_list):
            if old_idx in old_decode:
                self._decode_index[new_idx] = old_decode[old_idx]

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
        self._decode_index = {}

    # ── Decode index persistence ────────────────────────────────────

    def save_decode_index(self, path: str | Path) -> None:
        """Serialize ``_decode_index`` as JSON."""
        path = Path(path)
        with open(path, "w") as f:
            # JSON keys must be strings
            json.dump({str(k): v for k, v in self._decode_index.items()}, f)
        logger.info("Saved decode index (%d entries) to %s", len(self._decode_index), path)

    def load_decode_index(self, path: str | Path) -> None:
        """Load ``_decode_index`` from JSON."""
        path = Path(path)
        if not path.exists():
            logger.warning("Decode index not found at %s", path)
            return
        with open(path) as f:
            raw = json.load(f)
        self._decode_index = {int(k): v for k, v in raw.items()}
        logger.info("Loaded decode index (%d entries) from %s", len(self._decode_index), path)


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


# Backward-compat alias — existing imports of ``HippocampalMemory`` still work.
HippocampalMemory = LegacyHippocampalMemory


# ════════════════════════════════════════════════════════════════════
# Phase 2 — Modular Hopfield with Learned Routing
# ════════════════════════════════════════════════════════════════════


class HopfieldModule(nn.Module):
    """A single Hopfield sub-module with its own pattern store and decode index.

    Identical inner mechanics to ``LegacyHippocampalMemory`` but designed
    to be one of *M* modules inside a ``ModularHippocampalMemory``.
    """

    def __init__(
        self,
        pattern_dim: int = settings.embedding_dim,
        value_dim: int | None = None,
        beta_init: float = settings.hopfield_beta_init,
        max_patterns: int = settings.hopfield_patterns_per_module,
    ) -> None:
        super().__init__()
        self.pattern_dim = pattern_dim
        self.value_dim = value_dim or pattern_dim
        self.max_patterns = max_patterns

        self.log_beta = nn.Parameter(torch.tensor(float(beta_init)).log())
        self.separator = nn.Sequential(
            nn.Linear(pattern_dim, pattern_dim),
            nn.LayerNorm(pattern_dim),
        )
        self.query_proj = nn.Linear(pattern_dim, pattern_dim)

        self.register_buffer("patterns", torch.empty(0, pattern_dim))
        self.register_buffer("values", torch.empty(0, self.value_dim))
        self.register_buffer("access_counts", torch.empty(0))

        self._episode_ids: list[str] = []
        self._decode_index: dict[int, dict] = {}
        self.to(_device)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def beta(self) -> Tensor:
        return self.log_beta.exp()

    @property
    def num_patterns(self) -> int:
        return self.patterns.shape[0]

    def occupancy(self) -> float:
        """Fraction of capacity currently used."""
        if self.max_patterns == 0:
            return 0.0
        return self.num_patterns / self.max_patterns

    # ── Store ───────────────────────────────────────────────────────

    def store(
        self,
        embedding: Tensor,
        value: Tensor | None = None,
        episode_id: str = "",
        metadata: dict | None = None,
        strength: float = 1.0,
    ) -> None:
        with torch.no_grad():
            pattern = self.separator(embedding.to(_device).unsqueeze(0))
        if value is None:
            value = embedding.to(_device).unsqueeze(0)
        else:
            value = value.to(_device).unsqueeze(0)

        slot_index = self.num_patterns
        self.patterns = torch.cat([self.patterns, pattern], dim=0)
        self.values = torch.cat([self.values, value], dim=0)
        self.access_counts = torch.cat([
            self.access_counts,
            torch.zeros(1, device=_device),
        ])
        self._episode_ids.append(episode_id)

        if metadata is not None:
            self._decode_index[slot_index] = {
                "text": metadata.get("text", ""),
                "speaker": metadata.get("speaker", ""),
                "timestamp": metadata.get("timestamp", 0.0),
                "entities": metadata.get("entities", []),
                "topics": metadata.get("topics", []),
                "salience": metadata.get("salience", 0.0),
                "episode_id": episode_id,
            }

        if self.num_patterns > self.max_patterns:
            self._consolidate()

    # ── Retrieve ────────────────────────────────────────────────────

    def forward(
        self,
        query: Tensor,
        top_k: int | None = None,
    ) -> tuple[Tensor, Tensor, list[int]]:
        if self.num_patterns == 0:
            D = self.value_dim
            if query.dim() == 1:
                return torch.zeros(D, device=_device), torch.empty(0, device=_device), []
            return torch.zeros(query.shape[0], D, device=_device), torch.empty(0, device=_device), []

        was_1d = query.dim() == 1
        if was_1d:
            query = query.unsqueeze(0)

        query = query.to(_device)
        q = self.query_proj(query)
        logits = self.beta * (q @ self.patterns.t())
        attn = F.softmax(logits, dim=-1)

        with torch.no_grad():
            self.access_counts += attn.sum(dim=0)

        retrieved = attn @ self.values

        indices: list[int] = []
        if top_k is not None and top_k < self.num_patterns:
            _, topk_idx = attn.topk(top_k, dim=-1)
            indices = topk_idx.squeeze(0).tolist() if was_1d else topk_idx.tolist()
        else:
            indices = list(range(self.num_patterns))

        if was_1d:
            return retrieved.squeeze(0), attn.squeeze(0), indices
        return retrieved, attn, indices

    def retrieve_decoded(self, query: Tensor, top_k: int = 5) -> list[dict]:
        _, attn, _ = self.forward(query, top_k=None)
        if attn.numel() == 0:
            return []
        k = min(top_k, self.num_patterns)
        weights, indices = attn.topk(k)
        results = []
        for w, idx in zip(weights.tolist(), indices.tolist()):
            meta = self._decode_index.get(idx)
            if meta is not None:
                results.append({
                    "text": meta["text"],
                    "speaker": meta["speaker"],
                    "timestamp": meta["timestamp"],
                    "entities": meta.get("entities", []),
                    "topics": meta.get("topics", []),
                    "salience": meta.get("salience", 0.0),
                    "score": w,
                    "slot_index": idx,
                    "episode_id": meta.get("episode_id", ""),
                })
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    # ── Consolidation ───────────────────────────────────────────────

    def _consolidate(self) -> None:
        keep = self.max_patterns
        if self.num_patterns <= keep:
            return
        _, keep_idx = self.access_counts.topk(keep)
        keep_idx = keep_idx.sort().values

        self.patterns = self.patterns[keep_idx]
        self.values = self.values[keep_idx]
        self.access_counts = self.access_counts[keep_idx]

        old_ids = self._episode_ids
        keep_list = keep_idx.tolist()
        self._episode_ids = [old_ids[i] for i in keep_list]

        old_decode = self._decode_index
        self._decode_index = {}
        for new_idx, old_idx in enumerate(keep_list):
            if old_idx in old_decode:
                self._decode_index[new_idx] = old_decode[old_idx]

        logger.info("HopfieldModule consolidation: %d → %d", len(old_ids), len(self._episode_ids))

    # ── Utilities ───────────────────────────────────────────────────

    def clear(self) -> None:
        self.patterns = torch.empty(0, self.pattern_dim, device=_device)
        self.values = torch.empty(0, self.value_dim, device=_device)
        self.access_counts = torch.empty(0, device=_device)
        self._episode_ids = []
        self._decode_index = {}

    def save_decode_index(self, path: str | Path) -> dict:
        """Return serialisable decode-index dict (also writes to *path*)."""
        path = Path(path)
        data = {str(k): v for k, v in self._decode_index.items()}
        with open(path, "w") as f:
            json.dump(data, f)
        return data

    def load_decode_index(self, data: dict) -> None:
        """Load decode index from a dict (deserialized from JSON)."""
        self._decode_index = {int(k): v for k, v in data.items()}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Pre-resize buffers to match checkpoint shapes
        for key in ("patterns", "values", "access_counts"):
            if key in state_dict:
                val = state_dict[key]
                current = getattr(self, key)
                if isinstance(current, torch.Tensor) and current.shape != val.shape:
                    self.register_buffer(key, torch.empty_like(val))
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        n = self.patterns.shape[0]
        if len(self._episode_ids) != n:
            self._episode_ids = [f"__loaded_{i}" for i in range(n)]
        for i in range(n):
            if i not in self._decode_index:
                self._decode_index[i] = {
                    "text": "", "speaker": "", "timestamp": 0.0,
                    "entities": [], "topics": [], "salience": 0.0,
                    "episode_id": self._episode_ids[i],
                }
        return result


# ────────────────────────────────────────────────────────────────────
# MemoryRouter — learned routing across HopfieldModules
# ────────────────────────────────────────────────────────────────────


class MemoryRouter(nn.Module):
    """Learned router that dispatches embeddings to a subset of HopfieldModules.

    Each module has a learnable *key* vector.  Routing scores are computed as::

        score = softmax(proj(embedding) @ module_keys^T / (sqrt(d) * temperature))

    Parameters
    ----------
    num_modules :
        Number of HopfieldModules to route across.
    embedding_dim :
        Dimensionality of input embeddings.
    """

    def __init__(
        self,
        num_modules: int = settings.hopfield_num_modules,
        embedding_dim: int = settings.embedding_dim,
    ) -> None:
        super().__init__()
        self.num_modules = num_modules
        self.embedding_dim = embedding_dim

        self.module_keys = nn.Parameter(torch.randn(num_modules, embedding_dim) * 0.02)
        self.router_proj = nn.Linear(embedding_dim, embedding_dim)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))  # init temp=1

        self.to(_device)

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp()

    def _scores(self, embedding: Tensor) -> Tensor:
        """Compute routing scores for a single embedding. Returns [num_modules]."""
        emb = embedding.to(_device)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        proj = self.router_proj(emb)  # [1, D]
        scale = (self.embedding_dim ** 0.5) * self.temperature.clamp(min=1e-4)
        logits = (proj @ self.module_keys.t()) / scale  # [1, M]
        return F.softmax(logits, dim=-1).squeeze(0)  # [M]

    def route_write(
        self,
        embedding: Tensor,
        top_k: int = settings.hopfield_top_k_write,
    ) -> list[tuple[int, float]]:
        """Return ``(module_index, score)`` for the top-K write targets."""
        scores = self._scores(embedding)
        k = min(top_k, self.num_modules)
        vals, idxs = scores.topk(k)
        return list(zip(idxs.tolist(), vals.tolist()))

    def route_read(
        self,
        query: Tensor,
        top_k: int = settings.hopfield_top_k_read,
    ) -> list[tuple[int, float]]:
        """Return ``(module_index, score)`` for the top-K read targets."""
        scores = self._scores(query)
        k = min(top_k, self.num_modules)
        vals, idxs = scores.topk(k)
        return list(zip(idxs.tolist(), vals.tolist()))

    def scores_for(self, embedding: Tensor) -> Tensor:
        """Full routing score vector (for training the router)."""
        return self._scores(embedding)


# ────────────────────────────────────────────────────────────────────
# ModularHippocampalMemory — orchestrator
# ────────────────────────────────────────────────────────────────────


class ModularHippocampalMemory(nn.Module):
    """M HopfieldModules + a MemoryRouter — drop-in replacement for
    ``LegacyHippocampalMemory``.

    Public API is identical to the legacy class so the observer and
    trainer do not need special-casing.
    """

    def __init__(
        self,
        num_modules: int = settings.hopfield_num_modules,
        pattern_dim: int = settings.embedding_dim,
        value_dim: int | None = None,
        beta_init: float = settings.hopfield_beta_init,
        patterns_per_module: int = settings.hopfield_patterns_per_module,
    ) -> None:
        super().__init__()
        self.num_modules = num_modules
        self.pattern_dim = pattern_dim
        self.value_dim = value_dim or pattern_dim

        self.router = MemoryRouter(
            num_modules=num_modules,
            embedding_dim=pattern_dim,
        )
        self.modules_list = nn.ModuleList([
            HopfieldModule(
                pattern_dim=pattern_dim,
                value_dim=self.value_dim,
                beta_init=beta_init,
                max_patterns=patterns_per_module,
            )
            for _ in range(num_modules)
        ])
        self.to(_device)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def num_patterns(self) -> int:
        return sum(m.num_patterns for m in self.modules_list)

    # Expose separator/query_proj/log_beta from first module for
    # backward-compat training (trainer collects these params)
    @property
    def separator(self) -> nn.Module:
        return self.modules_list[0].separator

    @property
    def query_proj(self) -> nn.Module:
        return self.modules_list[0].query_proj

    @property
    def log_beta(self) -> nn.Parameter:
        return self.modules_list[0].log_beta

    @property
    def beta(self) -> Tensor:
        return self.modules_list[0].beta

    # ── Store ───────────────────────────────────────────────────────

    def store(
        self,
        embedding: Tensor,
        value: Tensor | None = None,
        episode_id: str = "",
        metadata: dict | None = None,
        strength: float = 1.0,
    ) -> None:
        """Route *embedding* to top-K modules and store in each."""
        targets = self.router.route_write(embedding)
        for mod_idx, _score in targets:
            self.modules_list[mod_idx].store(
                embedding, value=value, episode_id=episode_id, metadata=metadata,
            )

    # ── Retrieve ────────────────────────────────────────────────────

    def retrieve_decoded(
        self,
        query: Tensor,
        top_k: int = 5,
    ) -> list[dict]:
        """Query top-K modules, merge results, deduplicate by episode_id."""
        read_targets = self.router.route_read(query)
        all_results: list[dict] = []
        for mod_idx, route_score in read_targets:
            mod_results = self.modules_list[mod_idx].retrieve_decoded(query, top_k=top_k)
            # Weight each result's score by the routing score
            for r in mod_results:
                r["score"] = r["score"] * route_score
                r["source_module"] = mod_idx
            all_results.extend(mod_results)

        # Deduplicate by episode_id — keep highest-scoring entry
        seen: dict[str, dict] = {}
        for r in all_results:
            eid = r.get("episode_id", "")
            if eid and eid in seen:
                if r["score"] > seen[eid]["score"]:
                    seen[eid] = r
            else:
                key = eid or id(r)
                seen[str(key)] = r

        deduped = list(seen.values())
        deduped.sort(key=lambda r: r["score"], reverse=True)
        return deduped[:top_k]

    def forward(
        self,
        query: Tensor,
        top_k: int | None = None,
    ) -> tuple[Tensor, Tensor, list[int]]:
        """Forward pass — aggregates over routed modules.

        For backward compat with code expecting (retrieved, attn, indices).
        """
        read_targets = self.router.route_read(query)
        if not read_targets:
            D = self.value_dim
            return torch.zeros(D, device=_device), torch.empty(0, device=_device), []

        retrieved_sum = torch.zeros(self.value_dim, device=_device)
        total_weight = 0.0
        all_indices: list[int] = []

        for mod_idx, route_score in read_targets:
            mod = self.modules_list[mod_idx]
            if mod.num_patterns == 0:
                continue
            r, attn, idx = mod(query, top_k=top_k)
            retrieved_sum += r * route_score
            total_weight += route_score
            all_indices.extend(idx)

        if total_weight > 0:
            retrieved_sum /= total_weight

        # Return a dummy attention vector (full routing isn't 1-to-1 with legacy attn)
        return retrieved_sum, torch.empty(0, device=_device), all_indices

    # ── Utilities matching Legacy API ───────────────────────────────

    def clear(self) -> None:
        for m in self.modules_list:
            m.clear()

    def save_decode_index(self, path: str | Path) -> None:
        """Save all per-module decode indices as a single JSON file."""
        path = Path(path)
        combined: dict[str, dict] = {}
        for i, m in enumerate(self.modules_list):
            data = {str(k): v for k, v in m._decode_index.items()}
            if data:
                combined[str(i)] = data
        with open(path, "w") as f:
            json.dump(combined, f)
        logger.info("Saved modular decode index (%d modules) to %s", len(combined), path)

    def load_decode_index(self, path: str | Path) -> None:
        """Load per-module decode indices from a JSON file."""
        path = Path(path)
        if not path.exists():
            logger.warning("Modular decode index not found at %s", path)
            return
        with open(path) as f:
            combined = json.load(f)
        for mod_str, data in combined.items():
            mod_idx = int(mod_str)
            if mod_idx < len(self.modules_list):
                self.modules_list[mod_idx].load_decode_index(data)
        logger.info("Loaded modular decode index from %s", path)

    def retrieve_episode_ids(
        self,
        query: Tensor,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Convenience: return top-K episode IDs across routed modules."""
        results = self.retrieve_decoded(query, top_k=top_k)
        return [(r["episode_id"], r["score"]) for r in results if r.get("episode_id")]

    def decode(self, slot_indices: list[int]) -> list[dict]:
        """Decode from the first module that has the slot.

        Note: slot indices are module-local, so this is approximate.
        Prefer ``retrieve_decoded()`` for the modular path.
        """
        results = []
        for m in self.modules_list:
            for idx in slot_indices:
                if idx in m._decode_index:
                    results.append(m._decode_index[idx])
        return results

    # ── Specialization monitoring ───────────────────────────────────

    def module_summary(self) -> list[dict]:
        """Return per-module occupancy, num_patterns, top_entities, top_topics."""
        summaries = []
        for i, m in enumerate(self.modules_list):
            entities: dict[str, int] = {}
            topics: dict[str, int] = {}
            for meta in m._decode_index.values():
                for ent in meta.get("entities", []):
                    entities[ent] = entities.get(ent, 0) + 1
                for top in meta.get("topics", []):
                    topics[top] = topics.get(top, 0) + 1
            top_ents = sorted(entities.items(), key=lambda x: x[1], reverse=True)[:5]
            top_tops = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]
            summaries.append({
                "module_index": i,
                "num_patterns": m.num_patterns,
                "occupancy": m.occupancy(),
                "top_entities": top_ents,
                "top_topics": top_tops,
            })
        return summaries


# ────────────────────────────────────────────────────────────────────
# Router replay buffer & training
# ────────────────────────────────────────────────────────────────────


class RouterReplayBuffer:
    """Stores ``(query_embedding, module_index, reward)`` tuples for
    REINFORCE-style router training."""

    def __init__(self, capacity: int = settings.training_replay_buffer_size) -> None:
        self._buffer: deque[tuple[Tensor, int, float]] = deque(maxlen=capacity)

    def push(self, query: Tensor, module_idx: int, reward: float) -> None:
        self._buffer.append((query.detach().cpu(), module_idx, reward))

    def sample(self, batch_size: int) -> list[tuple[Tensor, int, float]] | None:
        if len(self._buffer) < batch_size:
            return None
        indices = random.sample(range(len(self._buffer)), batch_size)
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


def train_router_step(
    router: MemoryRouter,
    optimizer: torch.optim.Optimizer,
    replay_buffer: RouterReplayBuffer,
    batch_size: int = 8,
) -> dict[str, float] | None:
    """One REINFORCE training step for the MemoryRouter.

    Loss: ``-reward * log(routing_score[module_idx])``
    """
    samples = replay_buffer.sample(batch_size)
    if samples is None:
        return None

    router.train()
    total_loss = torch.tensor(0.0, device=_device)
    n = 0

    for query, mod_idx, reward in samples:
        query = query.to(_device)
        scores = router.scores_for(query)  # [M]
        log_prob = torch.log(scores[mod_idx] + 1e-8)
        total_loss = total_loss - reward * log_prob
        n += 1

    loss = total_loss / max(n, 1)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(router.parameters(), settings.training_grad_clip)
    optimizer.step()

    return {"loss": loss.item(), "total": loss.item()}


# ════════════════════════════════════════════════════════════════════
# Phase 3 — Continuous Weight Memory (Fast Weights)
# ════════════════════════════════════════════════════════════════════


class FastWeightModule(nn.Module):
    """Single fast-weight memory unit — stores memories as weight changes.

    Instead of a pattern buffer (explicit rows), memories are superposed into
    weight matrices via Hebbian outer-product writes.  Retrieval is a forward
    pass through the weight matrices, not a search over stored items.

    Brain analogy
    -------------
    * ``W_key``   — recurrent / autoassociative synapses (same-layer)
    * ``W_value`` — storage→output synapses
    * Hebbian write (``W += η · v ⊗ k``) — "cells that fire together wire together"
    * Slow parameters (separator, query_proj …) — the learned circuit architecture
    """

    def __init__(
        self,
        embedding_dim: int = settings.embedding_dim,
        hidden_dim: int = settings.fast_weight_hidden_dim,
        write_lr: float = settings.fast_weight_write_lr,
        capacity_estimate: int = settings.hopfield_patterns_per_module,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.write_lr = write_lr
        self.capacity_estimate = capacity_estimate

        # === SLOW PARAMETERS (learned via backprop) ===
        self.separator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.query_proj = nn.Linear(embedding_dim, hidden_dim)
        self.value_proj = nn.Linear(embedding_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        self.log_beta = nn.Parameter(torch.tensor(1.0))

        # === FAST WEIGHTS (Hebbian writes only — NOT in any optimizer) ===
        self.register_buffer("W_key", torch.zeros(hidden_dim, hidden_dim))
        self.register_buffer("W_value", torch.zeros(hidden_dim, hidden_dim))

        # === SYNAPTIC IMPORTANCE (interference protection) ===
        self.register_buffer("omega_key", torch.zeros(hidden_dim, hidden_dim))
        self.register_buffer("omega_value", torch.zeros(hidden_dim, hidden_dim))

        # === MEMORY STRENGTH TRACKING ===
        self.register_buffer("write_count", torch.tensor(0))
        self.register_buffer("total_energy", torch.tensor(0.0))

        # === DECODE INDEX ===
        self._decode_index: dict[str, dict] = {}
        self._write_history: list[tuple[str, Tensor]] = []
        self._max_history = capacity_estimate * 2

        self.to(_device)

    # ── Hashing ─────────────────────────────────────────────────────

    def _embedding_hash(self, embedding: Tensor) -> str:
        """Stable hash for an embedding (first 8 floats → hex)."""
        trunc = embedding.detach().cpu().flatten()[:8]
        return trunc.numpy().tobytes().hex()

    # ── Write (Hebbian) ────────────────────────────────────────────

    def write(
        self,
        embedding: Tensor,
        episode_id: str = "",
        metadata: dict | None = None,
        strength: float = 1.0,
    ) -> None:
        """Write a memory via Hebbian outer product: ``W += η · v ⊗ k``.

        Args:
            strength: Multiplier on write_lr. High-salience memories use
                      strength > 1 to create deeper attractor basins.
        """
        with torch.no_grad():
            emb = embedding.to(_device)
            key = self.separator(emb.unsqueeze(0)).squeeze(0)
            value = self.value_proj(emb.unsqueeze(0)).squeeze(0)

            key = F.normalize(key, dim=0)
            value = F.normalize(value, dim=0)

            # Interference-protected Hebbian write
            effective_lr = self.write_lr * strength
            delta_key = effective_lr * torch.outer(key, key)
            delta_value = effective_lr * torch.outer(value, key)

            protection_key = 1.0 / (1.0 + self.omega_key)
            protection_value = 1.0 / (1.0 + self.omega_value)

            self.W_key += delta_key * protection_key
            self.W_value += delta_value * protection_value

            self.write_count += 1
            self.total_energy += torch.norm(self.W_key).item()

            # Decode index
            emb_hash = self._embedding_hash(emb)
            if metadata is not None:
                self._decode_index[emb_hash] = {
                    "text": metadata.get("text", ""),
                    "speaker": metadata.get("speaker", ""),
                    "timestamp": metadata.get("timestamp", 0.0),
                    "entities": metadata.get("entities", []),
                    "topics": metadata.get("topics", []),
                    "salience": metadata.get("salience", 0.0),
                    "episode_id": episode_id,
                    "write_step": self.write_count.item(),
                }

            self._write_history.append((emb_hash, emb.clone()))
            self._trim_write_history()
            self._apply_homeostatic_decay()

    # Alias so that store() calls (from observer/modular) work unchanged
    def store(
        self,
        embedding: Tensor,
        value: Tensor | None = None,
        episode_id: str = "",
        metadata: dict | None = None,
        strength: float = 1.0,
    ) -> None:
        """Compatibility shim — delegates to :meth:`write`."""
        self.write(embedding, episode_id=episode_id, metadata=metadata, strength=strength)

    def replay_recent(self, n: int = 5, replay_strength: float = 0.3) -> None:
        """Re-write recent memories to strengthen their attractor basins.

        Brain analogy: hippocampal replay during consolidation.
        Each replay re-applies the Hebbian write with reduced strength,
        gradually deepening the attractor without overwhelming other memories.
        """
        if not self._write_history:
            return
        with torch.no_grad():
            recent = self._write_history[-n:]
            for _emb_hash, emb_tensor in recent:
                emb = emb_tensor.to(_device)
                key = self.separator(emb.unsqueeze(0)).squeeze(0)
                value = self.value_proj(emb.unsqueeze(0)).squeeze(0)
                key = F.normalize(key, dim=0)
                value = F.normalize(value, dim=0)
                effective_lr = self.write_lr * replay_strength
                self.W_key += effective_lr * torch.outer(key, key)
                self.W_value += effective_lr * torch.outer(value, key)

    # ── Homeostatic decay ───────────────────────────────────────────

    def _apply_homeostatic_decay(self) -> None:
        """Multiplicative decay to prevent unbounded weight growth."""
        if self.write_count % settings.fast_weight_decay_interval != 0:
            return
        factor = settings.fast_weight_decay_factor
        self.W_key *= factor
        self.W_value *= factor

    # ── Write-history management ────────────────────────────────────

    def _trim_write_history(self) -> None:
        if len(self._write_history) <= self._max_history:
            return
        active_hashes = set(self._decode_index.keys())
        new_history = [(h, k) for h, k in self._write_history if h in active_hashes]
        if len(new_history) > self._max_history:
            new_history = new_history[-self._max_history:]
        self._write_history = new_history

    def prune_decode_index(self, max_entries: int = 1000) -> None:
        """Remove oldest decode entries to bound index size."""
        if len(self._decode_index) <= max_entries:
            return
        entries = sorted(
            self._decode_index.items(),
            key=lambda x: x[1].get("write_step", 0),
        )
        for hash_key, _ in entries[: len(entries) - max_entries]:
            del self._decode_index[hash_key]

    # ── Retrieve ────────────────────────────────────────────────────

    def retrieve(self, query: Tensor, top_k: int = 5) -> tuple[Tensor, Tensor]:
        """Retrieve from fast weights via implicit attention.

        Returns ``(output_embedding, energy)``.
        """
        q = self.query_proj(query.to(_device).unsqueeze(0)).squeeze(0)
        q = F.normalize(q, dim=0)

        beta = self.log_beta.exp()
        key_response = beta * (q @ self.W_key)
        attention = F.softmax(key_response, dim=0)

        output_hidden = attention @ self.W_value
        output = self.output_proj(output_hidden.unsqueeze(0)).squeeze(0)

        energy = torch.dot(key_response, attention)
        return output, energy

    def retrieve_decoded(self, query: Tensor, top_k: int = 5) -> list[dict]:
        """Retrieve and decode back to text via the decode index."""
        if not self._decode_index:
            return []

        output, energy = self.retrieve(query, top_k)
        energy_val = abs(energy.item())

        if self._write_history:
            # Score by cosine similarity in the original embedding space
            # (same sentence-transformer space as the query).
            stored_embs = torch.stack([wh[1] for wh in self._write_history]).to(_device)

            # Backward compat: old checkpoints stored separator keys (hidden_dim).
            # Detect by checking tensor size against embedding_dim.
            if stored_embs.shape[-1] != query.shape[-1]:
                # Fall through to fallback path
                self._write_history.clear()
                return self.retrieve_decoded(query, top_k)

            q_emb = F.normalize(query.to(_device), dim=0)
            stored_norm = F.normalize(stored_embs, dim=1)
            similarities = stored_norm @ q_emb  # (N,)

            k = min(top_k, len(self._write_history))
            top_scores, top_indices = similarities.topk(k)

            results: list[dict] = []
            for score_val, idx_val in zip(top_scores.tolist(), top_indices.tolist()):
                emb_hash = self._write_history[idx_val][0]
                meta = self._decode_index.get(emb_hash)
                if meta is not None:
                    entry = dict(meta)
                    entry["score"] = max(score_val, 0.0)
                    entry["retrieval_energy"] = energy.item()
                    results.append(entry)
        else:
            # Fallback: no write_history (loaded from older checkpoint).
            # Return all decode entries sorted by recency so the pipeline
            # (spreading activation re-ranking, forgetting) can filter them.
            all_entries = sorted(
                self._decode_index.items(),
                key=lambda x: x[1].get("write_step", 0),
                reverse=True,
            )
            results = []
            for emb_hash, meta in all_entries[:top_k]:
                entry = dict(meta)
                entry["score"] = 0.1  # baseline score for unscored fallback
                entry["retrieval_energy"] = energy.item()
                results.append(entry)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ── Interference reinforcement ──────────────────────────────────

    def reinforce_retrieval(self, query: Tensor) -> None:
        """Mark activated weights as important (called after successful retrieval)."""
        with torch.no_grad():
            q = self.query_proj(query.to(_device).unsqueeze(0)).squeeze(0)
            q = F.normalize(q, dim=0)
            key_response = q @ self.W_key
            attention = F.softmax(self.log_beta.exp() * key_response, dim=0)

            self.omega_key += torch.outer(attention, q)
            self.omega_value += torch.outer(attention, q)

            self.omega_key *= 0.999
            self.omega_value *= 0.999

    # ── Occupancy ───────────────────────────────────────────────────

    def occupancy(self) -> float:
        """Estimate how 'full' the weight matrix is (nuclear norm proxy)."""
        if self.write_count == 0:
            return 0.0
        nuclear_norm = torch.linalg.norm(self.W_key, ord="nuc").item()
        theoretical_max = self.hidden_dim * self.write_lr * self.write_count.item()
        if theoretical_max == 0:
            return 0.0
        return min(1.0, nuclear_norm / (theoretical_max * 0.5))

    # ── Utilities ───────────────────────────────────────────────────

    def clear(self) -> None:
        """Reset all state — equivalent to amnesia."""
        self.W_key.zero_()
        self.W_value.zero_()
        self.omega_key.zero_()
        self.omega_value.zero_()
        self.write_count.zero_()
        self.total_energy.zero_()
        self._decode_index.clear()
        self._write_history.clear()

    # ── Decode-index persistence ────────────────────────────────────

    def save_decode_index_data(self) -> dict:
        """Return serialisable dict of decode index + write-history hashes."""
        return {
            "decode_index": self._decode_index,
            "write_history_hashes": [wh[0] for wh in self._write_history],
        }

    def save_write_history_tensors(self) -> list[tuple[str, Tensor]]:
        """Return (hash, key_tensor) pairs for persistence."""
        return [(h, k.clone()) for h, k in self._write_history]

    def load_decode_index_data(self, data: dict, key_tensors: dict[str, Tensor] | None = None) -> None:
        """Restore decode index and optionally rebuild write history from saved key tensors."""
        self._decode_index = data.get("decode_index", {})
        saved_hashes = data.get("write_history_hashes", [])
        if key_tensors and saved_hashes:
            self._write_history = [
                (h, key_tensors[h].to(_device))
                for h in saved_hashes
                if h in key_tensors
            ]
        else:
            self._write_history = []


# ────────────────────────────────────────────────────────────────────
# ModularFastWeightMemory — orchestrator (Phase 3)
# ────────────────────────────────────────────────────────────────────


class ModularFastWeightMemory(nn.Module):
    """M FastWeightModules + a MemoryRouter.

    Drop-in replacement for ``ModularHippocampalMemory`` — the observer
    calls the same ``store()`` / ``retrieve_decoded()`` / ``clear()`` API.
    """

    def __init__(
        self,
        embedding_dim: int = settings.embedding_dim,
        num_modules: int = settings.hopfield_num_modules,
        hidden_dim: int = settings.fast_weight_hidden_dim,
        write_lr: float = settings.fast_weight_write_lr,
        top_k_write: int = settings.hopfield_top_k_write,
        top_k_read: int = settings.hopfield_top_k_read,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_modules = num_modules
        self._top_k_write = top_k_write
        self._top_k_read = top_k_read

        self.router = MemoryRouter(
            num_modules=num_modules,
            embedding_dim=embedding_dim,
        )
        self.modules_list = nn.ModuleList([
            FastWeightModule(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                write_lr=write_lr,
            )
            for _ in range(num_modules)
        ])
        self.to(_device)

    # ── Properties for trainer backward compat ──────────────────────

    @property
    def separator(self) -> nn.Module:
        return self.modules_list[0].separator

    @property
    def query_proj(self) -> nn.Module:
        return self.modules_list[0].query_proj

    @property
    def log_beta(self) -> nn.Parameter:
        return self.modules_list[0].log_beta

    @property
    def beta(self) -> Tensor:
        return self.modules_list[0].log_beta.exp()

    @property
    def num_patterns(self) -> int:
        """Approximate — total writes across all modules."""
        return sum(int(m.write_count.item()) for m in self.modules_list)

    # ── Store ───────────────────────────────────────────────────────

    def store(
        self,
        embedding: Tensor,
        value: Tensor | None = None,
        episode_id: str = "",
        metadata: dict | None = None,
        strength: float = 1.0,
    ) -> None:
        """Route embedding to top-K modules and write into their weight matrices."""
        targets = self.router.route_write(embedding, top_k=self._top_k_write)
        for mod_idx, _score in targets:
            self.modules_list[mod_idx].write(
                embedding, episode_id=episode_id, metadata=metadata, strength=strength,
            )

    def replay_recent(self, n: int = 5, replay_strength: float = 0.3) -> None:
        """Replay recent memories across all active modules."""
        for m in self.modules_list:
            if m._write_history:
                m.replay_recent(n=n, replay_strength=replay_strength)

    # ── Retrieve ────────────────────────────────────────────────────

    def retrieve_decoded(self, query: Tensor, top_k: int = 5) -> list[dict]:
        """Query top-K modules and merge decoded results with deduplication."""
        read_targets = self.router.route_read(query, top_k=self._top_k_read)
        all_results: list[dict] = []

        for mod_idx, route_weight in read_targets:
            mod_results = self.modules_list[mod_idx].retrieve_decoded(query, top_k=top_k)
            for r in mod_results:
                # Use route_weight as a gentle tiebreaker, not a multiplier.
                # Raw cosine similarity IS the relevance signal.
                r["score"] = r["score"] + route_weight * 0.01
                r["source_module"] = mod_idx
            all_results.extend(mod_results)

        # Deduplicate by episode_id — keep highest score
        seen: dict[str, dict] = {}
        for r in all_results:
            eid = r.get("episode_id", "")
            if eid and eid in seen:
                if r["score"] > seen[eid]["score"]:
                    seen[eid] = r
            else:
                key = eid or str(id(r))
                seen[key] = r

        merged = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

    def forward(
        self,
        query: Tensor,
        top_k: int | None = None,
    ) -> tuple[Tensor, Tensor, list[int]]:
        """Compatibility forward — aggregates retrieve outputs."""
        read_targets = self.router.route_read(query, top_k=self._top_k_read)
        if not read_targets:
            return torch.zeros(self.embedding_dim, device=_device), torch.empty(0, device=_device), []

        retrieved_sum = torch.zeros(self.embedding_dim, device=_device)
        total_weight = 0.0
        for mod_idx, route_score in read_targets:
            mod = self.modules_list[mod_idx]
            if mod.write_count == 0:
                continue
            out, _energy = mod.retrieve(query)
            retrieved_sum += out * route_score
            total_weight += route_score
        if total_weight > 0:
            retrieved_sum /= total_weight
        return retrieved_sum, torch.empty(0, device=_device), []

    # ── Utilities ───────────────────────────────────────────────────

    def clear(self) -> None:
        for m in self.modules_list:
            m.clear()

    def total_writes(self) -> int:
        return sum(int(m.write_count.item()) for m in self.modules_list)

    def module_occupancies(self) -> list[float]:
        return [m.occupancy() for m in self.modules_list]

    def module_summary(self) -> list[dict]:
        """Per-module specialization summary."""
        from collections import Counter

        summaries: list[dict] = []
        for i, m in enumerate(self.modules_list):
            entities: list[str] = []
            topics: list[str] = []
            for entry in m._decode_index.values():
                entities.extend(entry.get("entities", []))
                topics.extend(entry.get("topics", []))
            summaries.append({
                "module_index": i,
                "occupancy": m.occupancy(),
                "write_count": int(m.write_count.item()),
                "w_key_norm": torch.norm(m.W_key).item(),
                "w_value_norm": torch.norm(m.W_value).item(),
                "top_entities": Counter(entities).most_common(5),
                "top_topics": Counter(topics).most_common(3),
            })
        return summaries

    def retrieve_episode_ids(self, query: Tensor, top_k: int = 5) -> list[tuple[str, float]]:
        results = self.retrieve_decoded(query, top_k=top_k)
        return [(r["episode_id"], r["score"]) for r in results if r.get("episode_id")]

    # ── Decode-index persistence ────────────────────────────────────

    def save_decode_index(self, path: str | Path) -> None:
        """Save all per-module decode indices as a single JSON file.

        Also saves write-history key tensors to a sibling ``.pt`` file
        so that ``retrieve_decoded`` works after reloading.
        """
        path = Path(path)
        combined: dict[str, dict] = {}
        all_tensors: dict[str, dict[str, Tensor]] = {}
        for i, m in enumerate(self.modules_list):
            data = m.save_decode_index_data()
            if data.get("decode_index"):
                combined[str(i)] = data
                wh = m.save_write_history_tensors()
                if wh:
                    all_tensors[str(i)] = {h: k for h, k in wh}
        with open(path, "w") as f:
            json.dump(combined, f)
        # Save key tensors alongside the JSON
        tensor_path = path.with_suffix(".keys.pt")
        torch.save(all_tensors, tensor_path)
        logger.info("Saved fast-weight decode index (%d modules) to %s", len(combined), path)

    def load_decode_index(self, path: str | Path) -> None:
        """Load per-module decode indices from a JSON file."""
        path = Path(path)
        if not path.exists():
            logger.warning("Fast-weight decode index not found at %s", path)
            return
        with open(path) as f:
            combined = json.load(f)
        # Load key tensors if available
        tensor_path = path.with_suffix(".keys.pt")
        all_tensors: dict[str, dict[str, Tensor]] = {}
        if tensor_path.exists():
            all_tensors = torch.load(tensor_path, map_location=_device, weights_only=True)
        for mod_str, data in combined.items():
            mod_idx = int(mod_str)
            if mod_idx < len(self.modules_list):
                key_tensors = all_tensors.get(mod_str)
                self.modules_list[mod_idx].load_decode_index_data(data, key_tensors=key_tensors)
        logger.info("Loaded fast-weight decode index from %s", path)


# ────────────────────────────────────────────────────────────────────
# Fast-weight replay buffer & slow-parameter training
# ────────────────────────────────────────────────────────────────────


class FastWeightReplayBuffer:
    """Stores ``(query_embedding, target_embedding)`` pairs for training
    the slow parameters of ``FastWeightModule``."""

    def __init__(self, capacity: int = settings.training_replay_buffer_size) -> None:
        self._buffer: deque[tuple[Tensor, Tensor]] = deque(maxlen=capacity)

    def push(self, query: Tensor, target: Tensor) -> None:
        self._buffer.append((query.detach().cpu(), target.detach().cpu()))

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor] | None:
        if len(self._buffer) < batch_size:
            return None
        indices = random.sample(range(len(self._buffer)), batch_size)
        batch = [self._buffer[i] for i in indices]
        queries = torch.stack([b[0] for b in batch])
        targets = torch.stack([b[1] for b in batch])
        return queries, targets

    def __len__(self) -> int:
        return len(self._buffer)


def train_fast_weight_step(
    module: FastWeightModule,
    optimizer: torch.optim.Optimizer,
    replay_buffer: FastWeightReplayBuffer,
    batch_size: int = 16,
) -> dict[str, float] | None:
    """Train slow parameters to improve retrieval quality.

    Loss = 1 − cosine_similarity(retrieved_output, target_embedding).
    """
    batch = replay_buffer.sample(batch_size)
    if batch is None:
        return None

    queries, targets = batch
    queries = queries.to(_device)
    targets = targets.to(_device)

    module.train()
    total_loss = torch.tensor(0.0, device=_device)
    n = 0

    for q, t in zip(queries, targets):
        output, _energy = module.retrieve(q)
        cos_sim = F.cosine_similarity(output.unsqueeze(0), t.unsqueeze(0))
        total_loss = total_loss + (1.0 - cos_sim.squeeze())
        n += 1

    loss = total_loss / max(n, 1)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(module.parameters(), settings.training_grad_clip)
    optimizer.step()

    return {"loss": loss.item(), "total": loss.item()}
