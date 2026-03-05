"""
Spreading Activation Engine — multi-channel seeding + sparse-matrix propagation.

Seeding is now a four-channel blend that mirrors how human recall works:

  1. **Entity channel**  — exact entity-label matches on the graph
  2. **Intent-cue channel** — "remember when …" / "what was my …" targets
  3. **Working-memory channel** — nodes similar to what's already "in mind"
  4. **Embedding-similarity channel** (fallback) — cosine similarity to a
     context vector; the old default behaviour, now weighted down

After seeding the combined activation vector goes through:

  propagate → lateral-inhibit → collect results

All matrix operations use ``torch.sparse`` so the engine can run on GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from config.settings import settings
from memory.semantic import SemanticGraph, SemanticNode

logger = logging.getLogger(__name__)
_device = settings.resolved_device


# ────────────────────────────────────────────────────────────────────
# Seed hints — structured cue packet passed into the engine
# ────────────────────────────────────────────────────────────────────


@dataclass
class SeedHints:
    """All the associative cues extracted from the current turn.

    The activation engine uses these to decide *which* graph nodes to
    wake up before propagation — just like a human mind is cued by
    names, intent, and recent context before memories "pop up".
    """

    entities: list[str] = field(default_factory=list)
    """Named entities (people, projects, places, tools) from the turn."""

    intent_targets: list[str] = field(default_factory=list)
    """Noun-phrase targets extracted from recall-intent patterns.
    E.g. ``["API key"]`` from "what was my API key?"."""

    intent_confidence: float = 0.0
    """0.0–1.0 confidence that this turn is a recall query."""

    working_memory_embeddings: list[Tensor] = field(default_factory=list)
    """Recent turn embeddings currently in the working-memory buffer."""

    context_vector: Tensor | None = None
    """GRU-projected context embedding (used for the fallback channel)."""


# ────────────────────────────────────────────────────────────────────
# Helper: build sparse adjacency matrix from graph
# ────────────────────────────────────────────────────────────────────


def _build_adjacency(graph: SemanticGraph) -> tuple[Tensor, list[str]]:
    """Return a sparse adjacency matrix and the ordered node-id list.

    Returns
    -------
    adj : torch.sparse_coo_tensor  [N, N]
        ``adj[i, j] == edge_weight`` if there is an edge *i → j*.
    node_ids : list[str]
        Ordered list of node IDs corresponding to matrix row/column indices.
    """
    node_ids = graph.node_ids()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for edge in graph.all_edges():
        src_idx = id_to_idx.get(edge.source)
        tgt_idx = id_to_idx.get(edge.target)
        if src_idx is not None and tgt_idx is not None:
            rows.append(src_idx)
            cols.append(tgt_idx)
            vals.append(edge.weight)

    if not rows:
        indices = torch.empty((2, 0), dtype=torch.long, device=_device)
        values = torch.empty(0, device=_device)
    else:
        indices = torch.tensor([rows, cols], dtype=torch.long, device=_device)
        values = torch.tensor(vals, dtype=torch.float32, device=_device)

    adj = torch.sparse_coo_tensor(indices, values, size=(n, n), device=_device)
    return adj, node_ids


# ────────────────────────────────────────────────────────────────────
# Core engine
# ────────────────────────────────────────────────────────────────────


class SpreadingActivationEngine:
    """Iterative spreading-activation over a semantic graph.

    Seeding uses four associative channels blended by configurable
    weights, making recall feel like *association* rather than search.

    Parameters
    ----------
    graph:
        The ``SemanticGraph`` to propagate activation through.
    max_iterations:
        Number of propagation hops.
    decay_factor:
        Per-hop multiplicative decay.
    seed_threshold:
        Minimum cosine similarity to seed a node (embedding channel).
    top_k:
        Nodes retained after lateral inhibition.
    """

    def __init__(
        self,
        graph: SemanticGraph,
        max_iterations: int = settings.spreading_activation_iterations,
        decay_factor: float = settings.activation_decay_factor,
        seed_threshold: float = settings.seed_similarity_threshold,
        top_k: int = settings.lateral_inhibition_k,
    ) -> None:
        self.graph = graph
        self.max_iterations = max_iterations
        self.decay_factor = decay_factor
        self.seed_threshold = seed_threshold
        self.top_k = top_k

        # Channel weights (from settings)
        self.w_entity = settings.seed_weight_entity
        self.w_intent = settings.seed_weight_intent
        self.w_wm = settings.seed_weight_working_memory
        self.w_embed = settings.seed_weight_embedding

        # Pre-compute adjacency on init; rebuild when graph mutates
        self._adj: Tensor | None = None
        self._node_ids: list[str] = []
        self._label_to_idx: dict[str, int] = {}
        self._rebuild_adjacency()

    def _rebuild_adjacency(self) -> None:
        """(Re-)build the sparse adjacency matrix from the current graph."""
        self._adj, self._node_ids = _build_adjacency(self.graph)
        # Index labels → matrix row for O(1) entity lookup
        self._label_to_idx = {}
        for i, nid in enumerate(self._node_ids):
            node = self.graph.get_node(nid)
            if node:
                self._label_to_idx[node.label.lower()] = i
                # Also index by node id
                self._label_to_idx[nid.lower()] = i

    def rebuild(self) -> None:
        """Public method to force adjacency rebuild after graph edits."""
        self._rebuild_adjacency()

    # ── channel 1: entity seeding ───────────────────────────────────

    def _seed_entities(self, entities: Sequence[str]) -> Tensor:
        """Activate nodes whose labels exactly match extracted entities.

        This is the strongest recall cue — names, projects, places.
        """
        n = len(self._node_ids)
        activation = torch.zeros(n, device=_device)
        for entity in entities:
            idx = self._label_to_idx.get(entity.lower())
            if idx is not None:
                activation[idx] = 1.0
                logger.debug("Entity seed: %r → node %s", entity, self._node_ids[idx])
        return activation

    # ── channel 2: intent-cue seeding ───────────────────────────────

    def _seed_intent_cues(
        self,
        targets: Sequence[str],
        confidence: float,
    ) -> Tensor:
        """Activate nodes that match intent-cue targets.

        Targets are noun-phrases like "API key" extracted from
        "what was my API key?".  We do fuzzy substring matching
        against node labels and boost by intent confidence.
        """
        n = len(self._node_ids)
        activation = torch.zeros(n, device=_device)
        if not targets or confidence < 0.1:
            return activation

        for target in targets:
            target_lower = target.lower().split()
            for label, idx in self._label_to_idx.items():
                # Substring / overlap match: any target word appears in the label
                if any(tw in label for tw in target_lower):
                    score = confidence * settings.intent_cue_boost
                    activation[idx] = max(activation[idx].item(), score)
                    logger.debug(
                        "Intent seed: %r matched label %r (score=%.2f)",
                        target, label, score,
                    )
        return activation

    # ── channel 3: working-memory focus ─────────────────────────────

    def _seed_working_memory(self, wm_embeddings: Sequence[Tensor]) -> Tensor:
        """Activate nodes similar to what's already in working memory.

        Averages the recent WM embeddings and computes cosine similarity
        against graph node embeddings.  Only nodes above `seed_threshold`
        are activated.  This represents "what's already on my mind".
        """
        n = len(self._node_ids)
        activation = torch.zeros(n, device=_device)
        if not wm_embeddings or n == 0:
            return activation

        # Compute centroid of recent WM items
        wm_stack = torch.stack([e.float().to(_device) for e in wm_embeddings])
        centroid = wm_stack.mean(dim=0)  # [D]

        node_matrix = self._node_embedding_matrix(centroid.shape[0])
        sims = F.cosine_similarity(node_matrix, centroid.unsqueeze(0), dim=1)
        activation = torch.where(sims >= self.seed_threshold, sims, torch.zeros_like(sims))
        return activation

    # ── channel 4: embedding similarity (fallback) ──────────────────

    def _seed_embedding(self, context_vector: Tensor | None) -> Tensor:
        """Classical cosine-similarity seeding — the original approach.

        Demoted to a secondary fallback channel so recall feels less
        like "search" and more like "association".
        """
        n = len(self._node_ids)
        activation = torch.zeros(n, device=_device)
        if context_vector is None or n == 0:
            return activation

        node_matrix = self._node_embedding_matrix(context_vector.shape[0])
        ctx = context_vector.float().to(_device).unsqueeze(0)  # [1, D]
        sims = F.cosine_similarity(node_matrix, ctx, dim=1)
        activation = torch.where(sims >= self.seed_threshold, sims, torch.zeros_like(sims))
        return activation

    # ── combined seeding ────────────────────────────────────────────

    def seed_nodes(
        self,
        context_vector: Tensor,
        hints: SeedHints | None = None,
    ) -> Tensor:
        """Multi-channel seeding: blend entity, intent, WM, and embedding signals.

        Parameters
        ----------
        context_vector:
            ``Tensor[D]`` — embedding-space context from working memory.
            Used by the embedding-similarity fallback channel.
        hints:
            Optional ``SeedHints`` carrying entities, intent cues, and
            WM focus.  When *None* the engine falls back to pure
            embedding similarity (backward-compatible).

        Returns
        -------
        activation:
            ``Tensor[N]`` — blended initial activation for every node.
        """
        if hints is None:
            # Legacy / backward-compat: pure embedding seeding
            return self.w_embed * self._seed_embedding(context_vector)

        # ── Run each channel ─────────────────────────────────────────
        a_entity = self._seed_entities(hints.entities)
        a_intent = self._seed_intent_cues(hints.intent_targets, hints.intent_confidence)
        a_wm = self._seed_working_memory(hints.working_memory_embeddings)
        embed_ctx = hints.context_vector if hints.context_vector is not None else context_vector
        a_embed = self._seed_embedding(embed_ctx)

        # ── Weighted blend ───────────────────────────────────────────
        blended = (
            self.w_entity * a_entity
            + self.w_intent * a_intent
            + self.w_wm * a_wm
            + self.w_embed * a_embed
        )

        logger.debug(
            "Seed channels — entity: %d, intent: %d, wm: %d, embed: %d active nodes",
            (a_entity > 0).sum().item(),
            (a_intent > 0).sum().item(),
            (a_wm > 0).sum().item(),
            (a_embed > 0).sum().item(),
        )

        return blended

    # ── helper: gather node embeddings ──────────────────────────────

    def _node_embedding_matrix(self, dim: int) -> Tensor:
        """Return ``Tensor[N, D]`` of node embeddings (on device)."""
        embeddings: list[list[float]] = []
        for nid in self._node_ids:
            node = self.graph.get_node(nid)
            if node and node.embedding:
                embeddings.append(node.embedding)
            else:
                embeddings.append([0.0] * dim)
        return torch.tensor(embeddings, dtype=torch.float32, device=_device)

    # ── step 2: propagate ───────────────────────────────────────────

    def propagate(self, activation: Tensor) -> Tensor:
        """Iteratively spread activation through the graph.

        .. code-block:: text

            for each iteration:
                activation = activation + decay * (adj^T @ activation)

        Parameters
        ----------
        activation:
            ``Tensor[N]`` — current activation vector.

        Returns
        -------
        activation:
            ``Tensor[N]`` — activation after all iterations.
        """
        if self._adj is None or self._adj.shape[0] == 0:
            return activation

        activation = activation.to(_device)
        adj_t = self._adj.t()  # transpose: incoming edges matter
        for _ in range(self.max_iterations):
            spread = torch.sparse.mm(adj_t, activation.unsqueeze(1)).squeeze(1)
            activation = activation + self.decay_factor * spread
        return activation

    # ── step 3: lateral inhibition ──────────────────────────────────

    def lateral_inhibition(self, activation: Tensor) -> Tensor:
        """Keep only the top-K activated nodes, zero the rest."""
        if activation.shape[0] <= self.top_k:
            return activation
        topk_vals, topk_idx = torch.topk(activation, self.top_k)
        inhibited = torch.zeros_like(activation)
        inhibited[topk_idx] = topk_vals
        return inhibited

    # ── full pipeline ───────────────────────────────────────────────

    def activate(
        self,
        context_vector: Tensor,
        hints: SeedHints | None = None,
    ) -> list[tuple[str, float]]:
        """Run the complete spreading-activation pipeline.

        Parameters
        ----------
        context_vector:
            ``Tensor[D]`` — embedding-space context from working memory.
        hints:
            Optional ``SeedHints`` for multi-channel seeding.  When
            *None* the engine falls back to embedding-only seeding.

        Returns
        -------
        list of (node_id, activation_strength) tuples, sorted by strength descending.
        """
        activation = self.seed_nodes(context_vector, hints=hints)
        activation = self.propagate(activation)
        activation = self.lateral_inhibition(activation)

        # Write activation back to graph nodes
        results: list[tuple[str, float]] = []
        for idx, nid in enumerate(self._node_ids):
            val = activation[idx].item()
            self.graph.set_activation(nid, val)
            if val > 0:
                results.append((nid, val))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
