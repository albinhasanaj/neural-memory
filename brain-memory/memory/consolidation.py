"""
Memory Consolidation — episodic → semantic knowledge extraction.

Runs periodically (every *N* turns or at session end).  High-salience
unconsolidated episodes are clustered by embedding similarity, then
each cluster is sent to an LLM to extract a typed fact (subject,
relation, object).  The extracted fact is upserted into the semantic
knowledge graph, and the source episodes are marked as consolidated.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage

from config.settings import settings
from memory.episodic import EpisodicEntry, EpisodicStore
from memory.semantic import SemanticEdge, SemanticGraph, SemanticNode
from memory.encoder import get_encoder

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Step 1: Select candidates
# ────────────────────────────────────────────────────────────────────


def select_candidates(
    store: EpisodicStore,
    min_salience: float = settings.salience_threshold,
) -> list[EpisodicEntry]:
    """Return unconsolidated episodes above the salience floor."""
    return store.get_unconsolidated(min_salience=min_salience)


# ────────────────────────────────────────────────────────────────────
# Step 2: Cluster episodes
# ────────────────────────────────────────────────────────────────────


def cluster_episodes(
    episodes: list[EpisodicEntry],
    distance_threshold: float = settings.consolidation_cluster_threshold,
) -> list[list[EpisodicEntry]]:
    """Agglomerative clustering on episode embeddings.

    Returns a list of clusters, each containing ≥ 1 episode.
    """
    if len(episodes) <= 1:
        return [episodes] if episodes else []

    matrix = np.array([e.embedding for e in episodes], dtype=np.float32)
    # Normalise so cosine distance ≈ 1 − dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    matrix = matrix / norms

    Z = linkage(matrix, method="average", metric="cosine")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    clusters: dict[int, list[EpisodicEntry]] = {}
    for label, episode in zip(labels, episodes):
        clusters.setdefault(int(label), []).append(episode)

    return list(clusters.values())


def cluster_episodes_vae(
    episodes: list[EpisodicEntry],
    vae: "ConsolidationVAE",
    n_clusters: int = 8,
) -> list[list[EpisodicEntry]]:
    """Cluster episodes using VAE latent-space K-means.

    Uses the trained VAE to embed episodes into latent space, then runs
    K-means clustering.  This replaces agglomerative clustering when
    ``use_vae_consolidation`` is enabled.
    """
    if len(episodes) <= 1:
        return [episodes] if episodes else []

    from memory.neural_consolidation import (
        build_metadata_vector,
        latent_cluster_assignments,
    )

    embeddings = torch.tensor(
        [e.embedding for e in episodes], dtype=torch.float32,
    )
    metadata = torch.stack([
        build_metadata_vector(
            salience=e.salience,
            entity_count=len(e.entities),
            speaker_is_user=(e.speaker == "user"),
        )
        for e in episodes
    ])

    # Move tensors to the same device as the VAE model
    device = next(vae.parameters()).device
    embeddings = embeddings.to(device)
    metadata = metadata.to(device)

    assignments = latent_cluster_assignments(
        vae, embeddings, metadata, n_clusters=n_clusters,
    )

    clusters: dict[int, list[EpisodicEntry]] = {}
    for label, episode in zip(assignments, episodes):
        clusters.setdefault(int(label), []).append(episode)

    return list(clusters.values())


# ────────────────────────────────────────────────────────────────────
# Step 3: Extract typed facts via LLM
# ────────────────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are a precise knowledge-extraction engine.  Given the following
conversation excerpts, extract ONE factual relationship as JSON.

Excerpts:
{texts}

Respond ONLY with valid JSON in this exact schema (no markdown, no explanation):
{{"subject": "<entity or concept>", "relation": "<relation verb phrase>", "object": "<entity, concept, or value>", "confidence": <float 0-1>}}
"""


async def extract_fact(
    texts: list[str],
) -> dict[str, Any] | None:
    """Call the configured LLM to extract a typed fact from a cluster of episode texts.

    Returns a dict with keys ``subject``, ``relation``, ``object``,
    ``confidence``, or *None* on failure.
    """
    prompt = _EXTRACTION_PROMPT.format(texts="\n---\n".join(texts))

    try:
        if settings.llm_provider == "openai":
            return await _call_openai(prompt)
        elif settings.llm_provider == "anthropic":
            return await _call_anthropic(prompt)
        else:
            logger.error("Unknown LLM provider: %s", settings.llm_provider)
            return None
    except Exception:
        logger.exception("Fact extraction LLM call failed")
        return None


async def _call_openai(prompt: str) -> dict[str, Any] | None:
    base = settings.llm_base_url or "https://api.openai.com/v1"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={
                "model": settings.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content)  # type: ignore[no-any-return]


async def _call_anthropic(prompt: str) -> dict[str, Any] | None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": settings.llm_model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        content = resp.json()["content"][0]["text"]
        return json.loads(content)  # type: ignore[no-any-return]


# ────────────────────────────────────────────────────────────────────
# Step 4+5: Full consolidation pipeline
# ────────────────────────────────────────────────────────────────────


async def consolidate(
    store: EpisodicStore,
    graph: SemanticGraph,
    vae: Any = None,
) -> list[SemanticEdge]:
    """Run the full consolidation pipeline and return newly created edges.

    1. Select high-salience unconsolidated episodes
    2. Cluster by embedding similarity (VAE or agglomerative)
    3. For clusters ≥ 2 episodes, extract a typed fact via LLM
    4. Upsert the fact into the semantic graph
    5. Decay salience of consolidated episodes
    """
    candidates = select_candidates(store)
    if not candidates:
        logger.info("No consolidation candidates.")
        return []

    # Use VAE-based clustering when enabled and a trained VAE is available
    if settings.use_vae_consolidation and vae is not None:
        clusters = cluster_episodes_vae(candidates, vae)
    else:
        clusters = cluster_episodes(candidates)
    new_edges: list[SemanticEdge] = []
    encoder = get_encoder()

    for cluster in clusters:
        if len(cluster) < 2:
            # Mark single episodes as consolidated anyway
            for ep in cluster:
                ep.consolidated = True
            continue

        texts = [ep.raw_text for ep in cluster]
        fact = await extract_fact(texts)
        if fact is None:
            continue

        subj = fact.get("subject", "")
        rel = fact.get("relation", "related_to")
        obj = fact.get("object", "")
        conf = float(fact.get("confidence", 0.5))

        if not subj or not obj:
            continue

        # Create / update nodes
        subj_id = subj.lower().replace(" ", "_")
        obj_id = obj.lower().replace(" ", "_")

        subj_emb = encoder.encode(subj).tolist()
        obj_emb = encoder.encode(obj).tolist()

        graph.upsert_node(SemanticNode(
            id=subj_id, label=subj, embedding=subj_emb, node_type="entity",
        ))
        graph.upsert_node(SemanticNode(
            id=obj_id, label=obj, embedding=obj_emb, node_type="entity",
        ))

        edge = SemanticEdge(
            source=subj_id,
            target=obj_id,
            relation=rel,
            weight=conf,
            confidence=conf,
            evidence=[ep.id for ep in cluster],
        )
        graph.upsert_edge(edge)
        new_edges.append(edge)

        # Mark episodes as consolidated + decay salience + populate links
        for ep in cluster:
            ep.consolidated = True
            ep.salience *= settings.consolidation_salience_decay
            if subj_id not in ep.links:
                ep.links.append(subj_id)
            if obj_id not in ep.links:
                ep.links.append(obj_id)

    logger.info(
        "Consolidation complete: %d clusters processed, %d new edges.",
        len(clusters),
        len(new_edges),
    )
    return new_edges


def decay_consolidated(store: EpisodicStore) -> None:
    """Apply salience decay to all consolidated (but not archived) episodes."""
    for entry in store.values():
        if entry.consolidated and not entry.archived:
            entry.salience *= settings.consolidation_salience_decay


# ────────────────────────────────────────────────────────────────────
# Background runner
# ────────────────────────────────────────────────────────────────────


_consolidation_lock = __import__("threading").Lock()


def run_consolidation_background(
    store: EpisodicStore,
    graph: SemanticGraph,
    vae: Any = None,
) -> None:
    """Fire-and-forget consolidation in a background thread.

    Safe to call from sync code — spins up a new event loop in a thread.
    Uses a module-level lock to prevent concurrent store/graph mutations.
    """
    import threading

    def _run() -> None:
        if not _consolidation_lock.acquire(blocking=False):
            logger.debug("Consolidation already running, skipping.")
            return
        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(consolidate(store, graph, vae=vae))
            finally:
                loop.close()
        finally:
            _consolidation_lock.release()

    t = threading.Thread(target=_run, daemon=True, name="consolidation")
    t.start()
