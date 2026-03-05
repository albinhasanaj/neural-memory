"""
Seed memory — pre-populate episodic and semantic stores with sample data.

Useful for testing and demo purposes.  Creates a small set of episodes
and a starter knowledge graph so the system has something to work with
before any real conversations happen.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone

from config.settings import settings
from memory.encoder import get_encoder
from memory.episodic import EpisodicEntry, EpisodicStore
from memory.semantic import SemanticEdge, SemanticGraph, SemanticNode
from storage.graph_store import save_graph
from storage.sqlite_store import SQLiteEpisodicStore

logger = logging.getLogger(__name__)

# ── sample data ──────────────────────────────────────────────────────

SAMPLE_TURNS: list[dict[str, str]] = [
    {"speaker": "user", "text": "My name is Alex and I'm a software developer."},
    {"speaker": "assistant", "text": "Nice to meet you, Alex! What kind of development do you do?"},
    {"speaker": "user", "text": "Mostly Python backend work. I love FastAPI."},
    {"speaker": "assistant", "text": "FastAPI is great for building APIs quickly!"},
    {"speaker": "user", "text": "I really dislike WordPress. It corrupted my database once."},
    {"speaker": "assistant", "text": "That sounds frustrating. Static site generators might be a better fit."},
    {"speaker": "user", "text": "I prefer dark mode in all my editors and tools."},
    {"speaker": "user", "text": "I'm currently learning about neural networks and transformers."},
    {"speaker": "user", "text": "My team uses PostgreSQL for our main database."},
    {"speaker": "user", "text": "We deploy everything on AWS using Docker containers."},
]

SAMPLE_FACTS: list[dict[str, str]] = [
    {"subject": "User", "relation": "is named", "object": "Alex"},
    {"subject": "User", "relation": "prefers", "object": "Python"},
    {"subject": "User", "relation": "dislikes", "object": "WordPress"},
    {"subject": "User", "relation": "uses", "object": "FastAPI"},
    {"subject": "User", "relation": "prefers", "object": "dark mode"},
    {"subject": "User", "relation": "is learning", "object": "neural networks"},
    {"subject": "User", "relation": "uses database", "object": "PostgreSQL"},
    {"subject": "User", "relation": "deploys on", "object": "AWS"},
    {"subject": "Python", "relation": "has framework", "object": "FastAPI"},
    {"subject": "FastAPI", "relation": "is a", "object": "web framework"},
]


def seed_episodic(db: SQLiteEpisodicStore) -> int:
    """Create sample episodic entries and persist them."""
    encoder = get_encoder()
    now = datetime.now(timezone.utc)
    count = 0

    for i, turn in enumerate(SAMPLE_TURNS):
        ts = now - timedelta(hours=len(SAMPLE_TURNS) - i)
        embedding = encoder.encode(turn["text"])
        entry = EpisodicEntry(
            timestamp=ts,
            speaker=turn["speaker"],
            raw_text=turn["text"],
            embedding=embedding.tolist(),
            salience=0.5 + (0.05 * i),  # increasing salience
            recall_times=[ts.timestamp()],
        )
        db.insert(entry)
        count += 1

    return count


def seed_semantic() -> SemanticGraph:
    """Create a sample semantic knowledge graph."""
    encoder = get_encoder()
    graph = SemanticGraph()

    # Collect all unique entities
    all_entities: set[str] = set()
    for fact in SAMPLE_FACTS:
        all_entities.add(fact["subject"])
        all_entities.add(fact["object"])

    # Create nodes
    for entity in all_entities:
        emb = encoder.encode(entity)
        graph.upsert_node(SemanticNode(
            id=entity.lower().replace(" ", "_"),
            label=entity,
            embedding=emb.tolist(),
            node_type="entity",
        ))

    # Create edges
    for fact in SAMPLE_FACTS:
        subj_id = fact["subject"].lower().replace(" ", "_")
        obj_id = fact["object"].lower().replace(" ", "_")
        graph.upsert_edge(SemanticEdge(
            source=subj_id,
            target=obj_id,
            relation=fact["relation"],
            weight=0.8,
            confidence=0.7,
            evidence=["seed_data"],
        ))

    return graph


def main() -> None:
    """CLI entry point for ``brain-seed``."""
    logging.basicConfig(level=logging.INFO)
    settings.ensure_data_dirs()

    logger.info("Seeding episodic memory...")
    db = SQLiteEpisodicStore()
    ep_count = seed_episodic(db)
    logger.info("  → %d episodes created.", ep_count)

    logger.info("Seeding semantic graph...")
    graph = seed_semantic()
    save_graph(graph)
    logger.info("  → %d nodes, %d edges created.", graph.num_nodes, graph.num_edges)

    db.close()
    logger.info("Seeding complete! Data written to %s", settings.data_dir)


if __name__ == "__main__":
    main()
