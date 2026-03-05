"""
Experiment 3: Semantic graph only (no episodic buffer).

Pre-populates the semantic graph with known facts and tests whether
structured knowledge alone is sufficient for recall.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx

from config.settings import settings
from experiments.eval_metrics import compute_all_metrics
from experiments.run_baseline import CONVERSATION
from memory.encoder import get_encoder
from memory.injector import build_memory_context, inject
from memory.semantic import SemanticEdge, SemanticGraph, SemanticNode

logger = logging.getLogger(__name__)


def build_test_graph() -> SemanticGraph:
    """Create a small pre-populated semantic graph for the experiment."""
    encoder = get_encoder()
    g = SemanticGraph()

    nodes = [
        ("user", "User"),
        ("python", "Python"),
        ("wordpress", "WordPress"),
        ("dark_mode", "dark mode"),
    ]
    for nid, label in nodes:
        g.upsert_node(SemanticNode(
            id=nid, label=label,
            embedding=encoder.encode(label).tolist(),
        ))

    edges = [
        SemanticEdge(source="user", target="python", relation="prefers", weight=0.8, confidence=0.72, evidence=["e1", "e2"]),
        SemanticEdge(source="user", target="wordpress", relation="dislikes", weight=0.85, confidence=0.85, evidence=["e3", "e4", "e5"]),
        SemanticEdge(source="user", target="dark_mode", relation="prefers", weight=0.9, confidence=0.9, evidence=["e6"]),
    ]
    for e in edges:
        g.upsert_edge(e)

    return g


async def run_semantic_only() -> dict[str, float]:
    """Run experiment with semantic graph only."""
    logger.info("Running semantic-only experiment...")

    graph = build_test_graph()
    activated = [("user", 1.0), ("python", 0.7), ("wordpress", 0.6), ("dark_mode", 0.5)]
    memory_block = build_memory_context(graph, activated, [])

    query_msg = CONVERSATION[-1]
    messages = [
        {"role": "system", "content": memory_block},
        *CONVERSATION[:-1],
        query_msg,
    ]

    base = settings.llm_base_url or "https://api.openai.com/v1"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{base}/chat/completions",
            headers=headers,
            json={"model": settings.llm_model, "messages": messages, "temperature": 0.0},
        )
        resp.raise_for_status()
        response_text = resp.json()["choices"][0]["message"]["content"]

    metrics = compute_all_metrics(
        expected_facts=["Python", "dark mode", "WordPress"],
        should_forget=[],
        retrieved_context=memory_block,
        activated_nodes=["user", "python", "wordpress", "dark_mode"],
        relevant_nodes={"user", "python", "wordpress", "dark_mode"},
        user_preferences={"language": "Python", "editor_theme": "dark mode"},
        response_text=response_text,
    )

    logger.info("Semantic-only results: %s", json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(run_semantic_only())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
