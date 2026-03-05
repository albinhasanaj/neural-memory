"""
Experiment 2: Episodic buffer + injection only (no semantic graph).

Tests whether raw episodic replay is sufficient for recall accuracy.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import httpx

from config.settings import settings
from experiments.eval_metrics import compute_all_metrics
from experiments.run_baseline import CONVERSATION
from memory.encoder import get_encoder
from memory.episodic import EpisodicEntry, EpisodicStore
from memory.injector import build_memory_context, inject
from memory.salience import SalienceScorer
from memory.semantic import SemanticGraph
from memory.working_memory import WorkingMemory
from nlp.entity_extractor import extract_entities

logger = logging.getLogger(__name__)


async def run_episodic_only() -> dict[str, float]:
    """Run experiment with episodic memory only (no semantic graph)."""
    logger.info("Running episodic-only experiment...")

    encoder = get_encoder()
    store = EpisodicStore()
    wm = WorkingMemory()
    scorer = SalienceScorer()
    empty_graph = SemanticGraph()

    # Process all turns except the last (the query)
    for msg in CONVERSATION[:-1]:
        emb = encoder.encode(msg["content"])
        ctx, pred = wm.update(emb)
        entities = extract_entities(msg["content"])
        salience = scorer.score(emb, pred, entities, msg["content"], empty_graph)

        if salience >= settings.salience_threshold:
            entry = EpisodicEntry(
                speaker=msg["role"],
                raw_text=msg["content"],
                embedding=emb.tolist(),
                entities=entities,
                salience=salience,
            )
            store.add(entry)

    # Build memory context from episodic store only
    episodes = store.get_all_active()
    memory_block = build_memory_context(empty_graph, [], episodes)

    # Build messages with memory context
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
        activated_nodes=[],
        relevant_nodes=set(),
        user_preferences={"language": "Python", "editor_theme": "dark mode"},
        response_text=response_text,
    )

    logger.info("Episodic-only results: %s", json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(run_episodic_only())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
