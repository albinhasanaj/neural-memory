"""
Experiment 5: Full system — all components enabled.

Runs the complete memory pipeline (observer → working memory →
episodic → semantic → activation → injection) over a multi-turn
conversation and measures all evaluation metrics.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx

from config.settings import settings
from experiments.eval_metrics import compute_all_metrics
from experiments.run_baseline import CONVERSATION
from memory.observer import MemoryObserver

logger = logging.getLogger(__name__)


async def run_full_system() -> dict[str, float]:
    """Run the full brain-memory system over the test conversation."""
    logger.info("Running full-system experiment...")

    observer = MemoryObserver()

    # Process all turns except the last (the query)
    for msg in CONVERSATION[:-1]:
        observer.observe(msg["content"], speaker=msg["role"])

    # Process the query turn with injection
    query = CONVERSATION[-1]
    modified_messages, info = observer.process_turn(
        text=query["content"],
        speaker=query["role"],
        messages=CONVERSATION,
    )

    # Extract the memory context block for metric evaluation
    memory_block = ""
    for msg in modified_messages:
        if msg.get("role") == "system" and "[MEMORY CONTEXT" in msg.get("content", ""):
            memory_block = msg["content"]
            break

    # Send to LLM
    base = settings.llm_base_url or "https://api.openai.com/v1"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{base}/chat/completions",
            headers=headers,
            json={
                "model": settings.llm_model,
                "messages": modified_messages,
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()
        response_text = resp.json()["choices"][0]["message"]["content"]

    # Observe the response
    observer.observe(response_text, speaker="assistant")

    activated_ids = [nid for nid, _ in info.get("activated_nodes", [])]
    metrics = compute_all_metrics(
        expected_facts=["Python", "dark mode", "WordPress"],
        should_forget=[],
        retrieved_context=memory_block,
        activated_nodes=activated_ids,
        relevant_nodes={"python", "wordpress", "dark_mode", "user"},
        user_preferences={"language": "Python", "editor_theme": "dark mode"},
        response_text=response_text,
    )

    logger.info("Full-system results: %s", json.dumps(metrics, indent=2))
    logger.info("Response: %s", response_text[:200])
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(run_full_system())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
