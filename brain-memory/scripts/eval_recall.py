"""
Evaluation harness — measures recall accuracy of the memory system.

Constructs conversations where turns 1-N establish facts, then a final
query asks "do you remember X?".  The eval measures whether the memory
system retrieves the correct fact.

Metrics
-------
* **Recall@K** — Was the correct memory in the top‑K retrieved?
* **MRR** — Mean Reciprocal Rank of the correct memory.
* **Storage rate** — What fraction of turns were stored by the gate?
* **Avg salience** — Mean salience of stored vs. unstored turns.

Usage::

    python -m scripts.eval_recall [--checkpoint PATH] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── Ensure project root is on path ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-28s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_recall")

# ────────────────────────────────────────────────────────────────────
# Evaluation dataset: conversations with ground-truth recall targets
# ────────────────────────────────────────────────────────────────────

EVAL_CONVERSATIONS: list[dict] = [
    {
        "id": "eval_python_prefs",
        "setup_turns": [
            {"role": "user", "text": "I primarily use Python for backend development."},
            {"role": "assistant", "text": "Python is a great choice for backend work!"},
            {"role": "user", "text": "My favorite framework is FastAPI because of its async support."},
            {"role": "assistant", "text": "FastAPI is excellent for async APIs. Do you use it with SQLAlchemy?"},
            {"role": "user", "text": "Yes, I pair FastAPI with SQLAlchemy and Pydantic for data validation."},
        ],
        "query": "What framework do I prefer for backend development?",
        "expected_keywords": ["FastAPI", "fastapi"],
        "expected_topic": "framework preference",
    },
    {
        "id": "eval_pet_name",
        "setup_turns": [
            {"role": "user", "text": "I have a golden retriever named Biscuit."},
            {"role": "assistant", "text": "Biscuit is an adorable name! How old is Biscuit?"},
            {"role": "user", "text": "She's 3 years old and loves swimming in the lake."},
            {"role": "assistant", "text": "Golden retrievers are natural swimmers!"},
        ],
        "query": "What's my dog's name?",
        "expected_keywords": ["Biscuit", "biscuit"],
        "expected_topic": "pet name",
    },
    {
        "id": "eval_work_project",
        "setup_turns": [
            {"role": "user", "text": "At work, I'm building a recommendation engine for e-commerce."},
            {"role": "assistant", "text": "That's an interesting project! What approach are you taking?"},
            {"role": "user", "text": "We're using collaborative filtering with a transformer model."},
            {"role": "assistant", "text": "Transformers work well for sequential recommendation."},
            {"role": "user", "text": "Our dataset has about 50 million user interactions from the past year."},
        ],
        "query": "What project am I working on at my job?",
        "expected_keywords": ["recommendation", "e-commerce", "collaborative filtering"],
        "expected_topic": "work project",
    },
    {
        "id": "eval_travel_plan",
        "setup_turns": [
            {"role": "user", "text": "I'm planning a trip to Japan next April for cherry blossom season."},
            {"role": "assistant", "text": "April is the perfect time for hanami! Which cities?"},
            {"role": "user", "text": "Tokyo for a week, then Kyoto for five days to visit the temples."},
            {"role": "assistant", "text": "Fushimi Inari and Kinkaku-ji are must-sees in Kyoto."},
        ],
        "query": "Where am I traveling to and when?",
        "expected_keywords": ["Japan", "April", "cherry blossom", "Tokyo", "Kyoto"],
        "expected_topic": "travel plans",
    },
    {
        "id": "eval_health_routine",
        "setup_turns": [
            {"role": "user", "text": "I've been doing intermittent fasting, 16:8 schedule."},
            {"role": "assistant", "text": "16:8 is a popular IF schedule. When is your eating window?"},
            {"role": "user", "text": "I eat between noon and 8pm. I also run 5K every morning at 6am."},
            {"role": "assistant", "text": "Running fasted can boost fat oxidation. How long have you been doing this?"},
            {"role": "user", "text": "About six months now. I've lost 15 pounds so far."},
        ],
        "query": "What's my exercise routine and diet approach?",
        "expected_keywords": ["intermittent fasting", "16:8", "5K", "run", "morning"],
        "expected_topic": "health routine",
    },
    {
        "id": "eval_music_taste",
        "setup_turns": [
            {"role": "user", "text": "I've been really into jazz lately, especially Miles Davis."},
            {"role": "assistant", "text": "Miles Davis is legendary! Kind of Blue is a masterpiece."},
            {"role": "user", "text": "That's my favorite album! I also love Coltrane's A Love Supreme."},
        ],
        "query": "What kind of music do I enjoy?",
        "expected_keywords": ["jazz", "Miles Davis", "Coltrane"],
        "expected_topic": "music taste",
    },
    {
        "id": "eval_tech_setup",
        "setup_turns": [
            {"role": "user", "text": "I just built a new PC with an RTX 4090 and 64GB RAM."},
            {"role": "assistant", "text": "That's a beast! What CPU did you go with?"},
            {"role": "user", "text": "AMD Ryzen 9 7950X. I use it mainly for deep learning research."},
            {"role": "assistant", "text": "The 7950X with a 4090 is a great combo for ML workloads."},
        ],
        "query": "What are my computer specs?",
        "expected_keywords": ["RTX 4090", "64GB", "Ryzen 9 7950X", "AMD"],
        "expected_topic": "computer specs",
    },
    {
        "id": "eval_learning_goals",
        "setup_turns": [
            {"role": "user", "text": "I want to learn Rust this year for systems programming."},
            {"role": "assistant", "text": "Rust is great for systems! Any specific projects in mind?"},
            {"role": "user", "text": "I want to rewrite our data pipeline's hot path from Python to Rust."},
            {"role": "assistant", "text": "PyO3 makes Python-Rust interop smooth."},
            {"role": "user", "text": "Exactly! I'm also interested in writing a custom memory allocator."},
        ],
        "query": "What programming language am I trying to learn and why?",
        "expected_keywords": ["Rust", "systems programming", "data pipeline"],
        "expected_topic": "learning goals",
    },
]


# ────────────────────────────────────────────────────────────────────
# Evaluator
# ────────────────────────────────────────────────────────────────────


class RecallEvaluator:
    """Evaluate the memory system's recall accuracy."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        self._enable_neural_modules()
        from memory.observer import NeuralMemoryObserver

        self.observer = NeuralMemoryObserver()

        if checkpoint_path:
            from pathlib import Path
            self.observer._trainer.load_checkpoint(Path(checkpoint_path))
            logger.info("Loaded checkpoint from %s", checkpoint_path)

    @staticmethod
    def _enable_neural_modules() -> None:
        """Enable all neural modules via env vars."""
        os.environ["BRAIN_USE_PATTERN_SEPARATION"] = "true"
        os.environ["BRAIN_USE_DOPAMINERGIC_GATE"] = "true"
        os.environ["BRAIN_USE_HOPFIELD_MEMORY"] = "true"
        os.environ["BRAIN_USE_VAE_CONSOLIDATION"] = "true"
        os.environ["BRAIN_USE_TRANSFORMER_WM"] = "true"
        os.environ["BRAIN_USE_LEARNED_FORGETTING"] = "true"

    def evaluate_conversation(self, conv: dict) -> dict:
        """Run one eval conversation and measure recall quality.

        Returns a result dict with metrics for this conversation.
        """
        # Reset memory state for a clean eval
        self.observer.working_memory.clear()
        self.observer.episodic_store._entries.clear()
        if self.observer._hopfield is not None:
            self.observer._hopfield.clear()
        self.observer.graph.reset_activations()

        # 1. Feed setup turns
        stored_turns: list[dict] = []
        for turn in conv["setup_turns"]:
            info = self.observer.observe(turn["text"], speaker=turn["role"])
            stored_turns.append({
                "text": turn["text"],
                "role": turn["role"],
                "stored": info["stored"],
                "salience": info["salience"],
                "episode_id": info.get("episode_id"),
            })

        # 1b. Run synchronous consolidation so the semantic graph is populated
        #     before we evaluate retrieval quality
        import asyncio
        from memory.consolidation import consolidate

        vae_model = None
        if hasattr(self.observer, "_trainer") and self.observer._trainer is not None:
            vae_comp = self.observer._trainer.components.get("vae")
            if vae_comp and vae_comp.enabled and vae_comp.model is not None:
                vae_model = vae_comp.model

        try:
            new_edges = asyncio.run(consolidate(
                self.observer.episodic_store, self.observer.graph, vae=vae_model,
            ))
            if new_edges:
                logger.info("Consolidation produced %d edges for %s", len(new_edges), conv["id"])
        except Exception as e:
            logger.debug("Consolidation skipped: %s", e)

        # 2. Encode the query
        query_embedding = self.observer.encoder.encode(conv["query"])

        # 3. Retrieve from episodic store — cosine similarity ranking
        all_episodes = self.observer.episodic_store.get_all_active()
        if not all_episodes:
            return {
                "id": conv["id"],
                "topic": conv["expected_topic"],
                "stored_count": sum(1 for t in stored_turns if t["stored"]),
                "total_turns": len(stored_turns),
                "recall_at_1": False,
                "recall_at_3": False,
                "recall_at_5": False,
                "mrr": 0.0,
                "keyword_found_in_top_k": False,
                "retrieved_texts": [],
            }

        # Rank episodes by cosine similarity to query
        # Move query to CPU for comparison (episode embeddings are stored as lists)
        query_cpu = query_embedding.detach().cpu()
        ranked = []
        for ep in all_episodes:
            ep_emb = torch.tensor(ep.embedding, dtype=torch.float32)
            sim = F.cosine_similarity(
                query_cpu.unsqueeze(0),
                ep_emb.unsqueeze(0),
            ).item()
            ranked.append((ep, sim))

        ranked.sort(key=lambda x: x[1], reverse=True)

        # 4. Also try Hopfield retrieval if available
        hopfield_retrieved: list[str] = []
        if self.observer._hopfield is not None and self.observer._hopfield.num_patterns > 0:
            with torch.no_grad():
                h_results = self.observer._hopfield.retrieve_episode_ids(
                    query_embedding.to(self.observer._hopfield.patterns.device), top_k=5,
                )
                hopfield_retrieved = [eid for eid, _ in h_results]

        # 5. Check if expected keywords appear in top-K retrieved
        expected_kws = [kw.lower() for kw in conv["expected_keywords"]]

        def has_keyword(text: str) -> bool:
            text_lower = text.lower()
            return any(kw in text_lower for kw in expected_kws)

        # Compute metrics
        recall_1 = has_keyword(ranked[0][0].raw_text) if len(ranked) >= 1 else False
        recall_3 = any(has_keyword(ep.raw_text) for ep, _ in ranked[:3])
        recall_5 = any(has_keyword(ep.raw_text) for ep, _ in ranked[:5])

        # MRR: reciprocal rank of first hit
        mrr = 0.0
        for i, (ep, _sim) in enumerate(ranked):
            if has_keyword(ep.raw_text):
                mrr = 1.0 / (i + 1)
                break

        # Hopfield recall
        hopfield_hit = False
        if hopfield_retrieved:
            for eid in hopfield_retrieved:
                ep = self.observer.episodic_store.get(eid)
                if ep and has_keyword(ep.raw_text):
                    hopfield_hit = True
                    break

        # 6. Full pipeline test: observe query, then activate_and_inject
        #    This tests the complete spreading activation + inject flow
        self.observer.observe(conv["query"], speaker="user")
        dummy_messages = [{"role": "user", "content": conv["query"]}]
        injected_messages, activated_nodes = self.observer.activate_and_inject(dummy_messages)

        # Check if injected context contains expected keywords
        injected_text = " ".join(m.get("content", "") for m in injected_messages)
        pipeline_hit = has_keyword(injected_text)
        graph_nodes_activated = len(activated_nodes)

        return {
            "id": conv["id"],
            "topic": conv["expected_topic"],
            "stored_count": sum(1 for t in stored_turns if t["stored"]),
            "total_turns": len(stored_turns),
            "storage_rate": sum(1 for t in stored_turns if t["stored"]) / len(stored_turns),
            "avg_salience": sum(t["salience"] for t in stored_turns) / len(stored_turns),
            "recall_at_1": recall_1,
            "recall_at_3": recall_3,
            "recall_at_5": recall_5,
            "mrr": mrr,
            "hopfield_hit": hopfield_hit,
            "pipeline_hit": pipeline_hit,
            "graph_nodes_activated": graph_nodes_activated,
            "keyword_found_in_top_k": recall_5,
            "num_episodes_in_store": len(all_episodes),
            "retrieved_texts": [ep.raw_text[:100] for ep, _ in ranked[:5]],
        }

    def run_full_eval(self) -> dict:
        """Run all eval conversations and aggregate metrics."""
        results: list[dict] = []
        for conv in EVAL_CONVERSATIONS:
            result = self.evaluate_conversation(conv)
            results.append(result)
            status = "✓" if result["recall_at_3"] else "✗"
            logger.info(
                "%s %s — R@1=%s R@3=%s R@5=%s MRR=%.3f stored=%d/%d hopfield=%s pipeline=%s nodes=%d",
                status,
                result["topic"],
                result["recall_at_1"],
                result["recall_at_3"],
                result["recall_at_5"],
                result["mrr"],
                result["stored_count"],
                result["total_turns"],
                result.get("hopfield_hit", "N/A"),
                result.get("pipeline_hit", "N/A"),
                result.get("graph_nodes_activated", 0),
            )

        # Aggregate
        n = len(results)
        agg = {
            "num_conversations": n,
            "recall_at_1": sum(r["recall_at_1"] for r in results) / n,
            "recall_at_3": sum(r["recall_at_3"] for r in results) / n,
            "recall_at_5": sum(r["recall_at_5"] for r in results) / n,
            "mean_mrr": sum(r["mrr"] for r in results) / n,
            "mean_storage_rate": sum(r.get("storage_rate", 0) for r in results) / n,
            "mean_salience": sum(r.get("avg_salience", 0) for r in results) / n,
            "hopfield_accuracy": sum(1 for r in results if r.get("hopfield_hit")) / n,
            "pipeline_accuracy": sum(1 for r in results if r.get("pipeline_hit")) / n,
            "mean_graph_nodes": sum(r.get("graph_nodes_activated", 0) for r in results) / n,
            "per_conversation": results,
        }

        logger.info("")
        logger.info("=" * 70)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 70)
        logger.info("Conversations:       %d", agg["num_conversations"])
        logger.info("Recall@1:            %.1f%%", agg["recall_at_1"] * 100)
        logger.info("Recall@3:            %.1f%%", agg["recall_at_3"] * 100)
        logger.info("Recall@5:            %.1f%%", agg["recall_at_5"] * 100)
        logger.info("Mean MRR:            %.3f", agg["mean_mrr"])
        logger.info("Mean storage rate:   %.1f%%", agg["mean_storage_rate"] * 100)
        logger.info("Mean salience:       %.3f", agg["mean_salience"])
        logger.info("Hopfield accuracy:   %.1f%%", agg["hopfield_accuracy"] * 100)
        logger.info("Pipeline accuracy:   %.1f%%", agg["pipeline_accuracy"] * 100)
        logger.info("Mean graph nodes:    %.1f", agg["mean_graph_nodes"])

        return agg


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate memory recall accuracy")
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained checkpoint directory",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Path to write JSON results",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 70)
    logger.info("MEMORY RECALL EVALUATION")
    logger.info("=" * 70)

    evaluator = RecallEvaluator(checkpoint_path=args.checkpoint)
    results = evaluator.run_full_eval()

    if args.output:
        # Ensure booleans are JSON-serializable
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
