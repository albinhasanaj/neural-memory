"""
Benchmark all three memory tiers and produce a comparison report.

Tier 1 — Legacy Hopfield (Phase 1)
Tier 2 — Modular Hopfield (Phase 2)
Tier 3 — Fast Weight Memory (Phase 3)

Usage::

    # Untrained Tier 3
    python -m scripts.benchmark_tiers --output benchmark_results_untrained.json

    # With trained Tier 3 checkpoint
    python -m scripts.benchmark_tiers \
        --tier3-checkpoint checkpoints/fast_weight_1k/final \
        --output benchmark_results_trained.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ── Ensure project root is on path ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-28s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_tiers")

# ────────────────────────────────────────────────────────────────────
# Evaluation dataset (same as eval_recall.py)
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
# Tier definitions
# ────────────────────────────────────────────────────────────────────

TIER_CONFIGS: list[dict] = [
    {
        "name": "Tier 1 (Legacy)",
        "env": {
            "BRAIN_USE_HOPFIELD_MEMORY": "true",
            "BRAIN_USE_MODULAR_HOPFIELD": "false",
            "BRAIN_USE_FAST_WEIGHT_MEMORY": "false",
        },
        "description": "Legacy Hopfield (Phase 1)",
        "memory_type": "Explicit patterns",
        "capacity_label": "2,048",
        "stores_data": "Yes (buffer rows)",
    },
    {
        "name": "Tier 2 (Modular)",
        "env": {
            "BRAIN_USE_HOPFIELD_MEMORY": "true",
            "BRAIN_USE_MODULAR_HOPFIELD": "true",
            "BRAIN_USE_FAST_WEIGHT_MEMORY": "false",
        },
        "description": "Modular Hopfield (Phase 2)",
        "memory_type": "Explicit patterns",
        "capacity_label": "8,192",
        "stores_data": "Yes (buffer rows)",
    },
    {
        "name": "Tier 3 (FastWeight)",
        "env": {
            "BRAIN_USE_HOPFIELD_MEMORY": "true",
            "BRAIN_USE_MODULAR_HOPFIELD": "true",
            "BRAIN_USE_FAST_WEIGHT_MEMORY": "true",
        },
        "description": "Fast Weight Memory (Phase 3)",
        "memory_type": "Weight matrices",
        "capacity_label": "Theoretically ∞",
        "stores_data": "No (weights only)",
    },
]


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

# Keys that must be cleaned between tiers to avoid leaking
_TIER_ENV_KEYS = [
    "BRAIN_USE_HOPFIELD_MEMORY",
    "BRAIN_USE_MODULAR_HOPFIELD",
    "BRAIN_USE_FAST_WEIGHT_MEMORY",
]

_NEURAL_FLAGS = [
    "BRAIN_USE_PATTERN_SEPARATION",
    "BRAIN_USE_DOPAMINERGIC_GATE",
    "BRAIN_USE_HOPFIELD_MEMORY",
    "BRAIN_USE_VAE_CONSOLIDATION",
    "BRAIN_USE_TRANSFORMER_WM",
    "BRAIN_USE_LEARNED_FORGETTING",
    "BRAIN_CONSOLIDATION_INTERVAL",
]


def _set_env(tier_env: dict[str, str]) -> None:
    """Set base neural flags + tier-specific overrides."""
    # Base neural flags
    os.environ["BRAIN_USE_PATTERN_SEPARATION"] = "true"
    os.environ["BRAIN_USE_DOPAMINERGIC_GATE"] = "true"
    os.environ["BRAIN_USE_VAE_CONSOLIDATION"] = "true"
    os.environ["BRAIN_USE_TRANSFORMER_WM"] = "true"
    os.environ["BRAIN_USE_LEARNED_FORGETTING"] = "true"
    os.environ["BRAIN_CONSOLIDATION_INTERVAL"] = "999999"
    # Tier overrides
    for key, val in tier_env.items():
        os.environ[key] = val


def _clear_env() -> None:
    """Remove all tier-relevant env vars so they don't leak."""
    for key in _TIER_ENV_KEYS + _NEURAL_FLAGS:
        os.environ.pop(key, None)


def _reload_settings():
    """Force-reload the settings singleton so it picks up new env vars."""
    # Ensure the module is in sys.modules before reload
    if "config.settings" not in sys.modules:
        import config.settings  # noqa: F811
    settings_mod = sys.modules["config.settings"]
    importlib.reload(settings_mod)
    return settings_mod.settings


def _has_keyword(text: str, expected_keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in expected_keywords)


# ────────────────────────────────────────────────────────────────────
# Single-tier evaluation
# ────────────────────────────────────────────────────────────────────


def evaluate_tier(
    tier_cfg: dict,
    checkpoint_path: str | None = None,
) -> dict:
    """Run all 8 eval conversations for one tier config and return metrics."""
    tier_name = tier_cfg["name"]
    logger.info("=" * 70)
    logger.info("EVALUATING: %s", tier_name)
    logger.info("=" * 70)

    # 1. Set env vars and reload settings
    _clear_env()
    _set_env(tier_cfg["env"])
    settings = _reload_settings()

    logger.info(
        "  hopfield=%s  modular=%s  fast_weight=%s",
        settings.use_hopfield_memory,
        settings.use_modular_hopfield,
        settings.use_fast_weight_memory,
    )

    # 2. Reimport observer with fresh settings
    # Reload all affected modules so they see the new settings singleton
    import config.settings
    import memory.observer as obs_mod
    importlib.reload(obs_mod)
    from memory.observer import NeuralMemoryObserver

    observer = NeuralMemoryObserver()

    # 3. Load checkpoint if available
    if checkpoint_path and Path(checkpoint_path).exists():
        observer._trainer.load_checkpoint(Path(checkpoint_path))
        logger.info("  Loaded checkpoint from %s", checkpoint_path)
    else:
        if checkpoint_path:
            logger.warning("  Checkpoint not found: %s — running untrained", checkpoint_path)
        else:
            logger.info("  No checkpoint — running untrained")

    # Disable online training during eval — we only want inference.
    # This avoids known issues with train_hopfield_step accessing .patterns
    # on ModularHippocampalMemory/ModularFastWeightMemory.
    for comp in observer._trainer.components.values():
        comp.enabled = False

    # 4. Run each eval conversation
    per_conv_results: list[dict] = []
    total_retrieval_latency = 0.0
    total_retrieval_calls = 0

    for conv in EVAL_CONVERSATIONS:
        result = _evaluate_single_conversation(observer, conv)
        total_retrieval_latency += result.get("retrieval_latency_ms", 0.0)
        total_retrieval_calls += 1
        per_conv_results.append(result)

        status = "✓" if result["recall_at_3"] else "✗"
        logger.info(
            "  %s %-22s R@1=%s R@3=%s R@5=%s MRR=%.3f hopfield=%s pipeline=%s lat=%.1fms",
            status,
            result["topic"],
            result["recall_at_1"],
            result["recall_at_3"],
            result["recall_at_5"],
            result["mrr"],
            result.get("hopfield_hit", "N/A"),
            result.get("pipeline_hit", "N/A"),
            result.get("retrieval_latency_ms", 0.0),
        )

    # 5. Aggregate
    n = len(per_conv_results)
    agg = {
        "tier": tier_name,
        "description": tier_cfg["description"],
        "memory_type": tier_cfg["memory_type"],
        "capacity_label": tier_cfg["capacity_label"],
        "stores_data": tier_cfg["stores_data"],
        "num_conversations": n,
        "recall_at_1": sum(r["recall_at_1"] for r in per_conv_results) / n,
        "recall_at_3": sum(r["recall_at_3"] for r in per_conv_results) / n,
        "recall_at_5": sum(r["recall_at_5"] for r in per_conv_results) / n,
        "mean_mrr": sum(r["mrr"] for r in per_conv_results) / n,
        "mean_storage_rate": sum(r.get("storage_rate", 0) for r in per_conv_results) / n,
        "hopfield_accuracy": sum(1 for r in per_conv_results if r.get("hopfield_hit")) / n,
        "pipeline_accuracy": sum(1 for r in per_conv_results if r.get("pipeline_hit")) / n,
        "mean_retrieval_latency_ms": (
            total_retrieval_latency / total_retrieval_calls
            if total_retrieval_calls > 0 else 0.0
        ),
        "per_conversation": per_conv_results,
    }

    logger.info("")
    logger.info("  Recall@1: %.1f%%  Recall@3: %.1f%%  Recall@5: %.1f%%  MRR: %.3f",
                agg["recall_at_1"] * 100, agg["recall_at_3"] * 100,
                agg["recall_at_5"] * 100, agg["mean_mrr"])
    logger.info("  Storage rate: %.1f%%  Retrieval latency: %.1fms",
                agg["mean_storage_rate"] * 100, agg["mean_retrieval_latency_ms"])
    logger.info("  Hopfield accuracy: %.1f%%  Pipeline accuracy: %.1f%%",
                agg["hopfield_accuracy"] * 100, agg["pipeline_accuracy"] * 100)
    return agg


def _evaluate_single_conversation(observer, conv: dict) -> dict:
    """Run one eval conversation on the given observer, resetting state first."""
    import asyncio
    from memory.consolidation import consolidate

    # Reset all state
    observer.working_memory.clear()
    observer.episodic_store._entries.clear()
    if observer._hopfield is not None:
        observer._hopfield.clear()
    observer.graph.reset_activations()

    # Feed setup turns
    stored_turns: list[dict] = []
    for turn in conv["setup_turns"]:
        info = observer.observe(turn["text"], speaker=turn["role"])
        stored_turns.append({
            "text": turn["text"],
            "role": turn["role"],
            "stored": info["stored"],
            "salience": info["salience"],
        })

    # Run synchronous consolidation
    vae_model = None
    if hasattr(observer, "_trainer") and observer._trainer is not None:
        vae_comp = observer._trainer.components.get("vae")
        if vae_comp and vae_comp.enabled and vae_comp.model is not None:
            vae_model = vae_comp.model
    try:
        new_edges = asyncio.run(consolidate(
            observer.episodic_store, observer.graph, vae=vae_model,
        ))
    except Exception:
        pass

    # Encode query
    query_embedding = observer.encoder.encode(conv["query"])

    # Rank episodes by cosine similarity
    all_episodes = observer.episodic_store.get_all_active()
    expected_kws = conv["expected_keywords"]

    if not all_episodes:
        return {
            "id": conv["id"], "topic": conv["expected_topic"],
            "stored_count": sum(1 for t in stored_turns if t["stored"]),
            "total_turns": len(stored_turns),
            "storage_rate": sum(1 for t in stored_turns if t["stored"]) / max(len(stored_turns), 1),
            "recall_at_1": False, "recall_at_3": False, "recall_at_5": False,
            "mrr": 0.0, "hopfield_hit": False, "pipeline_hit": False,
            "retrieval_latency_ms": 0.0, "retrieved_texts": [],
        }

    query_cpu = query_embedding.detach().cpu()
    ranked = []
    for ep in all_episodes:
        ep_emb = torch.tensor(ep.embedding, dtype=torch.float32)
        sim = F.cosine_similarity(query_cpu.unsqueeze(0), ep_emb.unsqueeze(0)).item()
        ranked.append((ep, sim))
    ranked.sort(key=lambda x: x[1], reverse=True)

    recall_1 = _has_keyword(ranked[0][0].raw_text, expected_kws) if ranked else False
    recall_3 = any(_has_keyword(ep.raw_text, expected_kws) for ep, _ in ranked[:3])
    recall_5 = any(_has_keyword(ep.raw_text, expected_kws) for ep, _ in ranked[:5])

    mrr = 0.0
    for i, (ep, _) in enumerate(ranked):
        if _has_keyword(ep.raw_text, expected_kws):
            mrr = 1.0 / (i + 1)
            break

    # Hopfield/fast-weight retrieve_decoded — measure latency
    hopfield_hit = False
    retrieval_latency_ms = 0.0
    if observer._hopfield is not None and observer._hopfield.num_patterns > 0:
        t0 = time.perf_counter()
        with torch.no_grad():
            h_results = observer._hopfield.retrieve_decoded(query_embedding, top_k=5)
        retrieval_latency_ms = (time.perf_counter() - t0) * 1000.0

        for r in h_results:
            if _has_keyword(r.get("text", ""), expected_kws):
                hopfield_hit = True
                break

    # Full pipeline test (may fail for modular hopfield + forgetting due to
    # known .values attribute gap — degrade gracefully)
    pipeline_hit = False
    try:
        observer.observe(conv["query"], speaker="user")
        dummy_messages = [{"role": "user", "content": conv["query"]}]
        injected_messages, activated_nodes = observer.activate_and_inject(dummy_messages)
        injected_text = " ".join(m.get("content", "") for m in injected_messages)
        pipeline_hit = _has_keyword(injected_text, expected_kws)
    except (AttributeError, RuntimeError, KeyError) as exc:
        logger.debug("Pipeline test failed for %s: %s", conv["id"], exc)

    return {
        "id": conv["id"],
        "topic": conv["expected_topic"],
        "stored_count": sum(1 for t in stored_turns if t["stored"]),
        "total_turns": len(stored_turns),
        "storage_rate": sum(1 for t in stored_turns if t["stored"]) / max(len(stored_turns), 1),
        "recall_at_1": recall_1,
        "recall_at_3": recall_3,
        "recall_at_5": recall_5,
        "mrr": mrr,
        "hopfield_hit": hopfield_hit,
        "pipeline_hit": pipeline_hit,
        "retrieval_latency_ms": retrieval_latency_ms,
        "retrieved_texts": [ep.raw_text[:100] for ep, _ in ranked[:5]],
    }


# ────────────────────────────────────────────────────────────────────
# Module specialization analysis (Tier 3)
# ────────────────────────────────────────────────────────────────────


def run_specialization_analysis(observer) -> dict | None:
    """Run module specialization analysis for fast-weight (Tier 3) models."""
    from memory.hopfield_memory import ModularFastWeightMemory

    if not isinstance(observer._hopfield, ModularFastWeightMemory):
        return None

    model = observer._hopfield

    # Feed all eval conversations through the model to build up state
    observer.working_memory.clear()
    observer.episodic_store._entries.clear()
    model.clear()
    observer.graph.reset_activations()

    for conv in EVAL_CONVERSATIONS:
        for turn in conv["setup_turns"]:
            observer.observe(turn["text"], speaker=turn["role"])

    # Get module summary
    summaries = model.module_summary()

    # Calculate weight matrix health
    w_key_norms = [s["w_key_norm"] for s in summaries]
    w_value_norms = [s["w_value_norm"] for s in summaries]

    report = {
        "module_summaries": summaries,
        "weight_health": {
            "w_key_norm_min": min(w_key_norms) if w_key_norms else 0.0,
            "w_key_norm_max": max(w_key_norms) if w_key_norms else 0.0,
            "w_key_norm_mean": sum(w_key_norms) / len(w_key_norms) if w_key_norms else 0.0,
            "w_value_norm_min": min(w_value_norms) if w_value_norms else 0.0,
            "w_value_norm_max": max(w_value_norms) if w_value_norms else 0.0,
            "w_value_norm_mean": sum(w_value_norms) / len(w_value_norms) if w_value_norms else 0.0,
        },
        "active_modules": sum(1 for s in summaries if s["write_count"] > 0),
        "total_modules": len(summaries),
    }

    # Print report
    logger.info("")
    logger.info("Module Specialization Report")
    logger.info("─" * 40)
    for s in summaries:
        if s["write_count"] > 0:
            ents = ", ".join(f"{e}({c})" for e, c in s["top_entities"][:3]) if s["top_entities"] else "none"
            logger.info(
                "  Module %2d:  occupancy=%.2f  w_key_norm=%.2f  top_entities: [%s]",
                s["module_index"], s["occupancy"], s["w_key_norm"], ents,
            )
    logger.info("")
    logger.info("Weight Matrix Health:")
    wh = report["weight_health"]
    logger.info("  w_key_norm:   min=%.3f  max=%.3f  mean=%.3f",
                wh["w_key_norm_min"], wh["w_key_norm_max"], wh["w_key_norm_mean"])
    logger.info("  w_value_norm: min=%.3f  max=%.3f  mean=%.3f",
                wh["w_value_norm_min"], wh["w_value_norm_max"], wh["w_value_norm_mean"])
    logger.info("Active modules: %d / %d", report["active_modules"], report["total_modules"])

    return report


# ────────────────────────────────────────────────────────────────────
# Comparison table
# ────────────────────────────────────────────────────────────────────


def print_comparison_table(all_results: list[dict]) -> None:
    """Print a formatted comparison table of all tier results."""
    # Column widths
    label_w = 22
    col_w = 22

    tiers = all_results
    tier_names = [r["tier"] for r in tiers]

    def sep_line():
        return "┌" + "─" * label_w + ("┬" + "─" * col_w) * len(tiers) + "┐"

    def mid_line():
        return "├" + "─" * label_w + ("┼" + "─" * col_w) * len(tiers) + "┤"

    def end_line():
        return "└" + "─" * label_w + ("┴" + "─" * col_w) * len(tiers) + "┘"

    def row(label: str, values: list[str]):
        cells = f"│{label:<{label_w}}"
        for v in values:
            cells += f"│{v:^{col_w}}"
        cells += "│"
        return cells

    print()
    print("=== BRAIN MEMORY BENCHMARK RESULTS ===")
    print()
    print(sep_line())
    print(row("Metric", tier_names))
    print(mid_line())

    # Metrics rows
    metrics = [
        ("Recall@1", lambda r: f"{r['recall_at_1'] * 100:.1f}%"),
        ("Recall@3", lambda r: f"{r['recall_at_3'] * 100:.1f}%"),
        ("Recall@5", lambda r: f"{r['recall_at_5'] * 100:.1f}%"),
        ("MRR", lambda r: f"{r['mean_mrr']:.3f}"),
        ("Storage Rate", lambda r: f"{r['mean_storage_rate'] * 100:.1f}%"),
        ("Retrieval Latency", lambda r: f"{r['mean_retrieval_latency_ms']:.1f}ms"),
        ("Hopfield Accuracy", lambda r: f"{r['hopfield_accuracy'] * 100:.1f}%"),
        ("Pipeline Hit", lambda r: f"{r['pipeline_accuracy'] * 100:.1f}%"),
        ("", lambda r: ""),
        ("Memory Type", lambda r: r.get("memory_type", "")),
        ("Total Capacity", lambda r: r.get("capacity_label", "")),
        ("Stores Data?", lambda r: r.get("stores_data", "")),
    ]

    for label, extractor in metrics:
        if label == "":
            print(mid_line())
            continue
        vals = [extractor(r) for r in tiers]
        print(row(label, vals))

    print(end_line())
    print()


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark all three memory tiers")
    p.add_argument(
        "--output", type=str, default="benchmark_results.json",
        help="Path to write JSON results (default: benchmark_results.json)",
    )
    # Default checkpoint is relative to workspace root (parent of brain-memory)
    _default_ckpt = str(PROJECT_ROOT.parent / "checkpoints" / "ultrachat_1k_trained" / "final")
    p.add_argument(
        "--tier12-checkpoint", type=str,
        default=_default_ckpt,
        help="Checkpoint path for Tier 1 and Tier 2",
    )
    p.add_argument(
        "--tier3-checkpoint", type=str, default=None,
        help="Checkpoint path for Tier 3 (default: None = untrained)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 70)
    logger.info("BRAIN MEMORY — THREE-TIER BENCHMARK")
    logger.info("=" * 70)
    logger.info("Tier 1/2 checkpoint: %s", args.tier12_checkpoint)
    logger.info("Tier 3 checkpoint:   %s", args.tier3_checkpoint or "(untrained)")
    logger.info("")

    all_results: list[dict] = []
    tier3_observer = None

    for i, tier_cfg in enumerate(TIER_CONFIGS):
        # Choose checkpoint
        if i < 2:
            ckpt = args.tier12_checkpoint
        else:
            ckpt = args.tier3_checkpoint

        result = evaluate_tier(tier_cfg, checkpoint_path=ckpt)
        all_results.append(result)

    # Print comparison table
    print_comparison_table(all_results)

    # Run specialization analysis for Tier 3
    logger.info("Running module specialization analysis for Tier 3...")
    _clear_env()
    _set_env(TIER_CONFIGS[2]["env"])
    _reload_settings()

    import memory.observer as obs_mod
    importlib.reload(obs_mod)
    from memory.observer import NeuralMemoryObserver as FreshObserver
    tier3_obs = FreshObserver()
    if args.tier3_checkpoint and Path(args.tier3_checkpoint).exists():
        tier3_obs._trainer.load_checkpoint(Path(args.tier3_checkpoint))

    # Disable online training during specialization analysis (same as eval)
    for comp in tier3_obs._trainer.components.values():
        comp.enabled = False

    spec_report = run_specialization_analysis(tier3_obs)
    if spec_report:
        all_results.append({"specialization_report": spec_report})

    # Clean up env
    _clear_env()

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Full results written to %s", output_path)


if __name__ == "__main__":
    main()
