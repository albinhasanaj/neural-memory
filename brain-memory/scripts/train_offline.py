"""
Offline Training Pipeline — train neural memory modules from conversation datasets.

This script simulates conversations through the NeuralMemoryObserver,
accumulating training signals into replay buffers and running gradient
updates. After training, checkpoints are saved for later use.

Usage::

    # Quick test (50 conversations, ~2 min)
    python -m scripts.train_offline --dataset ultrachat --max-conversations 50

    # Full training run
    python -m scripts.train_offline --dataset ultrachat --max-conversations 5000 --epochs 3

    # From local data
    python -m scripts.train_offline --dataset local --local-path data/my_convos.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch

# ── Ensure project root is on path ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _enable_neural_modules() -> None:
    """Set env vars to enable all neural modules before settings are loaded."""
    flags = [
        "BRAIN_USE_PATTERN_SEPARATION",
        "BRAIN_USE_DOPAMINERGIC_GATE",
        "BRAIN_USE_HOPFIELD_MEMORY",
        "BRAIN_USE_VAE_CONSOLIDATION",
        "BRAIN_USE_TRANSFORMER_WM",
        "BRAIN_USE_LEARNED_FORGETTING",
        "BRAIN_USE_GNN_ACTIVATION",
    ]
    for flag in flags:
        os.environ.setdefault(flag, "true")

    # Disable LLM-dependent consolidation during offline training
    # (no API key available, and it causes timeout spam)
    os.environ.setdefault("BRAIN_CONSOLIDATION_INTERVAL", "999999")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s │ %(name)-28s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    # Quiet noisy libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train neural memory modules from conversation datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["ultrachat", "oasst2", "local"],
        default="ultrachat",
        help="Dataset to use (default: ultrachat)",
    )
    parser.add_argument(
        "--max-conversations", "-n",
        type=int,
        default=500,
        help="Max conversations to load (default: 500)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of passes over the data (default: 1)",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=4,
        help="Min turns per conversation (default: 4)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/neural",
        help="Directory to save checkpoints (default: checkpoints/neural)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N conversations (default: 100)",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Path to local JSON/JSONL file (for --dataset local)",
    )
    parser.add_argument(
        "--train-every",
        type=int,
        default=1,
        help="Run training step every N turns (default: 1 = every turn)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging (DEBUG level)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint directory",
    )
    return parser.parse_args()


def load_conversations(args: argparse.Namespace) -> list:
    """Load conversation data based on args."""
    from data.conversation_loader import load_local_json, load_oasst2, load_ultrachat

    if args.dataset == "ultrachat":
        return load_ultrachat(
            max_conversations=args.max_conversations,
            min_turns=args.min_turns,
            seed=args.seed,
        )
    elif args.dataset == "oasst2":
        return load_oasst2(
            max_conversations=args.max_conversations,
            min_turns=args.min_turns,
            seed=args.seed,
        )
    elif args.dataset == "local":
        if not args.local_path:
            raise ValueError("--local-path is required for --dataset local")
        return load_local_json(
            args.local_path,
            max_conversations=args.max_conversations,
            min_turns=args.min_turns,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def main() -> None:
    # Parse args before enabling flags (so we can control behavior)
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("train_offline")

    # Enable neural modules BEFORE importing settings
    _enable_neural_modules()

    from config.settings import settings
    from data.conversation_loader import Conversation, stream_conversations

    settings.ensure_data_dirs()

    logger.info("=" * 70)
    logger.info("OFFLINE NEURAL MEMORY TRAINING")
    logger.info("=" * 70)
    logger.info("Device: %s", settings.resolved_device)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Max conversations: %d", args.max_conversations)
    logger.info("Epochs: %d", args.epochs)
    logger.info("Checkpoint dir: %s", args.checkpoint_dir)
    logger.info("")

    # Show which modules are enabled
    neural_flags = {
        "pattern_separation": settings.use_pattern_separation,
        "dopaminergic_gate": settings.use_dopaminergic_gate,
        "hopfield_memory": settings.use_hopfield_memory,
        "vae_consolidation": settings.use_vae_consolidation,
        "transformer_wm": settings.use_transformer_wm,
        "learned_forgetting": settings.use_learned_forgetting,
        "gnn_activation": settings.use_gnn_activation,
    }
    enabled = [k for k, v in neural_flags.items() if v]
    disabled = [k for k, v in neural_flags.items() if not v]
    logger.info("Enabled modules:  %s", ", ".join(enabled) or "(none)")
    logger.info("Disabled modules: %s", ", ".join(disabled) or "(none)")
    logger.info("")

    # ── Load data ───────────────────────────────────────────────────
    logger.info("Loading conversation data...")
    t0 = time.time()
    conversations = load_conversations(args)
    logger.info("Loaded %d conversations in %.1fs", len(conversations), time.time() - t0)

    if not conversations:
        logger.error("No conversations loaded! Check dataset and filters.")
        sys.exit(1)

    # ── Create observer ─────────────────────────────────────────────
    logger.info("Initializing NeuralMemoryObserver...")
    from memory.observer import NeuralMemoryObserver

    observer = NeuralMemoryObserver()

    # Resume from checkpoint if requested
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info("Resuming from checkpoint: %s", resume_path)
            observer._trainer.load_checkpoint(resume_path)
        else:
            logger.warning("Checkpoint path not found: %s", resume_path)

    # ── Training loop ───────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    total_turns = 0
    total_conversations = 0
    total_stored = 0
    train_step_count = 0
    loss_accumulator: dict[str, list[float]] = {}

    start_time = time.time()

    logger.info("")
    logger.info("Starting training loop (%d epochs)...", args.epochs)
    logger.info("-" * 70)

    for conv in stream_conversations(conversations, repeat=args.epochs, seed=args.seed):
        total_conversations += 1

        # Reset working memory between conversations
        observer.working_memory.clear()

        for turn in conv.turns:
            total_turns += 1

            # Process the turn through the full neural pipeline
            try:
                diag = observer.observe(turn.content, speaker=turn.role)
            except Exception as e:
                logger.warning("Error processing turn: %s", e)
                continue

            if diag.get("stored", False):
                total_stored += 1

            # Collect loss info from training step
            train_diag = diag.get("neural_training", {})
            for comp_name, comp_diag in train_diag.items():
                if isinstance(comp_diag, dict) and "total" in comp_diag:
                    loss_accumulator.setdefault(comp_name, []).append(comp_diag["total"])
                elif isinstance(comp_diag, dict) and "loss" in comp_diag:
                    loss_accumulator.setdefault(comp_name, []).append(comp_diag["loss"])

            train_step_count += 1

        # ── Progress logging ────────────────────────────────────────
        if total_conversations % 10 == 0:
            elapsed = time.time() - start_time
            turns_per_sec = total_turns / max(elapsed, 0.01)

            # Compute recent average losses
            recent_losses = {}
            for comp_name, losses in loss_accumulator.items():
                if losses:
                    recent = losses[-50:]  # last 50 values
                    recent_losses[comp_name] = sum(recent) / len(recent)

            loss_str = ", ".join(
                f"{k}={v:.4f}" for k, v in recent_losses.items()
            ) or "no losses yet"

            logger.info(
                "Conv %d/%d | turns=%d | stored=%d | %.1f turns/s | %s",
                total_conversations,
                len(conversations) * args.epochs,
                total_turns,
                total_stored,
                turns_per_sec,
                loss_str,
            )

        # ── Periodic checkpoint ─────────────────────────────────────
        if args.checkpoint_every > 0 and total_conversations % args.checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"step_{total_conversations}"
            observer._trainer.save_checkpoint(ckpt_path)
            logger.info("Saved checkpoint to %s", ckpt_path)

    # ── Final checkpoint ────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info("Conversations processed: %d", total_conversations)
    logger.info("Total turns processed:   %d", total_turns)
    logger.info("Episodes stored:         %d", total_stored)
    logger.info("Training steps:          %d", train_step_count)
    logger.info("Wall time:               %.1fs (%.1f turns/s)", elapsed, total_turns / max(elapsed, 0.01))

    # Final losses
    logger.info("")
    logger.info("Final losses:")
    for comp_name, losses in loss_accumulator.items():
        if losses:
            logger.info(
                "  %-20s first=%.4f  last=%.4f  mean=%.4f  (n=%d)",
                comp_name,
                losses[0],
                losses[-1],
                sum(losses) / len(losses),
                len(losses),
            )

    # Save final checkpoint
    final_path = checkpoint_dir / "final"
    observer._trainer.save_checkpoint(final_path)
    logger.info("")
    logger.info("Final checkpoint saved to: %s", final_path)

    # Print summary
    summary = observer._trainer.summary()
    logger.info("")
    logger.info("Component summary:")
    for name, info in summary.items():
        if isinstance(info, dict) and info.get("enabled"):
            logger.info(
                "  %-20s steps=%d  params=%s  last_loss=%s",
                name,
                info.get("total_steps", 0),
                f"{info.get('param_count', 0):,}",
                f"{info['last_loss']:.4f}" if info.get("last_loss") is not None else "N/A",
            )


if __name__ == "__main__":
    main()
