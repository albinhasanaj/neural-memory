"""
Offline training for Phase 3 — Fast Weight Memory slow parameters.

Trains the slow parameters (separator, query_proj, value_proj, output_proj,
log_beta) of FastWeightModule via cosine retrieval loss, plus the MemoryRouter
via REINFORCE.  Hebbian fast weights (W_key, W_value) are NOT in any optimizer;
they change only through Hebbian writes in observe().

Usage::

    python -m scripts.train_fast_weight \
        --dataset ultrachat \
        --max-conversations 1000 \
        --checkpoint-dir checkpoints/fast_weight_1k

    # Resume from a checkpoint
    python -m scripts.train_fast_weight \
        --dataset ultrachat --max-conversations 1000 \
        --checkpoint-dir checkpoints/fast_weight_1k \
        --resume checkpoints/fast_weight_1k/conv_400
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


def _enable_all_modules() -> None:
    """Set env vars to enable all neural modules + fast weight memory."""
    os.environ["BRAIN_USE_PATTERN_SEPARATION"] = "true"
    os.environ["BRAIN_USE_DOPAMINERGIC_GATE"] = "true"
    os.environ["BRAIN_USE_HOPFIELD_MEMORY"] = "true"
    os.environ["BRAIN_USE_MODULAR_HOPFIELD"] = "true"
    os.environ["BRAIN_USE_FAST_WEIGHT_MEMORY"] = "true"
    os.environ["BRAIN_USE_TRANSFORMER_WM"] = "true"
    os.environ["BRAIN_USE_LEARNED_FORGETTING"] = "true"
    os.environ["BRAIN_USE_VAE_CONSOLIDATION"] = "true"
    os.environ["BRAIN_CONSOLIDATION_INTERVAL"] = "999999"


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s │ %(name)-28s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Phase 3 fast-weight slow parameters from conversation datasets.",
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
        default=1000,
        help="Max conversations to train on (default: 1000)",
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
        default="checkpoints/fast_weight_1k",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="Save checkpoint every N conversations (default: 200)",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Path to local JSON/JSONL file (for --dataset local)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
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
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("train_fast_weight")

    # Enable all modules BEFORE importing settings
    _enable_all_modules()

    from config.settings import settings
    from data.conversation_loader import stream_conversations

    settings.ensure_data_dirs()

    logger.info("=" * 70)
    logger.info("PHASE 3 — FAST WEIGHT OFFLINE TRAINING")
    logger.info("=" * 70)
    logger.info("Device: %s", settings.resolved_device)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Max conversations: %d", args.max_conversations)
    logger.info("Checkpoint dir: %s", args.checkpoint_dir)
    logger.info("")

    # Show module flags
    logger.info("Module flags:")
    logger.info("  use_hopfield_memory:    %s", settings.use_hopfield_memory)
    logger.info("  use_modular_hopfield:   %s", settings.use_modular_hopfield)
    logger.info("  use_fast_weight_memory: %s", settings.use_fast_weight_memory)
    logger.info("  use_pattern_separation: %s", settings.use_pattern_separation)
    logger.info("  use_dopaminergic_gate:  %s", settings.use_dopaminergic_gate)
    logger.info("  use_transformer_wm:     %s", settings.use_transformer_wm)
    logger.info("  use_learned_forgetting: %s", settings.use_learned_forgetting)
    logger.info("  use_vae_consolidation:  %s", settings.use_vae_consolidation)
    logger.info("")

    if not settings.use_fast_weight_memory:
        logger.error("BRAIN_USE_FAST_WEIGHT_MEMORY is not enabled! Aborting.")
        sys.exit(1)

    # ── Load data ───────────────────────────────────────────────────
    logger.info("Loading conversation data...")
    t0 = time.time()
    conversations = load_conversations(args)
    logger.info("Loaded %d conversations in %.1fs", len(conversations), time.time() - t0)

    if not conversations:
        logger.error("No conversations loaded!")
        sys.exit(1)

    # ── Create observer ─────────────────────────────────────────────
    logger.info("Initializing NeuralMemoryObserver with fast-weight memory...")
    from memory.observer import NeuralMemoryObserver

    observer = NeuralMemoryObserver()

    # Verify fast-weight model is active
    from memory.hopfield_memory import ModularFastWeightMemory
    if not isinstance(observer._hopfield, ModularFastWeightMemory):
        logger.error("Expected ModularFastWeightMemory but got %s", type(observer._hopfield))
        sys.exit(1)

    logger.info("Fast-weight model: %d modules, hidden_dim=%d",
                observer._hopfield.num_modules,
                observer._hopfield.modules_list[0].hidden_dim)

    # Disable the `hopfield` training component — train_hopfield_step accesses
    # .patterns which doesn't exist on ModularFastWeightMemory. The `fast_weight`
    # component handles training the slow params we care about.
    hopfield_comp = observer._trainer.components.get("hopfield")
    if hopfield_comp:
        hopfield_comp.enabled = False

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
    prev_embedding: torch.Tensor | None = None

    start_time = time.time()

    logger.info("")
    logger.info("Starting training loop...")
    logger.info("-" * 70)

    for conv in stream_conversations(conversations, repeat=1, seed=args.seed):
        total_conversations += 1
        prev_embedding = None

        # Reset working memory between conversations
        # (but NOT the fast weights or decode index — they accumulate)
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

            # Push fast-weight training signal:
            # (query=current_embedding, target=previous_embedding)
            current_embedding = observer.encoder.encode(turn.content)
            if prev_embedding is not None:
                observer._trainer.push_fast_weight_experience(
                    current_embedding, prev_embedding,
                )
            prev_embedding = current_embedding

            # Collect loss info from training step
            train_diag = diag.get("neural_training", {})
            for comp_name, comp_diag in train_diag.items():
                if isinstance(comp_diag, dict):
                    loss_val = comp_diag.get("total", comp_diag.get("loss"))
                    if loss_val is not None:
                        loss_accumulator.setdefault(comp_name, []).append(loss_val)

            train_step_count += 1

        # ── Progress logging ────────────────────────────────────────
        if total_conversations % 50 == 0:
            elapsed = time.time() - start_time
            turns_per_sec = total_turns / max(elapsed, 0.01)

            # Compute recent average losses
            recent_losses = {}
            for comp_name, losses in loss_accumulator.items():
                if losses:
                    recent = losses[-100:]
                    recent_losses[comp_name] = sum(recent) / len(recent)

            loss_str = " ".join(
                f"{k}={v:.3f}" for k, v in sorted(recent_losses.items())
            ) or "no losses yet"

            logger.info(
                "[Conv %d/%d] %s | turns=%d stored=%d %.1f t/s",
                total_conversations,
                len(conversations),
                loss_str,
                total_turns,
                total_stored,
                turns_per_sec,
            )

        # ── Periodic checkpoint ─────────────────────────────────────
        if args.checkpoint_every > 0 and total_conversations % args.checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"conv_{total_conversations}"
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
    for comp_name, losses in sorted(loss_accumulator.items()):
        if losses:
            logger.info(
                "  %-20s first=%.4f  last=%.4f  mean=%.4f  (n=%d)",
                comp_name,
                losses[0],
                losses[-1],
                sum(losses) / len(losses),
                len(losses),
            )

    # Fast-weight specific stats
    fw_model = observer._hopfield
    logger.info("")
    logger.info("Fast-weight stats:")
    logger.info("  Total writes:     %d", fw_model.total_writes())
    for i, occ in enumerate(fw_model.module_occupancies()):
        if occ > 0:
            logger.info("  Module %2d:  occupancy=%.3f", i, occ)

    # Save final checkpoint
    final_path = checkpoint_dir / "final"
    observer._trainer.save_checkpoint(final_path)
    logger.info("")
    logger.info("Final checkpoint saved to: %s", final_path)

    # Summary
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
