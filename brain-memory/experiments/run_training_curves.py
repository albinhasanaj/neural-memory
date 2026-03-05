"""
Experiment: Training Curves — monitor loss convergence for all neural components.

Trains each enabled neural component for N steps on synthetic data and
plots/records loss curves for analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from config.settings import settings

_device = settings.resolved_device
N_STEPS = 100
BATCH_SIZE = 16


def train_vae_curves() -> list[float]:
    """Train VAE and record loss curve."""
    from memory.neural_consolidation import (
        ConsolidationReplayBuffer,
        ConsolidationVAE,
        train_vae_step,
    )

    vae = ConsolidationVAE()
    opt = torch.optim.Adam(vae.parameters(), lr=settings.consolidation_vae_lr)
    buf = ConsolidationReplayBuffer(capacity=500)

    # Fill buffer with synthetic data
    for _ in range(200):
        buf.push(
            torch.randn(settings.embedding_dim, device=_device) * 0.3,
            torch.rand(settings.vae_metadata_dim, device=_device),
        )

    losses = []
    for step in range(N_STEPS):
        diag = train_vae_step(vae, opt, buf, batch_size=BATCH_SIZE)
        if diag:
            losses.append(diag["total"])
    return losses


def train_gate_curves() -> list[float]:
    """Train Gate and record loss curve."""
    from memory.gate_network import DopaminergicGate, GateReplayBuffer, train_gate_step

    gate = DopaminergicGate()
    opt = torch.optim.Adam(gate.parameters(), lr=settings.gate_learning_rate)
    buf = GateReplayBuffer(capacity=500)

    for _ in range(200):
        buf.push(
            torch.randn(settings.embedding_dim, device=_device),
            torch.randn(settings.gru_hidden_dim, device=_device),
            torch.rand(4, device=_device),
            reward=float(torch.rand(1).item() > 0.5),
        )

    losses = []
    for step in range(N_STEPS):
        diag = train_gate_step(gate, opt, buf, batch_size=BATCH_SIZE)
        if diag:
            losses.append(diag["gate_loss"])
    return losses


def train_separator_curves() -> list[float]:
    """Train Pattern Separator and record loss curve."""
    from memory.pattern_separation import (
        PatternSeparator,
        SeparationReplayBuffer,
        train_separator_step,
    )

    sep = PatternSeparator()
    opt = torch.optim.Adam(sep.parameters(), lr=settings.pattern_sep_lr)
    buf = SeparationReplayBuffer(capacity=500)

    for _ in range(200):
        buf.push(torch.randn(settings.embedding_dim, device=_device) * 0.3)

    losses = []
    for step in range(N_STEPS):
        diag = train_separator_step(sep, opt, buf, batch_size=BATCH_SIZE)
        if diag:
            losses.append(diag["total"])
    return losses


def train_forgetting_curves() -> list[float]:
    """Train Forgetting Network and record loss curve."""
    from memory.forgetting import (
        ForgettingNetwork,
        ForgettingReplayBuffer,
        train_forgetting_step,
    )

    net = ForgettingNetwork()
    opt = torch.optim.Adam(net.parameters(), lr=settings.forgetting_lr)
    buf = ForgettingReplayBuffer(capacity=500)

    for _ in range(200):
        buf.push(
            torch.randn(settings.embedding_dim, device=_device),
            torch.rand(5, device=_device),
            delta_t=float(torch.rand(1).item() * 24.0),
            target_decay=0.1 + float(torch.rand(1).item() * 0.3),
            target_interference=float(torch.rand(1).item() * 0.2),
        )

    losses = []
    for step in range(N_STEPS):
        diag = train_forgetting_step(net, opt, buf, batch_size=BATCH_SIZE)
        if diag:
            losses.append(diag["total"])
    return losses


def main() -> None:
    print("=" * 60)
    print(" Training Curves — All Neural Components")
    print("=" * 60)

    results = {}

    print("\nTraining VAE...")
    results["vae"] = train_vae_curves()
    print(f"  Final loss: {results['vae'][-1]:.4f}" if results["vae"] else "  No data")

    print("Training Gate...")
    results["gate"] = train_gate_curves()
    print(f"  Final loss: {results['gate'][-1]:.4f}" if results["gate"] else "  No data")

    print("Training Pattern Separator...")
    results["pattern_sep"] = train_separator_curves()
    print(f"  Final loss: {results['pattern_sep'][-1]:.4f}" if results["pattern_sep"] else "  No data")

    print("Training Forgetting Network...")
    results["forgetting"] = train_forgetting_curves()
    print(f"  Final loss: {results['forgetting'][-1]:.4f}" if results["forgetting"] else "  No data")

    # Summary
    print("\n" + "-" * 40)
    for name, losses in results.items():
        if losses:
            print(f"  {name}: {losses[0]:.4f} → {losses[-1]:.4f} "
                  f"(Δ={losses[0]-losses[-1]:.4f})")

    output_dir = Path("data/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_curves.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'training_curves.json'}")


if __name__ == "__main__":
    main()
