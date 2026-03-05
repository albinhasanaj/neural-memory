# Brain Memory — Three-Tier Benchmark Report

## Overview

Three memory architecture tiers were benchmarked on 8 evaluation conversations, before and after training Phase 3 (FastWeight) on 1,000 UltraChat conversations.

| Tier | Architecture | Memory Type | Capacity |
|------|-------------|-------------|----------|
| 1 — Legacy | `LegacyHippocampalMemory` | Explicit pattern buffer | 2,048 slots |
| 2 — Modular | `ModularHippocampalMemory` | Explicit pattern buffer (modular) | 8,192 slots |
| 3 — FastWeight | `ModularFastWeightMemory` | Weight matrices (no explicit storage) | Theoretically ∞ |

Tiers 1 & 2 used the pre-existing trained checkpoint (`checkpoints/ultrachat_1k_trained/final`, step 6366).
Tier 3 was benchmarked **untrained** (random init), then **trained** on 1,000 UltraChat conversations and re-benchmarked.

---

## Full Results — All Tiers (Post-Training)

| Metric | Tier 1 (Legacy) | Tier 2 (Modular) | Tier 3 (FastWeight) |
|--------|:-:|:-:|:-:|
| **Recall@1** | **100.0%** | **100.0%** | 62.5% |
| **Recall@3** | **100.0%** | **100.0%** | **100.0%** |
| **Recall@5** | **100.0%** | **100.0%** | **100.0%** |
| **MRR** | **1.000** | **1.000** | 0.792 |
| Storage Rate | 78.8% | 66.2% | 71.2% |
| Retrieval Latency | 5.4 ms | **2.3 ms** | 3.4 ms |
| Hopfield Accuracy | **100.0%** | 25.0% | 37.5% |
| Pipeline Hit | **87.5%** | 0.0% | 0.0% |

---

## Phase 3 Training Impact — Tier 3 Before vs After

Training: 1,000 UltraChat conversations, 6,366 training steps, 1,467s wall time (4.3 turns/s).

| Metric | Untrained | Trained | Delta |
|--------|:-:|:-:|:-:|
| **Recall@1** | 50.0% | **62.5%** | **+12.5pp** |
| **Recall@3** | 62.5% | **100.0%** | **+37.5pp** |
| **Recall@5** | 62.5% | **100.0%** | **+37.5pp** |
| **MRR** | 0.562 | **0.792** | **+0.229** |
| Storage Rate | 62.1% | 71.2% | +9.2pp |
| Hopfield Accuracy | 50.0% | 37.5% | −12.5pp |
| Retrieval Latency | 2.5 ms | 3.4 ms | +0.9 ms |

**Key takeaways:**
- Recall@3 and Recall@5 jumped from 62.5% → 100.0% after training — every target memory is now retrievable within the top 3.
- MRR improved 41% (0.562 → 0.792), meaning the target is ranked higher on average.
- Recall@1 improved 25% (50.0% → 62.5%), showing better top-1 precision.
- Hopfield accuracy dipped slightly — expected since `ModularFastWeightMemory` routes through learned weight matrices rather than explicit pattern matching.

---

## Training Loss Curves

| Component | First Loss | Final Loss | Steps |
|-----------|:-:|:-:|:-:|
| fast_weight | 0.993 | **0.752** | 6,346 |
| transformer_wm | 1.037 | **0.640** | 6,357 |
| vae | 1.414 | **0.939** | 6,337 |
| pattern_sep | 0.044 | **0.001** | 6,351 |
| forgetting | 0.004 | **0.000** | 6,337 |

---

## Module Specialization (Tier 3 — Trained)

| Property | Untrained | Trained |
|----------|:-:|:-:|
| Active modules | 15 / 32 | **18 / 32** |
| w_key_norm mean | 0.060 | **0.084** |
| w_key_norm max | 0.241 | 0.241 |

Trained modules show emerging entity specialization:
- Module 11: FastAPI (2 occurrences) — w_key_norm=0.24
- Module 16: Python, Go — w_key_norm=0.23
- Module 29: Python (2) — w_key_norm=0.19
- Module 12: Miles Davis, Love Supreme — w_key_norm=0.18

3 additional modules activated after training (15 → 18), and the mean weight norm increased 40% (0.060 → 0.084), indicating the fast-weight matrices are learning to differentiate inputs.

---

## Architecture Trade-offs

| Property | Tier 1 | Tier 2 | Tier 3 |
|----------|--------|--------|--------|
| Top-1 accuracy | Best (100%) | Best (100%) | Good (62.5%) |
| Top-3 accuracy | 100% | 100% | 100% (trained) |
| Latency | 5.4 ms | 2.3 ms (fastest) | 3.4 ms |
| Explicit storage | Yes | Yes | **No** (weights only) |
| Privacy | Data stored | Data stored | **Data not stored** |
| Capacity scaling | Fixed (2K) | Fixed (8K) | Theoretically unbounded |
| Pipeline integration | ✅ 87.5% | ❌ 0% | ❌ 0% |

**Tier 1** achieves the best raw accuracy with full pipeline integration, at the cost of higher latency and a fixed-size explicit buffer.

**Tier 2** matches Tier 1's recall with the fastest latency, but the modular Hopfield routing doesn't produce clean pipeline hits.

**Tier 3** trades some top-1 precision for a fundamentally different storage paradigm — no explicit episodic copies are kept, only learned weight matrices. After just 1K conversations of training, it already reaches 100% Recall@3 and continues to develop module specialization.

---

## Files Produced

| File | Description |
|------|-------------|
| `benchmark_results_untrained.json` | Full results — all 3 tiers, Tier 3 untrained |
| `benchmark_results_trained.json` | Full results — all 3 tiers, Tier 3 with trained checkpoint |
| `brain-memory/scripts/benchmark_tiers.py` | Benchmark evaluation script |
| `brain-memory/scripts/train_fast_weight.py` | Phase 3 offline training script |
| `brain-memory/checkpoints/fast_weight_1k/` | Training checkpoints (conv_200 through final) |
