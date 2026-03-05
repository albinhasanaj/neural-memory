# Prompt: Upgrade Brain-Memory System with Real Neural Network Architectures

## Context

You are working on an existing project at `brain-memory/` — a brain-inspired memory system for LLMs. The project is already scaffolded and functional (~2,000 lines across 9 core modules). It has working episodic memory, semantic graphs, spreading activation, consolidation, salience scoring, and Open WebUI integration.

**The problem:** Almost everything is algorithmic — for-loops, cosine similarity, heuristic weights, sklearn clustering. Only two small `nn.Module` classes exist (a GRU encoder and a salience MLP). The system *describes* brain-like memory but doesn't *learn* like a brain. The activation engine is a handcoded graph traversal. Consolidation is clustering + an LLM call. Encoding is a fixed formula. None of these learn from experience.

**Your task:** Introduce real, trainable deep learning architectures into every component where the brain uses learned computation. The system should have neural networks that actually train, adapt, and improve as they process conversations — not just static algorithms with neural network embeddings bolted on.

Do NOT delete or rewrite the existing code. Add new neural modules alongside the existing implementations and make them swappable via config flags (e.g., `use_gnn_activation: bool = True` in settings.py). The existing algorithmic versions become the baselines for ablation experiments.

No Jupyter notebooks. No `.ipynb`. All regular `.py` files.

---

## Architecture Upgrades — What to Build

### 1. Graph Attention Network (GAT) for Spreading Activation

**File:** `memory/neural_activation.py`

**Why:** In the brain, activation propagation isn't a fixed algorithm — the connection strengths are *learned*. Different types of relationships should propagate activation differently. A Graph Attention Network learns which edges matter for which queries.

**What to build:**

```python
class MemoryGAT(nn.Module):
    """
    Graph Attention Network that learns activation propagation patterns
    over the memory graph.

    Instead of fixed edge_weight × decay spreading, the GAT learns:
    - Which neighbors are relevant given the current context (attention)
    - How much activation to propagate across different edge types
    - Multi-hop reasoning through multiple GAT layers
    """
```

Implementation details:

- Use `torch_geometric` (PyG) as the GNN framework. Add it to dependencies.
- **Node features:** Concatenation of [node_embedding (384d), node_type_onehot (5d), current_activation (1d), time_since_last_access (1d), access_count (1d)] = 392d input
- **Edge features:** Concatenation of [relation_type_onehot (10d), edge_weight (1d), evidence_count (1d), edge_age (1d)] = 13d
- **Architecture:**
  - 3 GATv2Conv layers (GAT v2 fixes the static attention problem in GAT v1)
  - Layer dims: 392 → 256 → 128 → 64
  - Multi-head attention: 4 heads per layer, concatenated (not averaged)
  - Edge features fed into attention computation via GATv2's edge_attr support
  - Skip connections between layers (ResGAT pattern)
  - Dropout 0.1 between layers
  - Final linear projection: 64 → 1 (activation score per node)
- **Forward pass:**
  - Input: full memory graph as PyG Data object + context_vector as query
  - Context vector is concatenated to every node feature (so the GAT is context-conditioned)
  - Output: activation score for every node in the graph
  - Top-K selection replaces lateral inhibition
- **Training signal:**
  - Self-supervised: after the LLM generates a response, check which memories were *actually useful* by measuring whether the response references or is consistent with the activated memories
  - Contrastive loss: activated memories that appeared in the response = positive, activated memories that didn't = negative
  - Train online with a small learning rate (1e-4) after each conversation turn
  - Use a replay buffer of recent (graph_state, context, relevance_labels) tuples

**Also create:** `memory/graph_converter.py`
- Utility to convert the existing NetworkX `SemanticGraph` + `EpisodicStore` into a PyG `Data` object
- Must handle dynamic graph size (nodes added/removed between turns)
- Efficient incremental updates (don't rebuild the full graph every turn)
- Cache the PyG representation and update incrementally

---

### 2. Modern Hopfield Network for Associative Memory

**File:** `memory/hopfield_memory.py`

**Why:** The hippocampus is fundamentally an associative memory — it stores patterns and retrieves them from partial cues (pattern completion). This is literally what Hopfield networks do. Modern Hopfield Networks (Ramsauer et al., 2020) have exponential storage capacity and a well-defined connection to transformer attention.

**What to build:**

```python
class HippocampalMemory(nn.Module):
    """
    Modern Hopfield Network that acts as the hippocampal associative store.

    Stores episodic memory patterns and retrieves them via pattern completion:
    - Given a partial cue (current context), retrieves the stored pattern
      most associated with that cue
    - Exponential storage capacity (unlike classical Hopfield nets)
    - The energy function is equivalent to transformer attention with
      large inverse temperature β
    """
```

Implementation details:

- Based on the Modern Hopfield Network formulation: retrieval = softmax(β × query @ stored_patterns.T) @ stored_values
- **Stored patterns:** episodic memory embeddings (384d each)
- **Stored values:** the full episodic entry metadata encoded as a vector (embedding + salience + entity_embedding + topic_embedding)
- **Query:** the current context vector from working memory
- **Inverse temperature β:** Learnable parameter, initialized to 1.0. Higher β = sharper retrieval (more like exact match), lower β = softer retrieval (more associative)
- **Separation function:** Apply a learned linear transform to stored patterns before attention — this implements pattern separation (making similar memories more distinct, like dentate gyrus does)
- **Pattern completion:** Given a partial query (e.g., just an entity name embedding), retrieve the full episodic memory associated with it
- **Capacity management:** When the store exceeds max capacity, consolidate least-activated patterns (mimicking hippocampal memory transfer to neocortex)
- **Interface with existing EpisodicStore:** HippocampalMemory reads from and writes to EpisodicStore but adds the neural pattern completion layer on top

---

### 3. Variational Autoencoder for Memory Consolidation

**File:** `memory/neural_consolidation.py`

**Why:** In the brain, consolidation happens when the hippocampus replays episodes and this replay trains the neocortex — the neocortex is literally a generative model being trained by replay. A VAE that is trained on replayed episodes and learns to generate compressed semantic representations is the direct computational analog.

**What to build:**

```python
class ConsolidationVAE(nn.Module):
    """
    Variational Autoencoder that models neocortical consolidation.

    Trained by "replaying" episodic memories (hippocampal replay).
    The encoder compresses episodes into a latent semantic space.
    The decoder reconstructs episodes from semantic representations.

    Over many replay cycles, the latent space organizes into clusters
    that correspond to semantic facts — this IS consolidation.
    """
```

Implementation details:

- **Encoder:** episodic_embedding (384d) + metadata_features (32d) → hidden (256d) → μ (64d), log_σ² (64d)
- **Decoder:** z (64d) → hidden (256d) → reconstructed_embedding (384d) + reconstructed_metadata (32d)
- **Metadata features (32d):** learned encoding of [speaker_type, entity_count, topic_ids, salience, turn_position]
- **Latent space (64d):** This is the "semantic representation" — the compressed knowledge that survives consolidation
- **Training loop — "Hippocampal Replay":**
  - Every N turns (configurable, e.g., 25), trigger a "replay session"
  - Sample a batch of episodic memories (weighted by salience × recency)
  - Replay each memory 3-5 times with noise (mimicking inexact replay / sharp-wave ripples)
  - Train the VAE with standard ELBO loss: reconstruction + KL divergence
  - After training, cluster the latent space using the VAE's own representations
  - Extract semantic facts from clusters where latent vectors are tightly grouped
- **Semantic fact extraction:**
  - After replay training, run all episodic embeddings through the encoder to get latent vectors
  - Cluster latent vectors (DBSCAN or k-means in latent space)
  - For each tight cluster (low intra-cluster variance), the cluster centroid IS the semantic representation
  - Decode the centroid back to embedding space and find the nearest entity/relation in the semantic graph
  - If no match exists, use the existing LLM call to label the new semantic fact
  - Upsert into the semantic knowledge graph with confidence = cluster_tightness × evidence_count
- **Key property:** Unlike the existing clustering approach, the VAE's latent space learns to organize semantically over time. Early in training, clustering is noisy. After many replay sessions, the latent space develops meaningful structure — just like how the neocortex gradually develops organized knowledge through sleep consolidation.

---

### 4. Gating Network for Memory Encoding (Dopaminergic Gate)

**File:** `memory/gate_network.py`

**Why:** The brain doesn't use a weighted sum of heuristics to decide what to remember. The dopaminergic system is a learned gating mechanism that is shaped by reward prediction errors — it learns from experience what kind of information tends to be important.

**What to build:**

```python
class DopaminergicGate(nn.Module):
    """
    Learned importance gating network that decides which experiences
    become long-term memories.

    Mimics the dopaminergic salience signal:
    - Novelty detection via prediction error
    - Learned importance scoring (not fixed weights)
    - Trained by delayed reward signal: was this memory actually useful later?
    """
```

Implementation details:

- **Input features (per turn):**
  - Turn embedding (384d)
  - Context vector from working memory (64d, from GRU)
  - Novelty score — cosine distance from nearest known concept (1d)
  - Prediction error — distance from GRU's predicted next embedding (1d)
  - Entity count (1d)
  - Text features: length, question_mark, exclamation, caps_ratio (4d)
  - Temporal features: time_since_last_turn, turn_position_in_session (2d)
  - Total: ~457d input
- **Architecture:**
  - Input projection: 457 → 128 (Linear + LayerNorm + GELU)
  - Residual block 1: 128 → 128 (Linear + LayerNorm + GELU + Dropout(0.1) + skip)
  - Residual block 2: 128 → 128 (same)
  - Output head: 128 → 1 (Linear + Sigmoid)
  - Output: probability that this turn should become a long-term memory (0.0 – 1.0)
- **Training — Delayed Reward Signal:**
  - The gate can't be trained in the moment (we don't know yet if a memory will be useful)
  - Instead, maintain a buffer of (turn_features, gate_decision, was_useful) triples
  - After each response, check: did the activated memories contribute to the response? If a memory was activated AND the response referenced its content → reward signal = 1.0 for the gate decision that encoded it
  - If a memory was encoded but NEVER activated in the next 50 turns → negative signal (wasted storage)
  - Train with binary cross-entropy: gate should output high probability for turns that later proved useful
  - Update every 10 turns with a small batch from the replay buffer
  - Learning rate 5e-5 (very slow — dopaminergic learning is gradual)
- **Exploration:** Add ε-greedy noise during gating (encode some low-confidence turns randomly) to ensure the gate doesn't become overly conservative and miss novel information

---

### 5. Sparse Autoencoder for Pattern Separation

**File:** `memory/pattern_separation.py`

**Why:** The dentate gyrus in the hippocampus performs "pattern separation" — taking similar inputs and mapping them to very different internal representations so they don't interfere with each other. This is critical for storing many similar memories without confusion. A sparse autoencoder with a wide hidden layer and L1 sparsity penalty is the standard computational model of this.

**What to build:**

```python
class PatternSeparator(nn.Module):
    """
    Sparse autoencoder that mimics dentate gyrus pattern separation.

    Takes dense input embeddings and produces sparse, high-dimensional
    representations where similar inputs are mapped to distinct codes.
    This prevents catastrophic interference between similar memories.
    """
```

Implementation details:

- **Encoder:** 384d → 2048d (expansion factor ~5x, mimicking DG/CA3 expansion)
- **Sparsity:** Top-K activation — only the K highest activations are kept, rest zeroed (K ≈ 50, meaning ~2.4% sparsity, matching biological estimates)
- **Decoder:** 2048d (sparse) → 384d (reconstruction)
- **Training:** Reconstruction loss (MSE) + sparsity penalty (L1 on hidden activations)
- **Usage:**
  - Every embedding that enters the episodic store goes through the pattern separator first
  - The sparse code is what's actually stored and used for associative retrieval
  - The dense embedding is kept as metadata for the semantic graph and consolidation
- **Why this matters for the system:** Without pattern separation, two similar conversations ("I had a problem with WordPress plugins" and "I had a problem with WordPress themes") would have nearly identical embeddings and might interfere. The sparse autoencoder maps them to distinct codes, enabling independent storage and retrieval.

---

### 6. Transformer Attention for Working Memory (Central Executive)

**File:** `memory/neural_working_memory.py`

**Why:** The existing GRU processes the working memory buffer sequentially. But the brain's central executive uses *selective attention* — it decides which items in working memory to focus on based on the current task. A transformer self-attention layer is the direct computational analog.

**What to build:**

```python
class CentralExecutive(nn.Module):
    """
    Transformer-based working memory with learned selective attention.

    Replaces the simple GRU encoder with a model that:
    - Attends selectively to relevant items in the working memory buffer
    - Learns which buffer positions and content types to prioritize
    - Produces a context vector that emphasizes task-relevant information
    """
```

Implementation details:

- **Input:** Working memory buffer as a sequence of embeddings [N, 384]
- **Positional encoding:** Learned positional embeddings (not sinusoidal) — position in the buffer matters (most recent vs. oldest)
- **Architecture:**
  - 2 transformer encoder layers (small — this is working memory, not a full LLM)
  - d_model=384, nhead=6, dim_feedforward=512
  - Layer norm + dropout 0.1
  - Learnable [CLS]-style query token prepended to sequence
  - Output: the CLS token's representation after self-attention = context vector
- **Attention heads specialization:** With 6 heads, encourage specialization:
  - Some heads attend to recent turns (recency bias via positional embeddings)
  - Some heads attend to entity-rich turns (learned from content)
  - Some heads attend to high-salience turns
  - This emerges naturally from training, but can be biased via auxiliary losses
- **Next-embedding prediction:** Replace the GRU's linear predictor with an MLP head on the context vector
- **Training:**
  - Primary: next-embedding prediction loss (predict the embedding of the next turn from context)
  - Auxiliary: use the attention weights to compute an "attention entropy" regularizer — encourage focused (low-entropy) attention rather than uniform attention

---

### 7. Forgetting Network (Learned Memory Decay)

**File:** `memory/forgetting.py`

**Why:** The existing system uses ACT-R power-law decay — a fixed mathematical function. In the brain, forgetting is context-dependent: you forget things that conflict with recent experience (interference), things that are no longer relevant to current goals, and things that have been superseded by semantic knowledge. A learned forgetting gate can capture all of this.

**What to build:**

```python
class ForgettingGate(nn.Module):
    """
    Learned forgetting mechanism that decides which memories to decay,
    strengthen, or archive — based on context, not just time.

    Mimics reconsolidation: every time a memory is accessed, it becomes
    labile and the gate decides its new activation level.
    """
```

Implementation details:

- **Input (per memory being evaluated):**
  - Memory embedding (384d)
  - Current context vector (64d from working memory)
  - Time since creation (1d, log-scaled)
  - Time since last access (1d, log-scaled)
  - Access count (1d, log-scaled)
  - Current activation (1d)
  - Salience score (1d)
  - Is_consolidated flag (1d)
  - Semantic graph degree (how many edges connect to this memory's entities) (1d)
  - Total: ~456d
- **Architecture:**
  - Two-head network:
    - **Decay head:** 456 → 128 → 64 → 1 (sigmoid) → decay_rate for this memory
    - **Interference head:** Compare memory embedding against recent buffer items via dot product → interference_score
  - Final activation update: new_activation = old_activation × (1 - decay_rate) - interference_score × interference_weight
  - If new_activation < archive_threshold → archive the memory
- **Training:**
  - When a memory is accessed and proves useful → positive signal (should not have been forgotten)
  - When a memory is accessed and is contradicted by newer information → negative signal (should decay faster)
  - When a memory is never accessed over a long window → soft negative (should decay, but slowly)
  - Binary cross-entropy on "should this memory still be active?"
- **Runs periodically:** Every M turns, evaluate all active memories through the forgetting gate

---

## New Dependencies to Add

Add these to `pyproject.toml`:

```toml
[project.dependencies]
# ... existing deps ...
torch-geometric = ">=2.5.0"
torch-scatter = ">=2.1.0"
torch-sparse = ">=0.6.0"  # already used, but pin version for PyG compat
hopfield-layers = ">=1.0.0"  # or implement from scratch
```

If `hopfield-layers` doesn't exist as a package or is hard to install, implement the Modern Hopfield Network from scratch — it's just: `softmax(β * Q @ K.T) @ V` with learnable β.

---

## New Config Flags in settings.py

Add these to the Settings class:

```python
# Neural architecture flags (toggle between algorithmic and neural versions)
use_gnn_activation: bool = True          # GAT vs algorithmic spreading activation
use_hopfield_memory: bool = True         # Hopfield associative store vs direct episodic retrieval
use_vae_consolidation: bool = True       # VAE consolidation vs clustering + LLM
use_dopaminergic_gate: bool = True       # Learned gating vs weighted-sum salience
use_pattern_separation: bool = True      # Sparse autoencoder vs raw embeddings
use_transformer_wm: bool = True          # Transformer working memory vs GRU
use_learned_forgetting: bool = True      # Forgetting network vs ACT-R power law

# Neural training hyperparameters
gat_learning_rate: float = 1e-4
gate_learning_rate: float = 5e-5
consolidation_vae_lr: float = 3e-4
forgetting_lr: float = 1e-4
pattern_sep_lr: float = 3e-4
wm_transformer_lr: float = 1e-4
training_replay_buffer_size: int = 500
gate_exploration_epsilon: float = 0.1

# Architecture dimensions
gat_hidden_dims: list[int] = [256, 128, 64]
gat_num_heads: int = 4
hopfield_beta_init: float = 1.0
vae_latent_dim: int = 64
pattern_sep_expansion: int = 2048
pattern_sep_top_k: int = 50
wm_transformer_layers: int = 2
wm_transformer_heads: int = 6
```

---

## Updated Module: memory/observer.py

The observer needs to be updated to route through the new neural modules. The flow becomes:

```
Input turn
  → PatternSeparator.encode() → sparse representation
  → DopaminergicGate.score() → encode decision
  → if encode: write to EpisodicStore AND HippocampalMemory
  → CentralExecutive.update() → new context vector
  → MemoryGAT.forward() → neural activation scores (OR fallback to algorithmic)
  → HippocampalMemory.retrieve() → pattern-completed episodic memories
  → Injector.inject() → memory context into LLM prompt
  → (Background) ConsolidationVAE.replay() → train on recent episodes
  → (Background) ForgettingGate.evaluate() → decay/archive memories
```

Add a `NeuralMemoryObserver` class that wraps the existing `MemoryObserver` and adds the neural pipeline. Make it configurable which components use neural vs algorithmic versions.

---

## Training Coordinator

**File:** `memory/trainer.py`

**Why:** Multiple neural components need to train concurrently with different schedules and signals. A training coordinator manages this.

```python
class MemoryTrainer:
    """
    Coordinates online training of all neural memory components.

    Training schedule:
    - DopaminergicGate: every 10 turns (from delayed reward buffer)
    - MemoryGAT: every turn (contrastive loss from activation relevance)
    - ConsolidationVAE: every 25 turns (replay session)
    - PatternSeparator: every 50 turns (reconstruction loss on stored memories)
    - CentralExecutive: every turn (next-embedding prediction)
    - ForgettingGate: every 20 turns (evaluate all active memories)
    """
```

- Maintains separate optimizers for each component
- Manages the replay buffer (shared across components)
- Logs training metrics (loss curves, gradient norms) for each component
- Saves/loads all model checkpoints together
- Can freeze specific components for ablation experiments

---

## New Tests

Add these test files:

```
tests/
├── test_gat_activation.py          # Test GAT forward pass, training step, comparison with algorithmic
├── test_hopfield_memory.py         # Test storage, retrieval, pattern completion, capacity
├── test_vae_consolidation.py       # Test encoding, decoding, latent clustering after replay
├── test_dopaminergic_gate.py       # Test gating decisions, training with delayed reward
├── test_pattern_separation.py      # Test sparsity, separation of similar inputs, reconstruction
├── test_transformer_wm.py          # Test attention patterns, context vector quality
├── test_forgetting_network.py      # Test decay decisions, interference detection
├── test_trainer.py                 # Test training coordinator scheduling and checkpointing
└── test_neural_integration.py      # End-to-end: full neural pipeline from input to injection
```

---

## New Experiment Scripts

```
experiments/
├── run_neural_vs_algorithmic.py    # Compare full neural stack vs full algorithmic stack
├── run_gat_vs_spreading.py         # GAT activation vs handcoded spreading activation
├── run_hopfield_vs_knn.py          # Hopfield pattern completion vs k-NN episodic retrieval
├── run_vae_vs_clustering.py        # VAE consolidation vs agglomerative clustering
├── run_gate_vs_heuristic.py        # Learned gating vs fixed weighted-sum salience
├── run_training_curves.py          # Monitor training loss curves for all neural components
└── run_capacity_scaling.py         # How does the neural system scale with 100, 1K, 10K memories?
```

---

## Critical Implementation Notes

1. **All neural components must be optional.** The existing algorithmic code is the baseline. Every neural module should be toggled by a config flag and fall back to the algorithmic version when disabled. This is essential for ablation research.

2. **Online training is tricky.** The neural components train while the system is in use — not offline. Keep batch sizes small (8-16), learning rates very low, and use gradient clipping (max_norm=1.0) everywhere to prevent instability.

3. **Memory efficiency matters.** The GAT rebuilds the graph representation every turn. Use incremental updates (add/remove nodes/edges) rather than full reconstruction. Cache PyG Data objects and update in-place.

4. **The training signals are weak and delayed.** The dopaminergic gate doesn't know if a memory was useful until many turns later. The GAT's contrastive signal is noisy. Use large replay buffers and slow learning rates. This is realistic — biological learning is slow too.

5. **Pattern separation MUST run before storage.** If you store raw dense embeddings, the Hopfield network will confuse similar memories. The sparse autoencoder separates them first, then the Hopfield net stores the separated codes.

6. **The VAE's latent space IS the semantic memory.** Over many replay sessions, the latent space self-organizes. Don't force external clustering early — let the VAE's latent geometry develop, then read off the semantic facts from the latent structure.

7. **Start with everything disabled.** First verify the existing algorithmic system still works. Then enable one neural component at a time, verify it trains and improves performance, and move to the next. The ablation experiments are built for exactly this workflow.

8. **GPU is optional but recommended.** Everything should work on CPU (with float32 tensors, small batch sizes). If CUDA is available, move all models to GPU. The config already has device auto-detection.
