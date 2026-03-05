# Brain Memory — Technical Audit

**Date:** 2026-03-05  
**Codebase:** `brain-memory` v0.1.0  
**Root:** `brain-memory/`  

---

## 1. File Structure

### `config/`
| File | Description |
|------|-------------|
| `__init__.py` | Re-exports the singleton `settings` object |
| `settings.py` | Master Pydantic Settings config — all hyperparams, feature flags, paths, LLM provider config; auto-reads `BRAIN_*` env vars and `.env` |

### `memory/` — Core cognitive architecture
| File | Description |
|------|-------------|
| `__init__.py` | Package docstring only |
| `encoder.py` | Thread-safe singleton wrapper around `sentence-transformers` (`all-MiniLM-L6-v2`); lazy-loads model on first call |
| `working_memory.py` | `RingBuffer` (fixed-capacity deque of tensors) + `GRUContextEncoder` (single-layer GRU → context vector + predicted-next embedding); combined as `WorkingMemory` |
| `salience.py` | Four-signal salience scorer: novelty (cosine distance to graph), prediction error (vs GRU prediction), emphasis (regex heuristics), entity density; weighted sum or optional MLP |
| `episodic.py` | `EpisodicEntry` Pydantic model + `EpisodicStore` in-memory dict with ACT-R power-law activation decay; `compute_activation()` implements B_i = ln(Σ(t-t_j)^{-d}) |
| `semantic.py` | `SemanticGraph` wrapping `networkx.DiGraph` with typed `SemanticNode`/`SemanticEdge` Pydantic models; supports upsert, traversal, activation values, JSON serialization |
| `activation.py` | `SpreadingActivationEngine`: 4-channel seeding (entity-match, intent-cue, working-memory focus, embedding-similarity fallback) → sparse-matrix propagation → top-K lateral inhibition |
| `observer.py` | `MemoryObserver` (orchestrates per-turn pipeline: encode → extract NLP → update WM → score → store → consolidate) + `NeuralMemoryObserver` (adds pattern sep, gate, Hopfield, VAE, forgetting, online training) |
| `injector.py` | Formats activated semantic facts + episodic memories into a `[MEMORY CONTEXT]` system message block; inserts after existing system messages |
| `consolidation.py` | Episodic → semantic extraction: selects high-salience unconsolidated episodes → agglomerative clustering → LLM fact extraction (OpenAI/Anthropic) → upserts typed edges into semantic graph |
| `gate_network.py` | `DopaminergicGate`: 2-residual-block MLP (embedding+context+signals → p(store)); epsilon-greedy decisions; `GateReplayBuffer` with delayed reward retroactively assigned via cosine similarity |
| `hopfield_memory.py` | `HippocampalMemory`: Modern Hopfield Network (Ramsauer 2020) with learnable β, separator transform, query projection; capacity-based consolidation evicts least-accessed patterns; `HopfieldReplayBuffer` for contrastive training |
| `pattern_separation.py` | Sparse autoencoder (384→2048→top-K→384): dentate gyrus analogue; reconstruction loss + L1 sparsity; `SeparationReplayBuffer` for online training |
| `neural_activation.py` | `MemoryGAT`: 3-layer GATv2 with skip connections for learned spreading activation; falls back to manual multi-head attention if `torch_geometric` not installed; `ActivationReplayBuffer` + BCE training on relevance labels |
| `neural_consolidation.py` | `ConsolidationVAE`: encoder-decoder VAE (embedding+metadata→latent→reconstruction); ELBO loss with β-annealing; K-means++ in latent space for semantic clustering; `ConsolidationReplayBuffer` |
| `neural_working_memory.py` | `TransformerContextEncoder`: 2-layer Transformer with learned [CLS] token + positional embeddings; drop-in replacement for GRU WM; cosine-similarity prediction loss; `TransformerWMReplayBuffer` |
| `forgetting.py` | `ForgettingNetwork`: two-headed MLP (decay-rate + interference score) conditioned on embedding, scalars, optional context; effective activation = base × (1−decay)^Δt × (1−interference); MSE training |
| `graph_converter.py` | `GraphConverter`: converts `SemanticGraph` + `EpisodicStore` into PyG-compatible `PyGData` objects (node features 392d, edge features 13d); incremental caching |
| `trainer.py` | `TrainingCoordinator`: manages 7 component optimizers, replay buffers, training dispatch, checkpoint save/load (model weights + optimizer states + metadata JSON) |

### `nlp/` — Text analysis
| File | Description |
|------|-------------|
| `__init__.py` | Package docstring only |
| `entity_extractor.py` | Regex-based NER (proper nouns, tech terms, emails, URLs) with optional spaCy `en_core_web_sm` upgrade |
| `intent_detector.py` | 10 regex patterns detecting recall-intent cues ("remember when...", "what was my...", etc.); returns `IntentCueResult` with confidence + noun-phrase targets |
| `topic_tagger.py` | Keyword dictionary → regex matching across 10 topic categories (programming, python, ML, personal, etc.); returns ranked topic labels |

### `pipeline/` — Integration layer
| File | Description |
|------|-------------|
| `__init__.py` | Package docstring only |
| `memory_pipeline.py` | `BrainMemoryPipeline`: Open WebUI Pipeline class; initializes all components, hydrates from SQLite, routes through `MemoryObserver` or `NeuralMemoryObserver`, persists on store/shutdown |
| `proxy_server.py` | FastAPI app at `/v1/chat/completions`; intercepts OpenAI-format requests, runs memory pipeline, forwards to real LLM (OpenAI or Anthropic), observes response; supports streaming |
| `llm_chat.py` | `MemoryChat`: interactive CLI chat; `LLMClient` for OpenAI/Anthropic; enables all neural modules, runs full observe→activate→inject→LLM→observe-response loop; supports checkpoint loading |

### `storage/` — Persistence
| File | Description |
|------|-------------|
| `__init__.py` | Package docstring only |
| `sqlite_store.py` | `SQLiteEpisodicStore`: SQLite DB with `episodes` table; embeddings as float32 BLOBs, structured fields as JSON text; WAL mode; CRUD + hydration + flush |
| `graph_store.py` | JSON file save/load for `SemanticGraph`; embeddings encoded as base64 float32 for compactness |

### `data/` — Training data
| File | Description |
|------|-------------|
| `__init__.py` | Package docstring only |
| `conversation_loader.py` | Loaders for UltraChat 200k (HuggingFace), OpenAssistant OASST2, and local JSON; yields `Conversation` objects with `Turn(role, content)` |
| `sample_conversations.json` | Sample conversation data file |

### `scripts/` — CLI tools
| File | Description |
|------|-------------|
| `train_offline.py` | Offline training pipeline: enables all neural modules → loads dataset → simulates conversations through `NeuralMemoryObserver` → periodic checkpointing |
| `eval_recall.py` | Evaluation harness: synthetic recall conversations measuring Recall@K, MRR, storage rate, avg salience |
| `seed_memory.py` | Seeds the memory system with sample data |
| `export_memory.py` | Exports memory state to disk |

### `experiments/` — Ablation studies
| File | Description |
|------|-------------|
| `eval_metrics.py` | Shared evaluation metric functions |
| `run_*.py` | Individual experiment scripts (activation vs KNN, baseline, capacity scaling, episodic only, forgetting, GAT vs spreading, gate vs heuristic, Hopfield vs KNN, neural vs algorithmic, semantic only, training curves, VAE vs clustering) |

### `tests/` — Test suite
| File | Description |
|------|-------------|
| `conftest.py` | Shared pytest fixtures |
| `test_*.py` | Per-module unit tests (activation, consolidation, data loader, dopaminergic gate, episodic, eval integration, forgetting, GAT activation, Hopfield, injector, etc.) |

### `visualization/` — Dashboards
| File | Description |
|------|-------------|
| Various files | Streamlit/Gradio dashboard and plotting utilities |

### Root
| File | Description |
|------|-------------|
| `pyproject.toml` | Project metadata, dependencies, CLI entry points, build config |
| `README.md` | Architecture diagram, data flow description, quickstart guide |

---

## 2. Module Map

### 2.1 Embedding Encoder
- **Files:** [memory/encoder.py](brain-memory/memory/encoder.py)
- **Input:** `str` (text)
- **Output:** `Tensor[384]` (embedding vector)
- **Architecture:** `sentence-transformers/all-MiniLM-L6-v2` via `SentenceTransformer`; thread-safe singleton, lazy-loaded
- **Status:** ✅ **Working** — fully implemented, used everywhere as the shared embedding backbone

### 2.2 Pattern Separator (Dentate Gyrus)
- **Files:** [memory/pattern_separation.py](brain-memory/memory/pattern_separation.py)
- **Input:** `Tensor[384]` (raw embedding)
- **Output:** `Tensor[384]` (separated embedding), also produces `Tensor[2048]` sparse code and `Tensor[384]` reconstruction
- **Architecture:** Sparse autoencoder: `384→2048 (Linear+LN+ReLU) → top-50 sparsity → 384 (bottleneck)`; decoder for reconstruction loss
- **Loss:** MSE reconstruction + L1 sparsity penalty
- **Config flag:** `use_pattern_separation` (default: `False`)
- **Status:** ✅ **Working** — fully implemented with replay buffer and training step; integrated into `NeuralMemoryObserver.observe()` (transforms embedding before all downstream use)

### 2.3 Working Memory (GRU / Transformer)
- **Files:** [memory/working_memory.py](brain-memory/memory/working_memory.py) (GRU baseline), [memory/neural_working_memory.py](brain-memory/memory/neural_working_memory.py) (Transformer upgrade)
- **Input:** `Tensor[384]` (per-turn embedding)
- **Output:** `Tensor[256]` (context vector), `Tensor[384]` (predicted-next embedding)
- **Architecture (GRU):** `RingBuffer(capacity=8)` → `GRU(384→256)` + linear predictor `(256→384)`
- **Architecture (Transformer):** `RingBuffer` → `Linear(384→256)` + learnable [CLS] + 2-layer Transformer encoder (4 heads, 512 FFN) → CLS output → predictor MLP `(256→256→384)`
- **Loss (Transformer):** `1 − cosine_similarity(predicted, actual_next)` trained from replay buffer
- **Config flag:** `use_transformer_wm` (default: `False`)
- **Status:** ✅ **Working** — both variants fully implemented; Transformer is a drop-in replacement via API compatibility; replay buffer training wired up

### 2.4 Salience Scorer
- **Files:** [memory/salience.py](brain-memory/memory/salience.py)
- **Input:** embedding, predicted-next embedding, entity list, raw text, semantic graph
- **Output:** `float` ∈ [0, 1] (salience score)
- **Architecture:** Computes 4 signals: `novelty` (max cosine distance to graph nodes), `prediction_error` (1 − cos_sim to GRU prediction), `emphasis` (regex pattern matching), `entity_density` (entities/words). Default: weighted sum (0.35/0.30/0.20/0.15). Optional: 3-layer MLP `(4→16→8→1, sigmoid)`
- **Status:** ✅ **Working** — weighted sum mode fully functional; MLP mode implemented but `use_mlp` never toggled via config (hardcoded `False` unless manually set)

### 2.5 Dopaminergic Gate Network
- **Files:** [memory/gate_network.py](brain-memory/memory/gate_network.py)
- **Input:** `Tensor[384]` (embedding) + `Tensor[256]` (context) + `Tensor[4]` (salience signals)
- **Output:** `float` ∈ [0, 1] (store probability) + boolean decision
- **Architecture:** `[384+256+4=644] → Linear(644→128) + LN + GELU → 2× ResidualBlock(128) → Linear(128→32→1, sigmoid)`
- **Loss:** BCE between predicted p(store) and retroactive reward (0 or 1) assigned via cosine similarity when memories are later retrieved
- **Config flag:** `use_dopaminergic_gate` (default: `False`)
- **Status:** ✅ **Working** — fully implemented with epsilon-greedy exploration, replay buffer, delayed reward mechanism, and training step (`train_gate_step` function imported in trainer); integrated into `NeuralMemoryObserver.observe()`

### 2.6 Hopfield Network (Hippocampal Memory)
- **Files:** [memory/hopfield_memory.py](brain-memory/memory/hopfield_memory.py)
- **Input:** `Tensor[384]` (query embedding)
- **Output:** `Tensor[384]` (pattern-completed value), attention weights, top-K indices; or `list[(episode_id, weight)]`
- **Architecture:** Modern Hopfield: `softmax(β · query_proj(q) @ separator(patterns)ᵀ) @ values`; learnable `log_beta`, `separator` (Linear+LN), `query_proj` (Linear); capacity-based consolidation evicts least-accessed when >2048 patterns
- **Loss:** Not explicitly in `train_hopfield_step` — the function signature is present but the body was not fully shown. The replay buffer stores `(query, positive)` pairs for contrastive retrieval training
- **Config flag:** `use_hopfield_memory` (default: `False`)
- **Status:** ✅ **Working** — store/retrieve/consolidate fully implemented; integrated into `NeuralMemoryObserver`'s `observe()` (store) and `activate_and_inject()` (retrieve + gate reward feedback)

### 2.7 Semantic Graph
- **Files:** [memory/semantic.py](brain-memory/memory/semantic.py), [storage/graph_store.py](brain-memory/storage/graph_store.py)
- **Input:** `SemanticNode` / `SemanticEdge` Pydantic models
- **Output:** Graph traversal results, activation values, neighbors, subgraphs
- **Architecture:** `networkx.DiGraph` with typed nodes (entity/concept/topic) carrying embeddings + activation floats; typed edges (relation, weight, confidence, evidence list); JSON serialization with base64-encoded embeddings
- **Status:** ✅ **Working** — fully implemented; used by spreading activation, consolidation, injector, and salience (novelty computation)

### 2.8 Spreading Activation Engine
- **Files:** [memory/activation.py](brain-memory/memory/activation.py) (algorithmic), [memory/neural_activation.py](brain-memory/memory/neural_activation.py) (GAT), [memory/graph_converter.py](brain-memory/memory/graph_converter.py)
- **Input:** `Tensor[384]` (context vector) + `SeedHints` (entities, intent targets, WM embeddings)
- **Output:** `list[(node_id, activation_strength)]` sorted descending
- **Architecture (Algorithmic):** 4-channel seeding → sparse-matrix propagation (`adj^T @ activation` for N iterations with decay) → top-K lateral inhibition
- **Architecture (GAT):** `GraphConverter` → `MemoryGAT` (3-layer GATv2 with skip connections, context-conditioned, sigmoid output); BCE loss against relevance labels
- **Config flag:** `use_gnn_activation` for GAT (default: `False`)
- **Status:** ✅ **Working (algorithmic)** — fully integrated into `MemoryObserver.activate_and_inject()`. ⚠️ **GAT: Implemented but not wired into the observer** — `MemoryGAT` and `GraphConverter` exist and train via `TrainingCoordinator`, but `NeuralMemoryObserver.activate_and_inject()` still calls the algorithmic engine; no codepath switches to GAT based on `use_gnn_activation`

### 2.9 VAE Consolidation (Replay)
- **Files:** [memory/neural_consolidation.py](brain-memory/memory/neural_consolidation.py)
- **Input:** `Tensor[384]` (embedding) + `Tensor[32]` (metadata vector: salience, entity count, speaker, age)
- **Output:** Reconstructed embedding + metadata; latent codes `Tensor[64]`; cluster centroids
- **Architecture:** VAE: encoder `(384+32→256→128→z)` + decoder `(z→128→256→384+32)`; K-means++ clustering in latent space
- **Loss:** ELBO = MSE reconstruction (embedding + metadata) + β × KL divergence
- **Config flag:** `use_vae_consolidation` (default: `False`)
- **Status:** ⚠️ **Partial** — VAE architecture, training step, and replay buffer are fully implemented. Data is pushed into the replay buffer during `NeuralMemoryObserver.observe()`. Training runs via `TrainingCoordinator.step()`. **However**, the consolidation output (latent clustering → semantic graph upsert) is **not connected** — `latent_cluster_centroids()` exists but is never called in the pipeline. The actual consolidation still uses the LLM-based path from `memory/consolidation.py`.

### 2.10 Forgetting Network
- **Files:** [memory/forgetting.py](brain-memory/memory/forgetting.py)
- **Input:** `Tensor[B, 384]` (embeddings) + `Tensor[B, 5]` (scalars: age, access count, salience, activation, context similarity) + optional `Tensor[256]` (context)
- **Output:** `Tensor[B]` decay rate ∈ (0,1), `Tensor[B]` interference ∈ [0,1]; effective activation = base × (1−decay)^Δt × (1−interference)
- **Architecture:** Shared trunk `(389+64→128→128)` + two heads `(128→32→1, sigmoid)` each; optional context projection `(256→64)`
- **Loss:** MSE on predicted decay vs target + MSE on predicted interference vs target
- **Config flag:** `use_learned_forgetting` (default: `False`)
- **Status:** ⚠️ **Partial** — network, replay buffer, and training step are fully implemented. Training data is pushed during `NeuralMemoryObserver.observe()`. **However**, the forgetting network is **never called at retrieval time** — the episodic store still uses `compute_activation()` (ACT-R power law) for all decay/archival decisions. No codepath applies `compute_effective_activation()` to modulate retrievals.

---

## 3. Data Flow — Single Message Trace

Below is the exact path a user message takes through the system, with file and function references.

### Step 0: Entry Point
```
User sends POST /v1/chat/completions with {"messages": [...]}
  ↓
proxy_server.py :: chat_completions()
  ↓
memory_pipeline.py :: BrainMemoryPipeline.pipe(body)
  → Extracts latest message text + role
  → Calls observer.process_turn(text, speaker, messages)
```

### Step 1: OBSERVE — `observer.py :: MemoryObserver.observe(text, speaker)`

```
1a. ENCODE
    encoder.py :: EmbeddingEncoder.encode(text) → Tensor[384]
    [If use_pattern_separation: pattern_separation.py :: separate(embedding) → Tensor[384]]

1b. NLP EXTRACTION
    entity_extractor.py :: extract_entities(text) → list[str]
    topic_tagger.py :: extract_topics(text) → list[str]
    intent_detector.py :: detect_intent_cues(text) → IntentCueResult

1c. WORKING MEMORY UPDATE
    working_memory.py :: WorkingMemory.update(embedding)
      → RingBuffer.append(embedding)          # push into deque
      → GRUContextEncoder.forward(seq)        # sequence → context_vector[256] + predicted_next[384]
    
    [If use_transformer_wm: neural_working_memory.py :: TransformerWorkingMemory.update()]

1d. SALIENCE SCORING
    salience.py :: compute_novelty(embedding, graph) → float
    salience.py :: compute_prediction_error(embedding, predicted_next) → float
    salience.py :: detect_emphasis(text) → float
    salience.py :: compute_entity_density(entities, text) → float
    
    [If use_dopaminergic_gate:
        gate_network.py :: DopaminergicGate.should_store(emb, ctx, signals) → (bool, float)
    Else:
        salience.py :: SalienceScorer.score() → weighted sum → float]
```

### Step 2: STORE (if salience ≥ threshold)

```
2a. EPISODIC STORAGE
    episodic.py :: EpisodicStore.add(EpisodicEntry{text, embedding, entities, topics, salience})
      → Sets initial recall_time, computes ACT-R activation
      → Stored in self._entries dict (in-memory)
    
    [If use_hopfield_memory:
        hopfield_memory.py :: HippocampalMemory.store(embedding, episode_id)
          → separator(embedding) → concat to patterns buffer
    ]
    
    [If use_vae_consolidation:
        neural_consolidation.py :: build_metadata_vector() → push to VAE replay buffer
    ]

2b. PERSISTENCE (in pipeline)
    memory_pipeline.py :: SQLiteEpisodicStore.insert(entry) → writes to episodes table

2c. CONSOLIDATION (every N turns)
    consolidation.py :: run_consolidation_background() → new thread:
      → select_candidates() → cluster_episodes() → extract_fact() via LLM
      → graph.upsert_node() + graph.upsert_edge()
```

### Step 3: ACTIVATE — `observer.py :: MemoryObserver.activate_and_inject(messages)`

```
3a. CONTEXT PROJECTION
    working_memory.py :: encoder.predictor(context_vector) → ctx_emb[384]
    
3b. BUILD SEED HINTS
    SeedHints(entities, intent_targets, intent_confidence, wm_embeddings, context_vector)
    
3c. SPREADING ACTIVATION
    activation.py :: SpreadingActivationEngine.activate(ctx_emb, hints)
      → seed_nodes(): 4-channel weighted blend → Tensor[N]
      → propagate(): sparse adj^T @ activation for 3 iterations with 0.7 decay
      → lateral_inhibition(): keep top-10 nodes
      → Returns [(node_id, activation_strength), ...]
    
3d. EPISODIC RETRIEVAL
    For each activated node: scan episodic_store for episodes linked by node_id or entity match
    episodic.py :: retrieve_by_ids() → records access event, recomputes activation
    
    [If use_hopfield_memory:
        hopfield_memory.py :: retrieve_episode_ids(ctx_emb, top_k=5)
          → Modern Hopfield attention → top-K episode IDs
          → episodic.py :: retrieve_by_ids()
    ]
```

### Step 4: INJECT — `injector.py :: inject(messages, graph, activated, episodes)`

```
4a. FORMAT
    format_semantic_memories(): activated nodes → outgoing edges → "[src] [relation] [tgt]" bullets
    format_episodic_memories(): episodes → "[date] speaker: text" bullets
    
4b. BUILD BLOCK
    "[MEMORY CONTEXT — Automatically activated based on current conversation]"
    "Semantic facts: ..."
    "Episodic memories: ..."
    "[END MEMORY CONTEXT]"
    
4c. INSERT
    Insert as {"role": "system"} message after existing system messages
```

### Step 5: FORWARD TO LLM

```
proxy_server.py :: Forward modified body to OpenAI/Anthropic API
  → Receive response
  → observer.observe(response_text, speaker="assistant")  # encode assistant turn too
  → Return response to client
```

### Step 6: ONLINE TRAINING (per-turn, in NeuralMemoryObserver)

```
trainer.py :: TrainingCoordinator.step()
  → For each enabled component: sample from replay buffer → one gradient step
  → Gate: delayed reward updated retroactively for retrieved memories
```

### Data Flow Gaps and Disconnects

| Gap | Description |
|-----|-------------|
| **GAT not wired** | `use_gnn_activation` flag exists but `NeuralMemoryObserver.activate_and_inject()` always uses the algorithmic `SpreadingActivationEngine`. No codepath substitutes `MemoryGAT`. |
| **Forgetting network unused at retrieval** | `ForgettingNetwork` trains but never modulates `EpisodicStore` retrieval/archival — ACT-R decay remains the only active mechanism. |
| **VAE consolidation never triggers** | `ConsolidationVAE` trains but `latent_cluster_centroids()` is never called — consolidation still goes through the LLM-based `consolidation.py`. |
| **Salience MLP never activated** | `SalienceScorer(use_mlp=True)` exists but no config flag / codepath enables it. |
| **No link back from consolidation to episodes** | When consolidation creates graph edges, it marks episodes `consolidated=True` but doesn't populate `EpisodicEntry.links` with the new node IDs — so the retrieval path in `activate_and_inject()` that checks `ep.links` against activated node IDs will never match via links (only via entity-label overlap). |
| **Graph persistence is periodic, not transactional** | Graph is saved every 5 turns in `BrainMemoryPipeline.pipe()` — a crash loses recent consolidation results. |
| **Double injection possible** | `NeuralMemoryObserver.activate_and_inject()` calls `super().activate_and_inject()` (which calls `inject()`) then calls `inject()` again with Hopfield retrieval results — this can produce two `[MEMORY CONTEXT]` system messages. |

---

## 4. Storage Layer

### Episodic Memory
| Layer | Technology | Details |
|-------|-----------|---------|
| **Runtime** | In-memory Python dict | `EpisodicStore._entries: dict[str, EpisodicEntry]`; all active operations (add, retrieve, decay, archive) happen here |
| **Persistence** | SQLite (WAL mode) | `storage/sqlite_store.py` → file at `./data/episodic.db`; table `episodes` with columns: id, timestamp, speaker, raw_text, embedding (BLOB), entities (JSON), topics (JSON), salience, activation, recall_times (JSON), links (JSON), consolidated, archived; indexed on salience, archived, consolidated |
| **Embedding format** | `numpy.float32.tobytes()` | Stored as raw BLOBs in SQLite, round-tripped via `np.frombuffer` |
| **Hydration** | On startup | `BrainMemoryPipeline.__init__()` loads all non-archived rows into in-memory `EpisodicStore` |
| **Flush** | On shutdown + after each store | Individual `insert()` on each new episode; `flush_store()` bulk-inserts on shutdown |

### Semantic Graph
| Layer | Technology | Details |
|-------|-----------|---------|
| **Runtime** | `networkx.DiGraph` | Node/edge attributes stored in-memory via `SemanticGraph._graph` |
| **Persistence** | JSON file | `storage/graph_store.py` → file at `./data/semantic_graph.json`; node embeddings as base64-encoded float32 |
| **Save frequency** | Every 5 turns | In `BrainMemoryPipeline.pipe()` + on shutdown |

### Hopfield Memory
| Layer | Technology | Details |
|-------|-----------|---------|
| **Runtime** | PyTorch buffers (tensors) | `patterns: Tensor[M, 384]`, `values: Tensor[M, 384]`, `access_counts: Tensor[M]`; plus `_episode_ids: list[str]` |
| **Persistence** | Checkpoint files | Saved as part of `TrainingCoordinator.save_checkpoint()` → `hopfield_model.pt` |

### Working Memory
| Layer | Technology | Details |
|-------|-----------|---------|
| **Runtime** | `deque[Tensor]` | `RingBuffer._items`; capacity 8; purely transient, never persisted |

### Neural Model Weights
| Layer | Technology | Details |
|-------|-----------|---------|
| **Persistence** | PyTorch `.pt` files | `TrainingCoordinator.save_checkpoint(path)` saves per-component `{name}_model.pt` + `{name}_optimizer.pt` + `coordinator_meta.json` |

### No Vector Database
There is **no dedicated vector DB** (no FAISS, Chroma, Qdrant, Pinecone, etc.). Similarity search is done via:
- **Spreading activation seeding**: brute-force `cosine_similarity` against all graph node embeddings (in `activation.py`)
- **Hopfield retrieval**: learned softmax attention over stored patterns
- **Salience novelty**: brute-force cosine distance to all graph embeddings (in `salience.py`)

---

## 5. Training Loop

### Online Training (per-turn)
**When:** After every `NeuralMemoryObserver.observe()` call (step 12 in the observe method).  
**Mechanism:** `TrainingCoordinator.step()` iterates over all enabled components and runs one mini-batch gradient step per component per turn.

| Component | Replay Buffer | Batch Size | Optimizer | LR | Loss Function |  Gradient Clip |
|-----------|--------------|------------|-----------|-----|--------------|----------------|
| **Pattern Separator** | `SeparationReplayBuffer` (raw embeddings) | 16 | Adam | 3e-4 | MSE reconstruction + L1 sparsity | 1.0 |
| **Dopaminergic Gate** | `GateReplayBuffer` (emb+ctx+signals+reward) | 16 | Adam | 5e-5 | BCE(predicted_prob, reward) | 1.0 |
| **Hopfield** | `HopfieldReplayBuffer` (query+positive pairs) | 8 | Adam | 1e-4 | Contrastive (cosine-based) | 1.0 |
| **VAE** | `ConsolidationReplayBuffer` (emb+metadata) | 16 | Adam | 3e-4 | ELBO (MSE recon + β×KL) | 1.0 |
| **Transformer WM** | `TransformerWMReplayBuffer` (seq+target) | 8 | Adam | 1e-4 | 1 − cosine_similarity(predicted, actual) | 1.0 |
| **Forgetting** | `ForgettingReplayBuffer` (emb+scalars+targets) | 16 | Adam | 1e-4 | MSE(pred_decay, target) + MSE(pred_interf, target) | 1.0 |
| **GAT** | `ActivationReplayBuffer` (graph+ctx+labels) | 8 | Adam | 1e-4 | BCE(predicted_scores, relevance_labels) | 1.0 |

### Replay Buffer Details
- All buffers are `deque(maxlen=500)` (configurable via `training_replay_buffer_size`)
- Data is pushed into buffers during `observe()` for each stored episode
- Gate reward is retroactively updated when memories are successfully retrieved (cosine similarity ≥ 0.85)
- Forgetting targets: `target_decay = 1 − salience`, `target_interference = 0.5 − entity_density×0.5`

### Offline Training
**Script:** `scripts/train_offline.py`  
- Enables all neural modules via env vars (except GAT, commented out)
- Disables LLM consolidation (interval=999999)
- Loads conversations from UltraChat 200k, OASST2, or local JSON
- Simulates each conversation through `NeuralMemoryObserver`
- Checkpoints every N conversations + final checkpoint
- Resets working memory between conversations

### What Does NOT Get Gradients
- `EmbeddingEncoder` (frozen `all-MiniLM-L6-v2` — no fine-tuning)
- `GRUContextEncoder` (never trained — used only in inference mode under `torch.no_grad()`)
- `SalienceScorer` (fixed weighted sum — MLP weights exist but `use_mlp` is never set)
- Algorithmic `SpreadingActivationEngine` (no parameters — pure computation)
- `EpisodicStore` / `SemanticGraph` (data structures, not neural modules)

---

## 6. Known Issues

### Critical / Architectural

1. **GAT spreading activation never used at inference** — `MemoryGAT` trains but no codepath in `NeuralMemoryObserver.activate_and_inject()` routes through it based on `use_gnn_activation`. The algorithmic engine is always used.

2. **Forgetting network never applied** — `ForgettingNetwork` trains online but `compute_effective_activation()` is never called during episodic retrieval or archival. The system still relies entirely on ACT-R decay.

3. **VAE consolidation disconnected** — The VAE trains but `latent_cluster_centroids()` is never invoked. The LLM-based consolidation pipeline remains the only active consolidation pathway.

4. **Double memory injection** — `NeuralMemoryObserver.activate_and_inject()` calls `super().activate_and_inject()` (which calls `inject()`) and then conditionally calls `inject()` again with Hopfield-retrieved episodes. This can produce two `[MEMORY CONTEXT]` blocks in the prompt.

5. **Episode links never populated** — `consolidation.py` creates graph edges with `evidence=[ep.id for ep in cluster]` but never writes back `node_id` into `EpisodicEntry.links`. The retrieval path in `activate_and_inject()` checks `ep.links` but it's always empty — retrieval falls back to entity-label string matching only.

6. **GRU context encoder never trained** — `WorkingMemory.update()` runs under `@torch.no_grad()`, and no training step exists for the GRU. The `predictor` layer (used for prediction error) is randomly initialized and never learns. This means prediction-error salience is meaningless noise.

### Moderate

7. **Settings singleton loaded at import time** — `settings = Settings()` executes at module import. The offline training script works around this by setting env vars _before_ importing settings, but this is fragile and order-dependent.

8. **No LR scheduling** — All optimizers use constant learning rates. No warmup, no decay, no cosine annealing.

9. **Gate intrinsic reward scaling is heuristic** — `min(1.0, raw_mean_salience * 3.0)` is a magic number. The gate's training signal quality depends heavily on this mapping.

10. **Streaming response parsing is fragile** — `proxy_server.py` extracts streamed content via a regex `'"content"\s*:\s*"([^"]*)"'` which will break on escaped quotes, multi-line content, or SSE formatting differences.

11. **Thread safety concerns** — `run_consolidation_background()` runs consolidation in a daemon thread that mutates `EpisodicStore` and `SemanticGraph` — both are plain Python objects with no locking. Concurrent reads during propagation or injection could see inconsistent state.

12. **No embedding cache** — `embeddings_cache_dir` path is configured but never used. Every text is re-encoded on every call.

### Minor

13. **`train_gate_step` imported but not defined in shown code** — The trainer imports `from memory.gate_network import train_gate_step` but this function wasn't visible in the file read. May be at the bottom of the file or missing.

14. **`stream_conversations` referenced in `train_offline.py`** — Imported from `data.conversation_loader` but not visible in the file read; may be at the bottom of the file.

15. **Offline training disables GAT** — The GAT flag is commented out in `_enable_neural_modules()` in `train_offline.py`, so it's never trained offline even though the training step exists.

16. **No validation/test split** — Offline training runs through all data sequentially; no held-out set for loss monitoring.

17. **`ensure_data_dirs()` only called in `BrainMemoryPipeline`** — Other entry points (CLI chat, offline training) don't call it; data dirs may not exist.

---

## 7. Dependencies

### Core (required)
| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.1.0 | All neural modules, tensor operations |
| `sentence-transformers` | ≥2.2.2 | Embedding encoder (`all-MiniLM-L6-v2`) |
| `networkx` | ≥3.2 | Semantic knowledge graph |
| `fastapi` | ≥0.104.0 | Proxy server HTTP API |
| `uvicorn` | ≥0.24.0 | ASGI server for FastAPI |
| `pydantic` | ≥2.5.0 | Data models (nodes, edges, episodes, settings) |
| `pydantic-settings` | ≥2.1.0 | Environment-based configuration |
| `httpx` | ≥0.25.0 | Async HTTP client for LLM API calls |
| `python-dotenv` | ≥1.0.0 | `.env` file loading |
| `numpy` | ≥1.26.0 | Embedding BLOBs, clustering math |
| `scipy` | ≥1.11.0 | Agglomerative clustering in consolidation |
| `scikit-learn` | ≥1.3.0 | Listed as dependency (used in experiments/eval) |

### Optional extras
| Extra | Libraries | Purpose |
|-------|-----------|---------|
| `neural` | `torch-geometric ≥2.5.0` | GATv2Conv (falls back to manual impl without it) |
| `data` | `datasets ≥2.14.0`, `huggingface-hub ≥0.17.0` | Loading UltraChat/OASST2 training data |
| `nlp` | `spacy ≥3.7.0` | Optional NER upgrade from regex |
| `viz` | `pyvis`, `matplotlib`, `plotly`, `streamlit`, `gradio` | Dashboards and visualization |
| `dev` | `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy` | Testing and linting |

### Implicit (via torch/sentence-transformers)
| Library | Purpose |
|---------|---------|
| `transformers` | Underlying HuggingFace model loading |
| `tokenizers` | Fast tokenization |
| `huggingface-hub` | Model downloading |

### Standard Library (notable usage)
| Module | Purpose |
|--------|---------|
| `sqlite3` | Episodic persistence |
| `threading` | Lazy-load locking, background consolidation |
| `asyncio` | Consolidation event loop in background thread |
| `json` | Serialization everywhere |
| `re` | Entity extraction, emphasis detection, intent detection, topic tagging |

---

## Summary Table

| Module | Files | Status | Trains? | Used at Inference? |
|--------|-------|--------|---------|-------------------|
| Encoder | `encoder.py` | ✅ Working | No (frozen) | ✅ Yes |
| Pattern Separator | `pattern_separation.py` | ✅ Working | ✅ Yes | ✅ Yes (when flag on) |
| Working Memory (GRU) | `working_memory.py` | ✅ Working | ❌ No (never trained) | ✅ Yes (default) |
| Working Memory (Transformer) | `neural_working_memory.py` | ✅ Working | ✅ Yes | ✅ Yes (when flag on) |
| Salience Scorer | `salience.py` | ✅ Working | ❌ No | ✅ Yes |
| Dopaminergic Gate | `gate_network.py` | ✅ Working | ✅ Yes | ✅ Yes (when flag on) |
| Hopfield Memory | `hopfield_memory.py` | ✅ Working | ✅ Yes | ✅ Yes (when flag on) |
| Semantic Graph | `semantic.py` | ✅ Working | N/A | ✅ Yes |
| Spreading Activation | `activation.py` | ✅ Working | N/A | ✅ Yes |
| GAT Activation | `neural_activation.py` | ⚠️ Implemented | ✅ Yes | ❌ **Never used** |
| VAE Consolidation | `neural_consolidation.py` | ⚠️ Partial | ✅ Yes | ❌ **Clustering never called** |
| Forgetting Network | `forgetting.py` | ⚠️ Partial | ✅ Yes | ❌ **Never applied** |
| Consolidation (LLM) | `consolidation.py` | ✅ Working | N/A | ✅ Yes |
| Injector | `injector.py` | ✅ Working | N/A | ✅ Yes |
| Observer | `observer.py` | ✅ Working | Via coordinator | ✅ Yes |
| Trainer | `trainer.py` | ✅ Working | ✅ Orchestrates all | N/A |

**Bottom line:** The algorithmic baseline pipeline (encode → salience → store → spread → inject) is fully functional. The neural upgrade modules (pattern sep, gate, Hopfield, Transformer WM) are implemented and integrated. Three neural modules (GAT, VAE consolidation output, forgetting network) **train but never affect inference** — they are dead code at runtime. The GRU encoder's prediction head is never trained, making prediction-error salience decorrelated noise.
