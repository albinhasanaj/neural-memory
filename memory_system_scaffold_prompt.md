# Prompt: Scaffold a Brain-Inspired Memory System for LLMs

You are building a research prototype of a brain-inspired memory system that operates alongside an LLM. This is NOT a RAG system. It is a cognitive memory architecture inspired by neuroscience — specifically the hippocampus-cortex interaction model — where memories activate automatically via spreading activation rather than being queried like a database.

## Your Task

Scaffold the full project: folder structure, configuration files, all Python modules with docstrings and type hints, a working dev environment, and integration with Open WebUI via its Pipelines feature. Use regular `.py` files throughout — no Jupyter notebooks, no `.ipynb` files whatsoever.

Every module should have real, runnable starter code with the core data structures and function signatures implemented. Not just empty files with `pass` — write the actual foundational logic so the system can be iterated on immediately.

## Tech Stack

- **Python 3.11+**
- **PyTorch** — for embeddings, the working memory GRU encoder, salience scoring MLP, and sparse matrix spreading activation
- **sentence-transformers** — for encoding conversation turns into embeddings (use `all-MiniLM-L6-v2`)
- **NetworkX** — for the semantic knowledge graph
- **SQLite** — for episodic memory persistence (with JSON columns for embeddings/metadata)
- **FastAPI** — for the memory system proxy server that sits between Open WebUI and the LLM API
- **Open WebUI Pipelines** — as the chat interface integration point
- **Pydantic** — for all data models
- **pytest** — for testing
- **uv** or **poetry** — for dependency management (prefer uv)

## Project Structure

Create exactly this folder structure:

```
brain-memory/
├── pyproject.toml                    # Project config, dependencies, scripts
├── README.md                         # Project overview, setup instructions, architecture diagram (ASCII)
├── .env.example                      # Template for API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)
├── .gitignore
│
├── config/
│   └── settings.py                   # Pydantic Settings class — all configurable params:
│                                     #   decay_rate, activation_threshold, salience_threshold,
│                                     #   working_memory_capacity, consolidation_interval,
│                                     #   spreading_activation_iterations, lateral_inhibition_k,
│                                     #   embedding_model_name, llm_provider, llm_model, etc.
│
├── memory/                           # Core memory system — this is the brain
│   ├── __init__.py
│   ├── observer.py                   # Memory Observer — entry point for every conversation turn
│   │                                 #   - Encodes turns into embeddings
│   │                                 #   - Extracts entities and topic tags
│   │                                 #   - Computes salience score
│   │                                 #   - Routes high-salience turns to episodic buffer
│   │
│   ├── working_memory.py             # Working Memory module
│   │                                 #   - RingBuffer class (capacity N, configurable)
│   │                                 #   - GRU encoder (PyTorch nn.Module) that encodes buffer → context vector
│   │                                 #   - predict_next_embedding() for prediction error calculation
│   │                                 #   - update() method that appends turn and returns new context vector
│   │
│   ├── episodic.py                   # Episodic Memory Buffer
│   │                                 #   - EpisodicEntry Pydantic model with fields:
│   │                                 #     id, timestamp, speaker, raw_text, embedding, entities,
│   │                                 #     topics, salience, activation, recall_times, links
│   │                                 #   - EpisodicStore class backed by SQLite
│   │                                 #   - compute_activation() using ACT-R power-law decay:
│   │                                 #     Activation(i,t) = ln( Σ(t - t_j)^(-d) )
│   │                                 #   - Methods: add(), retrieve_by_ids(), get_by_salience(),
│   │                                 #     apply_decay(), archive_below_threshold()
│   │
│   ├── semantic.py                   # Semantic Knowledge Graph
│   │                                 #   - SemanticNode and SemanticEdge Pydantic models
│   │                                 #   - SemanticGraph class wrapping NetworkX DiGraph
│   │                                 #   - Nodes: entity/concept with embedding, type, activation
│   │                                 #   - Edges: typed relations with weight, confidence, evidence list
│   │                                 #   - Methods: upsert_node(), upsert_edge(), get_neighbors(),
│   │                                 #     get_subgraph(), save_to_json(), load_from_json()
│   │
│   ├── activation.py                 # Spreading Activation Engine
│   │                                 #   - seed_nodes() — find nodes similar to context vector via cosine sim
│   │                                 #   - propagate() — iterative spreading activation:
│   │                                 #     for each iteration:
│   │                                 #       for each active node above threshold:
│   │                                 #         spread activation to neighbors proportional to edge weight × decay
│   │                                 #   - lateral_inhibition() — keep only top-K activated nodes
│   │                                 #   - activate() — full pipeline: seed → propagate → inhibit → return results
│   │                                 #   - Implement as sparse matrix multiplication (torch.sparse) for GPU support
│   │
│   ├── salience.py                   # Salience Detection / Importance Gating
│   │                                 #   - compute_novelty() — min cosine distance from known semantic nodes
│   │                                 #   - compute_prediction_error() — distance from working memory prediction
│   │                                 #   - detect_emphasis() — regex/heuristic detection of:
│   │                                 #     caps, exclamation, explicit memory requests ("remember this"),
│   │                                 #     corrections, repetition, questions
│   │                                 #   - compute_entity_density()
│   │                                 #   - SalienceScorer — PyTorch MLP that combines all signals
│   │                                 #     (initially use weighted sum, later train the MLP)
│   │                                 #   - score() method returning float 0-1
│   │
│   ├── consolidation.py              # Memory Consolidation Process
│   │                                 #   - select_candidates() — high-salience episodes not yet consolidated
│   │                                 #   - cluster_episodes() — agglomerative clustering on embeddings
│   │                                 #   - extract_fact() — structured LLM call to extract typed relation:
│   │                                 #     input: list of episode texts
│   │                                 #     output: {"subject": str, "relation": str, "object": str, "confidence": float}
│   │                                 #   - consolidate() — full pipeline: select → cluster → extract → upsert to graph
│   │                                 #   - decay_consolidated() — reduce salience of episodes after consolidation
│   │                                 #   - Should run as background task (asyncio or threading)
│   │
│   ├── injector.py                   # Memory Context Injector
│   │                                 #   - format_semantic_memories() — render facts as natural language
│   │                                 #   - format_episodic_memories() — render episodes with timestamps
│   │                                 #   - build_memory_context() — assemble the full [MEMORY CONTEXT] block
│   │                                 #   - inject() — prepend memory context to the LLM messages list
│   │
│   └── encoder.py                    # Shared embedding encoder
│                                     #   - EmbeddingEncoder class wrapping sentence-transformers
│                                     #   - encode(text) → Tensor[D]
│                                     #   - encode_batch(texts) → Tensor[N, D]
│                                     #   - Lazy-loads model on first use
│                                     #   - Singleton pattern so model is loaded once
│
├── pipeline/                         # Open WebUI Pipeline integration
│   ├── __init__.py
│   ├── memory_pipeline.py            # Open WebUI Pipeline class
│   │                                 #   - Implements the Pipeline protocol:
│   │                                 #     pipe() method that receives messages and returns modified messages
│   │                                 #   - Pre-processing: observe input → activate memories → inject context
│   │                                 #   - Post-processing: observe LLM response → encode if salient
│   │                                 #   - This is the main integration point with Open WebUI
│   │
│   └── proxy_server.py               # Standalone FastAPI proxy server (alternative to Pipeline)
│                                     #   - POST /v1/chat/completions — intercepts OpenAI-format requests
│                                     #   - Runs memory observation + activation + injection
│                                     #   - Forwards modified request to actual LLM API
│                                     #   - Observes response and encodes if salient
│                                     #   - Can be pointed at by ANY OpenAI-compatible chat UI
│
├── nlp/                              # NLP utilities
│   ├── __init__.py
│   ├── entity_extractor.py           # Named entity extraction (use spaCy small model or regex fallback)
│   └── topic_tagger.py               # Topic tagging (keyword-based initially, upgradeable to classifier)
│
├── storage/                          # Persistence layer
│   ├── __init__.py
│   ├── sqlite_store.py               # SQLite backend for episodic memory
│   │                                 #   - Schema: episodes table with columns for all EpisodicEntry fields
│   │                                 #   - Embeddings stored as BLOB (numpy tobytes/frombytes)
│   │                                 #   - JSON columns for entities, topics, recall_times, links
│   │                                 #   - Full CRUD + query methods
│   │
│   └── graph_store.py                # Persistence for semantic graph
│                                     #   - save_graph(graph, path) — serialize NetworkX to JSON
│                                     #   - load_graph(path) → SemanticGraph
│                                     #   - Embeddings serialized as base64-encoded numpy arrays
│
├── tests/                            # pytest test suite
│   ├── __init__.py
│   ├── conftest.py                   # Shared fixtures: sample episodes, sample graph, mock encoder
│   ├── test_working_memory.py        # Test ring buffer, GRU encoding, context vector shape
│   ├── test_episodic.py              # Test add/retrieve/decay/archive lifecycle
│   ├── test_semantic.py              # Test graph CRUD, edge upsert, subgraph extraction
│   ├── test_activation.py            # Test seeding, propagation, lateral inhibition, known graph patterns
│   ├── test_salience.py              # Test individual signals and combined scoring
│   ├── test_consolidation.py         # Test clustering and fact extraction pipeline
│   ├── test_injector.py              # Test memory context formatting
│   └── test_integration.py           # End-to-end: input turn → observe → activate → inject → verify output
│
├── experiments/                      # Research experiment scripts (regular .py, NOT notebooks)
│   ├── __init__.py
│   ├── run_baseline.py               # Experiment 1: LLM with no memory system
│   ├── run_episodic_only.py          # Experiment 2: Episodic buffer + injection only
│   ├── run_semantic_only.py          # Experiment 3: Semantic graph only
│   ├── run_activation_vs_knn.py      # Experiment 4: Spreading activation vs k-NN similarity search
│   ├── run_full_system.py            # Experiment 5: All components enabled
│   ├── run_forgetting.py             # Experiment 6: Introduce contradictions, test belief updates
│   └── eval_metrics.py               # Evaluation metrics:
│                                     #   - recall_accuracy: does the system remember what it should?
│                                     #   - consistency_score: does the system contradict itself?
│                                     #   - personalization_score: does the system use memories appropriately?
│                                     #   - forgetting_accuracy: does the system forget what it should?
│                                     #   - activation_precision: are activated memories actually relevant?
│
├── visualization/                    # Tools for inspecting memory state
│   ├── __init__.py
│   ├── graph_viz.py                  # Render semantic graph as interactive HTML (use pyvis or d3 export)
│   ├── activation_map.py             # Visualize which nodes activated for a given turn
│   ├── memory_timeline.py            # Plot episodic memories on a timeline with salience coloring
│   └── dashboard.py                  # Streamlit or Gradio dashboard combining all visualizations
│                                     #   - Live view of working memory contents
│                                     #   - Semantic graph explorer
│                                     #   - Activation trace for each conversation turn
│                                     #   - Episodic memory browser with search
│                                     #   - Consolidation log
│
├── scripts/                          # Utility scripts
│   ├── setup_openwebui.sh            # Download and configure Open WebUI with Docker
│   ├── seed_memory.py                # Pre-populate memory with sample data for testing
│   └── export_memory.py              # Export full memory state (episodes + graph) to JSON for analysis
│
└── data/                             # Persistent data directory (gitignored except structure)
    ├── .gitkeep
    ├── episodic.db                    # SQLite database (created at runtime)
    ├── semantic_graph.json            # Serialized knowledge graph (created at runtime)
    └── embeddings_cache/             # Cached embeddings to avoid recomputation
```

## Key Implementation Details

### ACT-R Activation Decay
Every episodic memory has an activation level that decays via power law:
```python
activation = math.log(sum((t_now - t_j) ** (-0.5) for t_j in recall_history) + 1e-9)
```
Memories below the retrieval threshold (-3.0) are archived — not deleted, just excluded from the active graph.

### Spreading Activation Algorithm
```
1. Seed: find nodes whose embeddings are cosine-similar (>0.5) to the current context vector
2. Propagate: for max_iter iterations, spread activation to neighbors (activation × edge_weight × decay_factor)
3. Inhibit: keep only top-K nodes (lateral inhibition)
4. Return: list of (node_id, activation_strength) tuples
```
Implement as sparse matrix-vector multiplication for GPU acceleration.

### Salience Scoring
Weighted combination of four signals:
- 0.35 × novelty (cosine distance from nearest known concept)
- 0.30 × prediction_error (deviation from expected next embedding)
- 0.20 × emphasis (user behavioral signals: caps, exclamation, repetition, explicit requests)
- 0.15 × entity_density (named entities per word)

Only turns scoring above 0.3 are written to episodic memory.

### Consolidation Pipeline
Runs every N turns or on session end:
1. Select high-salience unconsolidated episodes
2. Cluster by embedding similarity (agglomerative, threshold=0.4)
3. For clusters with 2+ episodes, call a structured LLM prompt to extract a typed fact
4. Upsert fact into semantic knowledge graph
5. Decay salience of consolidated episodes by 0.6×

### Memory Context Injection Format
```
[MEMORY CONTEXT — Automatically activated based on current conversation]

Semantic facts:
  - User dislikes WordPress  [confidence: 0.85, 3 episodes]
  - User prefers Python over JavaScript  [confidence: 0.72, 2 episodes]

Episodic memories:
  - [2026-02-14] User had a bad experience with a WordPress plugin corrupting their database.
  - [2026-02-28] User asked about migrating WordPress to a static site generator.

[END MEMORY CONTEXT]
```

### Open WebUI Pipeline Integration
The Pipeline class should implement:
```python
class BrainMemoryPipeline:
    def pipe(self, body: dict) -> dict:
        # 1. Extract the latest user message
        # 2. Run memory observer (encode, extract entities, score salience)
        # 3. Update working memory
        # 4. If salient, write to episodic buffer
        # 5. Run spreading activation engine
        # 6. Build memory context from activated memories
        # 7. Inject memory context into the messages list
        # 8. Return modified body
```

### FastAPI Proxy Server (Alternative Integration)
For chat interfaces that don't support Pipelines, provide a proxy server at `POST /v1/chat/completions` that:
1. Intercepts the OpenAI-format request
2. Runs the full memory pipeline (observe → activate → inject)
3. Forwards to the real LLM API
4. Observes the response
5. Returns the response to the client

Any OpenAI-compatible chat UI can point at this proxy.

## What NOT to Do

- Do NOT use Jupyter notebooks or `.ipynb` files anywhere
- Do NOT use or recommend existing AI memory libraries (LangChain memory, LlamaIndex, MemGPT, Mem0, etc.)
- Do NOT implement traditional RAG (no vector database similarity search as the primary retrieval mechanism)
- Do NOT build a chat UI — use Open WebUI or the FastAPI proxy approach
- Do NOT use placeholder `pass` statements — write real starter logic in every module
- Do NOT skip type hints or docstrings

## What TO Do

- Write real, runnable code in every module — data structures, core algorithms, function signatures with implementations
- Make every component independently testable
- Use dependency injection so components can be swapped (e.g., replace the activation engine with k-NN for ablation experiments)
- Include a `pyproject.toml` with all dependencies and script entry points
- Write the README with an ASCII architecture diagram showing the data flow
- Create the test suite with meaningful test cases (not just "test passes")
- Make the visualization dashboard actually work — this is critical for research introspection

Start by scaffolding the entire structure, then implement each module with working starter code. Prioritize the core memory loop: `observer → working_memory → episodic → activation → injector` — this should be runnable end-to-end as the first milestone.
