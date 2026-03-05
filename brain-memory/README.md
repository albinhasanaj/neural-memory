# 🧠 Brain Memory — Brain-Inspired Memory System for LLMs

A research prototype of a **cognitive memory architecture** inspired by the hippocampus-cortex interaction model from neuroscience. Unlike traditional RAG systems that treat memory as a retrieval database, Brain Memory uses **spreading activation** over a semantic knowledge graph, **ACT-R power-law decay** for episodic memories, and **salience-gated encoding** to decide what to remember.

## Architecture

```
                           ┌─────────────────────────────────────────┐
                           │            Open WebUI / Chat UI          │
                           └────────────────┬────────────────────────┘
                                            │ messages
                                            ▼
                    ┌───────────────────────────────────────────────────┐
                    │          Pipeline / FastAPI Proxy Server           │
                    │    POST /v1/chat/completions                       │
                    └───────┬───────────────────────────────┬──────────┘
                            │                               │
                     ┌──────▼──────┐                 ┌──────▼──────┐
                     │   OBSERVE   │                 │   INJECT    │
                     │  (encode,   │                 │  (build     │
                     │   extract,  │◄────────────────│   memory    │
                     │   score)    │  context vector │   context)  │
                     └──────┬──────┘                 └──────▲──────┘
                            │                               │
               ┌────────────┼────────────────────┐          │
               │            │                    │          │
        ┌──────▼──────┐ ┌───▼────────┐  ┌───────▼────┐     │
        │  WORKING    │ │  SALIENCE   │  │  EPISODIC  │     │
        │  MEMORY     │ │  SCORER     │  │  BUFFER    │     │
        │             │ │             │  │            │     │
        │  RingBuffer │ │ novelty     │  │ ACT-R      │     │
        │  GRU encoder│ │ pred error  │  │ decay      │     │
        │  context ──►│ │ emphasis    │  │ archive    │     │
        │  vector     │ │ entity dens │  │            │     │
        └──────┬──────┘ └─────┬──────┘  └─────┬──────┘     │
               │              │                │            │
               │              │  ┌─────────────┘            │
               │              │  │                          │
               │    ┌─────────▼──▼───────┐                  │
               │    │   SPREADING        │                  │
               └───►│   ACTIVATION       ├──────────────────┘
                    │   ENGINE           │
                    │                    │
                    │  1. Seed (cosine)  │
                    │  2. Propagate      │      ┌─────────────────┐
                    │     (sparse mm)    │◄────►│  SEMANTIC GRAPH  │
                    │  3. Inhibit (top-K)│      │  (NetworkX)      │
                    └────────────────────┘      └────────┬────────┘
                                                         │
                                                ┌────────▼────────┐
                                                │  CONSOLIDATION  │
                                                │                 │
                                                │  cluster →      │
                                                │  LLM extract →  │
                                                │  upsert graph   │
                                                └─────────────────┘
```

### Data Flow (per conversation turn)

1. **Observe** — The Memory Observer encodes the turn with `all-MiniLM-L6-v2`, extracts entities/topics, and updates Working Memory.
2. **Score** — The Salience Scorer combines novelty, prediction error, emphasis, and entity density to decide if the turn is worth remembering (threshold: 0.3).
3. **Store** — Salient turns are written to the Episodic Buffer with ACT-R activation tracking.
4. **Activate** — The Spreading Activation Engine seeds the semantic graph from the context vector, propagates activation through edges, and applies lateral inhibition.
5. **Inject** — Activated semantic facts and relevant episodic memories are formatted and prepended to the LLM prompt.
6. **Consolidate** — Periodically, high-salience episodes are clustered and facts are extracted via LLM into the semantic graph.

## Quick Start

### 1. Clone & Install

```bash
git clone <repo-url> brain-memory
cd brain-memory

# Using uv (recommended)
uv venv
uv pip install -e ".[all]"

# Or using pip
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[all]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Seed Sample Data

```bash
brain-seed
# or: python scripts/seed_memory.py
```

### 4. Run the Proxy Server

```bash
brain-proxy
# Starts on http://localhost:8800
```

### 5. Connect a Chat UI

**Option A: Open WebUI (Docker)**
```bash
bash scripts/setup_openwebui.sh
# Open http://localhost:8080
```

**Option B: Any OpenAI-compatible UI**
Point your chat client at `http://localhost:8800/v1` as the API base URL.

### 6. Launch the Dashboard

```bash
streamlit run visualization/dashboard.py
# or: brain-dashboard
```

## Running Tests

```bash
pytest                    # run all tests
pytest -x                 # stop on first failure
pytest -k "test_working"  # run working memory tests only
pytest --cov=memory       # with coverage
```

## Experiments

```bash
python experiments/run_baseline.py           # Exp 1: No memory
python experiments/run_episodic_only.py      # Exp 2: Episodic only
python experiments/run_semantic_only.py       # Exp 3: Semantic only
python experiments/run_activation_vs_knn.py   # Exp 4: SA vs k-NN
python experiments/run_full_system.py         # Exp 5: Full system
python experiments/run_forgetting.py          # Exp 6: Belief updates
```

## Key Algorithms

### ACT-R Activation Decay
```
Activation(i, t) = ln( Σ (t - t_j)^(-0.5) )
```
Memories below threshold (-3.0) are archived, not deleted.

### Spreading Activation
```
1. Seed: cosine_sim(node_embedding, context_vector) > 0.5
2. Propagate: activation += decay × (adj^T @ activation)  [sparse mm]
3. Inhibit: keep top-K activated nodes
```

### Salience Scoring
```
score = 0.35 × novelty + 0.30 × prediction_error + 0.20 × emphasis + 0.15 × entity_density
```

## Project Structure

```
brain-memory/
├── config/settings.py          — Pydantic Settings (all tuneable params)
├── memory/                     — Core memory system
│   ├── observer.py             — Entry point for every turn
│   ├── working_memory.py       — Ring buffer + GRU encoder
│   ├── episodic.py             — Episodic buffer with ACT-R decay
│   ├── semantic.py             — Knowledge graph (NetworkX)
│   ├── activation.py           — Spreading activation (sparse mm)
│   ├── salience.py             — Importance gating (4 signals)
│   ├── consolidation.py        — Episode → graph fact extraction
│   ├── injector.py             — Memory context formatting
│   └── encoder.py              — Sentence-transformer wrapper
├── pipeline/                   — Chat UI integration
│   ├── memory_pipeline.py      — Open WebUI Pipeline class
│   └── proxy_server.py         — FastAPI proxy (OpenAI-compatible)
├── nlp/                        — Entity extraction + topic tagging
├── storage/                    — SQLite + JSON persistence
├── tests/                      — pytest suite (8 test files)
├── experiments/                — Ablation study scripts
├── visualization/              — Dashboard + graph/timeline viz
└── scripts/                    — Setup, seed, export utilities
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Context encoder | PyTorch GRU |
| Knowledge graph | NetworkX DiGraph |
| Spreading activation | PyTorch sparse matrix ops |
| Episodic persistence | SQLite |
| API server | FastAPI + uvicorn |
| Data models | Pydantic v2 |
| NLP | spaCy (optional) / regex fallback |
| Visualization | Streamlit + pyvis + plotly |
| Testing | pytest |
| Dependencies | uv / pip |

## License

MIT
