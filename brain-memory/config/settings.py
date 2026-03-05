"""
Brain Memory — Global configuration via Pydantic Settings.

All tuneable parameters live here. Values are loaded from environment
variables first (with a ``BRAIN_`` prefix), then from a ``.env`` file,
and finally from the defaults declared below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Master configuration object for the brain-memory system."""

    model_config = SettingsConfigDict(
        env_prefix="BRAIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM provider ────────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o"
    llm_base_url: str | None = None
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Device ───────────────────────────────────────────────────────
    device: str = Field(
        default="auto",
        description="PyTorch device: 'auto' (detect GPU), 'cuda', 'cpu', 'mps'.",
    )

    # ── Embedding model ─────────────────────────────────────────────
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384  # dimension for all-MiniLM-L6-v2

    @property
    def resolved_device(self) -> str:
        """Return the actual torch device string (resolve 'auto')."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    # ── Working memory ──────────────────────────────────────────────
    working_memory_capacity: int = Field(
        default=8,
        description="Number of recent turns kept in the ring buffer.",
    )
    gru_hidden_dim: int = Field(
        default=256,
        description="Hidden dimension of the GRU context encoder.",
    )

    # ── Episodic memory ─────────────────────────────────────────────
    decay_rate: float = Field(
        default=0.5,
        description="Power-law decay exponent d in ACT-R activation.",
    )
    activation_threshold: float = Field(
        default=-3.0,
        description="Activation below this value → archive the episode.",
    )
    salience_threshold: float = Field(
        default=0.3,
        description="Minimum salience score to commit a turn to episodic memory.",
    )

    # ── Spreading activation ────────────────────────────────────────
    spreading_activation_iterations: int = Field(
        default=3,
        description="Number of propagation hops.",
    )
    activation_decay_factor: float = Field(
        default=0.7,
        description="Per-hop multiplicative decay during propagation.",
    )
    seed_similarity_threshold: float = Field(
        default=0.5,
        description="Min cosine similarity to seed a node.",
    )
    lateral_inhibition_k: int = Field(
        default=10,
        description="Top-K nodes retained after lateral inhibition.",
    )

    # ── Seed channel weights (multi-channel seeding) ────────────────
    # The four channels are blended:  final = Σ (w_i · channel_i).
    # Weights don't need to sum to 1 — they're relative strengths.
    seed_weight_entity: float = Field(
        default=1.0,
        description="Weight for entity-match seeding (names, projects, places).",
    )
    seed_weight_intent: float = Field(
        default=0.9,
        description="Weight for recall-intent cue seeding.",
    )
    seed_weight_working_memory: float = Field(
        default=0.7,
        description="Weight for working-memory focus seeding (what's 'in mind').",
    )
    seed_weight_embedding: float = Field(
        default=0.3,
        description="Weight for embedding-similarity fallback seeding.",
    )
    intent_cue_boost: float = Field(
        default=1.5,
        description="Multiplicative boost applied to nodes matched via intent-cue targets.",
    )

    # ── Salience weights ────────────────────────────────────────────
    salience_novelty_weight: float = 0.35
    salience_prediction_error_weight: float = 0.30
    salience_emphasis_weight: float = 0.20
    salience_entity_density_weight: float = 0.15

    # ── Consolidation ───────────────────────────────────────────────
    consolidation_interval: int = Field(
        default=10,
        description="Run consolidation every N conversation turns.",
    )
    consolidation_cluster_threshold: float = Field(
        default=0.4,
        description="Distance threshold for agglomerative clustering.",
    )
    consolidation_salience_decay: float = Field(
        default=0.6,
        description="Multiplicative decay applied to episode salience after consolidation.",
    )

    # ── Proxy server ────────────────────────────────────────────────
    proxy_host: str = "0.0.0.0"
    proxy_port: int = 8800

    # ── Data paths ──────────────────────────────────────────────────
    data_dir: Path = Path("./data")
    episodic_db_path: Path = Path("./data/episodic.db")
    semantic_graph_path: Path = Path("./data/semantic_graph.json")
    embeddings_cache_dir: Path = Path("./data/embeddings_cache")

    # ── Memory injection ────────────────────────────────────────────
    max_semantic_facts: int = 8
    max_episodic_memories: int = 6

    # ── Neural architecture flags ───────────────────────────────────
    # Toggle between algorithmic (baseline) and neural versions.
    use_gnn_activation: bool = Field(
        default=False,
        description="GAT spreading activation vs algorithmic.",
    )
    use_hopfield_memory: bool = Field(
        default=False,
        description="Hopfield associative store vs direct episodic retrieval.",
    )
    use_vae_consolidation: bool = Field(
        default=False,
        description="VAE consolidation vs clustering + LLM.",
    )
    use_dopaminergic_gate: bool = Field(
        default=False,
        description="Learned gating vs weighted-sum salience.",
    )
    use_pattern_separation: bool = Field(
        default=False,
        description="Sparse autoencoder vs raw embeddings.",
    )
    use_transformer_wm: bool = Field(
        default=False,
        description="Transformer working memory vs GRU.",
    )
    use_learned_forgetting: bool = Field(
        default=False,
        description="Forgetting network vs ACT-R power law.",
    )
    use_salience_mlp: bool = Field(
        default=False,
        description="Use learned MLP salience scorer instead of fixed weighted sum.",
    )

    # ── Forgetting network ──────────────────────────────────────────
    forgetting_threshold: float = Field(
        default=0.1,
        description="Minimum effective activation after forgetting network re-ranking. Episodes below this are dropped from injection.",
    )

    # ── Neural training hyperparameters ─────────────────────────────
    gat_learning_rate: float = 1e-4
    gate_learning_rate: float = 5e-5
    consolidation_vae_lr: float = 3e-4
    forgetting_lr: float = 1e-4
    gru_predictor_lr: float = 1e-4
    pattern_sep_lr: float = 3e-4
    wm_transformer_lr: float = 1e-4

    # ── Phase 4 — Neural injection into local LLM ───────────────
    use_neural_injection: bool = Field(
        default=False,
        description="Master switch for Phase 4 neural injection into local LLM.",
    )
    local_llm_model: str = Field(
        default="Qwen/Qwen2.5-3B-Instruct",
        description="HuggingFace model ID for local LLM (3B default for 11GB VRAM safety).",
    )
    local_llm_injection_layer: int | None = Field(
        default=None,
        description="Transformer layer to inject memory into. None = auto (middle layer).",
    )
    neural_injection_strength: float = Field(
        default=0.3,
        description="How strongly memory affects LLM hidden states.",
    )
    neural_projection_lr: float = Field(
        default=1e-4,
        description="Learning rate for projection layer training.",
    )
    training_replay_buffer_size: int = 500
    gate_exploration_epsilon: float = 0.1
    training_grad_clip: float = 1.0

    # ── Architecture dimensions ─────────────────────────────────────
    gat_hidden_dims: list[int] = Field(default=[256, 128, 64])
    gat_num_heads: int = 4
    hopfield_beta_init: float = 1.0
    hopfield_max_patterns: int = 2048

    # ── Modular Hopfield (Phase 2) ──────────────────────────────────
    use_modular_hopfield: bool = Field(
        default=False,
        description="Use M smaller HopfieldModules + learned MemoryRouter instead of monolithic HippocampalMemory.",
    )
    hopfield_num_modules: int = Field(
        default=32,
        description="Number of Hopfield sub-modules in modular mode.",
    )
    hopfield_patterns_per_module: int = Field(
        default=256,
        description="Max patterns per Hopfield sub-module.",
    )
    hopfield_top_k_write: int = Field(
        default=2,
        description="Number of modules to route each write to.",
    )
    hopfield_top_k_read: int = Field(
        default=3,
        description="Number of modules to query on each read.",
    )
    hopfield_router_lr: float = Field(
        default=1e-4,
        description="Learning rate for the MemoryRouter.",
    )

    # ── Fast-weight memory (Phase 3) ───────────────────────────────
    use_fast_weight_memory: bool = Field(
        default=False,
        description="Use FastWeightModule (Hebbian weight matrices) instead of explicit pattern buffers.",
    )
    fast_weight_write_lr: float = Field(
        default=0.1,
        description="Hebbian write learning rate for fast-weight outer-product updates.",
    )
    fast_weight_hidden_dim: int = Field(
        default=128,
        description="Hidden dimension for fast-weight key/value projections.",
    )
    fast_weight_decay_factor: float = Field(
        default=0.995,
        description="Homeostatic decay rate applied to fast weights periodically.",
    )
    fast_weight_decay_interval: int = Field(
        default=10,
        description="Apply homeostatic decay every N writes.",
    )

    vae_latent_dim: int = 64
    vae_metadata_dim: int = 32
    vae_replay_batch: int = 16
    vae_replay_noise: float = 0.05
    pattern_sep_expansion: int = 2048
    pattern_sep_top_k: int = 50
    wm_transformer_layers: int = 2
    wm_transformer_heads: int = 4
    wm_transformer_ff_dim: int = 512
    forgetting_eval_interval: int = 20
    forgetting_context_dim: int = 64

    def ensure_data_dirs(self) -> None:
        """Create data directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_cache_dir.mkdir(parents=True, exist_ok=True)


# Module-level singleton — import ``settings`` everywhere.
settings = Settings()
