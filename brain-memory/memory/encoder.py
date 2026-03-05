"""
Shared embedding encoder — singleton wrapper around sentence-transformers.

The model is lazy-loaded on first use so importing this module is cheap.
A module-level ``get_encoder()`` function provides the singleton instance.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from config.settings import settings


class EmbeddingEncoder:
    """Encode text into dense vectors using a sentence-transformer model.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g. ``"all-MiniLM-L6-v2"``.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        If *None* the best available device is auto-detected.
    cache_size:
        Maximum number of embeddings to cache in memory.
    """

    def __init__(
        self,
        model_name: str = settings.embedding_model_name,
        device: str | None = None,
        cache_size: int = 4096,
    ) -> None:
        self._model_name = model_name
        self._device = device or settings.resolved_device
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()
        self._cache: dict[str, Tensor] = {}
        self._cache_order: list[str] = []
        self._cache_size = cache_size

    # ── lazy loading ────────────────────────────────────────────────

    @property
    def model(self) -> SentenceTransformer:
        """Return the sentence-transformer model, loading it on first access."""
        if self._model is None:
            with self._lock:
                if self._model is None:  # double-check after acquiring lock
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(
                        self._model_name, device=self._device
                    )
        return self._model

    @property
    def dim(self) -> int:
        """Embedding dimensionality (e.g. 384 for all-MiniLM-L6-v2)."""
        return self.model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    # ── public API ──────────────────────────────────────────────────

    def encode(self, text: str) -> Tensor:
        """Encode a single string → ``Tensor[D]``.  Results are LRU-cached."""
        if text in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(text)
            self._cache_order.append(text)
            return self._cache[text]

        vec = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        vec = vec.to(self._device)  # type: ignore[union-attr]

        # Evict oldest entry if cache is full
        if len(self._cache) >= self._cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        self._cache[text] = vec
        self._cache_order.append(text)
        return vec

    def encode_batch(self, texts: list[str]) -> Tensor:
        """Encode a batch of strings → ``Tensor[N, D]``."""
        vecs = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return vecs.to(self._device)  # type: ignore[union-attr]


# ── module-level singleton ──────────────────────────────────────────

_encoder_instance: EmbeddingEncoder | None = None
_encoder_lock = threading.Lock()


def get_encoder() -> EmbeddingEncoder:
    """Return the global ``EmbeddingEncoder`` singleton."""
    global _encoder_instance
    if _encoder_instance is None:
        with _encoder_lock:
            if _encoder_instance is None:
                _encoder_instance = EmbeddingEncoder()
    return _encoder_instance
