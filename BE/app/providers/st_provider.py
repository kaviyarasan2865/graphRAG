"""sentence-transformers embedding provider (BAAI/bge-m3, 1024-dim).

Runs the embedding model in-process via sentence-transformers, so the app has
no external embedding dependency (no Ollama server, no API rate limits). This
is the deployable default for embeddings.

BAAI/bge-m3 is the same model Ollama's `bge-m3` wraps, producing 1024-dim
vectors in the same space, so it is compatible with a graph already loaded with
Ollama bge-m3 embeddings (no re-embed needed).
"""

from __future__ import annotations

from functools import lru_cache

from .base import EmbeddingProvider


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    # Imported lazily so the heavy torch/sentence-transformers import only
    # happens when this provider is actually used.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class SentenceTransformersEmbeddings(EmbeddingProvider):
    name = "st"

    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self.dim = dim
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # normalize_embeddings=True -> unit vectors, matching cosine metric and
        # Ollama's bge-m3 output convention.
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]
