"""Factory that builds LLM / embedding providers from settings.

Cached so the whole app shares one client instance per provider.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import settings

from .base import EmbeddingProvider, LLMProvider
from .fallback import FallbackLLM
from .gemini_provider import GeminiEmbeddings, GeminiLLM
from .ollama_provider import OllamaEmbeddings, OllamaLLM
from .st_provider import SentenceTransformersEmbeddings


def _ollama_llm() -> LLMProvider:
    return OllamaLLM(settings.ollama_base_url, settings.ollama_llm_model)


@lru_cache(maxsize=1)
def get_llm() -> LLMProvider:
    """Build the LLM.

    - LLM_PROVIDER=ollama: force Ollama.
    - LLM_PROVIDER=gemini (default):
        * with a key  -> Gemini primary, with AUTOMATIC RUNTIME FALLBACK to
          Ollama on any Gemini failure (429 rate limit, 503 outage, etc.).
        * without a key -> Ollama only.
    """
    provider = settings.llm_provider.lower().strip()
    if provider == "ollama":
        return _ollama_llm()
    if provider == "gemini":
        if settings.gemini_api_key:
            gemini = GeminiLLM(settings.gemini_api_key, settings.gemini_llm_model)
            return FallbackLLM(gemini, _ollama_llm())
        return _ollama_llm()
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


def active_llm_name() -> str:
    """Which LLM will actually be used, accounting for fallback."""
    provider = settings.llm_provider.lower().strip()
    if provider == "gemini":
        if not settings.gemini_api_key:
            return "ollama (fallback: no GEMINI_API_KEY)"
        return f"{settings.gemini_llm_model} (fallback: ollama {settings.ollama_llm_model})"
    return f"ollama {settings.ollama_llm_model}"


@lru_cache(maxsize=1)
def get_embeddings() -> EmbeddingProvider:
    provider = settings.embedding_provider.lower().strip()
    if provider in ("st", "sentence-transformers", "sentencetransformers"):
        return SentenceTransformersEmbeddings(settings.st_embed_model, settings.embed_dim)
    if provider == "ollama":
        return OllamaEmbeddings(
            settings.ollama_base_url, settings.ollama_embed_model, settings.embed_dim
        )
    if provider == "gemini":
        return GeminiEmbeddings(
            settings.gemini_api_key, settings.gemini_embed_model, settings.embed_dim
        )
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")
