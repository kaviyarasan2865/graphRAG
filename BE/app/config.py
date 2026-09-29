"""Central configuration loaded from environment / .env.

All tunables live here so nothing is hardwired to a provider or a host.
Switch providers by changing LLM_PROVIDER / EMBEDDING_PROVIDER.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# BE/ directory (parent of app/)
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider selection.
    # LLM: Gemini (falls back to Ollama only if no GEMINI_API_KEY).
    # Embeddings: sentence-transformers BAAI/bge-m3 in-process (deployable, no
    # server, no rate limits, 1024-dim; same space as the loaded graph).
    llm_provider: str = "gemini"
    embedding_provider: str = "st"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "qwen3:14b"
    ollama_embed_model: str = "bge-m3:latest"

    # Gemini
    gemini_api_key: str = ""
    gemini_llm_model: str = "gemini-3.7-flash"
    gemini_embed_model: str = "gemini-embedding-001"

    # sentence-transformers embedding model (in-process).
    st_embed_model: str = "BAAI/bge-m3"

    # Embeddings. bge-m3 = 1024-dim (matches the loaded graph vector schema).
    embed_dim: int = 1024

    # TigerGraph
    tg_host: str = "http://localhost"
    tg_graph: str = "OlympicsKG"
    tg_username: str = "tigergraph"
    tg_password: str = ""
    tg_secret: str = ""
    tg_restpp_port: int = 443
    tg_gsql_port: int = 443

    # Paths (relative to BE/ unless absolute)
    corpus_path: str = "../dataset/corpus/corpus.jsonl"
    eval_public_path: str = "../dataset/questions/eval_public.jsonl"
    eval_hidden_path: str = "../dataset/questions/eval_hidden.jsonl"

    # Retrieval knobs
    rag_top_k: int = 8
    graphrag_top_k: int = 8
    agent_max_steps: int = 8

    # Agentic backend: "graph" (custom LangGraph StateGraph, default) or
    # "react" (LangGraph prebuilt create_react_agent with native tool-calling).
    agent_impl: str = "graph"

    def resolve(self, p: str) -> Path:
        """Resolve a possibly-relative path against BE/."""
        path = Path(p)
        return path if path.is_absolute() else (BASE_DIR / path).resolve()

    @property
    def corpus_file(self) -> Path:
        return self.resolve(self.corpus_path)

    @property
    def eval_public_file(self) -> Path:
        return self.resolve(self.eval_public_path)

    @property
    def eval_hidden_file(self) -> Path:
        return self.resolve(self.eval_hidden_path)


settings = Settings()
