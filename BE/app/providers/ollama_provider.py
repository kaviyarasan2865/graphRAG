"""Ollama-backed LLM and embedding providers (local, default).

Uses the native Ollama REST API:
  - POST /api/chat       -> chat completions (returns prompt_eval_count / eval_count)
  - POST /api/embed      -> embeddings

qwen3 is a "thinking" model; for tool routing we disable thinking and request
raw JSON so downstream parsing is reliable.
"""

from __future__ import annotations

import json

import httpx

from .base import (
    ChatMessage,
    ChatResult,
    EmbeddingProvider,
    LLMProvider,
    TokenUsage,
)

_TIMEOUT = httpx.Timeout(300.0, connect=10.0)


class OllamaLLM(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "think": False,  # qwen3: skip reasoning trace for structured output
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        text = (data.get("message") or {}).get("content", "") or ""
        usage = TokenUsage(
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
        )
        return ChatResult(text=text.strip(), usage=usage, raw=data)


class OllamaEmbeddings(EmbeddingProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, dim: int):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts}
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(f"{self.base_url}/api/embed", json=payload)
            resp.raise_for_status()
            data = resp.json()
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise RuntimeError(f"Ollama embed returned no embeddings: {json.dumps(data)[:200]}")
        return embeddings
