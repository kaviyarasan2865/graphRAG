"""Gemini-backed LLM and embedding providers (switchable).

Uses the Generative Language REST API with an API key (free tier available at
https://aistudio.google.com/apikey). No SDK dependency: plain httpx so the
provider stays lightweight and mockable.
"""

from __future__ import annotations

import random
import time

import httpx

from .base import (
    ChatMessage,
    ChatResult,
    EmbeddingProvider,
    LLMProvider,
    TokenUsage,
)

_BASE = "https://generativelanguage.googleapis.com/v1beta"
_TIMEOUT = httpx.Timeout(120.0, connect=10.0)

# Transient statuses worth retrying (rate limit + server-side outages).
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 4


class GeminiError(RuntimeError):
    """Gemini API error with the API key scrubbed from the message."""


def _raise_clean(resp: httpx.Response, model: str) -> None:
    """Raise a GeminiError that never leaks the ?key=... query param.

    httpx's default HTTPStatusError includes the full request URL (with the
    API key). We raise our own error with only the status + model instead.
    """
    if resp.is_success:
        return
    body = ""
    try:
        body = resp.text[:200]
    except Exception:  # noqa: BLE001
        pass
    raise GeminiError(f"Gemini {model} request failed: HTTP {resp.status_code}. {body}")


class GeminiLLM(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        self.api_key = api_key
        self.model = model

    def _post_with_retry(self, url: str, payload: dict) -> dict:
        """POST with exponential backoff on transient 429/5xx responses.

        Never lets the API key leak into an exception (see _raise_clean)."""
        with httpx.Client(timeout=_TIMEOUT) as client:
            for attempt in range(_MAX_RETRIES):
                resp = client.post(url, params={"key": self.api_key}, json=payload)
                if resp.status_code in _RETRY_STATUS and attempt < _MAX_RETRIES - 1:
                    time.sleep((2 ** attempt) + random.uniform(0, 0.5))
                    continue
                _raise_clean(resp, self.model)
                return resp.json()
        raise GeminiError(f"Gemini {self.model} request failed after {_MAX_RETRIES} retries")

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        # Gemini takes a system_instruction separately from contents.
        system_parts = [m.content for m in messages if m.role == "system"]
        contents = []
        for m in messages:
            if m.role == "system":
                continue
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.content}]})

        gen_config: dict = {"temperature": temperature}
        if json_mode:
            gen_config["response_mime_type"] = "application/json"
        if max_tokens is not None:
            gen_config["maxOutputTokens"] = max_tokens

        payload: dict = {"contents": contents, "generationConfig": gen_config}
        if system_parts:
            payload["system_instruction"] = {"parts": [{"text": "\n".join(system_parts)}]}

        url = f"{_BASE}/models/{self.model}:generateContent"
        data = self._post_with_retry(url, payload)

        text = ""
        candidates = data.get("candidates") or []
        if candidates:
            parts = (candidates[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts)

        meta = data.get("usageMetadata") or {}
        usage = TokenUsage(
            prompt_tokens=int(meta.get("promptTokenCount", 0) or 0),
            completion_tokens=int(meta.get("candidatesTokenCount", 0) or 0),
        )
        return ChatResult(text=text.strip(), usage=usage, raw=data)


class GeminiEmbeddings(EmbeddingProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str, dim: int):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
        self.api_key = api_key
        self.model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # embedContent is single-text; batch client-side. gemini-embedding-001
        # supports outputDimensionality, so we pin it to self.dim to match the
        # TigerGraph vector schema (default 768).
        out: list[list[float]] = []
        url = f"{_BASE}/models/{self.model}:embedContent"
        with httpx.Client(timeout=_TIMEOUT) as client:
            for t in texts:
                payload = {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": self.dim,
                }
                data = self._post_embed_with_retry(client, url, payload)
                values = (data.get("embedding") or {}).get("values") or []
                out.append(values)
        return out

    def _post_embed_with_retry(self, client, url: str, payload: dict) -> dict:
        for attempt in range(_MAX_RETRIES):
            resp = client.post(url, params={"key": self.api_key}, json=payload)
            if resp.status_code in _RETRY_STATUS and attempt < _MAX_RETRIES - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 0.5))
                continue
            _raise_clean(resp, self.model)
            return resp.json()
        raise GeminiError(f"Gemini {self.model} embed failed after {_MAX_RETRIES} retries")
