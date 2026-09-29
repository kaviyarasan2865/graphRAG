"""Runtime fallback LLM: try a primary provider, fall back on failure.

If the primary (e.g. Gemini) raises for any reason — rate limit (429), server
outage (503), network error — the call is transparently retried on the fallback
(e.g. Ollama). This keeps the app answering even when the cloud LLM is
throttled or down, which is exactly what the free Gemini tier needs.
"""

from __future__ import annotations

from .base import ChatMessage, ChatResult, LLMProvider


class FallbackLLM(LLMProvider):
    name = "fallback"

    def __init__(self, primary: LLMProvider, fallback: LLMProvider):
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}->{fallback.name}"
        self.last_used = primary.name

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        try:
            res = self.primary.chat(
                messages, temperature=temperature, json_mode=json_mode, max_tokens=max_tokens
            )
            self.last_used = self.primary.name
            return res
        except Exception:  # noqa: BLE001 - any primary failure -> fall back
            res = self.fallback.chat(
                messages, temperature=temperature, json_mode=json_mode, max_tokens=max_tokens
            )
            self.last_used = self.fallback.name
            return res
