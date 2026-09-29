"""Provider-agnostic interfaces for LLM chat and embeddings.

Everything downstream (pipelines, agent, ingest) depends only on these
abstractions, so switching Ollama <-> Gemini is a config change, not a
code change.

Token usage is a first-class return value because token efficiency is a
scored benchmark metric.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatResult:
    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: dict | None = None


class LLMProvider(ABC):
    """Chat-completion interface."""

    name: str = "base"

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Return a completion for the given messages plus token usage."""
        raise NotImplementedError

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Convenience wrapper around chat() for a single-turn prompt."""
        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage("system", system))
        messages.append(ChatMessage("user", prompt))
        return self.chat(
            messages,
            temperature=temperature,
            json_mode=json_mode,
            max_tokens=max_tokens,
        )


class EmbeddingProvider(ABC):
    """Embedding interface. The SAME provider must index and query."""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
