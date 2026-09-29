"""Shared LLM answer synthesis with grounding + citations.

All three pipelines converge here so the comparison isolates the *retrieval*
strategy, not the prompt. The synthesizer is told to answer only from the
provided context and to stay terse (the gold answers are short strings).
"""

from __future__ import annotations

from app.providers.base import ChatMessage, TokenUsage
from app.providers.factory import get_llm

SYSTEM = (
    "You are an Olympic-events analyst. Answer ONLY from the provided context, "
    "which is derived from a fixed corpus that is the sole source of truth. "
    "If the context does not contain the answer, reply exactly: I don't know. "
    "Give the shortest possible answer: a name, number, or short phrase. "
    "Do not add explanation unless asked."
)


def synthesize(question: str, context: str, *, max_tokens: int = 256) -> tuple[str, TokenUsage]:
    llm = get_llm()
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer with only the essential value (name/number/phrase)."
    )
    res = llm.chat(
        [ChatMessage("system", SYSTEM), ChatMessage("user", prompt)],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return res.text.strip(), res.usage
