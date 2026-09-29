"""Shared result types for retrieval and pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.providers.base import TokenUsage


@dataclass
class RetrievedDoc:
    doc_id: str
    title: str
    text: str
    url: str = ""
    score: float = 0.0          # similarity (higher = closer); 1 - distance
    source: str = "vector"      # "vector" | "graph"


@dataclass
class EventFacts:
    """Structured facts about one Olympic event (from the graph)."""

    doc_id: str
    title: str
    event_name: str = ""
    year: int | None = None
    season: str = ""
    venue: str = ""
    event_date: str = ""
    competitors: int | None = None
    nations: int | None = None
    sport: str = ""
    medalists: list[dict] = field(default_factory=list)  # {rank, name, noc}

    def to_context(self) -> str:
        lines = [f"Event: {self.title}"]
        if self.year:
            lines.append(f"Games: {self.year} {self.season}")
        if self.sport:
            lines.append(f"Sport: {self.sport}")
        if self.venue:
            lines.append(f"Venue: {self.venue}")
        if self.event_date:
            lines.append(f"Date: {self.event_date}")
        if self.competitors is not None and self.competitors >= 0:
            lines.append(f"Competitors: {self.competitors}")
        if self.nations is not None and self.nations >= 0:
            lines.append(f"Nations: {self.nations}")
        for m in self.medalists:
            lines.append(f"{m['rank'].capitalize()}: {m['name']} ({m.get('noc', '')})")
        return "\n".join(lines)


@dataclass
class AnswerTraceStep:
    action: str
    detail: str
    result_summary: str = ""


@dataclass
class PipelineResult:
    question: str
    pipeline: str                       # "rag" | "graphrag" | "agentic"
    answer: str
    citations: list[str] = field(default_factory=list)   # doc_ids used
    retrieved_doc_ids: list[str] = field(default_factory=list)
    trace: list[AnswerTraceStep] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    # Agentic-only metadata (steps, tools, stop reason, per-op trace). Empty
    # for RAG/GraphRAG. Populated by the orchestrator.
    agentic_meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "question": self.question,
            "pipeline": self.pipeline,
            "answer": self.answer,
            "citations": self.citations,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "trace": [
                {"action": s.action, "detail": s.detail, "result": s.result_summary}
                for s in self.trace
            ],
            "usage": self.usage.as_dict(),
        }
        if self.agentic_meta:
            d["agentic_meta"] = self.agentic_meta
        return d
