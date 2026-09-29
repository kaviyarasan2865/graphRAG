"""Evidence state for an agentic investigation.

Tracks everything the orchestrator has learned so it can decide the next move
and, at the end, synthesize a grounded, cited answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.pipelines.types import AnswerTraceStep
from app.providers.base import TokenUsage


@dataclass
class Evidence:
    """One piece of evidence produced by a tool call."""

    tool: str
    summary: str                       # human/LLM-readable summary
    doc_ids: list[str] = field(default_factory=list)
    payload: dict = field(default_factory=dict)   # structured result


@dataclass
class ToolCallRecord:
    """Per-operation agentic trace (what the guidebook asks to report)."""

    step: int
    tool: str
    args: dict
    reason: str = ""
    seconds: float = 0.0
    tokens: int = 0                    # planner tokens spent choosing this step
    result_summary: str = ""
    n_docs: int = 0

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "tool": self.tool,
            "args": self.args,
            "reason": self.reason,
            "seconds": round(self.seconds, 3),
            "tokens": self.tokens,
            "n_docs": self.n_docs,
            "result": self.result_summary,
        }


@dataclass
class InvestigationState:
    question: str
    qtype_hint: str = ""               # optional classification
    evidence: list[Evidence] = field(default_factory=list)
    trace: list[AnswerTraceStep] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    steps_taken: int = 0
    used_tools: list[str] = field(default_factory=list)

    # richer agentic trace + stopping metadata
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    seen_calls: set[str] = field(default_factory=set)
    stop_reason: str = ""
    strategy_changed: bool = False
    max_steps: int = 8

    def add_evidence(self, ev: Evidence) -> None:
        self.evidence.append(ev)

    def all_doc_ids(self) -> list[str]:
        seen: list[str] = []
        for ev in self.evidence:
            for d in ev.doc_ids:
                if d not in seen:
                    seen.append(d)
        return seen

    def evidence_digest(self, max_chars: int = 4000) -> str:
        """Compact view of collected evidence for the orchestrator prompt."""
        lines = []
        for i, ev in enumerate(self.evidence, 1):
            lines.append(f"E{i} [{ev.tool}]: {ev.summary}")
        text = "\n".join(lines) if lines else "(no evidence yet)"
        return text[:max_chars]

    def add_usage(self, u: TokenUsage) -> None:
        self.usage = self.usage + u
