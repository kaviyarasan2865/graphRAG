"""Pipeline 3: Agentic GraphRAG entry point.

Dispatches to one of two interchangeable agent backends (both LangGraph):
  - "graph" (default): the custom StateGraph in app/agent/orchestrator.py, with
    explicit plan/act/synthesize nodes, loop guards, and decisive-stop rules.
  - "react": the prebuilt create_react_agent in app/agent/react_agent.py, using
    the model's native tool-calling.

Both share the same tools and return the same PipelineResult (pipeline=
"agentic") with an agentic_meta trace, so the API/benchmark treat them
uniformly. Select with AGENT_IMPL=graph|react.
"""

from __future__ import annotations

from app.config import settings

from .types import PipelineResult


def run_agentic(
    question: str, *, max_steps: int | None = None, impl: str | None = None
) -> PipelineResult:
    impl = (impl or settings.agent_impl).lower().strip()

    if impl == "react":
        from app.agent.react_agent import investigate_react

        return investigate_react(question, max_steps=max_steps)

    # default: custom LangGraph StateGraph
    from app.agent.orchestrator import investigate

    return investigate(question, max_steps=max_steps)
