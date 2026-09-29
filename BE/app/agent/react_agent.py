"""Prebuilt agent backend: LangGraph `create_react_agent` + native tool-calling.

This is an alternative to the custom StateGraph in orchestrator.py, selectable
via AGENT_IMPL=react. It binds the same underlying tools (app.agent.tools) as
LangChain tools and lets the model drive them through Ollama's native
tool-calling API.

It returns the same PipelineResult (pipeline="agentic") and reconstructs the
agentic_meta trace (steps, tools used, per-op summary, stop reason) from the
message history so the API/dashboard/benchmark treat both backends uniformly.
"""

from __future__ import annotations

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from app.config import settings
from app.pipelines.types import AnswerTraceStep, PipelineResult
from app.providers.base import TokenUsage

from . import tools as T

# --- LangChain tool wrappers over the shared Evidence tools ---------------
# Each returns a string (what the ReAct loop feeds back to the model) but we
# also stash the last Evidence so we can recover doc_ids + payload afterwards.

_LAST_EVIDENCE: list = []  # collected per-invocation; reset in investigate_react


def _record(ev):
    _LAST_EVIDENCE.append(ev)
    return ev.summary


@tool
def vector_search(query: str, k: int = 8) -> str:
    """Semantic search over all documents. Returns doc_id:title pairs. Use to
    find seed events or resolve names when exact fields are unknown."""
    return _record(T.tool_vector_search(query, k))


@tool
def event_lookup(doc_id: str) -> str:
    """Get full structured facts (medalists, venue, date, competitors) for one
    event by its doc_id (like 'Q303623')."""
    return _record(T.tool_event_lookup(doc_id))


@tool
def events_by_games_sport(year: int, season: str = "", sport: str = "") -> str:
    """List every event at a Games, optionally filtered by season
    ('Summer'/'Winter') and sport (e.g. 'Athletics'). Use before counting."""
    return _record(T.tool_events_by_games_sport(year, season, sport))


@tool
def count_over_threshold(year: int, threshold: int, metric: str = "competitors",
                         season: str = "", sport: str = "") -> str:
    """Count events at a Games whose metric ('competitors' or 'nations')
    exceeds a threshold. Best for 'how many ... more than N' questions.
    Always pass sport when the question names one."""
    return _record(T.tool_count_over_threshold(year, metric, threshold, season, sport))


@tool
def superlative(year: int, metric: str = "competitors", mode: str = "max",
                season: str = "", sport: str = "") -> str:
    """Find the event with the highest/lowest metric at a Games. mode is 'max'
    or 'min'. Always pass sport when the question names one (e.g. 'Athletics')."""
    return _record(T.tool_superlative(year, metric, mode, season, sport))


@tool
def medalists_at_venue_date(venue: str, date_fragment: str = "", rank: str = "gold") -> str:
    """Find medalists for events at a venue (optionally on a date fragment).
    Best for 'who won at VENUE on DATE' multi-hop questions."""
    return _record(T.tool_medalists_at_venue_date(venue, date_fragment, rank))


@tool
def temporal_neighbor(doc_id: str, direction: str = "prev") -> str:
    """Jump to the same event at the previous/next Games. direction is 'prev'
    or 'next'. doc_id MUST come from a prior vector_search result."""
    return _record(T.tool_temporal_neighbor(doc_id, direction))


_TOOLS = [vector_search, event_lookup, events_by_games_sport, count_over_threshold,
          superlative, medalists_at_venue_date, temporal_neighbor]

_SYSTEM = (
    "You investigate Olympic-events questions using the provided tools over a "
    "knowledge graph. Prefer structured tools (count_over_threshold, superlative, "
    "events_by_games_sport, medalists_at_venue_date, temporal_neighbor) for "
    "counting, ranking, and relationship questions; use vector_search to find "
    "unknown events or resolve names. Season is 'Summer' or 'Winter' "
    "(capitalized); sport names are capitalized like 'Biathlon'. When a question "
    "names a sport, ALWAYS pass it as the sport argument. NEVER invent a doc_id; "
    "obtain it from vector_search first. Your FINAL answer must be ONLY the "
    "essential value with no sentence around it: just the name, the number, or "
    "the exact event title. For example answer 'Chen Ding', '8', or "
    "\"Athletics at the 2008 Summer Olympics - Men's marathon\" -- never a full "
    "sentence."
)


def _build_agent():
    llm = ChatOllama(
        model=settings.ollama_llm_model,
        temperature=0.0,
        base_url=settings.ollama_base_url,
        reasoning=False,
    )
    return create_react_agent(llm, _TOOLS, prompt=_SYSTEM)


_AGENT = None


def _agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = _build_agent()
    return _AGENT


def _usage_from_messages(messages) -> TokenUsage:
    usage = TokenUsage()
    for m in messages:
        meta = getattr(m, "usage_metadata", None) or {}
        if meta:
            usage = usage + TokenUsage(
                prompt_tokens=int(meta.get("input_tokens", 0) or 0),
                completion_tokens=int(meta.get("output_tokens", 0) or 0),
            )
    return usage


def investigate_react(question: str, *, max_steps: int | None = None) -> PipelineResult:
    max_steps = max_steps or settings.agent_max_steps
    _LAST_EVIDENCE.clear()

    result = _agent().invoke(
        {"messages": [("user", question)]},
        config={"recursion_limit": max_steps * 2 + 6},
    )
    messages = result["messages"]

    # Build trace + agentic_meta from the message history.
    trace: list[AnswerTraceStep] = []
    tools_used: list[str] = []
    tool_calls_meta: list[dict] = []
    step = 0
    for m in messages:
        for tc in getattr(m, "tool_calls", []) or []:
            step += 1
            name = tc.get("name", "")
            args = tc.get("args", {})
            tools_used.append(name)
            trace.append(AnswerTraceStep("plan", f"call {name}({args})", ""))
            tool_calls_meta.append({"step": step, "tool": name, "args": args})
        if m.__class__.__name__ == "ToolMessage":
            trace.append(AnswerTraceStep(getattr(m, "name", "tool"), "",
                                         str(m.content)[:300]))

    answer = messages[-1].content if messages else ""
    trace.append(AnswerTraceStep("synthesize", "final answer", answer))

    # doc_ids from the Evidence objects the tools recorded
    doc_ids: list[str] = []
    for ev in _LAST_EVIDENCE:
        for d in ev.doc_ids:
            if d not in doc_ids:
                doc_ids.append(d)

    agentic_meta = {
        "backend": "react",
        "steps": step,
        "tools_used": tools_used,
        "distinct_tools": sorted(set(tools_used)),
        "strategy_changed": False,  # prebuilt loop has no explicit strategy flag
        "stop_reason": "model_final_answer" if step < max_steps else "max_steps",
        "n_citations": len(doc_ids),
        "tool_calls": tool_calls_meta,
    }

    return PipelineResult(
        question=question,
        pipeline="agentic",
        answer=answer,
        citations=doc_ids,
        retrieved_doc_ids=doc_ids,
        trace=trace,
        usage=_usage_from_messages(messages),
        agentic_meta=agentic_meta,
    )
