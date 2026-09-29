"""Orchestrator agent built on LangGraph.

The investigation is modeled as an explicit StateGraph:

        START
          |
          v
    +----------+      finish / max steps / no-tool
    |  plan    |------------------------------+
    +----------+                              |
          | tool                              |
          v                                   |
    +----------+   decisive answer / max      |
    |   act    |----------------+             |
    +----------+                |             |
          | continue            |             |
          +---------------------)-------------+
                                v             v
                          +-----------------------+
                          |      synthesize       |
                          +-----------------------+
                                    |
                                    v
                                   END

- plan:  the LLM picks the SINGLE next action (a tool + args, or finish),
         based on the question and the evidence gathered so far. Not a fixed
         sequence -- the next move depends on what is still missing.
- act:   run the chosen tool, record evidence + a per-operation trace entry
         (time, tokens, docs), and decide whether the answer is now decisive.
- synthesize: produce the grounded, cited answer from accumulated evidence.

Behavior (loop guard, action-shape normalization, decisive stop, doc_id
cleaning, stop-reason capture) is preserved from the original loop and made
explicit as graph routing.
"""

from __future__ import annotations

import json
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.pipelines.synthesize import synthesize
from app.pipelines.types import AnswerTraceStep, PipelineResult
from app.providers.base import ChatMessage
from app.providers.factory import get_llm

from .state import Evidence, InvestigationState, ToolCallRecord
from .tools import TOOL_SPECS, TOOLS

PLANNER_SYSTEM = (
    "You are the orchestrator of a multi-step investigation over an Olympic-events "
    "knowledge graph + document store. Decide the SINGLE next action that best "
    "reduces uncertainty about the question. Prefer structured graph tools "
    "(count_over_threshold, superlative, events_by_games_sport, "
    "medalists_at_venue_date, temporal_neighbor) for counting, ranking, and "
    "relationship questions; use vector_search to discover unknown events or "
    "resolve names. When you already have enough evidence to answer, finish.\n\n"
    "IMPORTANT: A doc_id looks like 'Q303623'. NEVER invent a doc_id. Any "
    "doc_id passed to event_lookup or temporal_neighbor MUST come from the "
    "evidence you already collected (usually from a prior vector_search). If "
    "you need an event's doc_id, run vector_search first. For 'immediately "
    "before/after YEAR' questions: vector_search to find the event at the "
    "reference year, then temporal_neighbor with that doc_id.\n\n"
    "Respond ONLY as compact JSON:\n"
    '{"action": "tool", "tool": "<name>", "args": {..}, "reason": ".."}\n'
    "or\n"
    '{"action": "finish", "reason": ".."}'
)


# --- LangGraph state ------------------------------------------------------


class GraphState(TypedDict):
    """LangGraph channel: carries the mutable InvestigationState + last plan."""

    inv: InvestigationState
    plan: dict[str, Any]
    route: str  # "act" | "synthesize"


def _tools_doc() -> str:
    return json.dumps(TOOL_SPECS, ensure_ascii=False)


def _run_tool(name: str, args: dict) -> Evidence:
    fn = TOOLS.get(name)
    if not fn:
        return Evidence(tool=name or "unknown", summary=f"unknown tool: {name}", doc_ids=[])
    try:
        return fn(**args)
    except TypeError as e:
        return Evidence(tool=name, summary=f"bad args for {name}: {e}", doc_ids=[])
    except Exception as e:  # noqa: BLE001
        return Evidence(tool=name, summary=f"{name} error: {e}", doc_ids=[])


# --- nodes ----------------------------------------------------------------


def plan_node(state: GraphState) -> GraphState:
    """Ask the LLM for the next action, then decide where to route."""
    inv = state["inv"]

    # Budget guard.
    if inv.steps_taken >= inv.max_steps:
        inv.stop_reason = "max_steps_reached"
        inv.trace.append(AnswerTraceStep("plan", "finish", inv.stop_reason))
        return {**state, "plan": {"action": "finish"}, "route": "synthesize"}

    llm = get_llm()
    prompt = (
        f"Question: {inv.question}\n\n"
        f"Available tools (name, args, use):\n{_tools_doc()}\n\n"
        f"Evidence collected so far:\n{inv.evidence_digest()}\n\n"
        f"Steps taken: {inv.steps_taken}/{inv.max_steps}. "
        "Choose the next action as JSON."
    )
    t0 = time.time()
    try:
        res = llm.chat(
            [ChatMessage("system", PLANNER_SYSTEM), ChatMessage("user", prompt)],
            temperature=0.0,
            json_mode=True,
            max_tokens=300,
        )
    except Exception as e:  # noqa: BLE001 - LLM outage: stop gracefully
        inv.stop_reason = f"planner_llm_error: {str(e)[:120]}"
        inv.trace.append(AnswerTraceStep("plan", "finish", inv.stop_reason))
        return {**state, "plan": {"action": "finish"}, "route": "synthesize"}
    plan_secs = time.time() - t0
    inv.add_usage(res.usage)

    try:
        plan = json.loads(res.text)
    except (json.JSONDecodeError, TypeError):
        inv.stop_reason = "planner_unparseable"
        inv.trace.append(AnswerTraceStep("plan", "finish", inv.stop_reason))
        return {**state, "plan": {"action": "finish"}, "route": "synthesize"}

    action = (plan.get("action") or "").strip()
    tool = (plan.get("tool") or "").strip()

    # Model often puts the tool name directly in "action".
    if action in TOOLS and not tool:
        tool = action
        action = "tool"

    if action == "finish":
        inv.stop_reason = "planner_finished"
        inv.trace.append(AnswerTraceStep("plan", "finish", plan.get("reason", "")))
        return {**state, "plan": plan, "route": "synthesize"}

    if not tool or tool not in TOOLS:
        inv.stop_reason = f"no_usable_tool:{tool!r}"
        inv.trace.append(AnswerTraceStep("plan", "finish", inv.stop_reason))
        return {**state, "plan": plan, "route": "synthesize"}

    # stash the planner cost + normalized fields on the plan for act_node
    plan["_tool"] = tool
    plan["_args"] = plan.get("args", {}) or {}
    plan["_plan_secs"] = plan_secs
    plan["_plan_tokens"] = res.usage.total_tokens
    inv.trace.append(
        AnswerTraceStep("plan", f"call {tool}({plan['_args']})", plan.get("reason", ""))
    )
    return {**state, "plan": plan, "route": "act"}


def act_node(state: GraphState) -> GraphState:
    """Execute the chosen tool, record evidence + per-op trace, decide route."""
    inv = state["inv"]
    plan = state["plan"]
    tool = plan["_tool"]
    args = plan["_args"]

    # Loop guard: skip an exact repeat and nudge toward a new strategy.
    call_key = f"{tool}:{json.dumps(args, sort_keys=True)}"
    if call_key in inv.seen_calls:
        inv.strategy_changed = True
        inv.trace.append(
            AnswerTraceStep("plan", "skip-repeat",
                            f"already tried {tool}({args}); forcing new strategy")
        )
        inv.add_evidence(
            Evidence(tool=tool,
                     summary=f"(skipped repeated call {tool}({args}); try a different tool "
                             f"such as vector_search to find the correct doc_id first)",
                     doc_ids=[])
        )
        inv.steps_taken += 1
        return {**state, "route": "plan"}
    inv.seen_calls.add(call_key)

    t0 = time.time()
    ev = _run_tool(tool, args)
    secs = time.time() - t0

    inv.add_evidence(ev)
    inv.used_tools.append(tool)
    inv.steps_taken += 1
    inv.tool_calls.append(
        ToolCallRecord(
            step=inv.steps_taken, tool=tool, args=args,
            reason=plan.get("reason", ""),
            seconds=secs, tokens=int(plan.get("_plan_tokens", 0)),
            result_summary=ev.summary[:300], n_docs=len(ev.doc_ids),
        )
    )
    inv.trace.append(AnswerTraceStep(tool, json.dumps(args), ev.summary[:300]))

    # Decisive stop: a structured tool produced a direct answer.
    if ev.tool == "superlative" and ev.payload.get("answer") is not None:
        inv.stop_reason = "decisive_superlative"
        return {**state, "route": "synthesize"}
    if ev.tool == "count_over_threshold" and ev.payload.get("count") is not None:
        inv.stop_reason = "decisive_count"
        return {**state, "route": "synthesize"}

    return {**state, "route": "plan"}


def _final_context(inv: InvestigationState) -> str:
    highlights = []
    for ev in inv.evidence:
        if ev.tool == "count_over_threshold" and "count" in ev.payload:
            highlights.append(f"COUNT RESULT: {ev.payload['count']} events match.")
        if ev.tool == "superlative" and ev.payload.get("answer"):
            highlights.append(
                f"SUPERLATIVE RESULT: {ev.payload['answer']} ({ev.payload.get('value')})."
            )
        if ev.tool == "medalists_at_venue_date" and ev.payload.get("winners"):
            names = ", ".join(w.get("name", "?") for w in ev.payload["winners"][:5])
            highlights.append(f"WINNERS: {names}")

    blocks = []
    for i, ev in enumerate(inv.evidence, 1):
        detail = ev.payload.get("facts") or ev.payload.get("neighbor") or ev.summary
        blocks.append(f"[E{i}] ({ev.tool}) {detail}")

    parts = []
    if highlights:
        parts.append("KEY FINDINGS:\n" + "\n".join(highlights))
    parts.append("EVIDENCE:\n" + ("\n\n".join(blocks) if blocks else "(none)"))
    return "\n\n".join(parts)


def _fallback_answer(inv: InvestigationState) -> str:
    """Best-effort answer from structured evidence if the LLM is unavailable."""
    for ev in reversed(inv.evidence):
        if ev.tool == "count_over_threshold" and "count" in ev.payload:
            return str(ev.payload["count"])
        if ev.tool == "superlative" and ev.payload.get("answer"):
            return str(ev.payload["answer"])
        if ev.tool == "medalists_at_venue_date" and ev.payload.get("winners"):
            w = ev.payload["winners"][0]
            return w.get("name", "") if isinstance(w, dict) else str(w)
    return "I don't know."


def synthesize_node(state: GraphState) -> GraphState:
    inv = state["inv"]
    context = _final_context(inv)
    try:
        answer, usage = synthesize(inv.question, context)
        inv.add_usage(usage)
    except Exception as e:  # noqa: BLE001 - LLM outage: fall back to evidence
        answer = _fallback_answer(inv)
        inv.stop_reason = (inv.stop_reason or "") + f" | synth_llm_error: {str(e)[:80]}"
    inv.trace.append(AnswerTraceStep("synthesize", "answer from evidence", answer))
    state["plan"] = {**state.get("plan", {}), "_answer": answer}
    return state


# --- graph assembly -------------------------------------------------------


def _route_from(state: GraphState) -> str:
    return state["route"]


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("plan", plan_node)
    g.add_node("act", act_node)
    g.add_node("synthesize", synthesize_node)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", _route_from,
                            {"act": "act", "synthesize": "synthesize"})
    g.add_conditional_edges("act", _route_from,
                            {"plan": "plan", "synthesize": "synthesize"})
    g.add_edge("synthesize", END)
    return g.compile()


# Compiled once; reused across questions.
_APP = build_graph()


def investigate(question: str, *, max_steps: int | None = None) -> PipelineResult:
    inv = InvestigationState(question=question, max_steps=max_steps or settings.agent_max_steps)
    # recursion_limit must exceed 2 nodes/step * max_steps + synthesize.
    final = _APP.invoke(
        {"inv": inv, "plan": {}, "route": "plan"},
        config={"recursion_limit": inv.max_steps * 2 + 10},
    )
    inv = final["inv"]
    answer = final.get("plan", {}).get("_answer", "")

    agentic_meta = {
        "steps": inv.steps_taken,
        "tools_used": inv.used_tools,
        "distinct_tools": sorted(set(inv.used_tools)),
        "strategy_changed": inv.strategy_changed,
        "stop_reason": inv.stop_reason,
        "n_citations": len(inv.all_doc_ids()),
        "tool_calls": [tc.as_dict() for tc in inv.tool_calls],
    }

    return PipelineResult(
        question=question,
        pipeline="agentic",
        answer=answer,
        citations=inv.all_doc_ids(),
        retrieved_doc_ids=inv.all_doc_ids(),
        trace=inv.trace,
        usage=inv.usage,
        agentic_meta=agentic_meta,
    )
