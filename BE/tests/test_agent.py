"""Agent orchestrator tests with a fake LLM + fake retriever (no network)."""

import json

import app.agent.orchestrator as orch
import app.agent.tools as tools_mod
from app.agent.tools import _clean_doc_id
from app.pipelines.types import EventFacts
from app.providers.base import ChatResult, TokenUsage


class FakeRetriever:
    def __init__(self):
        self.events = {
            "Q1": EventFacts("Q1", "Shooting A", year=2004, season="Summer",
                             sport="Shooting", competitors=40),
            "Q2": EventFacts("Q2", "Shooting B", year=2004, season="Summer",
                             sport="Shooting", competitors=30),
            "Q3": EventFacts("Q3", "Shooting C", year=2004, season="Summer",
                             sport="Shooting", competitors=50),
        }

    def vector_search(self, q, k):
        return []

    def event_facts(self, doc_id):
        return self.events.get(doc_id)

    def events_by_games_sport(self, year, season, sport):
        return [e for e in self.events.values()
                if e.year == year
                and (not season or e.season == season)
                and (not sport or e.sport.lower() == sport.lower())]

    def medalists_at_venue_date(self, *a):
        return ([], [])

    def temporal_neighbor(self, *a):
        return None

    def events_by_athlete(self, *a):
        return []


def test_clean_doc_id():
    assert _clean_doc_id("Q123") == "Q123"
    assert _clean_doc_id("Q123:Some Title: with colon") == "Q123"
    assert _clean_doc_id("  Q9 ") == "Q9"


def test_count_tool_scans_all_events(monkeypatch):
    monkeypatch.setattr(tools_mod, "get_retriever", lambda: FakeRetriever())
    ev = tools_mod.tool_count_over_threshold(2004, "competitors", 37, "Summer", "Shooting")
    # Q1(40) and Q3(50) exceed 37 -> count 2
    assert ev.payload["count"] == 2


def test_superlative_tool_handles_ties(monkeypatch):
    r = FakeRetriever()
    # make two events tie so sort must not compare EventFacts
    r.events["Q2"].competitors = 50
    monkeypatch.setattr(tools_mod, "get_retriever", lambda: r)
    ev = tools_mod.tool_superlative(2004, "competitors", "max", "Summer", "Shooting")
    assert ev.payload["value"] == 50  # no crash on tie


def test_orchestrator_normalizes_action_as_tool(monkeypatch):
    """Planner returns tool name directly in 'action' -> treated as a tool call."""
    r = FakeRetriever()
    monkeypatch.setattr(tools_mod, "get_retriever", lambda: r)

    calls = {"n": 0}

    class FakeLLM:
        def chat(self, messages, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                out = {"action": "count_over_threshold",
                       "args": {"year": 2004, "metric": "competitors",
                                "threshold": 37, "season": "Summer", "sport": "Shooting"},
                       "reason": "count"}
            else:
                out = {"action": "finish", "reason": "done"}
            return ChatResult(text=json.dumps(out), usage=TokenUsage(5, 5))

    monkeypatch.setattr(orch, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(orch, "synthesize",
                        lambda q, c, **kw: ("2", TokenUsage(1, 1)))

    res = orch.investigate("how many shooting events at 2004 had more than 37 competitors?")
    assert res.pipeline == "agentic"
    assert res.answer == "2"
    assert any(s.action == "count_over_threshold" for s in res.trace)


def test_orchestrator_loop_guard(monkeypatch):
    """Repeated identical failing call is skipped, not spun forever."""
    r = FakeRetriever()
    monkeypatch.setattr(tools_mod, "get_retriever", lambda: r)

    class FakeLLM:
        def chat(self, messages, **kw):
            out = {"action": "temporal_neighbor",
                   "args": {"doc_id": "NOPE", "direction": "prev"}, "reason": "x"}
            return ChatResult(text=json.dumps(out), usage=TokenUsage(5, 5))

    monkeypatch.setattr(orch, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(orch, "synthesize",
                        lambda q, c, **kw: ("I don't know.", TokenUsage(1, 1)))

    res = orch.investigate("bad question", max_steps=5)
    # first call runs, subsequent identical calls are skipped -> loop terminates
    assert res.pipeline == "agentic"
    # trace should contain a skip-repeat marker
    assert any("skip-repeat" in s.detail or s.result_summary.startswith("already tried")
               for s in res.trace)
