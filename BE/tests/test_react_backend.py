"""Tests for the react agent backend selection + shared tool robustness."""

import app.agent.tools as tools_mod
from app.pipelines.types import EventFacts, PipelineResult


class FakeRetriever:
    def __init__(self):
        self.events = {
            "Q1": EventFacts("Q1", "A", year=2008, season="Summer",
                             sport="Athletics", competitors=40, nations=20),
            "Q2": EventFacts("Q2", "B", year=2008, season="Summer",
                             sport="Athletics", competitors=90, nations=10),
        }

    def events_by_games_sport(self, year, season, sport):
        return [e for e in self.events.values()
                if e.year == year
                and (not season or e.season == season)
                and (not sport or e.sport.lower() == sport.lower())]


def test_metric_normalization_phrasing(monkeypatch):
    monkeypatch.setattr(tools_mod, "get_retriever", lambda: FakeRetriever())
    # "number of competitors" must resolve to competitors, not nations
    ev = tools_mod.tool_superlative(2008, "number of competitors", "max", "Summer", "Athletics")
    assert ev.payload["answer"] == "B"   # Q2 has 90 competitors
    assert ev.payload["value"] == 90


def test_metric_normalization_nations(monkeypatch):
    monkeypatch.setattr(tools_mod, "get_retriever", lambda: FakeRetriever())
    ev = tools_mod.tool_superlative(2008, "number of nations", "max", "Summer", "Athletics")
    assert ev.payload["answer"] == "A"   # Q1 has 20 nations
    assert ev.payload["value"] == 20


def test_count_metric_phrasing(monkeypatch):
    monkeypatch.setattr(tools_mod, "get_retriever", lambda: FakeRetriever())
    ev = tools_mod.tool_count_over_threshold(2008, "the number of competitors", 50, "Summer", "Athletics")
    assert ev.payload["count"] == 1     # only Q2 (90) > 50


def test_agentic_dispatch_selects_backend(monkeypatch):
    import app.pipelines.agentic as ag

    calls = {}

    def fake_react(q, max_steps=None):
        calls["react"] = q
        return PipelineResult(question=q, pipeline="agentic", answer="R")

    def fake_graph(q, max_steps=None):
        calls["graph"] = q
        return PipelineResult(question=q, pipeline="agentic", answer="G")

    import app.agent.react_agent as ra
    import app.agent.orchestrator as orch
    monkeypatch.setattr(ra, "investigate_react", fake_react)
    monkeypatch.setattr(orch, "investigate", fake_graph)

    r1 = ag.run_agentic("q", impl="react")
    assert r1.answer == "R" and calls.get("react") == "q"

    r2 = ag.run_agentic("q", impl="graph")
    assert r2.answer == "G" and calls.get("graph") == "q"
