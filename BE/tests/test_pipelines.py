"""Pipeline structural tests using a fake retriever + fake LLM (no network)."""

import app.pipelines.retrieval as retrieval_mod
import app.pipelines.synthesize as synth_mod
from app.pipelines.rag import run_rag
from app.pipelines.graphrag import run_graphrag
from app.pipelines.types import EventFacts, RetrievedDoc
from app.providers.base import TokenUsage


class FakeRetriever:
    def vector_search(self, query, k):
        return [
            RetrievedDoc("Q1", "Event One", "text one", score=0.9),
            RetrievedDoc("Q2", "Event Two", "text two", score=0.8),
        ][:k]

    def event_facts(self, doc_id):
        if doc_id == "Q1":
            return EventFacts("Q1", "Event One", year=2012, season="Summer",
                              venue="V", medalists=[{"rank": "gold", "name": "A", "noc": "USA"}])
        return None

    def temporal_neighbor(self, doc_id, direction):
        return None

    def events_by_games_sport(self, *a): return []
    def medalists_at_venue_date(self, *a): return ([], [])
    def events_by_athlete(self, *a): return []


def _patch(monkeypatch):
    fake = FakeRetriever()
    monkeypatch.setattr(retrieval_mod, "get_retriever", lambda: fake)
    # patch the name imported into each pipeline module
    import app.pipelines.rag as rag_mod
    import app.pipelines.graphrag as graph_mod
    monkeypatch.setattr(rag_mod, "get_retriever", lambda: fake)
    monkeypatch.setattr(graph_mod, "get_retriever", lambda: fake)

    def fake_synth(question, context, **kw):
        return "FAKE_ANSWER", TokenUsage(prompt_tokens=10, completion_tokens=5)

    monkeypatch.setattr(rag_mod, "synthesize", fake_synth)
    monkeypatch.setattr(graph_mod, "synthesize", fake_synth)


def test_rag_shape(monkeypatch):
    _patch(monkeypatch)
    r = run_rag("who won?", top_k=2)
    assert r.pipeline == "rag"
    assert r.answer == "FAKE_ANSWER"
    assert r.retrieved_doc_ids == ["Q1", "Q2"]
    assert r.usage.total_tokens == 15
    assert any(s.action == "vector_search" for s in r.trace)


def test_graphrag_enriches_and_cites(monkeypatch):
    _patch(monkeypatch)
    r = run_graphrag("who won?", top_k=2)
    assert r.pipeline == "graphrag"
    assert "Q1" in r.citations
    assert any(s.action == "graph_enrich" for s in r.trace)
