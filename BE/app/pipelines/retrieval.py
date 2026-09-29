"""Retrieval primitives shared by all three pipelines.

Backed exclusively by TigerGraph: vector search + structured graph queries run
as installed GSQL queries on Savanna. A live TigerGraph connection is required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

from app.providers.factory import get_embeddings

from .types import EventFacts, RetrievedDoc


class Retriever(ABC):
    """Primitive retrieval operations used to compose pipelines."""

    @abstractmethod
    def vector_search(self, query: str, k: int) -> list[RetrievedDoc]: ...

    @abstractmethod
    def event_facts(self, doc_id: str) -> EventFacts | None: ...

    @abstractmethod
    def events_by_games_sport(
        self, year: int, season: str, sport: str | None
    ) -> list[EventFacts]: ...

    @abstractmethod
    def medalists_at_venue_date(
        self, venue: str, date_fragment: str, rank: str
    ) -> tuple[list[EventFacts], list[dict]]: ...

    @abstractmethod
    def temporal_neighbor(self, doc_id: str, direction: str) -> EventFacts | None: ...

    @abstractmethod
    def events_by_athlete(self, athlete: str) -> list[EventFacts]: ...


# ---------------------------------------------------------------------------
# TigerGraph backend
# ---------------------------------------------------------------------------


class TigerGraphRetriever(Retriever):
    def __init__(self) -> None:
        from app.graph.client import get_client

        self.client = get_client()
        self.emb = get_embeddings()

    @staticmethod
    def _norm_attr(attr: dict) -> dict:
        """GSQL prints attributes prefixed with the SELECT alias, e.g.
        'result.competitors' or 'v.title'. Strip the prefix so lookups by
        plain field name work regardless of the alias used in the query."""
        out = {}
        for k, v in (attr or {}).items():
            key = k.split(".", 1)[1] if "." in k else k
            out[key] = v
        return out

    def vector_search(self, query: str, k: int) -> list[RetrievedDoc]:
        qv = self.emb.embed_one(query)
        res = self.client.run_query("vector_search", {"query_vector": qv, "k": k})
        docs, distances = [], {}
        for block in res:
            if "docs" in block:
                for row in block["docs"]:
                    attr = self._norm_attr(row.get("attributes", row))
                    docs.append((row.get("v_id") or attr.get("doc_id"), attr))
            if "distances" in block:
                distances = block["distances"]
        out = []
        for doc_id, attr in docs:
            dist = float(distances.get(doc_id, 0.0)) if isinstance(distances, dict) else 0.0
            out.append(
                RetrievedDoc(
                    doc_id=doc_id,
                    title=attr.get("title", ""),
                    text=attr.get("text", ""),
                    url=attr.get("url", ""),
                    score=1.0 - dist,
                    source="vector",
                )
            )
        return out

    def _facts_from_row(self, attr: dict) -> EventFacts:
        attr = self._norm_attr(attr)
        return EventFacts(
            doc_id=attr.get("doc_id", ""),
            title=attr.get("title", ""),
            event_name=attr.get("event_name", ""),
            year=attr.get("year"),
            season=attr.get("season", ""),
            venue=attr.get("venue", ""),
            event_date=attr.get("event_date", ""),
            competitors=attr.get("competitors"),
            nations=attr.get("nations"),
        )

    def event_facts(self, doc_id: str) -> EventFacts | None:
        res = self.client.run_query("event_by_doc", {"doc_id": doc_id})
        facts: EventFacts | None = None
        medalists: list[dict] = []
        for block in res:
            if block.get("ev"):
                facts = self._facts_from_row(block["ev"][0].get("attributes", {}))
            # event_by_doc returns medalists as {name: [ranks]} in its own block
            mmap = block.get("medalists")
            if isinstance(mmap, dict):
                for name, ranks in mmap.items():
                    for rank in (ranks if isinstance(ranks, list) else [ranks]):
                        medalists.append({"rank": rank, "name": name, "noc": ""})
        if facts is not None:
            # order gold, silver, bronze for readable context
            order = {"gold": 0, "silver": 1, "bronze": 2}
            facts.medalists = sorted(medalists, key=lambda m: order.get(m["rank"], 9))
        return facts

    def events_by_games_sport(self, year, season, sport):
        res = self.client.run_query(
            "events_by_games_sport",
            {"year": year, "season": season or "", "sport": sport or ""},
        )
        out = []
        for block in res:
            for row in block.get("events", []):
                out.append(self._facts_from_row(row.get("attributes", row)))
        return out

    def medalists_at_venue_date(self, venue, date_fragment, rank):
        res = self.client.run_query(
            "medalists_at_venue_date",
            {"venue_name": venue, "date_fragment": date_fragment or "", "rank": rank or ""},
        )
        events, winners = [], []
        for block in res:
            for row in block.get("events", []):
                events.append(self._facts_from_row(row.get("attributes", row)))
            # winners is a MapAccum: {athlete_name: ["rank @ event title", ...]}
            wmap = block.get("winners")
            if isinstance(wmap, dict):
                for name, entries in wmap.items():
                    for entry in (entries if isinstance(entries, list) else [entries]):
                        rk, _, ev_title = str(entry).partition(" @ ")
                        winners.append({"name": name, "rank": rk, "event": ev_title})
        return events, winners

    def temporal_neighbor(self, doc_id, direction):
        res = self.client.run_query(
            "event_temporal_neighbor", {"doc_id": doc_id, "direction": direction}
        )
        for block in res:
            for row in block.get("neighbor", []):
                return self._facts_from_row(row.get("attributes", row))
        return None

    def events_by_athlete(self, athlete):
        res = self.client.run_query("events_by_athlete", {"athlete_name": athlete})
        out = []
        for block in res:
            for row in block.get("events", []):
                out.append(self._facts_from_row(row.get("attributes", row)))
        return out


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Return the TigerGraph-backed retriever (the only backend)."""
    return TigerGraphRetriever()
