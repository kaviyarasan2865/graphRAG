"""Specialized tools the orchestrator can invoke.

Each tool wraps retrieval primitives and returns an Evidence object with a
summary + doc_ids + structured payload. Tools are the ONLY way the agent
touches the graph/vector store, which keeps the investigation auditable.

The aggregation/superlative tools are what let the agent beat plain RAG: they
scan ALL matching events in the graph rather than a top-k slice.
"""

from __future__ import annotations

from typing import Callable

from app.pipelines.retrieval import get_retriever

from .state import Evidence


def _metric_value(ev, metric: str):
    """Resolve a metric name (robust to phrasing like 'number of competitors',
    'nation', 'countries') to the right EventFacts attribute value."""
    m = (metric or "").lower()
    if "nation" in m or "countr" in m:
        return ev.nations
    return ev.competitors  # default + any 'competitor' phrasing


def _fmt_event(ev) -> str:
    parts = [ev.title]
    if ev.competitors is not None and ev.competitors >= 0:
        parts.append(f"competitors={ev.competitors}")
    if ev.nations is not None and ev.nations >= 0:
        parts.append(f"nations={ev.nations}")
    return " | ".join(parts)


# --- tool implementations -------------------------------------------------


def tool_vector_search(query: str, k: int = 8) -> Evidence:
    r = get_retriever()
    docs = r.vector_search(query, k)
    summary = "; ".join(f"{d.doc_id}:{d.title}" for d in docs[:k]) or "no results"
    return Evidence(
        tool="vector_search",
        summary=f"top-{k} for '{query}': {summary}",
        doc_ids=[d.doc_id for d in docs],
        payload={"docs": [{"doc_id": d.doc_id, "title": d.title, "text": d.text[:1200]} for d in docs]},
    )


def _clean_doc_id(doc_id: str) -> str:
    """Accept 'Q123' or a copied 'Q123:Title ...' summary; return just 'Q123'."""
    doc_id = (doc_id or "").strip()
    if ":" in doc_id:
        head = doc_id.split(":", 1)[0].strip()
        if head:
            return head
    return doc_id


def tool_event_lookup(doc_id: str) -> Evidence:
    doc_id = _clean_doc_id(doc_id)
    r = get_retriever()
    facts = r.event_facts(doc_id)
    if not facts:
        return Evidence("event_lookup", f"no event for doc_id={doc_id}", [])
    return Evidence(
        tool="event_lookup",
        summary=facts.to_context().replace("\n", " | "),
        doc_ids=[doc_id],
        payload={"facts": facts.to_context()},
    )


def tool_events_by_games_sport(year: int, season: str = "", sport: str = "") -> Evidence:
    r = get_retriever()
    events = r.events_by_games_sport(year, season or None, sport or None)
    listing = "; ".join(_fmt_event(e) for e in events[:40])
    return Evidence(
        tool="events_by_games_sport",
        summary=f"{len(events)} events at {year} {season} sport={sport or 'any'}: {listing}",
        doc_ids=[e.doc_id for e in events],
        payload={"events": [
            {"doc_id": e.doc_id, "title": e.title, "competitors": e.competitors,
             "nations": e.nations} for e in events
        ]},
    )


def tool_count_over_threshold(
    year: int, metric: str = "competitors", threshold: int = 0,
    season: str = "", sport: str = "",
) -> Evidence:
    """Count events at a Games (optionally a sport) whose metric > threshold."""
    r = get_retriever()
    events = r.events_by_games_sport(year, season or None, sport or None)
    matched = []
    for e in events:
        val = _metric_value(e, metric)
        if val is not None and val >= 0 and val > threshold:
            matched.append(e)
    return Evidence(
        tool="count_over_threshold",
        summary=(f"{len(matched)} of {len(events)} events at {year} {season} "
                 f"sport={sport or 'any'} have {metric} > {threshold}: "
                 + "; ".join(_fmt_event(e) for e in matched[:40])),
        doc_ids=[e.doc_id for e in matched],
        payload={"count": len(matched), "metric": metric, "threshold": threshold,
                 "matched": [e.title for e in matched]},
    )


def tool_superlative(
    year: int, metric: str = "competitors", mode: str = "max",
    season: str = "", sport: str = "",
) -> Evidence:
    """Find the event with the max/min metric at a Games (optionally a sport)."""
    r = get_retriever()
    events = r.events_by_games_sport(year, season or None, sport or None)
    # Include an index as a tiebreaker so equal metric values never force a
    # comparison between EventFacts objects (which are not orderable).
    scored = [
        (_metric_value(e, metric), i, e)
        for i, e in enumerate(events)
    ]
    scored = [(v, i, e) for v, i, e in scored if v is not None and v >= 0]
    if not scored:
        return Evidence("superlative", f"no events with {metric} at {year} {season} {sport}", [])
    scored.sort(key=lambda t: (t[0], -t[1]), reverse=(mode == "max"))
    best_val, _, best = scored[0]
    return Evidence(
        tool="superlative",
        summary=(f"{mode} {metric} at {year} {season} sport={sport or 'any'}: "
                 f"{best.title} ({metric}={best_val})"),
        doc_ids=[best.doc_id],
        payload={"answer": best.title, "value": best_val, "doc_id": best.doc_id},
    )


def tool_medalists_at_venue_date(venue: str, date_fragment: str = "", rank: str = "gold") -> Evidence:
    r = get_retriever()
    events, winners = r.medalists_at_venue_date(venue, date_fragment, rank)
    wsum = "; ".join(f"{w.get('name','?')} ({w.get('rank','')}) @ {w.get('event','')}" for w in winners[:20])
    return Evidence(
        tool="medalists_at_venue_date",
        summary=(f"{len(events)} events at venue='{venue}' date~'{date_fragment}'; "
                 f"{len(winners)} {rank or 'any'} winners: {wsum}"),
        doc_ids=[e.doc_id for e in events],
        payload={"winners": winners, "events": [e.title for e in events]},
    )


def tool_temporal_neighbor(doc_id: str, direction: str = "prev") -> Evidence:
    doc_id = _clean_doc_id(doc_id)
    r = get_retriever()
    nb = r.temporal_neighbor(doc_id, direction)
    if not nb:
        return Evidence("temporal_neighbor", f"no {direction} edition for {doc_id}", [])
    return Evidence(
        tool="temporal_neighbor",
        summary=f"{direction} edition of {doc_id}: {nb.title} [{nb.year} {nb.season}] (doc_id={nb.doc_id})",
        doc_ids=[nb.doc_id],
        payload={"neighbor": nb.to_context(), "doc_id": nb.doc_id},
    )


def tool_events_by_athlete(athlete: str) -> Evidence:
    r = get_retriever()
    events = r.events_by_athlete(athlete)
    return Evidence(
        tool="events_by_athlete",
        summary=f"{athlete} appears in {len(events)} events: " + "; ".join(e.title for e in events[:20]),
        doc_ids=[e.doc_id for e in events],
        payload={"events": [e.title for e in events]},
    )


# --- registry + schema for the orchestrator -------------------------------

TOOLS: dict[str, Callable[..., Evidence]] = {
    "vector_search": tool_vector_search,
    "event_lookup": tool_event_lookup,
    "events_by_games_sport": tool_events_by_games_sport,
    "count_over_threshold": tool_count_over_threshold,
    "superlative": tool_superlative,
    "medalists_at_venue_date": tool_medalists_at_venue_date,
    "temporal_neighbor": tool_temporal_neighbor,
    "events_by_athlete": tool_events_by_athlete,
}

# Compact descriptions the orchestrator sees when choosing the next action.
TOOL_SPECS = [
    {"name": "vector_search",
     "args": {"query": "str", "k": "int?"},
     "use": "Semantic search over all documents. Use to find seed events or when you don't know exact fields."},
    {"name": "event_lookup",
     "args": {"doc_id": "str"},
     "use": "Get full structured facts (medalists, venue, date, competitors) for one event by doc_id."},
    {"name": "events_by_games_sport",
     "args": {"year": "int", "season": "str?", "sport": "str?"},
     "use": "List every event at a Games, optionally filtered by sport. Use before counting/superlatives."},
    {"name": "count_over_threshold",
     "args": {"year": "int", "metric": "competitors|nations", "threshold": "int", "season": "str?", "sport": "str?"},
     "use": "Count events at a Games whose metric exceeds a threshold. Best for 'how many ... more than N' questions."},
    {"name": "superlative",
     "args": {"year": "int", "metric": "competitors|nations", "mode": "max|min", "season": "str?", "sport": "str?"},
     "use": "Find the event with the highest/lowest metric at a Games. Best for 'which event had the most/fewest' questions."},
    {"name": "medalists_at_venue_date",
     "args": {"venue": "str", "date_fragment": "str?", "rank": "gold|silver|bronze?"},
     "use": "Find medalists for events at a venue (optionally on a date). Best for multi-hop 'who won at VENUE on DATE'."},
    {"name": "temporal_neighbor",
     "args": {"doc_id": "str", "direction": "prev|next"},
     "use": "Jump to the same event at the previous/next Games. Best for 'immediately before/after YEAR' questions."},
    {"name": "events_by_athlete",
     "args": {"athlete": "str"},
     "use": "Find events an athlete medaled in."},
]
