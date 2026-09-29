"""Transform parsed docs into TigerGraph vertex/edge upsert rows.

Pure functions (no network) so they can be unit-tested. The run script feeds
these rows to the GraphClient upsert methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import OlympicEvent, RawDoc
from .parser import is_olympic_event, parse_olympic_event


@dataclass
class GraphRows:
    """Accumulated upsert rows keyed by vertex/edge type."""

    # vertices: type -> list[(primary_id, attrs)]
    documents: list[tuple[str, dict]] = field(default_factory=list)
    events: list[tuple[str, dict]] = field(default_factory=list)
    games: dict[str, dict] = field(default_factory=dict)      # dedup by key
    venues: set[str] = field(default_factory=set)
    sports: set[str] = field(default_factory=set)
    athletes: set[str] = field(default_factory=set)
    countries: set[str] = field(default_factory=set)

    # edges: list[(from_id, to_id, attrs)]
    has_document: list[tuple[str, str, dict]] = field(default_factory=list)
    part_of: list[tuple[str, str, dict]] = field(default_factory=list)
    held_at: list[tuple[str, str, dict]] = field(default_factory=list)
    in_sport: list[tuple[str, str, dict]] = field(default_factory=list)
    won_medal: list[tuple[str, str, dict]] = field(default_factory=list)
    represents: list[tuple[str, str, dict]] = field(default_factory=list)
    followed_by: list[tuple[str, str, dict]] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "Document": len(self.documents),
            "Event": len(self.events),
            "Games": len(self.games),
            "Venue": len(self.venues),
            "Sport": len(self.sports),
            "Athlete": len(self.athletes),
            "Country": len(self.countries),
            "HAS_DOCUMENT": len(self.has_document),
            "PART_OF": len(self.part_of),
            "HELD_AT": len(self.held_at),
            "IN_SPORT": len(self.in_sport),
            "WON_MEDAL": len(self.won_medal),
            "REPRESENTS": len(self.represents),
            "FOLLOWED_BY": len(self.followed_by),
        }


def _add_event(rows: GraphRows, ev: OlympicEvent) -> None:
    rows.events.append(
        (
            ev.doc_id,
            {
                "title": ev.title,
                "event_name": ev.event or "",
                "year": ev.year or 0,
                "season": ev.season or "",
                "venue": ev.venue or "",
                "event_date": ev.date or "",
                "competitors": ev.competitors if ev.competitors is not None else -1,
                "nations": ev.nations if ev.nations is not None else -1,
                "win_value": ev.win_value or "",
                "url": ev.url,
            },
        )
    )
    rows.has_document.append((ev.doc_id, ev.doc_id, {}))

    gk = ev.games_key()
    if gk:
        rows.games[gk] = {"year": ev.year or 0, "season": ev.season or ""}
        rows.part_of.append((ev.doc_id, gk, {}))

    if ev.venue:
        rows.venues.add(ev.venue)
        rows.held_at.append((ev.doc_id, ev.venue, {}))

    if ev.sport:
        rows.sports.add(ev.sport)
        rows.in_sport.append((ev.doc_id, ev.sport, {}))

    for m in ev.medalists:
        for name in m.names:
            rows.athletes.add(name)
            rows.won_medal.append((ev.doc_id, name, {"rank": m.rank}))
            if m.noc:
                rows.countries.add(m.noc)
                rows.represents.append((name, m.noc, {}))


def build_rows(docs: list[RawDoc]) -> GraphRows:
    """Build all vertex/edge rows from a list of raw docs.

    Every doc becomes a Document vertex (for vector search). Olympic-event docs
    additionally become Event vertices with their full neighborhood. prev/next
    year links are resolved into FOLLOWED_BY edges between events of the same
    (sport, event_name) across consecutive Games.
    """
    rows = GraphRows()

    # first pass: documents + events; index events for temporal linking
    events_by_key: dict[tuple[str, str, int], str] = {}  # (sport, event, year) -> doc_id
    parsed: list[OlympicEvent] = []

    for doc in docs:
        rows.documents.append(
            (
                doc.doc_id,
                {
                    "title": doc.title,
                    "url": doc.url,
                    "infobox_type": doc.infobox_type or "",
                    "approx_tokens": doc.approx_tokens or 0,
                    "text": doc.text,
                },
            )
        )
        if is_olympic_event(doc):
            ev = parse_olympic_event(doc)
            _add_event(rows, ev)
            parsed.append(ev)
            if ev.sport and ev.event and ev.year:
                events_by_key[(ev.sport, ev.event, ev.year)] = ev.doc_id

    # second pass: temporal edges via prev/next year
    for ev in parsed:
        if not (ev.sport and ev.event):
            continue
        if ev.next_year:
            nxt = events_by_key.get((ev.sport, ev.event, ev.next_year))
            if nxt:
                rows.followed_by.append((ev.doc_id, nxt, {}))

    return rows
