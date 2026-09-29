"""Unit tests for graph row building (no network)."""

from app.ingest.graph_builder import build_rows
from app.ingest.models import RawDoc
from app.ingest.parser import parse_infobox_fields

EV_2012 = """[Infobox Olympic event]
  event: Men's canoe sprint K-2 1000 metres
  games: 2012 Summer
  venue: Eton Dorney
  competitors: 24
  nations: 12
  gold: Rudolf DombiRoland Kökény
  goldNOC: HUN
  silver: Fernando PimentaEmanuel Silva
  silverNOC: POR
  prev: 2008
  next: 2016
Body text.
"""

EV_2016 = """[Infobox Olympic event]
  event: Men's canoe sprint K-2 1000 metres
  games: 2016 Summer
  venue: Lagoa Stadium
  competitors: 20
  nations: 10
  gold: Some AthleteOther Athlete
  goldNOC: GER
  prev: 2012
  next: 2020
Body text.
"""

FILM = """[Infobox film]
  name: A Movie
  director: Someone
Body.
"""


def _doc(doc_id, text, title):
    itype, _ = parse_infobox_fields(text)
    return RawDoc(doc_id=doc_id, title=title, url="http://x", text=text, infobox_type=itype)


def test_every_doc_becomes_document_vertex():
    docs = [
        _doc("Q1", EV_2012, "Canoeing at the 2012 Summer Olympics – Men's K-2 1000 metres"),
        _doc("F1", FILM, "A Movie"),
    ]
    rows = build_rows(docs)
    assert len(rows.documents) == 2
    # only the olympic event becomes an Event
    assert len(rows.events) == 1


def test_medal_edges_and_teams():
    docs = [_doc("Q1", EV_2012, "Canoeing at the 2012 Summer Olympics – Men's K-2 1000 metres")]
    rows = build_rows(docs)
    # 2 gold + 2 silver athletes = 4 WON_MEDAL edges
    assert len(rows.won_medal) == 4
    assert {"Rudolf Dombi", "Roland Kökény", "Fernando Pimenta", "Emanuel Silva"} <= rows.athletes
    assert {"HUN", "POR"} <= rows.countries


def test_temporal_followed_by():
    docs = [
        _doc("Q2012", EV_2012, "Canoeing at the 2012 Summer Olympics – Men's K-2 1000 metres"),
        _doc("Q2016", EV_2016, "Canoeing at the 2016 Summer Olympics – Men's K-2 1000 metres"),
    ]
    rows = build_rows(docs)
    # 2012 -> next 2016 should create one FOLLOWED_BY edge
    assert ("Q2012", "Q2016", {}) in rows.followed_by


def test_games_dedup():
    docs = [
        _doc("Q1", EV_2012, "Canoeing at the 2012 Summer Olympics – A"),
        _doc("Q2", EV_2012.replace("K-2 1000", "K-1 500"), "Canoeing at the 2012 Summer Olympics – B"),
    ]
    rows = build_rows(docs)
    assert "2012 Summer" in rows.games
    assert len(rows.games) == 1  # deduped
