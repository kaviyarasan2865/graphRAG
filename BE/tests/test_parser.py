"""Unit tests for the infobox parser (no network / DB needed)."""

from app.ingest.models import RawDoc
from app.ingest.parser import (
    _split_names,
    is_olympic_event,
    parse_infobox_fields,
    parse_olympic_event,
)

SAMPLE = """[Infobox Olympic event]
  event: Bantamweight boxing
  games: 1996 Summer
  venue: Alexander Memorial Coliseum
  dates: 20 July – 3 August 1996
  competitors: 31
  nations: 31
  gold: István Kovács
  goldNOC: HUN
  silver: Arnaldo Mesa
  silverNOC: CUB
  bronze: Vichairachanon Khadpo
  bronzeNOC: THA
  bronze2: Raimkul Malakhbekov
  bronzeNOC2: RUS
  prev: 1992
  next: 2000
The Bantamweight class ...
"""


def _doc(text: str, title: str = "Boxing at the 1996 Summer Olympics – Bantamweight") -> RawDoc:
    itype, _ = parse_infobox_fields(text)
    return RawDoc(doc_id="Q1", title=title, url="http://x", text=text, infobox_type=itype)


def test_split_names_team():
    assert _split_names("Rudolf DombiRoland Kökény") == ["Rudolf Dombi", "Roland Kökény"]


def test_split_names_mc_prefix_not_split():
    assert _split_names("Danny McFarlane") == ["Danny McFarlane"]


def test_split_names_single_accented():
    assert _split_names("Félix Sánchez") == ["Félix Sánchez"]


def test_infobox_type_detected():
    doc = _doc(SAMPLE)
    assert is_olympic_event(doc)


def test_parse_event_core_fields():
    ev = parse_olympic_event(_doc(SAMPLE))
    assert ev.event == "Bantamweight boxing"
    assert ev.year == 1996
    assert ev.season == "Summer"
    assert ev.venue == "Alexander Memorial Coliseum"
    assert ev.competitors == 31
    assert ev.nations == 31
    assert ev.prev_year == 1992
    assert ev.next_year == 2000
    assert ev.sport == "Boxing"


def test_parse_tied_bronze():
    ev = parse_olympic_event(_doc(SAMPLE))
    bronze = [m for m in ev.medalists if m.rank == "bronze"]
    assert len(bronze) == 2
    nocs = {m.noc for m in bronze}
    assert nocs == {"THA", "RUS"}


def test_non_olympic_infobox():
    text = "[Infobox film]\n  name: Some Movie\n  director: Someone\nA film about ...\n"
    doc = _doc(text, title="Some Movie")
    assert not is_olympic_event(doc)
