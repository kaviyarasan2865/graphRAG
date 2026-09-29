"""Structured records extracted from corpus documents.

The corpus is mixed: ~2,162 Olympic-event pages plus films, people, companies,
etc. Only Olympic-event pages become rich graph structure; every document is
still embedded for vector search.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Medalist:
    """One athlete (or a single named competitor) on a medal line.

    Team events can list multiple athletes on one line; `names` holds each
    individual, `noc` the shared country code.
    """

    names: list[str]
    noc: str | None
    rank: str  # "gold" | "silver" | "bronze"


@dataclass
class RawDoc:
    """A corpus document as loaded from JSONL, plus its infobox type."""

    doc_id: str
    title: str
    url: str
    text: str
    infobox_type: str | None
    wikidata_qid: str | None = None
    wikipedia_pageid: int | None = None
    approx_tokens: int | None = None


@dataclass
class OlympicEvent:
    """A parsed 'Infobox Olympic event' document -> graph entity bundle."""

    doc_id: str
    title: str
    url: str

    # infobox fields
    event: str | None = None           # e.g. "Bantamweight boxing"
    games_raw: str | None = None       # e.g. "1996 Summer"
    year: int | None = None
    season: str | None = None          # "Summer" | "Winter"
    venue: str | None = None
    date: str | None = None            # date or dates, raw string
    competitors: int | None = None
    nations: int | None = None
    win_value: str | None = None
    sport: str | None = None           # inferred from title/event
    prev_year: int | None = None
    next_year: int | None = None

    medalists: list[Medalist] = field(default_factory=list)

    def games_key(self) -> str | None:
        if self.year and self.season:
            return f"{self.year} {self.season}"
        return self.games_raw
