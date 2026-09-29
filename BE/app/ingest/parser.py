"""Infobox parser: corpus.jsonl -> RawDoc / OlympicEvent records.

The infobox is a block of `  key: value` lines that appears right after the
`[Infobox ...]` header at the top of each document's text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from .models import Medalist, OlympicEvent, RawDoc

_HEADER_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$", re.M)
_FIELD_RE = re.compile(r"^\s{2}([A-Za-z_][\w]*):\s?(.*)$")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Split a concatenated multi-athlete medal line like
# "Rudolf DombiRoland Kökény" -> ["Rudolf Dombi", "Roland Kökény"].
# Heuristic: a lowercase letter immediately followed by an uppercase letter
# marks a name boundary in these Wikipedia infoboxes.
_NAME_SPLIT_RE = re.compile(r"(?<=[a-zß])(?=[A-ZÀ-Þ])")

# Prefixes where an internal capital is part of ONE surname, not a boundary
# (e.g. "McFarlane", "MacDonald", "O'Brien", "DeBruyne").
_NAME_PREFIX_MERGE_RE = re.compile(r"\b(Mc|Mac|O['\u2019]|De|Van|Von|Le|La|Di|Du)$")


def _to_int(val: str | None) -> int | None:
    if not val:
        return None
    m = re.search(r"\d+", val.replace(",", ""))
    return int(m.group()) if m else None


def _split_names(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    parts = _NAME_SPLIT_RE.split(raw)
    # Re-merge splits that landed inside a compound surname (McFarlane, etc.).
    merged: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if merged and _NAME_PREFIX_MERGE_RE.search(merged[-1]):
            merged[-1] = merged[-1] + part
        else:
            merged.append(part)
    return merged


def parse_infobox_fields(text: str) -> tuple[str | None, dict[str, str]]:
    """Return (infobox_type, {field: value}) from a document's text."""
    header = _HEADER_RE.search(text)
    infobox_type = header.group(1).strip() if header else None

    fields: dict[str, str] = {}
    start = header.end() if header else 0
    for line in text[start:].splitlines():
        if not line.strip():
            # a blank line after fields have begun ends the infobox
            if fields:
                break
            continue
        m = _FIELD_RE.match(line)
        if m:
            fields[m.group(1)] = m.group(2).strip()
        elif fields:
            # non-field line after fields started -> infobox is over
            break
    return infobox_type, fields


def infer_sport(title: str, event: str | None) -> str | None:
    """Olympic titles look like 'Sport at the YYYY ... – Event'."""
    m = re.match(r"(.+?)\s+at the\b", title)
    if m:
        return m.group(1).strip()
    return None


def _medalists_from_fields(fields: dict[str, str]) -> list[Medalist]:
    out: list[Medalist] = []
    for rank in ("gold", "silver", "bronze"):
        # primary + tie variants: gold, gold2, bronze2, etc.
        for suffix in ("", "2", "3"):
            name_key = f"{rank}{suffix}"
            noc_key = f"{rank}NOC{suffix}"
            raw = fields.get(name_key)
            if not raw:
                continue
            names = _split_names(raw)
            if not names:
                continue
            out.append(Medalist(names=names, noc=fields.get(noc_key), rank=rank))
    return out


def parse_olympic_event(doc: RawDoc) -> OlympicEvent:
    _, fields = parse_infobox_fields(doc.text)

    games_raw = fields.get("games")
    year = None
    season = None
    if games_raw:
        ym = _YEAR_RE.search(games_raw)
        year = int(ym.group()) if ym else None
        if "Summer" in games_raw:
            season = "Summer"
        elif "Winter" in games_raw:
            season = "Winter"

    prev_year = _to_int(fields.get("prev"))
    next_year = _to_int(fields.get("next"))

    return OlympicEvent(
        doc_id=doc.doc_id,
        title=doc.title,
        url=doc.url,
        event=fields.get("event"),
        games_raw=games_raw,
        year=year,
        season=season,
        venue=fields.get("venue"),
        date=fields.get("date") or fields.get("dates"),
        competitors=_to_int(fields.get("competitors")),
        nations=_to_int(fields.get("nations")),
        win_value=fields.get("win_value"),
        sport=infer_sport(doc.title, fields.get("event")),
        prev_year=prev_year,
        next_year=next_year,
        medalists=_medalists_from_fields(fields),
    )


def iter_raw_docs(corpus_path: str | Path) -> Iterator[RawDoc]:
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            infobox_type, _ = parse_infobox_fields(d.get("text", ""))
            yield RawDoc(
                doc_id=d["doc_id"],
                title=d.get("title", ""),
                url=d.get("url", ""),
                text=d.get("text", ""),
                infobox_type=infobox_type,
                wikidata_qid=d.get("wikidata_qid"),
                wikipedia_pageid=d.get("wikipedia_pageid"),
                approx_tokens=d.get("approx_tokens"),
            )


def is_olympic_event(doc: RawDoc) -> bool:
    return (doc.infobox_type or "").strip().lower() == "infobox olympic event"
