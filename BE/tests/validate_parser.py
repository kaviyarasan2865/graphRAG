"""Validate the infobox parser against the real corpus. Not a pytest test."""

import collections

from app.config import settings
from app.ingest.parser import iter_raw_docs, is_olympic_event, parse_olympic_event


def main() -> None:
    path = settings.corpus_file
    total = 0
    olympic = 0
    infobox_types = collections.Counter()
    missing_year = 0
    missing_competitors = 0
    no_medalists = 0
    tie_bronze = 0
    team_events = 0
    examples = []

    for doc in iter_raw_docs(path):
        total += 1
        infobox_types[doc.infobox_type or "<none>"] += 1
        if not is_olympic_event(doc):
            continue
        olympic += 1
        ev = parse_olympic_event(doc)
        if ev.year is None:
            missing_year += 1
        if ev.competitors is None:
            missing_competitors += 1
        if not ev.medalists:
            no_medalists += 1
        bronze = [m for m in ev.medalists if m.rank == "bronze"]
        if len(bronze) >= 2:
            tie_bronze += 1
        if any(len(m.names) >= 2 for m in ev.medalists):
            team_events += 1
        if len(examples) < 3 and ev.medalists and any(len(m.names) >= 2 for m in ev.medalists):
            examples.append(ev)

    print(f"total docs           : {total}")
    print(f"olympic events       : {olympic}")
    print(f"missing year         : {missing_year}")
    print(f"missing competitors  : {missing_competitors}")
    print(f"olympic w/o medalists: {no_medalists}")
    print(f"tied-bronze events   : {tie_bronze}")
    print(f"team (multi-athlete) : {team_events}")
    print("\ntop infobox types:")
    for k, c in infobox_types.most_common(8):
        print(f"  {c:5d}  {k}")

    print("\nsample parsed team events:")
    for ev in examples:
        print(f"  [{ev.doc_id}] {ev.event} | {ev.games_key()} | venue={ev.venue} | "
              f"competitors={ev.competitors} | sport={ev.sport}")
        for m in ev.medalists:
            print(f"      {m.rank:6s} {m.noc}: {m.names}")


if __name__ == "__main__":
    main()
