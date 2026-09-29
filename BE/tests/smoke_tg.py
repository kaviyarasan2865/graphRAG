"""Live smoke test against the connected TigerGraph. Not part of pytest."""

from app.pipelines.retrieval import get_retriever


def main() -> None:
    r = get_retriever()

    print("=== vector_search ===")
    for d in r.vector_search("men canoe sprint K-2 1000 metres 2012", 3):
        print(f"  {d.doc_id} score={d.score:.3f} title={d.title[:55]!r}")

    print("=== events_by_games_sport (2018 Winter Biathlon) ===")
    evs = r.events_by_games_sport(2018, "Winter", "Biathlon")
    comps = sorted([e.competitors for e in evs if e.competitors is not None], reverse=True)
    print(f"  {len(evs)} events; competitors={comps}")
    over = [e for e in evs if e.competitors and e.competitors > 73]
    print(f"  events with >73 competitors: {len(over)} (expect 5)")

    print("=== event_by_doc ===")
    f = r.event_facts("Q303623")
    if f:
        print(f"  {f.title[:50]!r} year={f.year} competitors={f.competitors} medalists={len(f.medalists)}")


if __name__ == "__main__":
    main()
