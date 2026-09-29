"""End-to-end ingestion: parse -> build graph rows -> embed -> load to TigerGraph.

Usage:
  python -m app.ingest.run                 # full ingest to TigerGraph
  python -m app.ingest.run --dry-run       # parse + build + embed, no DB writes
  python -m app.ingest.run --limit 200     # only first 200 docs (dev/testing)
  python -m app.ingest.run --skip-embed    # graph only, skip embeddings
  python -m app.ingest.run --skip-graph    # embeddings only (e.g. re-embed)
"""

from __future__ import annotations

import argparse

from app.config import settings
from app.graph.client import get_client
from app.ingest.embedder import embed_documents
from app.ingest.graph_builder import build_rows
from app.ingest.parser import iter_raw_docs

# Truncate very long docs before embedding to keep within model context.
EMBED_CHAR_LIMIT = 6000


def _doc_embed_text(title: str, text: str) -> str:
    body = text[:EMBED_CHAR_LIMIT]
    return f"{title}\n\n{body}" if title else body


def load_graph(client, rows) -> None:
    print("[graph] upserting vertices...")
    client.upsert_vertices("Document", rows.documents)
    client.upsert_vertices("Event", rows.events)
    client.upsert_vertices("Games", [(k, v) for k, v in rows.games.items()])
    client.upsert_vertices("Venue", [(v, {}) for v in rows.venues])
    client.upsert_vertices("Sport", [(s, {}) for s in rows.sports])
    client.upsert_vertices("Athlete", [(a, {}) for a in rows.athletes])
    client.upsert_vertices("Country", [(c, {}) for c in rows.countries])

    print("[graph] upserting edges...")
    client.upsert_edges("Event", "HAS_DOCUMENT", "Document", rows.has_document)
    client.upsert_edges("Event", "PART_OF", "Games", rows.part_of)
    client.upsert_edges("Event", "HELD_AT", "Venue", rows.held_at)
    client.upsert_edges("Event", "IN_SPORT", "Sport", rows.in_sport)
    client.upsert_edges("Event", "WON_MEDAL", "Athlete", rows.won_medal)
    client.upsert_edges("Athlete", "REPRESENTS", "Country", rows.represents)
    client.upsert_edges("Event", "FOLLOWED_BY", "Event", rows.followed_by)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no DB writes")
    ap.add_argument("--limit", type=int, default=0, help="limit docs (0 = all)")
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--skip-graph", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    print(f"[ingest] corpus = {settings.corpus_file}")
    docs = list(iter_raw_docs(settings.corpus_file))
    if args.limit:
        docs = docs[: args.limit]
    print(f"[ingest] loaded {len(docs)} docs")

    rows = build_rows(docs)
    print("[ingest] graph rows:")
    for k, v in rows.counts().items():
        print(f"    {k:14s}: {v}")

    embeddings: dict[str, list[float]] = {}
    if not args.skip_embed:
        pairs = [(d.doc_id, _doc_embed_text(d.title, d.text)) for d in docs]
        embeddings = embed_documents(pairs, batch_size=args.batch_size)
        print(f"[ingest] embedded {len(embeddings)} docs")

    if args.dry_run:
        print("[ingest] dry-run: skipping all DB writes")
        return

    client = get_client()
    if not client.ping():
        print("[ingest] WARNING: TigerGraph not reachable. Skipping DB writes.")
        print("         Configure TG_* in .env and run schema first "
              "(gsql/01_schema.gsql, gsql/02_queries.gsql).")
        return

    if not args.skip_graph:
        load_graph(client, rows)

    if embeddings and not args.skip_embed:
        print(f"[graph] upserting {len(embeddings)} document embeddings...")
        client.upsert_document_embeddings(list(embeddings.items()))

    print("[ingest] done.")


if __name__ == "__main__":
    main()
