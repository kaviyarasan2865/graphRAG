"""Embed corpus documents with the configured provider, cached to disk.

Embeddings are cached in data/embeddings/ keyed by (provider, model, doc_id) so
re-running ingestion doesn't re-embed unchanged documents. The cache stores a
JSONL sidecar of doc_id -> vector.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import BASE_DIR, settings
from app.providers.factory import get_embeddings

CACHE_DIR = BASE_DIR / "data" / "embeddings"


def _model_name_for(emb) -> str:
    if emb.name == "ollama":
        return settings.ollama_embed_model
    if emb.name == "gemini":
        return settings.gemini_embed_model
    if emb.name == "st":
        return settings.st_embed_model
    return getattr(emb, "model_name", emb.name)


def _cache_path() -> Path:
    emb = get_embeddings()
    tag = f"{emb.name}_{_model_name_for(emb)}"
    tag = tag.replace(":", "-").replace("/", "-")
    return CACHE_DIR / f"{tag}.jsonl"


def load_cache() -> dict[str, list[float]]:
    path = _cache_path()
    if not path.exists():
        return {}
    cache: dict[str, list[float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cache[rec["doc_id"]] = rec["emb"]
    return cache


def _append_cache(records: list[tuple[str, list[float]]]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for doc_id, vec in records:
            f.write(json.dumps({"doc_id": doc_id, "emb": vec}) + "\n")


def embed_documents(
    docs: list[tuple[str, str]],
    *,
    batch_size: int = 32,
    use_cache: bool = True,
    progress: bool = True,
) -> dict[str, list[float]]:
    """Embed [(doc_id, text), ...] -> {doc_id: vector}, using/refreshing cache."""
    emb = get_embeddings()
    cache = load_cache() if use_cache else {}

    todo = [(d, t) for d, t in docs if d not in cache]
    if progress:
        print(f"[embed] provider={emb.name} total={len(docs)} cached={len(docs) - len(todo)} to_embed={len(todo)}")

    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        vectors = emb.embed([t for _, t in batch])
        new_records = [(d, v) for (d, _), v in zip(batch, vectors)]
        for d, v in new_records:
            cache[d] = v
        if use_cache:
            _append_cache(new_records)
        if progress:
            done = min(i + batch_size, len(todo))
            print(f"[embed] {done}/{len(todo)}", end="\r", flush=True)

    if progress and todo:
        print()
    return {d: cache[d] for d, _ in docs}
