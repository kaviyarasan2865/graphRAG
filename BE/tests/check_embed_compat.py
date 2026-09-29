"""Verify sentence-transformers bge-m3 vectors match Ollama bge-m3 vectors.

If the cross-model cosine similarity for the same text is very high, the two
embedders share the same space and the graph loaded with Ollama bge-m3 does not
need re-embedding when we switch the query embedder to sentence-transformers.
"""

import math

from app.providers.ollama_provider import OllamaEmbeddings
from app.providers.st_provider import SentenceTransformersEmbeddings
from app.config import settings


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


TEXTS = [
    "Canoeing at the 2012 Summer Olympics - Men's K-2 1000 metres",
    "How many biathlon events at the 2018 Winter Olympics had more than 73 competitors?",
    "Naim Suleymanoglu weightlifting gold medal 1988",
]


def main():
    st = SentenceTransformersEmbeddings(settings.st_embed_model, 1024)
    ol = OllamaEmbeddings(settings.ollama_base_url, settings.ollama_embed_model, 1024)

    st_vecs = st.embed(TEXTS)
    ol_vecs = ol.embed(TEXTS)

    print(f"ST dim={len(st_vecs[0])}  Ollama dim={len(ol_vecs[0])}")
    print("\nCross-model cosine (same text, ST vs Ollama):")
    same = []
    for i, t in enumerate(TEXTS):
        c = cos(st_vecs[i], ol_vecs[i])
        same.append(c)
        print(f"  {c:.4f}  {t[:55]}")

    print("\nCross-text cosine (ST[0] vs Ollama[1], should be LOWER):")
    print(f"  {cos(st_vecs[0], ol_vecs[1]):.4f}")

    avg = sum(same) / len(same)
    print(f"\navg same-text cross-model cosine = {avg:.4f}")
    print("VERDICT:", "COMPATIBLE (no re-embed needed)" if avg > 0.95
          else "NOT compatible -- re-embed required")


if __name__ == "__main__":
    main()
