"""Smoke test RAG + GraphRAG against the local backend on a few public Qs."""

import json

from app.config import settings
from app.pipelines.rag import run_rag
from app.pipelines.graphrag import run_graphrag


def load_questions(n=3):
    out = []
    with open(settings.eval_public_file) as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def main():
    qs = load_questions(3)
    for q in qs:
        print("=" * 80)
        print(f"[{q['qid']}] ({q['qtype']}) {q['question']}")
        print(f"GOLD: {q['answer']}  gold_docs={q['gold_doc_ids'][:3]}...")
        for fn, name in ((run_rag, "RAG"), (run_graphrag, "GraphRAG")):
            r = fn(q["question"])
            hit = any(d in q["gold_doc_ids"] for d in r.retrieved_doc_ids)
            print(f"  {name:9s} answer={r.answer!r}  tokens={r.usage.total_tokens}  gold_retrieved={hit}")


if __name__ == "__main__":
    main()
