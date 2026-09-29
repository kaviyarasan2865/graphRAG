"""Smoke test the agentic pipeline on a spread of public question types."""

import json

from app.config import settings
from app.pipelines.agentic import run_agentic


def load_questions(qids):
    out = {}
    with open(settings.eval_public_file) as f:
        for line in f:
            q = json.loads(line)
            if q["qid"] in qids:
                out[q["qid"]] = q
    return out


def main():
    # aggregation (pub-001, pub-003), superlative (pub-004),
    # temporal (pub-002), multi-hop (pub-005)
    qids = ["pub-001", "pub-002", "pub-003", "pub-004", "pub-005"]
    qs = load_questions(qids)
    for qid in qids:
        q = qs[qid]
        r = run_agentic(q["question"])
        hit = any(d in q["gold_doc_ids"] for d in r.retrieved_doc_ids)
        print("=" * 80)
        print(f"[{qid}] ({q['qtype']}) {q['question']}")
        print(f"GOLD: {q['answer']}")
        print(f"AGENT: {r.answer!r}  tokens={r.usage.total_tokens}  gold_retrieved={hit}")
        print("tools used:", [s.action for s in r.trace if s.action not in ('plan', 'synthesize')])


if __name__ == "__main__":
    main()
