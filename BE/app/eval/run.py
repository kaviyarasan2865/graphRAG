"""Benchmark runner: score RAG vs GraphRAG vs Agentic on the public eval set.

Usage:
  python -m app.eval.run                    # all public Qs, all 3 pipelines
  python -m app.eval.run --limit 20         # first 20 questions
  python -m app.eval.run --pipelines agentic
  python -m app.eval.run --predict-hidden   # produce answers for eval_hidden

Writes JSON + CSV to results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from app.config import BASE_DIR, settings
from app.pipelines.agentic import run_agentic
from app.pipelines.graphrag import run_graphrag
from app.pipelines.rag import run_rag

from .metrics import answer_matches, completeness

RUNNERS = {"rag": run_rag, "graphrag": run_graphrag, "agentic": run_agentic}
RESULTS_DIR = BASE_DIR / "results"


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def run_benchmark(
    *, limit: int = 0, pipelines: list[str] | None = None, verbose: bool = False
) -> dict:
    pipelines = pipelines or ["rag", "graphrag", "agentic"]
    questions = _load_jsonl(settings.eval_public_file)
    if limit:
        questions = questions[:limit]

    per_question: list[dict] = []
    # aggregates: pipeline -> metric sums
    agg = {p: defaultdict(float) for p in pipelines}
    agg_by_type = {p: defaultdict(lambda: defaultdict(float)) for p in pipelines}

    for q in questions:
        row: dict = {"qid": q["qid"], "qtype": q["qtype"], "question": q["question"],
                     "gold": q["answer"]}
        for p in pipelines:
            t0 = time.time()
            res = RUNNERS[p](q["question"])
            dt = time.time() - t0

            acc = 1.0 if answer_matches(res.answer, q["answer"]) else 0.0
            comp = completeness(res.retrieved_doc_ids, q.get("gold_doc_ids", []))
            toks = res.usage.total_tokens

            row[f"{p}_answer"] = res.answer
            row[f"{p}_acc"] = acc
            row[f"{p}_completeness"] = round(comp, 3)
            row[f"{p}_tokens"] = toks
            row[f"{p}_seconds"] = round(dt, 2)

            agg[p]["n"] += 1
            agg[p]["accuracy"] += acc
            agg[p]["completeness"] += comp
            agg[p]["tokens"] += toks
            agg[p]["seconds"] += dt

            qt = q["qtype"]
            agg_by_type[p][qt]["n"] += 1
            agg_by_type[p][qt]["accuracy"] += acc

            if verbose:
                mark = "OK " if acc else "XX "
                print(f"  [{p:8s}] {mark} {res.answer!r} (gold {q['answer']}) "
                      f"comp={comp:.2f} tok={toks}")
        if verbose:
            print(f"[{q['qid']}] {q['qtype']}: {q['question'][:70]}")
        per_question.append(row)

    summary = {}
    for p in pipelines:
        n = agg[p]["n"] or 1
        summary[p] = {
            "n": int(agg[p]["n"]),
            "accuracy": round(agg[p]["accuracy"] / n, 4),
            "completeness": round(agg[p]["completeness"] / n, 4),
            "avg_tokens": round(agg[p]["tokens"] / n, 1),
            "total_tokens": int(agg[p]["tokens"]),
            "avg_seconds": round(agg[p]["seconds"] / n, 2),
            "by_qtype": {
                qt: round(v["accuracy"] / (v["n"] or 1), 4)
                for qt, v in sorted(agg_by_type[p].items())
            },
        }

    result = {"n_questions": len(questions), "pipelines": pipelines, "summary": summary,
              "per_question": per_question}
    _write_results(result)
    return result


def _write_results(result: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = result["per_question"]
    if rows:
        fields = list(rows[0].keys())
        with open(RESULTS_DIR / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, list) else v)
                            for k, v in r.items()})


def predict_hidden(pipeline: str = "agentic") -> Path:
    """Produce answers for the hidden eval set (no scoring)."""
    questions = _load_jsonl(settings.eval_hidden_file)
    runner = RUNNERS[pipeline]
    out = []
    for q in questions:
        res = runner(q["question"])
        out.append({"qid": q["qid"], "question": q["question"],
                    "answer": res.answer, "citations": res.citations[:10]})
        print(f"[{q['qid']}] {res.answer!r}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"hidden_predictions_{pipeline}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _print_summary(result: dict) -> None:
    print("\n" + "=" * 70)
    print(f"BENCHMARK SUMMARY  (n={result['n_questions']})")
    print("=" * 70)
    header = f"{'pipeline':10s} {'accuracy':>9s} {'complete':>9s} {'avg_tok':>9s} {'avg_sec':>8s}"
    print(header)
    for p, s in result["summary"].items():
        print(f"{p:10s} {s['accuracy']:>9.3f} {s['completeness']:>9.3f} "
              f"{s['avg_tokens']:>9.1f} {s['avg_seconds']:>8.2f}")
    print("\nAccuracy by question type:")
    for p, s in result["summary"].items():
        print(f"  {p:10s} " + "  ".join(f"{qt}={acc:.2f}" for qt, acc in s["by_qtype"].items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pipelines", nargs="+",
                    default=["rag", "graphrag", "agentic"],
                    choices=["rag", "graphrag", "agentic"])
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--predict-hidden", action="store_true")
    ap.add_argument("--hidden-pipeline", default="agentic",
                    choices=["rag", "graphrag", "agentic"])
    args = ap.parse_args()

    if args.predict_hidden:
        path = predict_hidden(args.hidden_pipeline)
        print(f"\nWrote hidden predictions to {path}")
        return

    result = run_benchmark(limit=args.limit, pipelines=args.pipelines, verbose=args.verbose)
    _print_summary(result)
    print(f"\nWrote results/benchmark.json and results/benchmark.csv")


if __name__ == "__main__":
    main()
