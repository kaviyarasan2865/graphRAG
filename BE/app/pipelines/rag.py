"""Pipeline 1: plain RAG.

Embed the question -> vector-search the corpus -> stuff top-k document text
into the LLM -> answer. No graph. This is the baseline; it does well on lookup
questions and poorly on aggregation/superlative/multi-hop.
"""

from __future__ import annotations

from app.config import settings

from .retrieval import get_retriever
from .synthesize import synthesize
from .types import AnswerTraceStep, PipelineResult

DOC_CHAR_LIMIT = 1500


def _context_from_docs(docs) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(f"[{i}] doc_id={d.doc_id} | {d.title}\n{d.text[:DOC_CHAR_LIMIT]}")
    return "\n\n".join(blocks)


def run_rag(question: str, *, top_k: int | None = None) -> PipelineResult:
    top_k = top_k or settings.rag_top_k
    retriever = get_retriever()

    docs = retriever.vector_search(question, top_k)
    trace = [
        AnswerTraceStep(
            action="vector_search",
            detail=f"top_k={top_k}",
            result_summary=f"{len(docs)} docs: {[d.doc_id for d in docs]}",
        )
    ]

    context = _context_from_docs(docs)
    answer, usage = synthesize(question, context)
    trace.append(AnswerTraceStep("synthesize", "LLM answer from top-k docs", answer))

    return PipelineResult(
        question=question,
        pipeline="rag",
        answer=answer,
        citations=[d.doc_id for d in docs],
        retrieved_doc_ids=[d.doc_id for d in docs],
        trace=trace,
        usage=usage,
    )
