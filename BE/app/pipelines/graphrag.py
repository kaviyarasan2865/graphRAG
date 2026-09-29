"""Pipeline 2: GraphRAG.

Vector-search to find seed documents, then enrich with the graph: for any
retrieved doc that is an Olympic event, pull its structured facts (medalists,
venue, games, competitors) AND its temporal neighbors. The structured facts are
added to the context alongside the raw text.

This helps multi-hop and temporal questions (relationships are explicit) but,
like RAG, it still only sees the top-k seed set, so it remains weak on
aggregation/superlative questions that require scanning ALL matching events.
"""

from __future__ import annotations

from app.config import settings

from .retrieval import get_retriever
from .synthesize import synthesize
from .types import AnswerTraceStep, PipelineResult

DOC_CHAR_LIMIT = 800


def run_graphrag(question: str, *, top_k: int | None = None) -> PipelineResult:
    top_k = top_k or settings.graphrag_top_k
    retriever = get_retriever()

    docs = retriever.vector_search(question, top_k)
    trace = [
        AnswerTraceStep(
            "vector_search", f"top_k={top_k}",
            f"{len(docs)} docs: {[d.doc_id for d in docs]}",
        )
    ]

    context_blocks: list[str] = []
    cited: list[str] = []
    enriched = 0
    for i, d in enumerate(docs, 1):
        cited.append(d.doc_id)
        block = [f"[{i}] doc_id={d.doc_id} | {d.title}"]
        facts = retriever.event_facts(d.doc_id)
        if facts:
            enriched += 1
            block.append("STRUCTURED FACTS:\n" + facts.to_context())
            # temporal neighbors help "immediately before/after" questions
            for direction in ("prev", "next"):
                nb = retriever.temporal_neighbor(d.doc_id, direction)
                if nb:
                    block.append(f"({direction} edition) {nb.title} [{nb.year} {nb.season}]")
                    if nb.doc_id not in cited:
                        cited.append(nb.doc_id)
        block.append(d.text[:DOC_CHAR_LIMIT])
        context_blocks.append("\n".join(block))

    trace.append(
        AnswerTraceStep(
            "graph_enrich", "pull structured facts + temporal neighbors",
            f"{enriched}/{len(docs)} docs enriched",
        )
    )

    context = "\n\n".join(context_blocks)
    answer, usage = synthesize(question, context)
    trace.append(AnswerTraceStep("synthesize", "LLM answer from enriched context", answer))

    return PipelineResult(
        question=question,
        pipeline="graphrag",
        answer=answer,
        citations=cited,
        retrieved_doc_ids=cited,
        trace=trace,
        usage=usage,
    )
