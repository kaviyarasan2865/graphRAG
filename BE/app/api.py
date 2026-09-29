"""FastAPI service exposing the three pipelines + a benchmark trigger.

Run:  uvicorn app.api:app --reload

Endpoints:
  GET  /health              provider + retriever status
  POST /ask                 answer one question with a chosen pipeline
  POST /compare             answer one question with ALL three pipelines
  POST /benchmark           run the benchmark over the public eval set
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import BASE_DIR, settings
from app.pipelines.agentic import run_agentic
from app.pipelines.graphrag import run_graphrag
from app.pipelines.rag import run_rag

app = FastAPI(
    title="Olympics Investigator - Agentic GraphRAG",
    version="0.1.0",
    description="Answers Olympic-events questions three ways and compares them.",
)

_STATIC_DIR = BASE_DIR / "static"

Pipeline = Literal["rag", "graphrag", "agentic"]

_RUNNERS = {"rag": run_rag, "graphrag": run_graphrag, "agentic": run_agentic}


AgentImpl = Literal["graph", "react"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    pipeline: Pipeline = "agentic"
    agent_impl: AgentImpl | None = Field(
        None, description="Agentic backend override: 'graph' (default) or 'react'."
    )


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=3)
    pipelines: list[Pipeline] = ["rag", "graphrag", "agentic"]
    agent_impl: AgentImpl | None = None


class BenchmarkRequest(BaseModel):
    limit: int = Field(0, ge=0, description="0 = all public questions")
    pipelines: list[Pipeline] = ["rag", "graphrag", "agentic"]


@app.get("/health")
def health() -> dict:
    from app.graph.client import get_client
    from app.providers.factory import active_llm_name

    tg_ok = False
    try:
        tg_ok = get_client().ping()
    except Exception:  # noqa: BLE001
        tg_ok = False

    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "active_llm": active_llm_name(),
        "embedding_provider": settings.embedding_provider,
        "retriever": "TigerGraphRetriever",
        "tigergraph_reachable": tg_ok,
        "graph": settings.tg_graph,
        "agent_impl": settings.agent_impl,
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if req.pipeline == "agentic":
        result = run_agentic(req.question, impl=req.agent_impl)
    else:
        result = _RUNNERS[req.pipeline](req.question)
    return result.as_dict()


@app.post("/compare")
def compare(req: CompareRequest) -> dict:
    out: dict = {}
    for name in req.pipelines:
        if name == "agentic":
            out[name] = run_agentic(req.question, impl=req.agent_impl).as_dict()
        else:
            out[name] = _RUNNERS[name](req.question).as_dict()
    return {"question": req.question, "results": out}


@app.post("/benchmark")
def benchmark(req: BenchmarkRequest) -> dict:
    # Imported lazily so the API can boot even if eval deps aren't ready.
    from app.eval.run import run_benchmark

    return run_benchmark(limit=req.limit, pipelines=req.pipelines)


@app.get("/benchmark/latest")
def benchmark_latest() -> dict:
    """Return the most recently saved benchmark results (results/benchmark.json)."""
    import json

    from app.config import BASE_DIR

    path = BASE_DIR / "results" / "benchmark.json"
    if not path.exists():
        return {"available": False, "message": "No benchmark run yet."}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"available": True, **data}


# --- static HTML dashboard ------------------------------------------------
# Served last so API routes take precedence. GET / returns the dashboard.


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
