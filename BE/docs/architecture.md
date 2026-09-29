# Architecture — Olympics Investigator (Agentic GraphRAG)

Agentic GraphRAG over an Olympic-events corpus, **powered by TigerGraph Savanna**.
Every question is answered three ways — RAG, GraphRAG, and Agentic GraphRAG — and
benchmarked on accuracy, retrieval completeness, and token efficiency.

> Rendered images: [architecture.png](./architecture.png) · [architecture.svg](./architecture.svg).
> The Mermaid source below is kept for reference.

```mermaid
flowchart TB
    %% ---------------- Client ----------------
    subgraph CLIENT["Client"]
        UI["HTML / JS Dashboard — served at /<br/>pick pipelines · answers · trace · benchmark"]
    end

    %% ---------------- API ----------------
    subgraph SVC["API — FastAPI · app/api.py"]
        API["/ask · /compare · /benchmark · /health"]
    end

    %% ---------------- Pipelines ----------------
    subgraph PIPE["Answering pipelines · app/pipelines"]
        RAG["RAG<br/>vector top-k → synthesize"]
        GRAG["GraphRAG<br/>vector + graph facts + prev/next → synthesize"]
        AG["Agentic — LangGraph StateGraph<br/>plan → act → synthesize<br/>tools · stop + loop-guard · alt: create_react_agent"]
    end

    %% ---------------- Shared services ----------------
    subgraph CORE["Shared services · app/providers"]
        EMB["Embeddings<br/>sentence-transformers BAAI/bge-m3<br/>1024-dim · in-process"]
        LLM["FallbackLLM<br/>Gemini gemini-3.7-flash → Ollama qwen3:14b<br/>fallback on 429 / 503"]
        TGR["TigerGraphRetriever<br/>REST++ · installed GSQL"]
    end

    %% ---------------- Data store ----------------
    subgraph DB["TigerGraph Savanna v4.2.5"]
        TG["Graph — Event / Games / Venue / Sport / Athlete / Country<br/>Vector — Document.emb · HNSW · 1024-dim · COSINE"]
    end

    %% ---------------- Offline build + eval ----------------
    CORPUS["dataset/corpus.jsonl · 2,951 docs"]
    INGEST["Ingestion · app/ingest<br/>parse infobox → vertices/edges + embed"]
    BENCH["Benchmark · app/eval<br/>accuracy · completeness · tokens<br/>→ results/*.json,csv"]

    %% ---------------- Flows ----------------
    UI -->|HTTP JSON| API
    API --> RAG
    API --> GRAG
    API --> AG
    API --> BENCH

    RAG --> EMB & LLM & TGR
    GRAG --> EMB & LLM & TGR
    AG --> EMB & LLM & TGR

    EMB -.query vector.-> TGR
    TGR --> TG

    CORPUS --> INGEST -->|upsert graph + embeddings| TG

    %% ---------------- Colors ----------------
    classDef client  fill:#eef4ff,stroke:#2f6bff,color:#12203f;
    classDef api     fill:#e7f0ff,stroke:#2f6bff,color:#12203f;
    classDef rag     fill:#eaf1ff,stroke:#2f6bff,color:#12203f;
    classDef graphrag fill:#f3ecff,stroke:#8a4fff,color:#2a1a4f;
    classDef agentic fill:#e7f8f0,stroke:#12a66a,color:#0c3d29;
    classDef core    fill:#fffdf5,stroke:#c9a227,color:#3d3410;
    classDef db      fill:#fff3e6,stroke:#ee7600,color:#4a2c00;
    classDef offline fill:#f4f6fa,stroke:#8a93a6,color:#33415c;

    class UI client;
    class API api;
    class RAG rag;
    class GRAG graphrag;
    class AG agentic;
    class EMB,LLM,TGR core;
    class TG db;
    class CORPUS,INGEST,BENCH offline;

    style CLIENT fill:#f7faff,stroke:#c7d6f5;
    style SVC fill:#f7faff,stroke:#c7d6f5;
    style PIPE fill:#fbfaff,stroke:#dfd6f0;
    style CORE fill:#fffef8,stroke:#ece0b8;
    style DB fill:#fff8f0,stroke:#f6d6ad;
```

## Stack

| Layer | Choice |
|---|---|
| Graph + Vector DB | TigerGraph Savanna v4.2.5 (GSQL, native HNSW vector index) |
| DB client | pyTigerGraph (REST++, secret auth) |
| Retrieval | TigerGraph only — installed GSQL queries |
| Embeddings | sentence-transformers `BAAI/bge-m3`, 1024-dim, in-process |
| LLM | Gemini `gemini-3.7-flash` → automatic fallback to Ollama `qwen3:14b` |
| Agent framework | LangGraph `StateGraph` (default) / prebuilt `create_react_agent` |
| API + Dashboard | FastAPI serving JSON + static HTML/JS at `/` |
| Eval / tests | pandas metrics · pytest |

**Where each pipeline wins:** lookup → RAG; temporal → GraphRAG; multi-hop /
aggregation / superlative → Agentic (scans all matching events via graph
queries, which top-k retrieval cannot).
