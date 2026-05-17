#!/usr/bin/env bash
set -euo pipefail

DOCS_DIR="$(cd "$(dirname "$0")" && pwd)"
ASSETS_DIR="$DOCS_DIR/assets/embedding-platform"
mkdir -p "$ASSETS_DIR"

echo "Rendering Patra Embedding Platform diagrams..."

mmdc -i <(cat <<'MERMAID'
graph TB
    subgraph Clients
        WebUI["Web Frontend"]
        SDK["Patra Toolkit"]
        Agent["AI Agent"]
        ChatBot["RAG Chat Bot"]
    end
    subgraph Backend["Patra Backend"]
        REST["REST Server"]
        MCP_GW["MCP Gateway"]
        EMB["Embedding Engine"]
        IDX["Indexing Service"]
        RET["Retrieval Service"]
        PG[("PostgreSQL")]
        MCP_GW --> REST
        REST --> PG
        REST --> IDX
        REST --> RET
        IDX --> EMB
        RET --> EMB
        IDX --> PG
    end
    subgraph ICICLE["Shared ICICLE Infrastructure"]
        QDRANT[("Qdrant")]
        LLM["LLM Gateway"]
    end
    WebUI --> REST
    SDK --> REST
    Agent --> MCP_GW
    Agent --> LLM
    ChatBot --> REST
    IDX --> QDRANT
    RET --> QDRANT
    EMB -.-> LLM
    REST --> LLM
MERMAID
) -o "$ASSETS_DIR/architecture.svg" -t default
echo "  ✓ architecture.svg"

mmdc -i <(cat <<'MERMAID'
sequenceDiagram
    actor User
    participant REST as REST Server
    participant PG as PostgreSQL
    participant IDX as Indexing Service
    participant EMB as Embedding Engine
    participant QD as Qdrant
    User->>REST: Create / Update model card
    REST->>PG: INSERT / UPDATE
    REST-->>User: 201 Created
    REST->>IDX: index(asset_type, id)
    IDX->>PG: SELECT metadata fields
    IDX->>EMB: embed(name + desc + keywords + ...)
    EMB-->>IDX: 384-dim vector
    IDX->>QD: upsert(id, vector, payload)
MERMAID
) -o "$ASSETS_DIR/indexing-flow.svg" -t default
echo "  ✓ indexing-flow.svg"

mmdc -i <(cat <<'MERMAID'
sequenceDiagram
    actor User
    participant REST as REST Server
    participant RET as Retrieval Service
    participant EMB as Embedding Engine
    participant QD as Qdrant
    participant LLM as LLM Gateway
    User->>REST: Semantic search / Ask Patra chat
    REST->>RET: search(query)
    RET->>EMB: embed(query)
    EMB-->>RET: query vector
    RET->>QD: search(vector, filters)
    QD-->>RET: top-K results
    RET-->>REST: ranked hits
    Note over REST,LLM: RAG Chat Bot path only
    REST->>LLM: chat(system + context + query)
    LLM-->>REST: grounded response
    REST-->>User: answer + citations
MERMAID
) -o "$ASSETS_DIR/retrieval-flow.svg" -t default
echo "  ✓ retrieval-flow.svg"

mmdc -i <(cat <<'MERMAID'
gantt
    title Patra Embedding Platform — Year 5
    dateFormat YYYY-MM-DD
    axisFormat %b
    section Foundation
    Config + Embedding Engine         :f1, 2026-04-21, 5d
    Qdrant client + collection        :f2, after f1, 3d
    section Indexing
    Index service (MC + DS)           :i1, after f2, 5d
    Hook into asset create/update     :i2, after i1, 3d
    Bulk reindex endpoint             :i3, after i2, 2d
    section Search API
    Semantic search endpoint          :r1, after i1, 4d
    Privacy filtering                 :r2, after r1, 2d
    section HF Ingestion Enhancement
    Similarity detection on ingest    :h1, 2026-05-15, 5d
    Augmentation via nearest cards    :h2, after h1, 5d
    section RAG Chat Bot
    Ask Patra vector retrieval        :c1, 2026-07-01, 5d
    LLM context + fallback            :c2, after c1, 3d
    section Schema Discovery
    DenseEmbeddingIndex               :s1, 2026-07-07, 5d
    Hybrid scoring                    :s2, after s1, 4d
    section Demo
    PEARC tutorial prep               :d1, 2026-07-28, 5d
MERMAID
) -o "$ASSETS_DIR/timeline.svg" -t default
echo "  ✓ timeline.svg"

echo "Done. SVGs written to $ASSETS_DIR/"
