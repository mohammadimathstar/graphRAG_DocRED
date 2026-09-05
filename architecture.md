# Architecture & Design Decisions

This document outlines the architectural best practices implemented in this GraphRAG system.

## 1. Hybrid Agentic RAG vs Standard RAG

Standard RAG relies solely on vector similarity search. This fails when users ask specific multi-hop questions (e.g., "Who is the CEO of the company that founded X?").

**Solution:** We implemented an LLM-based Query Router.

- If the query contains specific entities, the router extracts them and routes to the Graph Retriever.
- If the query is thematic, it routes to the Vector Retriever.
- This optimizes both precision (graph) and recall (vector).

## 2. Strict vs Soft Validation

LLMs are non-deterministic and often output malformed data.

- **Strict Validation (Pydantic):** We enforce a strict schema for JSON structure, local IDs, and referential integrity (no dangling entity references). If these fail, the pipeline retries.
- **Soft Validation (Python Post-processing):** We check semantic constraints (e.g., a place of birth relation must have a Person as the subject and a Place as the object). Instead of raising errors, we flag these triples as valid=False in the database, ensuring no data is lost while keeping the active retrieval graph clean.

## 3. Connection Pooling & Transaction Safety

To handle concurrent web requests in FastAPI, we use psycopg-pool.

- **Retrieval connections** are checked out from the pool per request.
- **Logging connections** are isolated. If the RAG pipeline crashes, the logging transaction is unaffected, ensuring errors are always captured in production_traces.

## 4. Separation of Offline vs Online Pipelines

- **Offline (Scripts):** The `scripts/` folder handles batch processing, LLM extraction, and evaluation. These scripts commit transactions per document to ensure progress isn't lost during long runs.
- **Online (FastAPI):** The `app.py` server only reads from the graph and generates answers. It is lightweight and stateless, allowing it to scale horizontally if needed.
