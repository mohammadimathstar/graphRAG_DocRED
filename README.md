uv run uvicorn app:app --reload --port 8000

Add latency_ms to extraction
Add judge to RAG (after finishing RAG answer, we ask the judge to give RELEVANT/PARTIALY_Relevant/NON_RELEVANT
(an explanation)
