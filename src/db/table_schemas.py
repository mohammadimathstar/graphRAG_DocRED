# Enable extensions
ENABLE_EXTENSIONS_SQL = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
"""

# Drop statements (order matters due to Foreign Keys)
DROP_TABLES_SQL = """
DROP TABLE IF EXISTS rag_evaluations CASCADE;
DROP TABLE IF EXISTS qa_pairs CASCADE;
DROP TABLE IF EXISTS triples CASCADE;
DROP TABLE IF EXISTS entities CASCADE;
DROP TABLE IF EXISTS extractions CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS runs CASCADE;  
"""


# Table: documents
DOCUMENTS_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id UUID PRIMARY KEY,
    title TEXT,
    text TEXT,
    text_embedding VECTOR(768),
    ground_truth_entities JSONB,
    ground_truth_triples JSONB,
    ground_truth_triples_idx JSONB,
    is_holdout BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);
"""


RUNS_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_name TEXT,                -- e.g., "gpt-4o-mini-v1"
    instruction_version TEXT,         
    instruction TEXT,
    model_name TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

# Table: extractions
EXTRACTIONS_SQL = """
CREATE TABLE IF NOT EXISTS extractions (
    extraction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID REFERENCES runs(run_id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(document_id) ON DELETE CASCADE,
    model_name TEXT,    
    cost_usd NUMERIC(10, 6),
    input_tokens INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,    
    status TEXT,
    raw_output JSONB,
    validation_errors TEXT,    
    created_at TIMESTAMP DEFAULT NOW()
);
"""


EVALUATIONS_SQL = """
CREATE TABLE IF NOT EXISTS extraction_evals (
    evaluation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    extraction_id UUID REFERENCES extractions(extraction_id) ON DELETE CASCADE,
    evaluated_at TIMESTAMP DEFAULT NOW(),    
    entity_precision NUMERIC(10, 4),
    entity_recall NUMERIC(10, 4),
    entity_f1 NUMERIC(10, 4),
    entity_tp INTEGER,
    entity_fp INTEGER,
    entity_fn INTEGER,
    triple_precision NUMERIC(10, 4),
    triple_recall NUMERIC(10, 4),
    triple_f1 NUMERIC(10, 4),
    triple_tp INTEGER,
    triple_fp INTEGER,
    triple_fn INTEGER
);
"""


# Table: entities
ENTITIES_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    entity_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    extraction_id UUID REFERENCES extractions(extraction_id) ON DELETE CASCADE,
    local_id TEXT,
    canonical_mention TEXT,
    aliases TEXT[],
    entity_type TEXT,
    description TEXT,
    valid BOOLEAN DEFAULT TRUE,
    validation_errors JSONB,
    embedding_text TEXT,
    embedding VECTOR(768)
);
CREATE INDEX IF NOT EXISTS idx_entities_extraction ON entities(extraction_id);
CREATE INDEX IF NOT EXISTS idx_entities_embedding ON entities USING hnsw (embedding vector_cosine_ops);
"""
# Table: triples
TRIPLES_SQL = """
CREATE TABLE IF NOT EXISTS triples (
    triple_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    extraction_id UUID REFERENCES extractions(extraction_id) ON DELETE CASCADE,
    subject_entity_id UUID REFERENCES entities(entity_id) ON DELETE CASCADE,
    object_entity_id UUID REFERENCES entities(entity_id) ON DELETE CASCADE,
    relation TEXT,
    evidence_span TEXT,
    triple_text TEXT,
    triple_embedding VECTOR(768),
    valid BOOLEAN DEFAULT TRUE,
    validation_errors JSONB
);
CREATE INDEX IF NOT EXISTS idx_triples_extraction ON triples(extraction_id);
CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object_entity_id);
CREATE INDEX IF NOT EXISTS idx_triples_evidence_embedding ON triples USING hnsw (triple_embedding vector_cosine_ops);
"""

QA_PAIRS_SQL = """
-- 1. Static Q&A Dataset (Created once from your ground truth)
CREATE TABLE IF NOT EXISTS qa_pairs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id UUID REFERENCES documents(document_id) ON DELETE CASCADE,
    question TEXT,
    gold_entity TEXT,
    ground_truth_answer TEXT,
    gold_relation TEXT,
    gold_chunk_text TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

RAG_EVALUATIONS_SQL = """
CREATE TABLE IF NOT EXISTS rag_evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Grouping for Grafana
    -- run_id UUID REFERENCES runs(run_id) ON DELETE CASCADE,
    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(), 
    
    -- The specific question
    qa_id UUID REFERENCES qa_pairs(id) ON DELETE CASCADE,
    
    -- Configs
    retrieval_method TEXT,       -- e.g., 'hybrid_v1'
    generator_model TEXT,        -- e.g., 'gpt-4o'
    
    -- Retrieval Metrics (Per Question - stored as booleans for easy SQL aggregation)
    entity_resolved BOOLEAN,     -- Did the resolver find the correct entity?
    hit_at_k BOOLEAN,            -- Did we retrieve the gold chunk at all?
    hit_at_1 BOOLEAN,            -- Was the gold chunk at rank 1?
    hit_at_3 BOOLEAN,            -- Was the gold chunk in the top 3?
    mrr NUMERIC(10, 4),          -- Reciprocal rank for this specific question (1.0, 0.5, 0.0)
    
    -- LLM-as-a-Judge (Synthetic QA)
    judge_is_correct BOOLEAN,     -- True if generated answer matches ground truth
    judge_explanation TEXT,       -- Why it was correct/incorrect
    
    -- Debugging / Error Analysis payloads
    retrieved_context TEXT[],    -- What did the graph actually return?
    generated_answer TEXT
);
"""

PRODUCTION_TRACES_SQL = """
CREATE TABLE IF NOT EXISTS production_traces (
    trace_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- User Context
    session_id TEXT,                   -- Group multi-turn conversations
    user_id TEXT,                      -- Track who is asking
    
    -- The Core Data
    user_question TEXT NOT NULL,
    retrieved_context TEXT[],          -- The chunks fetched by the retriever
    generated_answer TEXT,             -- The final LLM response
    
    -- RAG Configuration
    retrieval_method TEXT,             -- 'GRAPH', 'VECTOR', etc.
    generator_model TEXT,              -- 'gpt-4o', 'gpt-4o-mini'
    
    -- Operational Metrics (Cost & Performance)    
    retrieval_input_tokens INTEGER,    -- retrieval 
    retrieval_output_tokens INTEGER,
    retrieval_cached_tokens INTEGER,
    retrieval_cost_usd NUMERIC(10, 6),
    
    generator_input_tokens INTEGER,    -- generation
    generator_output_tokens INTEGER,
    generator_cached_tokens INTEGER,
    generator_cost_usd NUMERIC(10, 6),

    latency_ms INTEGER,                -- Total time to answer (retrieval + generation)
    
    -- System Health
    status TEXT,                       -- 'success', 'failed_retrieval', 'failed_generation', 'rate_limited'
    error_message TEXT,
    
    -- Feedback (Optional, for RLHF later)
    thumbs_up BOOLEAN,
    user_feedback TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for fast querying of recent production logs
CREATE INDEX IF NOT EXISTS idx_traces_created_at ON production_traces(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_traces_session ON production_traces(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_status ON production_traces(status);
"""
