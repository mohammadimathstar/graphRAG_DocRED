
import uuid
from psycopg import Connection
from psycopg.types.json import Jsonb

from src.db.connection import get_conn_from_pool, release_conn
from src.db.manager import init_db

from src.utils.structures import (
    Document, Metrics, Usage, ExtractionResult,  
    EntityEmbedding, TripleEmbedding
)
from src.utils.llm_schemas import Entity, Triple


def insert_run(
    conn: Connection,
    run: dict,
):
    
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO runs (
                run_id, run_name, instruction_version, instruction, model_name
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO NOTHING;
        """, (
            run['run_id'],
            run['run_name'],
            run['instruction_version'],
            run['instruction'],
            run['model']
        ))


def insert_document(
    conn: Connection,  
    doc: Document, 
    is_holdout: bool = False
):
    """Inserts or updates a document. Expects a dataclass/dict with .title, .text, etc."""
    
    # 1. Prepare the embedding string
    embedding_str = str(doc.text_embedding[0]) if doc.text_embedding else None

    with conn.cursor() as cursor:

        # Format the ground truth entities
        formatted_gt_entities = [
            {"canonical": ent[0], "type": ent[1], "aliases": ent[2]}
            for ent in doc.entities
        ] if doc.entities else []
        
        cursor.execute("""
            INSERT INTO documents (
                document_id, title, text, text_embedding, 
                ground_truth_entities, ground_truth_triples, 
                ground_truth_triples_idx, is_holdout
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id) DO UPDATE SET 
                title=EXCLUDED.title, 
                text=EXCLUDED.text, 
                text_embedding=EXCLUDED.text_embedding,
                ground_truth_entities=EXCLUDED.ground_truth_entities, 
                ground_truth_triples=EXCLUDED.ground_truth_triples,
                ground_truth_triples_idx=EXCLUDED.ground_truth_triples_idx,
                is_holdout=EXCLUDED.is_holdout;
        """, (
            doc.document_id, 
            doc.title, 
            doc.text, 
            embedding_str,
            Jsonb(formatted_gt_entities), 
            Jsonb(doc.triples), 
            Jsonb(doc.triples_idx), 
            is_holdout
        ))




def insert_extraction(
    conn: Connection, 
    extraction: ExtractionResult,
):
    """Inserts an LLM extraction and returns the new UUID."""

    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO extractions (
                extraction_id, run_id, document_id, model_name,                 
                cost_usd, input_tokens, output_tokens, cached_tokens,                 
                status, raw_output, validation_errors
            ) VALUES (
                %s, %s, %s, %s, 
                %s, %s, %s, %s, 
                %s, %s, %s
            )
        """, (extraction.extraction_id,
              extraction.run_id,
              extraction.document_id,
              extraction.model,
              extraction.usage.cost,
              extraction.usage.input_tokens,
              extraction.usage.output_tokens,
              extraction.usage.cached_tokens,
              extraction.status,
              extraction.raw_output,
              extraction.error            
        ))




def insert_entity(
    conn: Connection,
    entity: Entity,
    entity_embedding: EntityEmbedding,
    extraction_id: str
) -> str:
    """Inserts an entity and returns the new UUID."""
    embedding_str = str(entity_embedding.embedding) if entity_embedding.embedding else None
    
    """Inserts an entity and returns the new UUID."""
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO entities (
                extraction_id, local_id, 
                canonical_mention, aliases, entity_type, description, 
                valid, validation_errors, embedding_text, embedding
            ) VALUES (
                %s, %s, 
                %s, %s, %s, %s, 
                %s, %s, %s, %s
            ) 
            RETURNING entity_id
        """, (
            extraction_id, entity.id, 
            entity.canonical_mention, entity.aliases, entity.entity_type, entity.description, 
            entity.valid, Jsonb(entity.validation_errors) if entity.validation_errors else Jsonb([]),
            entity_embedding.text, embedding_str
        ))
        # Fetch the randomly generated UUID
        db_entity_id = cursor.fetchone()[0]        
    return db_entity_id


def insert_triple(
    conn: Connection, 
    triple: Triple, 
    triple_embedding: TripleEmbedding,
    extraction_id: str, 
    subject_uuid: str,  
    object_uuid: str    
):
    """Inserts a triple."""
    embedding_str = str(triple_embedding.embedding) if triple_embedding.embedding else None
    
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO triples (
                extraction_id, subject_entity_id, object_entity_id, 
                relation, evidence_span, triple_text, triple_embedding, 
                valid, validation_errors
            ) VALUES (
                %s, %s, %s, 
                %s, %s, %s, %s, 
                %s, %s
            );
        """, (
            extraction_id, 
            subject_uuid,  
            object_uuid,   
            triple.relation, 
            triple.evidence_span, 
            triple_embedding.text, 
            embedding_str, 
            triple.valid, 
            Jsonb(triple.validation_errors) if triple.validation_errors else Jsonb([])
        ))



def insert_evaluation(
    conn: Connection, 
    metrics: dict[str, Metrics],
    extraction_id: str,
):
    """Inserts evaluation metrics for a specific extraction."""    
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO extraction_evals (
                extraction_id,       
                entity_precision, entity_recall, entity_f1, entity_tp, entity_fp, entity_fn,          
                triple_precision, triple_recall, triple_f1, triple_tp, triple_fp, triple_fn
            ) VALUES (
                %s, 
                %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s
            )
        """, (extraction_id,
              metrics['entity_metrics'].precision,
              metrics['entity_metrics'].recall,
              metrics['entity_metrics'].f1,
              metrics['entity_metrics'].tp,
              metrics['entity_metrics'].fp,                 
              metrics['entity_metrics'].fn, 
              metrics['triple_metrics'].precision,
              metrics['triple_metrics'].recall,
              metrics['triple_metrics'].f1,
              metrics['triple_metrics'].tp,
              metrics['triple_metrics'].fp,                 
              metrics['triple_metrics'].fn, 
        ))


def insert_qa_pair(
    conn: Connection,
    doc_qa: dict,
):
    with conn.cursor() as cursor:
        try:
            cursor.execute("""
                INSERT INTO qa_pairs (
                    doc_id, question, gold_entity, 
                    ground_truth_answer, gold_relation, gold_chunk_text
                ) VALUES (
                    %s, %s, %s, %s, %s, %s
                );
            """, (
                doc_qa['doc_id'],
                doc_qa['question'],
                doc_qa['gold_entity'],
                doc_qa['gold_answer_entity'],
                doc_qa['gold_relation'],
                doc_qa['gold_chunk_text']
            ))
        except Exception as e:
            print(f"SQL Error during inserting QA pair: {e}")            
            conn.rollback()


def insert_rag_evaluation(
    conn: Connection,
    metrics: dict, 
    generator_model: str,
    run_id: str = None,
):
    """Inserts RAG evaluation metrics for Grafana dashboards."""
    with conn.cursor() as cursor:
        try:
            cursor.execute("""
                INSERT INTO rag_evaluations (
                    qa_id, -- run_id
                    retrieval_method, generator_model, entity_resolved, 
                    hit_at_k, hit_at_1, hit_at_3,
                    mrr, retrieved_context
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s
                );
            """, (
                metrics['qa_id'], metrics['retrieval_method'], 
                generator_model, metrics['entity_resolved'], 
                metrics['hit_at_k'], metrics['hit_at_1'], metrics['hit_at_3'],
                metrics['mrr'], metrics['retrieved_context']
            ))
        except Exception as e:
            print(f"SQL Error during inserting evaluation of QA: {e}")            
            conn.rollback()



def log_production_trace(
    user_question: str,
    retrieved_context: str,
    retrieval_method: str,
    retrieval_usage: Usage,
    generated_answer: str,
    generator_model: str,
    generator_usage: Usage,
    latency_ms: int,
    status: str = "success",
    session_id: str = None,
    error_message: str = None
):
    """Inserts a production RAG trace. Fails silently to not break the app."""
    trace_id = str(uuid.uuid4())

    try:
        conn = get_conn_from_pool()
    except Exception as e:
        print(f"[POOL ERROR] Failed to get connection for logging: {e}")
        return None
    
    init_db()            
    
    try:        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO production_traces (
                    trace_id, session_id,  user_question, retrieved_context, 
                    generated_answer, retrieval_method, generator_model, 
                    retrieval_input_tokens, retrieval_output_tokens,
                    retrieval_cached_tokens, retrieval_cost_usd,
                    generator_input_tokens, generator_output_tokens,
                    generator_cached_tokens, generator_cost_usd,
                    latency_ms, status, error_message
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                    %s, %s, %s, %s, %s, %s, %s, %s, %s          
                )
            """, (
                trace_id, session_id, user_question, 
                retrieved_context, generated_answer,
                retrieval_method, generator_model,                
                retrieval_usage.input_tokens, 
                retrieval_usage.output_tokens, 
                retrieval_usage.cached_tokens, 
                retrieval_usage.cost,
                generator_usage.input_tokens, generator_usage.output_tokens, 
                generator_usage.cached_tokens, generator_usage.cost, 
                latency_ms, status, error_message
            ))

        conn.commit()
    except Exception as e:
        print(f"[DB ERROR] Failed to log production trace: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            release_conn(conn)
        
    return trace_id
    
def update_trace_feedback(trace_id: str, thumbs_up: bool, user_feedback: str):
    """Updates a production trace with user feedback."""
    conn = get_conn_from_pool()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE production_traces 
                SET thumbs_up = %s, user_feedback = %s 
                WHERE trace_id = %s;
            """, (thumbs_up, user_feedback, trace_id))
        conn.commit()
    except Exception as e:
        print(f"Error updating feedback: {e}")
        conn.rollback()
    finally:
        release_conn(conn)
        
