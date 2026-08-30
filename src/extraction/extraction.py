from psycopg import Connection

from src.extraction.extractor import InformationExtractor
from src.extraction.validator import soft_validate

from src.evaluation.extraction_evaluator import evaluate_document

from src.utils.embedder import embed_extraction
from src.utils.structures import Document

from src.db.ingest import (
    insert_document,
    insert_extraction,
    insert_evaluation,
    insert_entity,
    insert_triple,
)


def process_single_document(
    conn: Connection, doc: Document, extractor: InformationExtractor
):
    """
    a function to process a SINGLE document atomically
    """
    doc_obj = Document.from_dict(doc)
    print(f"\n--- Processing document: {doc_obj.document_id} ---")

    try:
        # A. Insert Document
        insert_document(conn, doc_obj)

        # B. Extract & Validate
        extraction_result = extractor.extract(
            doc_obj.text, document_id=doc_obj.document_id
        )
        extraction_result.data = soft_validate(
            extraction_result.data, source_text=doc_obj.text
        )
        extraction_id = extraction_result.extraction_id

        insert_extraction(conn, extraction_result)

        # C. Embed & Insert Entities
        entity_embs, triple_embs = embed_extraction(extraction_result.data)
        local_id_to_db_id = {}

        for entity, entity_emb in zip(extraction_result.data.entities, entity_embs):
            db_entity_id = insert_entity(conn, entity, entity_emb, extraction_id)
            local_id_to_db_id[entity.id] = db_entity_id

        # D. Insert Triples
        for triple, triple_emb in zip(extraction_result.data.triples, triple_embs):
            subject_uuid = local_id_to_db_id.get(triple.subject_id)
            object_uuid = local_id_to_db_id.get(triple.object_id)

            if not subject_uuid or not object_uuid:
                print(
                    f"  [WARN] Skipping hallucinated triple: {triple.subject_id} -> {triple.object_id}"
                )
                continue

            insert_triple(
                conn=conn,
                triple=triple,
                triple_embedding=triple_emb,
                extraction_id=extraction_id,
                subject_uuid=subject_uuid,
                object_uuid=object_uuid,
            )

        # E. Evaluate
        if doc_obj.entities or doc_obj.triples:
            metrics = evaluate_document(doc_obj, extraction_result.data)
            insert_evaluation(conn, metrics, extraction_id)

        # F. COMMIT per document (Atomic Transaction)
        conn.commit()
        print(f"  [SUCCESS] Document {doc_obj.document_id} committed.\n")
        return True

    except Exception as e:
        # If ANYTHING fails for this document, roll back THIS document only
        print(f"  [ERROR] Failed to process document {doc_obj.document_id}: {e}\n")
        conn.rollback()
        return False
