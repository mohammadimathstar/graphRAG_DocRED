from psycopg import Connection

from src.retrieval.resolver import resolve_entity
from src.retrieval.graph_retriever import get_entity_context_by_vector
from src.retrieval.vector_retriever import get_unique_vector_context
from src.utils.embedder import generate_embeddings_batch
from src.extraction.extractor import InformationExtractor


def retrieve_context(
    conn: Connection,
    router: InformationExtractor,
    query: str,
    max_spans_per_entity: int = 3,
) -> dict[str]:
    result = router.extract(
        text=query,
    )

    strategy = result.data.strategy

    query_embedding = generate_embeddings_batch([query])[0]

    if strategy == "GRAPH":
        entities_uuid = [
            resolve_entity(conn=conn, user_entity_string=ent)
            for ent in result.data.entities
        ]

        context = get_entity_context_by_vector(
            conn=conn,
            entity_ids=entities_uuid,
            entity_names=result.data.entities,
            query_embedding=query_embedding,
            max_spans_per_entity=max_spans_per_entity,
        )

    elif strategy == "VECTOR":
        context = get_unique_vector_context(
            conn=conn, query_embedding=query_embedding, top_k=max_spans_per_entity
        )

    else:
        raise f"The strategy should be 'GRAPH' or 'VECTOR', but it is {strategy}!"

    return {
        "strategy": strategy,
        "context": context,
        "usage": result.usage,
        "extractor_status": result.status,
        "extractor_error": result.error,
    }
