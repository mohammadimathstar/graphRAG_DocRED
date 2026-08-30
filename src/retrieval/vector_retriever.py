from psycopg import Connection


def get_unique_vector_context(
    conn: Connection, query_embedding: list[float], top_k: int = 5
) -> str:
    # 1. Over-fetch to account for duplicates we will remove
    fetch_k = top_k * 3  # Fetch 3x more than needed

    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT evidence_span 
                FROM triples 
                ORDER BY triple_embedding <-> %s::vector 
                LIMIT %s;
            """,
                (str(query_embedding), fetch_k),
            )

            # 2. Deduplicate in Python while preserving relevance order
            seen_spans = set()
            unique_contexts = []

            for row in cur.fetchall():
                span = row[0]
                if span and span not in seen_spans:
                    seen_spans.add(span)
                    unique_contexts.append(span)

                # Stop early if we hit our target top_k
                if len(unique_contexts) == top_k:
                    break

            return unique_contexts

        except Exception as e:
            print(f"Database error retrieving context using vector search: {e}")
            conn.rollback()
    return []
