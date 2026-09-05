from psycopg import Connection


def get_entity_context_by_vector(
    conn: Connection,
    entity_ids: list[str],
    entity_names: list[str],
    query_embedding: list[float],
    max_spans_per_entity: int = 3,
) -> str:
    """
    Retrieves the most semantically relevant evidence spans for specific entities,
    regardless of the exact relation type.
    """
    all_contexts = []

    for ent_id, ent_name in zip(entity_ids, entity_names):
        try:
            with conn.cursor() as cur:
                # Single query with conditional JOIN (no UNION needed!)
                cur.execute(
                    """
                    SELECT t.evidence_span, t.relation, 
                           other.canonical_mention AS connected_entity
                    FROM triples t
                    JOIN entities other ON 
                        (t.subject_entity_id = %s AND other.entity_id = t.object_entity_id) OR
                        (t.object_entity_id = %s AND other.entity_id = t.subject_entity_id)
                    WHERE t.valid = TRUE
                      AND (t.subject_entity_id = %s OR t.object_entity_id = %s)
                    ORDER BY t.triple_embedding <-> %s::vector
                    LIMIT %s;
                """,
                    (
                        ent_id,
                        ent_id,
                        ent_id,
                        ent_id,
                        str(query_embedding),
                        max_spans_per_entity,
                    ),
                )

                rows = cur.fetchall()

            if rows:
                for evidence, relation, connected_entity in rows:
                    chunk = f"[Context about {ent_name}] {evidence} (Relation: {relation} -> {connected_entity})"
                    all_contexts.append(chunk)

        except Exception as e:
            print(f"Database error retrieving context for entity {ent_id}: {e}")
            conn.rollback()

    return all_contexts

