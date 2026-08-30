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


# def get_entity_context_by_vector(
#     conn: Connection,
#     entity_ids: list[str],       # The resolved UUIDs from the DB
#     query_embedding: list[float], # The embedded user query
#     max_spans_per_entity: int = 3
# ) -> str:
#     """
#     Retrieves the most semantically relevant evidence spans for specific entities,
#     regardless of the exact relation type.
#     """
#     all_contexts = []

#     for ent_id in entity_ids:
#         try:
#             with conn.cursor() as cur:
#                 # Fetch spans where the entity is EITHER the subject or the object
#                 # NOTE: Added 't.' prefix to triple_embedding to avoid ambiguous column error
#                 cur.execute("""
#                     SELECT t.evidence_span, t.relation,
#                            obj.canonical_mention AS connected_entity
#                     FROM triples t
#                     JOIN entities obj ON t.object_entity_id = obj.entity_id
#                     WHERE t.subject_entity_id = %s AND t.valid = TRUE
#                     UNION
#                     SELECT t.evidence_span, t.relation,
#                            subj.canonical_mention AS connected_entity
#                     FROM triples t
#                     JOIN entities subj ON t.subject_entity_id = subj.entity_id
#                     WHERE t.object_entity_id = %s AND t.valid = TRUE
#                     ORDER BY t.triple_embedding <-> %s::vector  -- Vector similarity!
#                     LIMIT %s;
#                 """, (ent_id, ent_id, str(query_embedding), max_spans_per_entity))

#                 rows = cur.fetchall()

#             if rows:
#                 entity_context = f"Relevant information found:\n"
#                 for evidence, relation, connected_entity in rows:
#                     entity_context += f"- {evidence} (Relation: {relation} -> {connected_entity})\n"
#                 all_contexts.append(entity_context)

#         # except psycopg.Error as db_err:
#         #     # Catch database-specific errors (e.g., syntax error, bad vector format)
#         #     print(f"Database error retrieving context for entity {ent_id}: {db_err}")
#         #     conn.rollback()  # CRITICAL: Resets the transaction so the next entity can try again
#         #     continue

#         except Exception as e:
#             # Catch any other unexpected Python errors
#             print(f"Unexpected error processing entity {ent_id}: {e}")
#             conn.rollback()
#             continue

#     # If all entities failed, this will be an empty string, which is safe to pass to the LLM
#     return "\n---\n".join(all_contexts)
