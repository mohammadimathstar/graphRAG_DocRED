from typing import Optional
from psycopg import Connection

from src.utils.embedder import generate_embeddings_batch



def resolve_entity(conn: Connection, user_entity_string: str) -> Optional[str]:
    """
    Resolves a string to an entity_id (UUID) in the database.
    Strategy 1: Exact / Fuzzy string match on canonical name and aliases.
    Strategy 2: Semantic vector similarity.
    """
    with conn.cursor() as cur:
        try:
            # ==========================================
            # 1. Exact & Fuzzy String Match (pg_trgm)
            # ==========================================
            # We use ORDER BY similarity() DESC to get the CLOSEST match, not just any match.
            # We use array_to_string to safely search the aliases array.
            cur.execute("""
                SELECT entity_id
                FROM entities
                WHERE 
                    canonical_mention ILIKE %s  -- 1a. Try exact case-insensitive match first
                    OR %s ILIKE ANY(aliases)   -- 1b. Try exact alias match
                    OR similarity(canonical_mention, %s) > 0.3
                    OR similarity(array_to_string(aliases, ' '), %s) > 0.3
                ORDER BY 
                    similarity(canonical_mention, %s) DESC,
                    similarity(array_to_string(aliases, ' '), %s) DESC
                LIMIT 1;
            """, (
                user_entity_string, user_entity_string, 
                user_entity_string, user_entity_string, 
                user_entity_string, user_entity_string
            ))
            
            result = cur.fetchone()
            if result:
                return result[0]  # Found a reliable string match!

            # ==========================================
            # 2. Semantic Fallback (pgvector)
            # ==========================================
            # If fuzzy fails (e.g., user said "Steve Jobs' company" instead of "Apple")
            # Generate embedding for the user's string
            query_embedding = generate_embeddings_batch([user_entity_string])[0]
            
            cur.execute("""
                SELECT entity_id 
                FROM entities 
                -- Use cosine distance operator (<->). 
                -- filter out extremely distant matches
                WHERE 
                    embedding <-> %s::vector < 0.5
                ORDER BY 
                    embedding <-> %s::vector 
                LIMIT 1;
            """, (str(query_embedding),))
            
            result = cur.fetchone()
            if result:
                return result[0]
        except Exception as e:
            # 1. Print the ACTUAL error (e.g., function similarity() does not exist)
            print(f"SQL Error during entity resolution: {e}")
            
            # 2. CRITICAL: Rollback the transaction so the connection is usable again!
            conn.rollback()
            
    return None # Could not resolve entity


