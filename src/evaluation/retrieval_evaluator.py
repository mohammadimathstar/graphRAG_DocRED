from psycopg import Connection
from tqdm import tqdm
import re
import string
from typing import Any

from src.retrieval.retrieval import retrieve_context
from src.extraction.extractor import InformationExtractor


def normalize_text(text: str) -> str:
    """Normalizes text for robust substring matching."""
    if not text:
        return ""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def get_evaluation_set(conn: Connection) -> list[dict]:
    with conn.cursor() as cur:
        try:
            cur.execute("""
            SELECT *
            FROM qa_pairs
            """)
            rows = cur.fetchall()

            eval_list = []
            if rows:
                for row in rows:
                    eval_list.append(
                        {
                            "qa_id": row[0],
                            "doc_id": row[1],
                            "question": row[2],
                            "gold_entity": row[3],
                            "ground_truth_answer": row[4],
                            "gold_relation": row[5],
                            "gold_chunk_text": row[6],
                        }
                    )
                return eval_list
        except Exception as e:
            print(f"Database error getting evaluation set: {e}")
            conn.rollback()
    return []


def evaluate_retrieval(
    result: Any, qa: list[dict]
) -> dict:
    
    gold_chunk_norm = normalize_text(qa["gold_chunk_text"])
    gold_entity_norm = normalize_text(qa["gold_entity"])

    # 1. Retrieve context
    retrieved_list = result["context"]

    if not retrieved_list:
        return {
            "hit_at_k": False,
            "hit_at_1": False,  # Top 1 retrieval
            "hit_at_3": False,  # Top 3 retrieval
            "mrr": round(0, 4),  # Mean Reciprocal Rank
            "entity_resolved": False,
            "retrieved_context": [""],
        }

    # 2. Evaluate Entity Resolution (across the whole retrieved context)
    combined_context_norm = normalize_text(" ".join(retrieved_list))
    if gold_entity_norm in combined_context_norm:
        entity_resolve = True

    # 3. Evaluate Chunk Retrieval (Rank-aware)
    found_rank = None
    for rank, chunk in enumerate(retrieved_list):
        chunk_norm = normalize_text(chunk)

        # If the gold chunk is found in this ranked position
        if gold_chunk_norm in chunk_norm:
            found_rank = rank + 1  # ranks are 1-indexed
            break

    hit_1 = False
    hit_3 = False
    hit_k = False
    if found_rank is not None:
        hit_k = True

        if found_rank == 1:
            hit_1 = True
        if found_rank <= 3:
            hit_3 = True

        # MRR: 1/rank. e.g., if found at rank 2, score is 0.5
        reciprocal_rank = 1.0 / found_rank
    else:
        reciprocal_rank = 0.0  # Not found, adds 0 to MRR

    return {
        "hit_at_k": hit_k,
        "hit_at_1": hit_1,  # Top 1 retrieval
        "hit_at_3": hit_3,  # Top 3 retrieval
        "mrr": round(reciprocal_rank, 4),  # Mean Reciprocal Rank
        "entity_resolved": entity_resolve,
        "retrieved_context": retrieved_list,
    }


def evaluate_retrieval_batch(
    conn: Connection, router: InformationExtractor, eval_dataset: list[dict]
) -> dict:
    # Initialize counters
    hit_1_count = 0
    hit_3_count = 0
    hit_k_count = 0
    entity_resolve_count = 0

    graph_strategy = 0
    vector_strategy = 0

    reciprocal_ranks = []  # For MRR calculation

    for item in tqdm(eval_dataset, desc="Evaluating Retrieval"):
        question = item["question"]
        gold_chunk_norm = normalize_text(item["gold_chunk_text"])
        gold_entity_norm = normalize_text(item["gold_entity"])

        # 1. Retrieve context
        result = retrieve_context(conn, router, question)
        retrieved_list = result["context"]

        if result["strategy"] == "GRAPH":
            graph_strategy += 1
        elif result["strategy"] == "VECTOR":
            vector_strategy += 1

        # 2. Evaluate Entity Resolution (across the whole retrieved context)
        combined_context_norm = normalize_text(" ".join(retrieved_list))
        if gold_entity_norm in combined_context_norm:
            entity_resolve_count += 1

        # 3. Evaluate Chunk Retrieval (Rank-aware)
        found_rank = None
        for rank, chunk in enumerate(retrieved_list):
            chunk_norm = normalize_text(chunk)

            # If the gold chunk is found in this ranked position
            if gold_chunk_norm in chunk_norm:
                found_rank = rank + 1  # ranks are 1-indexed
                break

        if found_rank is not None:
            hit_k_count += 1  # It was found somewhere in the retrieved list

            if found_rank == 1:
                hit_1_count += 1
            if found_rank <= 3:
                hit_3_count += 1

            # MRR: 1/rank. e.g., if found at rank 2, score is 0.5
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)  # Not found, adds 0 to MRR

    # Calculate final metrics
    total = len(eval_dataset)
    metrics = {
        "hit_rate": round(hit_k_count / total, 4),  # Overall Hit@K
        "hit_at_1": round(hit_1_count / total, 4),  # Top 1 retrieval
        "hit_at_3": round(hit_3_count / total, 4),  # Top 3 retrieval
        "mrr": round(sum(reciprocal_ranks) / total, 4),  # Mean Reciprocal Rank
        "entity_resolution_rate": round(entity_resolve_count / total, 4),
    }

    return metrics
