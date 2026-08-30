from psycopg import Connection

from src.extraction.extractor import InformationExtractor, LLMProvider
from src.evaluation.prompt import USER_PROMPT
from src.utils.llm_schemas import QuestionList

from src.db.ingest import insert_qa_pair


def generate_questions_for_document(
    doc: dict, provider: LLMProvider, params: dict
) -> list[dict]:
    """
    Generates questions for all GROUND TRUTH triples in a document in a single API call.
    doc: The processed ground truth dict containing 'text', 'entities', 'triples', 'triples_idx'
    """

    # 1. Prepare the ground truth triples for the prompt
    triples_prompt_list = []
    for i, triple_str in enumerate(doc["triples"]):
        # triple_str format is "canonical_name1:relation_type:canonical_name2"
        parts = triple_str.split(":", 2)
        if len(parts) < 3:
            continue
        head, rel, tail = parts[0], parts[1], parts[2]

        triples_prompt_list.append(
            f"Triple {i}: Subject: {head} | Relation: {rel} | Object: {tail}"
        )

    triples_text = "\n".join(triples_prompt_list)

    triple_str = f"""
# Ground Truth Triples
{triples_text}

# Task
Generate a question and extract the supporting sentence (gold_chunk_text) for each triple. 
Return a JSON object containing a list of questions, where each question specifies the 'triple_index' it corresponds to."""

    generator = InformationExtractor(
        provider=provider,
        schema=QuestionList,
        input_template=USER_PROMPT + triple_str,
        params=params,
    )

    response = generator.extract(
        text=doc["text"],
    )

    generated_questions = response.data.questions

    # 4. Map back to the ground truth entities for evaluation
    eval_dataset = []

    for gen_q in generated_questions:
        idx = gen_q.triple_index
        if idx < len(doc["triples"]):
            parts = doc["triples"][idx].split(":", 2)
            if len(parts) < 3:
                continue
            head, rel, tail = parts[0], parts[1], parts[2]

            eval_dataset.append(
                {
                    "doc_id": doc["document_id"],
                    "question": gen_q.question,
                    "gold_entity": head,  # For Entity Resolution evaluation
                    "gold_answer_entity": tail,  # For Answer Correctness evaluation
                    "gold_relation": rel,  # For relation filtering evaluation
                    "gold_chunk_text": gen_q.gold_chunk_text,  # For Retrieval (Hit@K, MRR) evaluation
                }
            )

    return eval_dataset


def generate_QAs(
    conn: Connection, docs: list[dict], provider: LLMProvider, params: dict
):
    for doc in docs:
        print(f"Generate Question for document {doc['document_id']}...")
        doc_qas = generate_questions_for_document(doc, provider, params)

        print(f"{len(doc_qas)} questions has been generated!\n")
        for qa in doc_qas:
            insert_qa_pair(conn=conn, doc_qa=qa)

    conn.commit()
