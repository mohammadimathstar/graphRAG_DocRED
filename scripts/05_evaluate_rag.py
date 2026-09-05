from openai import OpenAI
from tqdm import tqdm
import yaml
from dotenv import load_dotenv

from src.db.connection import get_connection
from src.db.ingest import insert_rag_evaluation

from src.evaluation.retrieval_evaluator import get_evaluation_set, evaluate_retrieval

from src.rag.rags import RAGBase
from src.rag.prompt import SYSTEM_PROMPT
from src.db.ingest import update_rag_evaluation_judgment
from src.evaluation.judge import judge_offline_qa

load_dotenv()


with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)


def run_evaluate_retrieval(run_id: str = None):
        
    llm_rag = config["llm"]["model"]
    llm_judge = config["judgement"]["model"]

    # ==========================================
    # 1. Setup DB & Run
    # ==========================================
    conn = get_connection()

    # ==========================================
    # 2. Initialize RAG System
    # ==========================================
    client = OpenAI()
    rag_system = RAGBase(client, instructions=SYSTEM_PROMPT, model="gpt-5.6-luna")        

    # ==========================================
    # 3. Load evaluation set (for QA)
    # ==========================================
    eval_set = get_evaluation_set(conn)

    # ==========================================
    # 4. Evaluation Loop: Retrieve, Generate, Judge, and Store Results
    # ==========================================
    for qa_pair in tqdm(eval_set):
        query = qa_pair["question"]

        # 1. Retrieve and Generate
        retrieval_result = rag_system.search(query)
        retrieved_contexts = retrieval_result.get("context", [])

        prompt = rag_system.build_prompt(query, retrieved_contexts)        
        response = rag_system.llm(prompt)

        answer = response.output_text

        # 2. Calculate Retrieval Metrics (Hit@K, MRR) for each QA pair
        retrieval_metrics = evaluate_retrieval(result=retrieval_result, qa=qa_pair)
        retrieval_metrics["qa_id"] = qa_pair["qa_id"]
        retrieval_metrics['generated_answer'] = answer
        retrieval_metrics["retrieval_method"] = retrieval_result["strategy"] if retrieval_result["context"] else None

        # 3. Insert into rag_evaluations table
        eval_id =insert_rag_evaluation(
            conn=conn,
            metrics=retrieval_metrics,
            generator_model=llm_rag,
        )
        
        try:
            # 4. Run LLM-as-a-Judge
            judge_result = judge_offline_qa(
                question=query,
                generated_answer=answer,
                ground_truth_answer=qa_pair['gold_entity'],
                gold_chunk=qa_pair['gold_chunk_text']
            )
            
            # 5. Update the row with judge results
            update_rag_evaluation_judgment(
                conn=conn,
                evaluation_id=eval_id,
                judge_model=llm_judge,
                is_correct=judge_result.is_correct,
                explanation=judge_result.explanation
            )
            
        except Exception as e:
            print(f"\n[ERROR] Judge failed for QA {qa_pair['qa_id']}: {e}")
            conn.rollback()
            continue
        
        conn.commit()
        


if __name__ == "__main__":
    run_evaluate_retrieval()
