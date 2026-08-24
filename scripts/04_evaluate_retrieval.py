from openai import OpenAI
from tqdm import tqdm
import yaml

from src.extraction.extractor import OpenAIProvider, InformationExtractor

from src.db.connection import get_connection
from src.db.ingest import insert_rag_evaluation

from src.utils.llm_schemas import RouterDecision
from src.utils.identifiers import generate_run_name
from src.evaluation.retrieval_evaluator import get_evaluation_set, evaluate_retrieval

from src.retrieval.prompt import ROUTER_PROMPT, USER_TEMPLATE





with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)


def run_evaluate_retrieval():
    # ==========================================
    # 1. Loading Parameters
    # ==========================================
    params = {
        'model': config['llm']['model'],
        'run_id': 'c2095213-4cc7-4000-9655-b72da434e28d',
        'instruction_version': config['generateQA']['instruction_version'],
        'instruction': ROUTER_PROMPT,
    }

    params['run_name'] = generate_run_name(
        model_name=params['model'], 
        prompt_version=params['instruction_version']
    )    

    # ==========================================
    # 2. Setup DB & Run
    # ==========================================
    conn = get_connection()


    # ==========================================
    # 3. Initialize extractor
    # ==========================================
    client = OpenAI()
    provider = OpenAIProvider(client)

    router = InformationExtractor(
        provider=provider,
        schema=RouterDecision,
        input_template=USER_TEMPLATE,
        params=params,
    )

    # ==========================================
    # 4. Load evaluation set (for QA)
    # ==========================================
    eval_set = get_evaluation_set(conn)

    # ==========================================
    # 2. Retrieve Context
    # ==========================================
    for qa_pair in tqdm(eval_set):
        retrieval_metrics = evaluate_retrieval(
            conn=conn,
            router=router,
            qa=qa_pair
        )

        insert_rag_evaluation(
            conn=conn,
            metrics=retrieval_metrics,
            generator_model=params['model'],
            run_id = params['run_id']
        )

        conn.commit()
    


if __name__ == "__main__":
    
    run_evaluate_retrieval()
    