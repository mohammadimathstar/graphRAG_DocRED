from openai import OpenAI
from dotenv import load_dotenv
import yaml

from src.extraction.extractor import OpenAIProvider, InformationExtractor

from src.db.connection import get_connection

from src.retrieval.prompt import ROUTER_PROMPT, USER_TEMPLATE
from src.retrieval.retrieval import retrieve_context

from src.utils.llm_schemas import RouterDecision
from src.utils.identifiers import generate_run_name, get_runid


with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)


def run_retrieve(query: str):

    # ==========================================
    # 1. Loading Parameters
    # ==========================================
    params = {
        'model': config['llm']['model'],
        'run_id': get_runid(),
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
    # 2. Retrieve Context
    # ==========================================
    result = retrieve_context(
        conn=conn,
        router=router,
        query=query
    )

    return result


if __name__ == "__main__":

    query = 'give me the name of an airplane?'        
    result = run_retrieve(query)
    print(result)