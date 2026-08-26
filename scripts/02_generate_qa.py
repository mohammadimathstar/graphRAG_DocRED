from openai import OpenAI
from dotenv import load_dotenv
import yaml
from src.utils.io import read_jsonl
from src.utils.identifiers import generate_run_name, get_runid

from src.extraction.extractor import OpenAIProvider

from src.db.connection import get_connection

from src.evaluation.prompt import SYSTEM_PROMPT
from src.evaluation.generate_QA import generate_QAs


load_dotenv()

with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

def run_generateQA():

    # ==========================================
    # 1. Loading Parameters
    # ==========================================
    ndocs_to_process = config['extraction']['num_docs']
    
    params = {
        'model': config['llm']['model'],
        'run_id': get_runid(),
        'instruction_version': config['extraction']['instruction_version'],
        'instruction': SYSTEM_PROMPT,
    }

    params['run_name'] = generate_run_name(
        model_name=params['model'], 
        prompt_version=params['instruction_version']
    )

    # ==========================================
    # 2. Initialize extractor
    # ==========================================
    file_path = f"{config['data']['processed_path']}/{config['extraction']['file_name']}"
    docs = read_jsonl(file_path)[:ndocs_to_process]

    # ==========================================
    # 3. Initialize provider
    # ==========================================
    client = OpenAI()
    provider = OpenAIProvider(client)    

    # ==========================================
    # 4. Setup DB & Run
    # ==========================================
    conn = get_connection()

    # ==========================================
    # 5. Generate QA
    # ==========================================
    generate_QAs(
        conn=conn,
        docs=docs, 
        provider=provider, 
        params=params
    )

    conn.close()


if __name__ == "__main__":
    run_generateQA()