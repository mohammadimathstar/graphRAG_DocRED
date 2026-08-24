from openai import OpenAI
from dotenv import load_dotenv
import yaml

from src.extraction.prompt import SYSTEM_PROMPT, USER_TEMPLATE
from src.extraction.extractor import OpenAIProvider, InformationExtractor
from src.utils.llm_schemas import ExtractionDocument
from src.extraction.extraction import process_single_document

from src.utils.identifiers import generate_run_name, get_runid
from src.utils.io import read_jsonl

from src.db.connection import get_connection
from src.db.manager import init_db
from src.db.ingest import insert_run


load_dotenv()

with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

def run_extraction():

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
    # 2. Initialize extractor
    # ==========================================
    client = OpenAI()
    provider = OpenAIProvider(client)

    extractor = InformationExtractor(
        provider=provider,
        schema=ExtractionDocument,
        input_template=USER_TEMPLATE,
        params=params,
    )

    # ==========================================
    # 2. Setup DB & Run
    # ==========================================
    conn = get_connection()
    init_db(drop_if_exists=False)
    insert_run(conn, params)

    # ==========================================
    # 3. Main Execution Loop
    # ==========================================
    try:
        print(f"Starting batch processing for {len(docs)} documents...")
        success_count = 0
        
        for doc in docs:
            # Process the document atomically
            if process_single_document(conn, doc, extractor):
                success_count += 1
                
        print(f"\nBatch complete! Successfully processed {success_count}/{len(docs)} documents.")

    except Exception as e:
        print(f"Fatal error during batch processing: {e}")
        
    finally:
        # ALWAYS close the connection when the script ends!
        print("Closing database connection.")
        conn.close()


if __name__ == "__main__":
    run_extraction()