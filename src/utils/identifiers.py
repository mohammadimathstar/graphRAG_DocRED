import uuid 
import datetime

def get_runid():
    return str(uuid.uuid4())


def generate_run_name(model_name: str, prompt_version: str, dataset_name: str = "docred") -> str:
    """ 
    Generates a human-readable run name.
    Example: 'gpt-4o-mini_docred_prompt-v2-strict_20231106_1430'
    """
    # Get current date and time (YYYYMMDD_HHMM)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # Clean up strings (replace spaces with hyphens, just in case)
    model = model_name.replace(" ", "-").lower()
    prompt = prompt_version.replace(" ", "-").lower()
    dataset = dataset_name.replace(" ", "-").lower()
    
    return f"{model}_{dataset}_{prompt}_{timestamp}"