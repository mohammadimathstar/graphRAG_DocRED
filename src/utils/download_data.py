import json
from datasets import load_dataset
import os
import re
from tqdm import tqdm


# List of files required for processing
FILES_TO_DOWNLOAD = [
    "train_annotated",
    # "validation",
    # "test",
    'rel_info'
]


def download_docred_files(output_dir: str = 'data/raw/docred'):
    os.makedirs(output_dir, exist_ok=True)
    try:
        data = load_dataset('thunlp/docred')
        for file_name in FILES_TO_DOWNLOAD:
            raw_list = list(data[file_name])

            file_path = f"{output_dir}/{file_name if 'train' not in file_name else 'train'}.json"

            with open(file_path, 'w', encoding="utf-8") as f:
                json.dump(raw_list, f, ensure_ascii=False, indent=2)
            print(f"'{file_name}' set has been saved in `{file_path}`!\n")
            
    except Exception as e:
        print(f"Hugging Face download failed: {e}")



if __name__ == "__main__":
    download_docred_files(output_dir='data/raw/docred')
    
