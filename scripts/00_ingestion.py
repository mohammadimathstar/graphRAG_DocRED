from src.utils.download_data import download_docred_files
from src.utils.preprocess_data import convert_dataset_to_jsonl


download_docred_files(output_dir='data/raw/docred')
convert_dataset_to_jsonl(
        input_dir='data/raw/docred',
        output_dir='data/processed/docred'
    )