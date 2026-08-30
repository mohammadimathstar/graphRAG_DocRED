from src.utils.download_data import download_docred_files
from src.utils.preprocess_data import convert_dataset_to_jsonl
import yaml


with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)


def run_preprocessing():
    """
    It downloads the data set and then perform preprocessing, including:
    - generating vector embedding for each text
    """

    raw_path = config["data"]["raw_path"]
    processed_path = config["data"]["processed_path"]

    download_docred_files(output_dir=raw_path)

    convert_dataset_to_jsonl(input_dir=raw_path, output_dir=processed_path)


if __name__ == "__main__":
    run_preprocessing()
