import json
import os
from dataclasses import asdict

from .structures import ExtractionResult


def read_jsonl(input_jsonl_path: str):
    docs = []
    with open(input_jsonl_path, "r") as f:
        for line in f.readlines():
            docs.append(json.loads(line))

    return docs


def save_llm_to_jsonl(results: list[ExtractionResult], filepath: str):
    file_dir = "/".join(filepath.split("/")[:-1])
    os.makedirs(file_dir, exist_ok=True)

    with open(filepath, "a", encoding="utf-8") as f:
        for r in results:
            row = asdict(r)
            if r.data is not None:
                row["data"] = r.data.model_dump()  # Convert Pydantic model to dict
            if r.status:
                row["status"] = r.status.value  # Convert Enum to string

            f.write(json.dumps(row) + "\n")


def save_evaluation_to_jsonl(
    entity_metrics,  #: Dict[Metrics],
    triple_metrics,  #: Dict[Metrics],
    filepath: str,
):
    file_dir = "/".join(filepath.split("/")[:-1])
    os.makedirs(file_dir, exist_ok=True)

    with open(filepath, "a", encoding="utf-8") as f:
        for extraction_id in entity_metrics:
            row = {
                "extraction_id": extraction_id,
                "entity_metrics": asdict(entity_metrics[extraction_id]),
                "triple_metrics": asdict(triple_metrics[extraction_id]),
            }
            # Prepare the row dictionary
            f.write(json.dumps(row) + "\n")


def save_runs(params: dict, output_path: str):
    file_dir = "/".join(output_path.split("/")[:-1])
    os.makedirs(file_dir, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(params) + "\n")
