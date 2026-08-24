import os
import json
import re
from tqdm import tqdm
from dataclasses import asdict
from typing import Dict

from .structures import Document
from .embedder import generate_embeddings_batch

PUNC_PATTERN = re.compile(r'\s([?.,!":;](?:\s|$))')

FILES_TO_PROCESS = [
    'train',
    # 'validation',
    # 'test'
]



def process_doc(doc: Dict) -> Document:
    # 1. Clean the text
    title = doc['title']
    raw_sentences = [" ".join(sent) for sent in doc["sents"]]
    raw_paragraph = " ".join(raw_sentences)
    clean_text = PUNC_PATTERN.sub(r'\1', raw_paragraph)

    # Compute the embedding for the cleaned text
    text_embedding = generate_embeddings_batch([clean_text])

    # 2. Extract entity strings and their types
    entities = []
    idx2ent = {}
    for ent_idx, cluster in enumerate(doc["vertexSet"]):
        names = list(set([mention["name"] for mention in cluster]))
        canonical_name = max(names, key=len)
        idx2ent[ent_idx] = canonical_name
        ent_type = cluster[0]["type"]
        entities.append((canonical_name, ent_type, ";".join(names)))

    # 3. Process triples/relationships 
    triples = []
    triples_idx = []
    labels_dict = doc.get("labels", {})
    
    # Extract the four parallel lists from the dictionary
    heads = labels_dict.get("head", [])
    tails = labels_dict.get("tail", [])
    relation_texts = labels_dict.get("relation_text", [])
    
    # Use zip to iterate through all four lists in parallel
    for head_idx, tail_idx, relation_text in zip(heads, tails, relation_texts):
        # Ensure the entity indices exist in our mapping
        if head_idx in idx2ent and tail_idx in idx2ent:
            head_name = idx2ent[head_idx]
            tail_name = idx2ent[tail_idx]
            
            # Format as a single string
            triple_idx = f"{head_idx}:{relation_text}:{tail_idx}"
            triple = f"{head_name}:{relation_text}:{tail_name}"

            triples_idx.append(triple_idx)
            triples.append(triple)
        else:
            print(f"Warning: Missing entity index in doc. Head: {head_idx}, Tail: {tail_idx}")

    processed = Document(
        title=title, 
        text=clean_text, 
        text_embedding=text_embedding,
        entities=entities, 
        triples=triples,
        triples_idx=triples_idx
    )
    
    return processed

def convert_dataset_to_jsonl(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in FILES_TO_PROCESS:

        input_json_path = f"{input_dir}/{filename}.json"
        output_jsonl_path = f"{output_dir}/{filename}.jsonl"

        print(f"Loading {input_json_path}...")        
        with open(input_json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        
        print(f"Processing and streaming to {output_jsonl_path}...")        
        with open(output_jsonl_path, "w", encoding="utf-8") as out_f:
            for doc in tqdm(dataset):
                processed = process_doc(doc)
                out_f.write(json.dumps(asdict(processed), ensure_ascii=False) + "\n")
                
        print(f"Successfully processed {len(dataset)} records!\n")



if __name__ == "__main__":
    convert_dataset_to_jsonl(
        input_dir='data/raw/docred',
        output_dir='data/processed/docred'
    )
