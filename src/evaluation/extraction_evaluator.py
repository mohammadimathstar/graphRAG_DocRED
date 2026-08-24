import re
import string
from typing import List, Dict, Set, Tuple

from src.utils.llm_schemas import ExtractionDocument
from src.utils.structures import Document, Metrics


def normalize_text(text: str) -> str:
    """Normalize entity strings for fair comparison."""

    if not text:
        return ""
    
    text = text.strip().lower()

    # remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text

def build_entity_mappings(
    llm_doc: ExtractionDocument, 
    gt_entities: List[Dict]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Maps LLM local IDs (e.g., 'E1') and GT entity indices to a unified ID.
    Matches if ANY alias overlaps between the two.
    
    gt_entities format [canonical_name, ent_type, aliases]:
    [
      ["Barack Obama", "Person", "Barak Obama;Obama"], # GT Entity 0
      ...
    ]
    """
    # 1. Prepare Ground Truth alias sets
    gt_alias_sets = {}
    for idx, ent in enumerate(gt_entities):  # FIXED: idx, ent order
        # ent[0] is canonical_name, ent[2] is the semicolon-separated aliases
        aliases = set(normalize_text(m) for m in ent[2].split(';') if m.strip())
        aliases.add(normalize_text(ent[0]))  # Add canonical name to be safe
        gt_alias_sets[idx] = aliases 
    
    # 2. Prepare LLM alias sets
    llm_alias_sets = {}
    for e in llm_doc.entities:
        aliases = set(normalize_text(a) for a in e.aliases)
        aliases.add(normalize_text(e.canonical_mention))
        llm_alias_sets[e.id] = set(aliases)

    # 3. Match via intersection
    llm_to_gt = {}
    gt_to_llm = {}
    matched_llm_ids = set()  

    for gt_idx, gt_alias_set in gt_alias_sets.items():
        for llm_id, llm_aliases in llm_alias_sets.items():
            # Skip if this LLM entity was already matched to a previous GT entity
            if llm_id in matched_llm_ids:
                continue
                
            if llm_aliases.intersection(gt_alias_set):
                llm_to_gt[llm_id] = gt_idx
                gt_to_llm[gt_idx] = llm_id
                matched_llm_ids.add(llm_id)  # Mark as matched
                break  # Move to the next GT entity
                
    return llm_to_gt, gt_to_llm


def extract_llm_triples(
    doc: ExtractionDocument, 
    llm_to_gt: Dict[str, int]
) -> Set[Tuple[int, str, int]]:
    """Converts LLM triples to a set of (gt_idx, relation, gt_idx)."""
    triples = set()
    
    for t in doc.triples:
        if not t.valid:
            continue
            
        s_id = llm_to_gt.get(t.subject_id)
        o_id = llm_to_gt.get(t.object_id)
        
        if s_id is not None and o_id is not None:
            rel = t.relation.strip().lower()
            triples.add((s_id, rel, o_id)) # id from GT
    return triples


def calculate_metrics(pred_set: Set, gold_set: Set) -> Metrics:
    """Calculates Precision, Recall, F1 for sets (used for triples)."""
    if not pred_set and not gold_set:        
        return Metrics(
            precision= 1.0, 
            recall=1.0,
            f1=1.0, 
            tp=0, 
            fp=0, 
            fn=0)
    
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    perf = Metrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        tp=tp, fp=fp, fn=fn
    )
    
    return perf


def evaluate_document(gt_doc: Document, 
                      llm_doc: ExtractionDocument) -> Dict[str, Metrics]:
    """
    Main evaluation function.
    gt_data format: {"text": "...", "entities": [(name, type, aliases)], "triples": ["h:r:t"]}
    """
    gt_entities = gt_doc.entities
    gt_triple_idx = gt_doc.triples_idx  # Load directly as list of strings 'head_idx:relation_type:tail_idx'
    gt_triple_set = [] # convert gt_triples to a set of tuples (head_idx, relation_type, tail_idx)
    for triple_idx in gt_triple_idx:
        h_idx, rel, t_idx = triple_idx.split(":")
        gt_triple_set.append((int(h_idx), rel.strip().lower(), int(t_idx)))
    gt_triple_set = set(gt_triple_set)
    
    # 1. Align Entities
    llm_to_gt, gt_to_llm = build_entity_mappings(llm_doc, gt_entities)
    
    
    # ------------------- ENTITY METRICS -----------------
    e_tp = len(llm_to_gt)
    e_fp = len(llm_doc.entities) - e_tp
    e_fn = len(gt_entities) - len(gt_to_llm)
    
    e_prec = e_tp / (e_tp + e_fp) if (e_tp + e_fp) > 0 else 0.0
    e_rec = e_tp / (e_tp + e_fn) if (e_tp + e_fn) > 0 else 0.0
    e_f1 = 2 * (e_prec * e_rec) / (e_prec + e_rec) if (e_prec + e_rec) > 0 else 0.0

    entity_metrics = Metrics(
        precision=round(e_prec, 4),
        recall=round(e_rec, 4),
        f1=round(e_f1, 4),
        tp=e_tp, fp=e_fp, fn=e_fn
    )

    print(f"Entity Metrics: {entity_metrics}")
    
    # 2. Align Triples

    # LLM triples translated to GT indices
    llm_triples = extract_llm_triples(llm_doc, llm_to_gt)    
    
    # Calculate Triple Metrics
    triple_metrics = calculate_metrics(llm_triples, gt_triple_set)
    print(f"Triple Metrics: {triple_metrics}")

    return {
        "entity_metrics": entity_metrics,
        "triple_metrics": triple_metrics,
    }

