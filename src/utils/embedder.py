from sentence_transformers import SentenceTransformer
from typing import List

from .structures import EntityEmbedding, TripleEmbedding
from .llm_schemas import Entity, Triple, ExtractionDocument


# We use nomic-embed-text because it handles 450+ words easily (8192 token limit)
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
# For docker version
# model = SentenceTransformer('/app/local_model') 

def build_entity_embedding_text(entity: Entity) -> str:
        # canonical_mention: str, entity_type: str, aliases: list, description: str = None) -> str:
    """
    Dynamically constructs a rich text string for entity embedding.
    """
    parts = [entity.canonical_mention]
    
    if entity.entity_type:
        parts.append(f"({entity.entity_type})")
        
    if entity.aliases:
        parts.append(f"Also known as: {', '.join(entity.aliases)}")
        
    if entity.description:
        parts.append(f"Description: {entity.description}")
        
    return ". ".join(parts) + "."


def build_triple_embedding_text(triple: Triple) -> str:
    """
    Constructs the text string for triple/evidence embedding.
    """
    return f"Relation: {triple.relation}. Evidence: {triple.evidence_span}"



def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using SentenceTransformers.
    """
    if not texts:
        return []
    
    
    embeddings = model.encode(
        texts, 
        convert_to_numpy=True, 
        show_progress_bar=False, 
        batch_size=32 
    )
    
    # Convert numpy arrays to lists of floats for psycopg
    return [emb.tolist() for emb in embeddings]


def embed_extraction(llm_output: ExtractionDocument):
    """
    Generates embeddings for a single document, including its text, entities, and triples.
    """
    
    entity_texts = []
    entity_ids = []
    # Generate embeddings for each entity
    for entity in llm_output.entities:
        entity_text = build_entity_embedding_text(entity)
        entity_texts.append(entity_text)
        entity_ids.append(entity.id)
        
    entity_embeddings = generate_embeddings_batch(entity_texts)

    entities = [
        EntityEmbedding(id=i, text=t, embedding=e) 
        for i, t, e in zip(entity_ids, entity_texts, entity_embeddings)
    ]
    
    triple_texts = []
    # Generate embeddings for each triple
    for triple in llm_output.triples:
        triple_text = build_triple_embedding_text(triple)
        triple_texts.append(triple_text)

    triple_embeddings = generate_embeddings_batch(triple_texts)
    
    triples = [
        TripleEmbedding(text=t, embedding=e) for t, e in zip(triple_texts, triple_embeddings)
    ]

    return entities, triples


