from typing import Callable, Optional

from src.utils.extraction_types import RELATION_CONSTRAINTS
from src.utils.llm_schemas import Entity, Triple, ExtractionDocument


# =====================================================================
# 1. ENTITIY-LEVEL CHECKS
# Signature: (Entity, source_text) -> Optional[str]
# =====================================================================


def check_mention_nonempty(entity: Entity, source_text: str) -> Optional[str]:
    if not entity.canonical_mention.strip():
        return "canonical_mention is empty"


def check_aliases_in_text(entity: Entity, source_text: str) -> Optional[str]:
    """Ensure all aliases actually appear in the source text."""
    missing = [a for a in entity.aliases if a.strip() not in source_text]
    if missing:
        return f"aliases not found in source text: {missing}"


def check_description_length(entity: Entity, source_text: str) -> Optional[str]:
    """GraphRAG best practice: descriptions should be concise for embeddings."""
    if entity.description and len(entity.description) > 300:
        return "description exceeds 300 chars (may degrade embedding quality)"


EntityCheck = Callable[[Entity, str], Optional[str]]
ENTITY_CHECKS: tuple[EntityCheck, ...] = (
    check_mention_nonempty,
    check_aliases_in_text,
    check_description_length,
)

# =====================================================================
# 2. TRIPLE-LEVEL CHECKS
# Signature: (Triple, subj: Entity, obj: Entity, source_text) -> Optional[str]
# =====================================================================

ASYMMETRIC_NO_SELFLOOP = frozenset(
    {
        "father",
        "mother",
        "spouse",
        "child",
        "sibling",
        "place of birth",
        "place of death",
        "educated at",
        "employer",
        "founded by",
        "capital",
        "capital of",
        "head of government",
        "head of state",
        "chairperson",
        "parent organization",
        "parent taxon",
        "subclass of",
        "headquarters location",
        "residence",
        "work location",
    }
)


def check_domain_range(t: Triple, s: Entity, o: Entity, text: str) -> Optional[str]:
    s_ok, o_ok = RELATION_CONSTRAINTS.get(t.relation, (None, None))
    if s_ok is not None and s.entity_type not in s_ok:
        return f"domain mismatch: subject type '{s.entity_type}' not in {s_ok}"
    if o_ok is not None and o.entity_type not in o_ok:
        return f"range mismatch: object type '{o.entity_type}' not in {o_ok}"
    return None


def check_evidence_grounded(
    t: Triple, s: Entity, o: Entity, text: str
) -> Optional[str]:
    ev = (t.evidence_span or "").strip()
    if not ev:
        return "evidence_span is empty"
    norm_src = " ".join(text.split())
    norm_ev = " ".join(ev.split())
    return None if norm_ev in norm_src else "evidence not grounded in source text"


def check_self_loop(t: Triple, s: Entity, o: Entity, text: str) -> Optional[str]:
    if t.subject_id == t.object_id and t.relation in ASYMMETRIC_NO_SELFLOOP:
        return f"self-loop on asymmetric relation '{t.relation}'"
    return None


TripleCheck = Callable[[Triple, Entity, Entity, str], Optional[str]]
TRIPLE_CHECKS: tuple[TripleCheck, ...] = (
    check_domain_range,
    check_evidence_grounded,
    check_self_loop,
)

# =====================================================================
# 3. DOCUMENT-LEVEL CHECKS
# Signature: (ExtractionDocument) -> Optional[str]
# =====================================================================


def check_duplicate_entities(doc: ExtractionDocument) -> Optional[str]:
    """Flags if the LLM created two entities with the exact same canonical mention."""
    mentions = [
        e.canonical_mention.lower() for e in doc.entities if e.canonical_mention
    ]
    duplicates = {m for m in mentions if mentions.count(m) > 1}
    if duplicates:
        return f"duplicate canonical_mentions found: {duplicates}"


def check_min_triple_yield(doc: ExtractionDocument) -> Optional[str]:
    """Flags documents where the LLM failed to extract any relationships."""
    if len(doc.triples) == 0:
        return "document yielded 0 triples (possible extraction failure)"


def check_orphan_entities(doc: ExtractionDocument) -> Optional[str]:
    """(OPTIONAL) Flags if there are entities that participate in zero triples."""
    referenced_ids = {t.subject_id for t in doc.triples} | {
        t.object_id for t in doc.triples
    }
    orphans = [e.id for e in doc.entities if e.id not in referenced_ids]

    if orphans:
        return f"orphan entities (not in any triple): {orphans}"


DocCheck = Callable[[ExtractionDocument], Optional[str]]
DOC_CHECKS: tuple[DocCheck, ...] = (
    check_duplicate_entities,
    check_min_triple_yield,
    # check_orphan_entities,  # Uncomment if you want to flag orphan nodes
)

# =====================================================================
# 4. ORCHESTRATOR (Runs all checks, mutates flags, never raises)
# =====================================================================


def soft_validate(doc: ExtractionDocument, source_text: str) -> ExtractionDocument:
    """
    Runs Entity -> Triple -> Document checks sequentially.
    Mutates objects in place. Never raises.
    """

    # --- Phase 1: Entity Checks ---
    print(f"Running {len(ENTITY_CHECKS)} entity-level checks...")
    for e in doc.entities:
        errors = [
            msg for msg in (check(e, source_text) for check in ENTITY_CHECKS) if msg
        ]
        e.validation_errors = errors
        e.valid = not errors
        if errors:
            print(f"Entity '{e.id}' failed {len(errors)} checks: {errors}")

    # --- Phase 2: Triple Checks ---
    print(f"Running {len(TRIPLE_CHECKS)} triple-level checks...")

    id_to_entity = {e.id: e for e in doc.entities}
    for t in doc.triples:
        subj = id_to_entity.get(t.subject_id)
        obj = id_to_entity.get(t.object_id)
        if not subj or not obj:
            # Pydantic hard validator should catch this, but defensive guard
            t.validation_errors = ["referential integrity broken (fatal)"]
            t.valid = False
            continue

        errors = [
            msg
            for msg in (check(t, subj, obj, source_text) for check in TRIPLE_CHECKS)
            if msg
        ]
        t.validation_errors = errors
        t.valid = not errors

        if errors:
            print(
                f"Triple '{t.subject_id}-{t.relation}-{t.object_id}' failed {len(errors)} checks: {errors}"
            )

    # --- Phase 3: Document Checks ---
    doc_errors = [msg for msg in (check(doc) for check in DOC_CHECKS) if msg]
    doc.validation_errors = doc_errors
    doc.valid = not doc_errors
    print(f"Document validation errors: {doc_errors}")

    return doc


# =====================================================================
# 5. REPORTING HELPERS
# =====================================================================


def stats(doc: ExtractionDocument) -> dict:
    valid_t = sum(1 for t in doc.triples if t.valid)
    valid_e = sum(1 for e in doc.entities if e.valid)

    return {
        "doc_valid": doc.valid,
        "entities_valid": f"{valid_e}/{len(doc.entities)}",
        "triples_valid": f"{valid_t}/{len(doc.triples)}",
        "doc_errors": doc.validation_errors,
    }
