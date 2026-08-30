from typing import Literal, List, TypeVar
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .extraction_types import EntityType, RelationType

# **************************************
#     Output of LLM for Extraction
# **************************************

T = TypeVar("T", bound=BaseModel)

"""Pydantic schemas used to enforce and validate structured LLM generation constraints."""


class Entity(BaseModel):
    """An entity mentioned explicitly inside the source text document."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(
        ...,
        description="A unique local identifier used for internal linking, e.g., 'E1', 'E2'.",
    )
    canonical_mention: str = Field(
        ...,
        description="The most informative, standardized name or surface string for this entity.",
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Alternative name variations, acronyms, or coreferences found for this entity.",
    )
    entity_type: EntityType = Field(
        ...,
        description="The classification category matching this entity's taxonomy profile.",
    )
    description: str | None = Field(
        default=None,
        description="A concise, one-sentence contextual summary explaining who or what this entity is.",
    )

    valid: bool = Field(
        default=True, description="Internal validation status flag (DO NOT MODIFY)."
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Collected list of evaluation validation error descriptions (DO NOT MODIFY).",
    )


class Triple(BaseModel):
    """A semantic relationship statement connecting a subject entity to an object entity."""

    model_config = ConfigDict(populate_by_name=True)

    subject_id: str = Field(
        ...,
        description="The unique local 'id' matching the subject entity (e.g., 'E1').",
    )
    relation: RelationType = Field(
        ...,
        description="The formal semantic link predicate connecting the subject to the object.",
    )
    object_id: str = Field(
        ...,
        description="The unique local 'id' matching the target object entity (e.g., 'E2').",
    )
    evidence_span: str | None = Field(
        default=None,
        description="The exact, unaltered verbatim sentence extracted directly from the source text where this relation is stated.",
    )

    valid: bool = Field(
        default=True, description="Internal validation status flag (DO NOT MODIFY)."
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Collected list of evaluation validation error descriptions (DO NOT MODIFY).",
    )


class ExtractionDocument(BaseModel):
    """The master collection schema capturing all discovered text entities and relationship mappings."""

    model_config = ConfigDict(populate_by_name=True)

    entities: List[Entity] = Field(
        ...,
        description="Comprehensive list of distinct entities identified within the document.",
    )
    triples: List[Triple] = Field(
        ...,
        description="List of relationship connections built directly from the identified entities.",
    )

    # Internal orchestration properties hidden from the LLM JSON Schema target
    valid: bool = Field(
        default=True, description="Internal validation status flag (DO NOT MODIFY)."
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Collected list of evaluation validation error descriptions (DO NOT MODIFY).",
    )

    @model_validator(mode="after")
    def check_referential_integrity(self) -> "ExtractionDocument":
        """Enforces entity reference tracking rules across extracted structural components.

        Validates:
          1. Unique Entity Identification (Throws ValueError if IDs clash)
          2. Complete Link Integrity (Throws ValueError if a triple points to a missing Entity ID)
        """
        entity_ids = [e.id for e in self.entities]

        # Check 1: Duplicate Entity Identifiers
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError(
                f"Duplicate entity IDs detected: {entity_ids}. Extraction unrecoverable."
            )

        # Check 2: Missing Graph Graph Links
        id_set = set(entity_ids)
        dangling_links = []

        for index, triple in enumerate(self.triples):
            if triple.subject_id not in id_set:
                dangling_links.append(
                    f"triple[{index}] ({triple.subject_id}-{triple.relation}-{triple.object_id}): "
                    f"Subject ID '{triple.subject_id}' missing from entities."
                )
            if triple.object_id not in id_set:
                dangling_links.append(
                    f"triple[{index}] ({triple.subject_id}-{triple.relation}-{triple.object_id}): "
                    f"Object ID '{triple.object_id}' missing from entities."
                )

        if dangling_links:
            raise ValueError(
                "Dangling graph reference exceptions found:\n  - "
                + "\n  - ".join(dangling_links)
            )

        return self


# **************************************
#    Output of LLM for QA Generator
# **************************************


class GeneratedQuestion(BaseModel):
    triple_index: int = Field(
        ..., description="The 0-based index of the triple this question is based on."
    )
    question: str = Field(
        ...,
        description="A natural language question whose answer is the object of the triple.",
    )
    gold_chunk_text: str = Field(
        ...,
        description="The exact, verbatim sentence from the document text that supports this triple and answers the question.",
    )


class QuestionList(BaseModel):
    questions: List[GeneratedQuestion]


# **************************************
#     Output of LLM for Router
# **************************************


class RouterDecision(BaseModel):
    strategy: Literal["GRAPH", "VECTOR"]
    entities: List[str] = Field(default_factory=list, description="Entity names")
