"""
Domain data models and schemas for knowledge graph extraction processing
pipelines.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, List
import uuid
from .llm_schemas import T


# --- Low-Level Support Types & Enums ---


class ExtractionStatus(str, Enum):
    """The runtime operational status of a single metadata schema extraction task."""

    SUCCESS = "success"
    INVALID_JSON = "invalid_json"
    SCHEMA_MISMATCH = "schema_mismatch"
    EMPTY_OUTPUT = "empty_output"
    PROVIDER_ERROR = "provider_error"


@dataclass(frozen=True)
class Usage:
    """Read-only token usage statistics, cost estimations, and performance latency numbers."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0
    latency_ms: int = 0


@dataclass
class Metrics:
    """Performance tracking scores measuring extraction truth matrices and alignment metrics."""

    precision: float
    recall: float
    f1: float
    tp: float
    fp: float
    fn: float

    def __add__(self, other: "Metrics") -> "Metrics":
        if not isinstance(other, Metrics):
            return NotImplemented
        return Metrics(
            precision=self.precision + other.precision,
            recall=self.recall + other.recall,
            f1=self.f1 + other.f1,
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )


# --- Primary Core Entities ---


@dataclass
class Document:
    """Source text entity capturing target content fields, extracted objects, and raw vector representations."""

    title: str
    text: str
    entities: List[str] = field(default_factory=list)
    triples: List[str] = field(default_factory=list)
    triples_idx: List[str] = field(default_factory=list)
    text_embedding: List[float] = field(default_factory=list)
    document_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.document_id = str(
            uuid.uuid5(uuid.NAMESPACE_OID, self.title.strip().lower())
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """Instantiates a valid Document object from dictionary records while preserving pre-existing IDs."""
        saved_id = data.pop("document_id", None)
        instance = cls(**data)
        if saved_id is not None:
            instance.document_id = saved_id
        return instance


# --- Vector Embeddings & Processing Outputs ---


@dataclass
class EntityEmbedding:
    """The localized numeric vector signature mapped directly to an extracted document entity string."""

    id: str
    text: str
    embedding: List[float]


@dataclass
class TripleEmbedding:
    """The holistic numeric vector representation capturing context of a structured relationship statement."""

    text: str
    embedding: List[float]


@dataclass
class ExtractionResult(Generic[T]):
    """Unified container mapping structured runtime outputs, token usage, and lifecycle identifiers together."""

    data: T | None
    raw_output: str | None
    status: ExtractionStatus
    error: str | None
    usage: Usage
    model: str | None
    document_id: str | None
    run_id: str | None
    extraction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
