"""Pydantic models describing the semantic schema registry.

These models are the canonical, deterministic representation of an indexed IRIS
FHIR SQL Builder projection. They are produced without any LLM involvement and
are later consumed by the runtime semantic query planner.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Confidence(StrEnum):
    """How certain an inference is, for explainability and weighting."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RelationshipType(StrEnum):
    """The mechanism by which a relationship between two tables was found."""

    PHYSICAL_FOREIGN_KEY = "physical_fk"
    FHIR_REFERENCE = "fhir_reference"
    NESTED_COMPONENT = "nested_component"
    INFERRED = "inferred"


class SemanticType(StrEnum):
    """Closed vocabulary classifying the clinical meaning of a column."""

    IDENTIFIER = "identifier"
    REFERENCE = "reference"
    CODE = "code"
    CODING = "coding"
    DATE = "date"
    DATETIME = "datetime"
    QUANTITY = "quantity"
    STRING = "string"
    BOOLEAN = "boolean"
    NARRATIVE = "narrative"


class ParsedFHIRPath(BaseModel):
    """A canonical FHIR path parsed from a column ``DESCRIPTION``."""

    raw: str
    resource_type: str | None = None
    segments: list[str] = Field(default_factory=list)
    is_reference: bool = False
    terminal_field: str | None = None


class ColumnMetadata(BaseModel):
    """A semantically meaningful column within a projected table."""

    column_name: str
    data_type: str
    fhir_path_raw: str | None = None
    parsed_fhir_path: ParsedFHIRPath | None = None
    semantic_type: SemanticType | None = None


class TableMetadata(BaseModel):
    """A table in the projection, with its semantic columns and inferred role."""

    table_name: str
    inferred_resource_type: str | None = None
    columns: list[ColumnMetadata] = Field(default_factory=list)
    semantic_tags: list[str] = Field(default_factory=list)


class RelationshipMetadata(BaseModel):
    """A directed relationship between two tables.

    ``relationship_type`` records the *mechanism* (physical FK, FHIR reference,
    inferred). ``is_nested`` flags that the edge also represents a FHIR nested
    component (child table belonging to a parent resource). When a reference
    target cannot be resolved to a discovered table, ``target_table`` is ``None``
    and ``target_hint`` preserves the best guess for explainability.
    """

    source_table: str
    source_column: str
    target_table: str | None = None
    relationship_type: RelationshipType
    is_nested: bool = False
    confidence: Confidence
    rationale: str
    target_hint: str | None = None


class SemanticGraphNode(BaseModel):
    """A node in the semantic graph (a root resource or nested component table)."""

    name: str
    resource_type: str | None = None
    is_nested: bool = False


class SemanticGraphEdge(BaseModel):
    """A directed edge in the semantic graph, mirroring a relationship."""

    source: str
    target: str | None = None
    relationship_type: RelationshipType
    is_nested: bool = False


class SemanticGraph(BaseModel):
    """Plain adjacency representation of the schema's semantic structure."""

    nodes: list[SemanticGraphNode] = Field(default_factory=list)
    edges: list[SemanticGraphEdge] = Field(default_factory=list)


class SchemaRegistry(BaseModel):
    """The persisted, canonical semantic layer for one indexed schema."""

    schema_name: str
    namespace: str
    generated_at: datetime
    tables: dict[str, TableMetadata] = Field(default_factory=dict)
    relationships: list[RelationshipMetadata] = Field(default_factory=list)
    graph: SemanticGraph = Field(default_factory=SemanticGraph)
    stats: dict[str, int] = Field(default_factory=dict)
