"""Pydantic models for the runtime semantic-retrieval stage.

``RuntimeContext`` is the compact, registry-derived payload shown to the LLM.
``ResourceSelectionResult`` is the LLM's structured output. ``NarrowedSubgraph``
is the deterministic result that feeds the TUI render and the query planner.
``SemanticQueryPlan`` (and its parts) is the planner's structured output: the
canonical, SQL-independent intermediate representation for later deterministic
SQL generation.
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.schema.models.registry import SemanticGraphEdge

# Comparison operators the planner may emit. Constrained so downstream
# deterministic SQL generation never has to guess at operator spelling.
ComparisonOperator = Literal[">", ">=", "<", "<=", "=", "!=", "in", "contains"]


class RuntimeResource(BaseModel):
    """A root semantic resource as exposed to the LLM for reasoning.

    Intentionally compact: a name, its related root resources, and a sample of
    relevant FHIR paths (root columns plus folded nested-component paths). Raw
    SQL types and nested component tables are deliberately omitted.
    """

    name: str
    relationships: list[str] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)


class RuntimeContext(BaseModel):
    """The semantic context derived from the indexed registry for one question."""

    resources: list[RuntimeResource] = Field(default_factory=list)

    def resource_names(self) -> set[str]:
        """Return the set of resource names available for selection."""
        return {r.name for r in self.resources}


class ResourceSelectionResult(BaseModel):
    """Structured LLM output: the resources relevant to the question."""

    resources: list[str] = Field(default_factory=list)
    reasoning: str | None = None


class NarrowedSubgraph(BaseModel):
    """Deterministic narrowing result.

    ``resources`` are the LLM-selected roots; ``bridge_connectors`` are extra
    roots pulled in to connect otherwise-disconnected selections. ``relationships``
    are the edges among the combined node set; ``paths`` maps each node to its
    relevant FHIR paths. ``reasoning`` is carried through from selection.
    """

    resources: list[str] = Field(default_factory=list)
    bridge_connectors: list[str] = Field(default_factory=list)
    relationships: list[SemanticGraphEdge] = Field(default_factory=list)
    paths: dict[str, list[str]] = Field(default_factory=dict)
    reasoning: str | None = None

    def all_nodes(self) -> list[str]:
        """Selected resources followed by any bridge connectors."""
        return self.resources + self.bridge_connectors


class TemporalConstraint(BaseModel):
    """A structured, SQL-independent temporal window.

    ``relative`` windows use ``direction``/``amount``/``unit`` (e.g. "last 6
    months" -> last/6/month; "this year" -> this/-/year). ``absolute`` windows
    use ISO ``start``/``end``. ``label`` carries the original phrase for
    explainability and rendering.
    """

    kind: Literal["relative", "absolute"]
    direction: Literal["last", "this", "next"] | None = None
    amount: int | None = None
    unit: Literal["day", "week", "month", "year"] | None = None
    start: str | None = None
    end: str | None = None
    label: str | None = None


class SemanticFilter(BaseModel):
    """A single semantic constraint on a resource.

    ``concept`` stays semantic (e.g. "diabetes", "A1c"); ``path`` is an optional
    concrete FHIR path drawn from the narrowed context when one clearly applies.
    ``operator``/``value`` express comparisons; ``temporal_constraint`` expresses
    a time window. Any subset of fields may be present.
    """

    resource: str
    path: str | None = None
    concept: str | None = None
    operator: ComparisonOperator | None = None
    value: str | int | float | None = None
    temporal_constraint: TemporalConstraint | None = None


class TraversalPath(BaseModel):
    """A semantic hop between two resources via a FHIR reference path."""

    source_resource: str
    target_resource: str
    relationship_path: str


class SemanticQueryPlan(BaseModel):
    """The planner's structured output: the canonical semantic IR.

    SQL-independent by design — it names intent, resources, filters, semantic
    traversal paths, and aggregation, never joins or SQL. Validated against the
    narrowed subgraph so it cannot reference resources or paths that were not
    selected.
    """

    intent: str
    resources: list[str] = Field(default_factory=list)
    filters: list[SemanticFilter] = Field(default_factory=list)
    traversal_paths: list[TraversalPath] = Field(default_factory=list)
    aggregation: str | None = None
    reasoning: str | None = None
