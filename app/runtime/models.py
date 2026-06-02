"""Pydantic models for the runtime semantic-retrieval stage.

``RuntimeContext`` is the compact, registry-derived payload shown to the LLM.
``ResourceSelectionResult`` is the LLM's structured output. ``NarrowedSubgraph``
is the deterministic result that feeds the TUI render and later query planning.
"""

from pydantic import BaseModel, Field

from app.schema.models.registry import SemanticGraphEdge


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
