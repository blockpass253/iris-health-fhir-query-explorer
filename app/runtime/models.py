"""Pydantic models for the runtime semantic query pipeline.

Two LLM stages produce these. :class:`QueryPlan` is the *ungrounded* extraction
output: generic FHIR resource names and semantic concepts taken straight from the
question, with no reference to the indexed schema. :class:`BoundPlan` is the
*grounded* binding output: the same intent mapped onto the real registry
tables/paths, with concept codings attached and a feasibility verdict. Both are
SQL-independent — SQL generation is a later step.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Comparison operators the extractor may emit. Constrained so downstream
# deterministic SQL generation never has to guess at operator spelling.
ComparisonOperator = Literal[">", ">=", "<", "<=", "=", "!=", "in", "contains"]

# The high-level shape of the answer the question asks for.
QueryIntent = Literal["list", "count", "trend"]


class Filter(BaseModel):
    """A single constraint on a resource, as extracted from the question.

    ``concept`` stays semantic (e.g. "diabetes", "A1c", "metformin") and is later
    resolved to terminology codes. ``path`` is a direct attribute (e.g. "gender",
    "age") compared via ``operator``/``value``. A filter typically carries either
    a ``concept`` or a ``path``, not both.
    """

    resource: str
    concept: str | None = None
    path: str | None = None
    operator: ComparisonOperator | None = None
    value: str | int | float | None = None


class TemporalConstraint(BaseModel):
    """A relative time window on a resource (e.g. "in the last 6 months").

    Exactly one of the ``last_n_*`` fields is set. ``label`` preserves the
    original phrase for explainability.
    """

    resource: str
    last_n_days: int | None = None
    last_n_months: int | None = None
    last_n_years: int | None = None
    label: str | None = None


class QueryPlan(BaseModel):
    """Ungrounded extraction output: the question as structured intent.

    Uses generic FHIR resource names and semantic concepts only; it never
    references the indexed schema. Grounding happens in :class:`BoundPlan`.
    """

    intent: QueryIntent = "list"
    resources: list[str] = Field(default_factory=list)
    filters: list[Filter] = Field(default_factory=list)
    temporal_constraints: list[TemporalConstraint] = Field(default_factory=list)


class Coding(BaseModel):
    """A terminology code for a semantic concept (from the synonym dictionary)."""

    system: str
    code: str
    display: str | None = None


class BoundFilter(BaseModel):
    """An extracted filter resolved against the indexed schema.

    ``table`` is the real registry table the filter binds to; ``column_path`` is
    the concrete FHIR path/column when one applies; ``codings`` are the codes a
    semantic concept resolves to (empty for direct path filters).
    """

    filter: Filter
    table: str
    column_path: str | None = None
    codings: list[Coding] = Field(default_factory=list)


class BoundTemporal(BaseModel):
    """An extracted time window resolved to a real date column."""

    constraint: TemporalConstraint
    table: str
    column_path: str


class Feasibility(BaseModel):
    """Whether the indexed schema can fully answer the question."""

    can_answer: bool = True
    missing: list[str] = Field(default_factory=list)


class BoundPlan(BaseModel):
    """Grounded binding output: the plan mapped onto the real schema.

    ``resource_tables`` maps each abstract resource to its real registry table.
    ``feasibility`` records whether everything mapped; when it cannot, the
    pipeline stops rather than producing a partial answer.
    """

    intent: QueryIntent
    resource_tables: dict[str, str] = Field(default_factory=dict)
    filters: list[BoundFilter] = Field(default_factory=list)
    temporal_constraints: list[BoundTemporal] = Field(default_factory=list)
    feasibility: Feasibility = Field(default_factory=Feasibility)
