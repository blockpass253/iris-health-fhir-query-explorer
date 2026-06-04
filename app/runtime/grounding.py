"""Deterministic projection of the indexed registry for the binding step.

Builds a compact, LLM-facing :class:`SchemaView` of the real schema: the root
semantic resources (table names as projected by the FHIR SQL Builder), a sample
of their FHIR paths, the subset of those paths that are dates (for temporal
constraints), and the coding ``system`` URIs observed in their data.

The structural helpers (``root_resources``, ``nested_parents``,
``derive_nested_element_path`` …) are deterministic schema math salvaged from the
retired ``runtime.context`` module; they are not part of any LLM flow. Pure
functions only — no LLM, no I/O.
"""

import re

from pydantic import BaseModel, Field

from app.schema.models.registry import (
    ColumnMetadata,
    RelationshipType,
    SchemaRegistry,
    SemanticType,
    TableMetadata,
)

# Keep each resource's path list compact for prompt economy.
MAX_PATHS_PER_RESOURCE = 16

_CAMEL = re.compile(r"[A-Z][a-z0-9]*")
_DATE_TYPES = {SemanticType.DATE, SemanticType.DATETIME}


class SchemaResource(BaseModel):
    """One real root resource as exposed to the binding LLM."""

    name: str
    resource_type: str | None = None
    paths: list[str] = Field(default_factory=list)
    date_paths: list[str] = Field(default_factory=list)
    coding_systems: list[str] = Field(default_factory=list)


class SchemaView(BaseModel):
    """The compact, LLM-facing view of the indexed schema for binding."""

    resources: list[SchemaResource] = Field(default_factory=list)

    def table_names(self) -> set[str]:
        """The set of real table names available to bind to."""
        return {r.name for r in self.resources}


def root_resources(registry: SchemaRegistry) -> set[str]:
    """Return the names of tables that are root semantic resources."""
    return {
        name
        for name, meta in registry.tables.items()
        if meta.inferred_resource_type is not None
    }


def nested_parents(registry: SchemaRegistry) -> dict[str, str]:
    """Map each nested component table to its owning root resource.

    Primary signal is the ``is_nested`` physical-FK edge to a root; a name-prefix
    fallback covers tables without such an edge.
    """
    roots = root_resources(registry)
    parents: dict[str, str] = {}
    for rel in registry.relationships:
        if (
            rel.relationship_type == RelationshipType.PHYSICAL_FOREIGN_KEY
            and rel.is_nested
            and rel.target_table in roots
        ):
            parents[rel.source_table] = rel.target_table

    for name in registry.tables:
        if name in roots or name in parents:
            continue
        matches = [r for r in roots if name.startswith(r) and r != name]
        if matches:
            parents[name] = max(matches, key=len)
    return parents


def _depluralize(word: str) -> str:
    """Crude singularization sufficient for projection table-name suffixes."""
    return word[:-1] if word.endswith("s") and len(word) > 1 else word


def derive_nested_element_path(
    parent_root: str, nested_table: str, segments: list[str]
) -> str:
    """Reconstruct a best-effort qualified path for a nested-component column.

    The nested table name encodes the element chain relative to its parent (with
    no delimiter), so reconstruction is approximate: strip the parent prefix,
    split the remainder on CamelCase, de-pluralize the final segment and lowercase
    each, then append the column's own local path segments.

    Examples (parent ``Observation``):
        ``ObservationCodeCodings`` + ``[code]`` -> ``Observation.code.coding.code``
        ``ConditionEncounters``    + ``[reference]`` (parent ``Condition``)
            -> ``Condition.encounter.reference``
    """
    remainder = (
        nested_table[len(parent_root) :]
        if nested_table.startswith(parent_root)
        else nested_table
    )
    words = _CAMEL.findall(remainder)
    if words:
        words[-1] = _depluralize(words[-1])
    element_segments = [w.lower() for w in words]
    parts = [parent_root, *element_segments, *segments]
    return ".".join(p for p in parts if p)


def _dedup(items: list[str]) -> list[str]:
    """Order-preserving de-duplication."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _column_path(root: str, table: str, col: ColumnMetadata) -> str | None:
    """Qualified FHIR path for ``col``, on the root itself or a nested child."""
    parsed = col.parsed_fhir_path
    if not parsed or not parsed.segments:
        return None
    if table == root:
        return ".".join(parsed.segments)
    return derive_nested_element_path(root, table, parsed.segments)


def _resource_columns(
    registry: SchemaRegistry, root: str, parents: dict[str, str]
) -> list[tuple[str, TableMetadata, ColumnMetadata]]:
    """Yield ``(table, table_meta, column)`` for the root and its nested children."""
    out: list[tuple[str, TableMetadata, ColumnMetadata]] = []
    meta = registry.tables[root]
    out.extend((root, meta, col) for col in meta.columns)
    for table, parent in parents.items():
        if parent == root:
            child = registry.tables[table]
            out.extend((table, child, col) for col in child.columns)
    return out


def build_schema_view(registry: SchemaRegistry) -> SchemaView:
    """Build the compact, LLM-facing schema view from the registry."""
    parents = nested_parents(registry)
    resources: list[SchemaResource] = []

    for root in sorted(root_resources(registry)):
        paths: list[str] = []
        date_paths: list[str] = []
        systems: list[str] = []
        for table, _meta, col in _resource_columns(registry, root, parents):
            path = _column_path(root, table, col)
            if path is not None:
                paths.append(path)
                if col.semantic_type in _DATE_TYPES:
                    date_paths.append(path)
            systems.extend(usage.system for usage in col.coding_systems)

        resources.append(
            SchemaResource(
                name=root,
                resource_type=registry.tables[root].inferred_resource_type,
                paths=_dedup(paths)[:MAX_PATHS_PER_RESOURCE],
                date_paths=_dedup(date_paths),
                coding_systems=_dedup(systems),
            )
        )

    return SchemaView(resources=resources)
