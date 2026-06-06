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


class ColumnRef(BaseModel):
    """A resolved physical column: real table and column names plus its role.

    ``table`` / ``column`` are the *actual* projected SQL identifiers (which the
    projection author may have customized); ``semantic_type`` is the metadata used
    to select the column. SQL generation reads these to emit physical names — it
    never derives a name from a FHIR path.
    """

    table: str
    column: str
    semantic_type: SemanticType | None = None


class CodingChild(BaseModel):
    """A resource's coding child table with the columns needed to match codes."""

    table: str
    fk_column: str
    code: ColumnRef
    system: ColumnRef | None = None


def _ref(table: str, col: ColumnMetadata) -> ColumnRef:
    return ColumnRef(
        table=table, column=col.column_name, semantic_type=col.semantic_type
    )


def find_column(
    table: TableMetadata,
    *,
    semantic_type: SemanticType | None = None,
    terminal: str | None = None,
) -> ColumnRef | None:
    """Find a column on ``table`` by its FHIR-element role, not by its name.

    Matches on ``semantic_type`` and/or the column's parsed FHIR ``terminal_field``
    (the last path segment), so it works even when the physical column was renamed.
    Returns the first match, or ``None``.
    """
    for col in table.columns:
        if semantic_type is not None and col.semantic_type != semantic_type:
            continue
        if terminal is not None:
            parsed = col.parsed_fhir_path
            if not parsed or parsed.terminal_field != terminal:
                continue
        return _ref(table.table_name, col)
    return None


def patient_reference_column(table: TableMetadata) -> ColumnRef | None:
    """The reference column on ``table`` that points at the Patient (subject).

    Selects the ``reference`` column whose FHIR path runs through ``subject`` or
    ``patient`` (e.g. ``Condition.subject.reference``,
    ``AllergyIntolerance.patient.reference``), so it is not confused with other
    references such as ``encounter.reference``. Selection is by metadata; the
    returned name is the real projected column.
    """
    for col in table.columns:
        parsed = col.parsed_fhir_path
        if not parsed or not parsed.is_reference:
            continue
        lowered = {seg.lower() for seg in parsed.segments}
        if "subject" in lowered or "patient" in lowered:
            return _ref(table.table_name, col)
    return None


def resolve_column_path(
    registry: SchemaRegistry, root: str, column_path: str
) -> ColumnRef | None:
    """Resolve a bound ``column_path`` (a qualified FHIR path) to a physical column.

    Uses the same path construction as :func:`build_schema_view`, so a path that
    binding accepted resolves back to its column even when that construction is
    only approximate. Returns ``None`` if nothing matches.
    """
    parents = nested_parents(registry)
    for table, _meta, col in _resource_columns(registry, root, parents):
        if _column_path(root, table, col) == column_path:
            return _ref(table, col)
    return None


def coding_child(registry: SchemaRegistry, resource_table: str) -> CodingChild | None:
    """Locate ``resource_table``'s primary coding child table.

    Candidates are nested children (via :func:`nested_parents`) carrying a column
    whose FHIR terminal element is ``code``, with a physical FK back to the parent.
    A resource often has several (``code``, ``clinicalStatus``, ``category`` …);
    we prefer the one bound to the resource's primary ``code`` element — the
    CodeableConcept that holds diagnosis/observation/medication codes — identified
    by its reconstructed element chain. The FK is the physical-FK relationship's
    ``source_column``; all returned names are the real projected identifiers.
    """
    parents = nested_parents(registry)
    fk_by_child = {
        rel.source_table: rel.source_column
        for rel in registry.relationships
        if rel.relationship_type == RelationshipType.PHYSICAL_FOREIGN_KEY
        and rel.target_table == resource_table
    }

    candidates: list[tuple[str, TableMetadata, ColumnRef, str]] = []
    for child, parent in parents.items():
        if parent != resource_table:
            continue
        meta = registry.tables[child]
        code = find_column(meta, terminal="code")
        fk_column = fk_by_child.get(child)
        if code is None or fk_column is None:
            continue
        candidates.append((child, meta, code, fk_column))
    if not candidates:
        return None

    def prefers_primary_code(child: str) -> int:
        # 0 == element chain begins with the resource's primary `code` element.
        chain = derive_nested_element_path(resource_table, child, []).split(".")
        return 0 if chain[1:2] == ["code"] else 1

    child, meta, code, fk_column = min(
        candidates, key=lambda c: prefers_primary_code(c[0])
    )
    return CodingChild(
        table=child,
        fk_column=fk_column,
        code=code,
        system=find_column(meta, terminal="system"),
    )


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
