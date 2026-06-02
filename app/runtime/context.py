"""Deterministic construction of the runtime semantic context.

Derives a compact, LLM-facing view from the indexed :class:`SchemaRegistry`:
root semantic resources, their root-to-root relationships, and a sample of
relevant FHIR paths. Nested component tables never appear as resources; instead
their paths are folded up into the parent root as best-effort qualified hints
(e.g. ``ObservationCodeCodings.Code`` -> ``Observation.code.coding.code``).

Pure functions only — no LLM, no I/O.
"""

import re

from app.runtime.models import RuntimeContext, RuntimeResource
from app.schema.models.registry import (
    RelationshipType,
    SchemaRegistry,
    TableMetadata,
)

# Keep each resource's path list compact for prompt economy.
MAX_PATHS_PER_RESOURCE = 12

_CAMEL = re.compile(r"[A-Z][a-z0-9]*")


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


def _owning_root(
    table: str | None, roots: set[str], parents: dict[str, str]
) -> str | None:
    """Resolve a table to the root resource it belongs to (itself if a root)."""
    if table is None:
        return None
    if table in roots:
        return table
    return parents.get(table)


def root_adjacency(registry: SchemaRegistry) -> dict[str, set[str]]:
    """Undirected adjacency between root resources.

    Endpoints of every resolved relationship are mapped to their owning root;
    edges between two distinct roots become undirected neighbor links. This
    collapses nested-component intermediaries into the roots they describe.
    """
    roots = root_resources(registry)
    parents = nested_parents(registry)
    adjacency: dict[str, set[str]] = {r: set() for r in roots}
    for rel in registry.relationships:
        src = _owning_root(rel.source_table, roots, parents)
        tgt = _owning_root(rel.target_table, roots, parents)
        if src and tgt and src != tgt:
            adjacency[src].add(tgt)
            adjacency[tgt].add(src)
    return adjacency


def _qualified_root_paths(meta: TableMetadata) -> list[str]:
    """FHIR paths physically present on a root resource table."""
    paths: list[str] = []
    for col in meta.columns:
        parsed = col.parsed_fhir_path
        if parsed and parsed.segments:
            paths.append(".".join(parsed.segments))
    return paths


def derive_nested_element_path(
    parent_root: str, nested_table: str, segments: list[str]
) -> str:
    """Reconstruct a best-effort qualified path for a nested-component column.

    The nested table name encodes the element chain relative to its parent (with
    no delimiter), so reconstruction is approximate: strip the parent prefix,
    split the remainder on CamelCase, de-pluralize and lowercase each segment to
    form the element prefix, then append the column's own local path segments.

    Only the final CamelCase segment is de-pluralized (it is the array/component
    marker, e.g. ``Codings`` -> ``coding``); interior segments are element names
    that may legitimately end in ``s`` (e.g. ``Status``) and are left intact.

    Examples (parent ``Observation``):
        ``ObservationCodeCodings`` + ``[code]``  -> ``Observation.code.coding.code``
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


def _depluralize(word: str) -> str:
    """Crude singularization sufficient for projection table-name suffixes."""
    return word[:-1] if word.endswith("s") and len(word) > 1 else word


def _folded_nested_paths(
    registry: SchemaRegistry, root: str, parents: dict[str, str]
) -> list[str]:
    """Qualified paths from the nested children belonging to ``root``."""
    paths: list[str] = []
    for table, parent in parents.items():
        if parent != root:
            continue
        for col in registry.tables[table].columns:
            parsed = col.parsed_fhir_path
            if parsed and parsed.segments:
                paths.append(derive_nested_element_path(root, table, parsed.segments))
    return paths


def _dedup(items: list[str]) -> list[str]:
    """Order-preserving de-duplication."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_runtime_context(registry: SchemaRegistry) -> RuntimeContext:
    """Build the compact, LLM-facing semantic context from the registry."""
    roots = sorted(root_resources(registry))
    parents = nested_parents(registry)
    adjacency = root_adjacency(registry)

    resources: list[RuntimeResource] = []
    for root in roots:
        meta = registry.tables[root]
        paths = _dedup(
            _qualified_root_paths(meta) + _folded_nested_paths(registry, root, parents)
        )[:MAX_PATHS_PER_RESOURCE]
        resources.append(
            RuntimeResource(
                name=root,
                relationships=sorted(adjacency.get(root, set())),
                paths=paths,
            )
        )
    return RuntimeContext(resources=resources)
