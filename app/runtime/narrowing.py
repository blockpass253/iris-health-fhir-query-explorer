"""Deterministic semantic-subgraph narrowing.

Given the LLM-selected root resources, produce a connected subgraph for later
query planning: the selected resources plus any single intermediate root needed
to connect two otherwise-disconnected selections (bridge connectors), the edges
among that node set, and the relevant FHIR paths per node.

Pure functions only — no LLM, no I/O.
"""

from itertools import combinations

from app.runtime.context import (
    _owning_root,
    build_runtime_context,
    nested_parents,
    root_adjacency,
    root_resources,
)
from app.runtime.models import NarrowedSubgraph
from app.schema.models.registry import (
    RelationshipType,
    SchemaRegistry,
    SemanticGraphEdge,
)

# When two root edges collapse onto the same (source, target) pair, keep the most
# meaningful mechanism for explainability.
_EDGE_PRIORITY = {
    RelationshipType.FHIR_REFERENCE: 0,
    RelationshipType.PHYSICAL_FOREIGN_KEY: 1,
    RelationshipType.NESTED_COMPONENT: 2,
    RelationshipType.INFERRED: 3,
}


def _bridge_connectors(
    selected: list[str], adjacency: dict[str, set[str]]
) -> list[str]:
    """Roots that connect two selected resources lacking a direct edge."""
    connectors: set[str] = set()
    chosen = set(selected)
    for a, b in combinations(selected, 2):
        if b in adjacency.get(a, set()):
            continue  # already directly connected
        common = adjacency.get(a, set()) & adjacency.get(b, set())
        connectors.update(c for c in common if c not in chosen)
    return sorted(connectors)


def _root_edges(
    registry: SchemaRegistry, node_set: set[str]
) -> list[SemanticGraphEdge]:
    """Directed root-to-root edges among ``node_set``, collapsing nested tables."""
    roots = root_resources(registry)
    parents = nested_parents(registry)
    best: dict[tuple[str, str], SemanticGraphEdge] = {}
    for rel in registry.relationships:
        src = _owning_root(rel.source_table, roots, parents)
        tgt = _owning_root(rel.target_table, roots, parents)
        if not src or not tgt or src == tgt:
            continue
        if src not in node_set or tgt not in node_set:
            continue
        key = (src, tgt)
        candidate = SemanticGraphEdge(
            source=src,
            target=tgt,
            relationship_type=rel.relationship_type,
            is_nested=False,
        )
        current = best.get(key)
        if current is None or (
            _EDGE_PRIORITY[rel.relationship_type]
            < _EDGE_PRIORITY[current.relationship_type]
        ):
            best[key] = candidate
    return [best[k] for k in sorted(best)]


def narrow_subgraph(
    registry: SchemaRegistry,
    selected: list[str],
    reasoning: str | None = None,
) -> NarrowedSubgraph:
    """Narrow the semantic graph to the selected resources plus bridge connectors."""
    roots = root_resources(registry)
    selected = [r for r in selected if r in roots]  # defensive: roots only

    adjacency = root_adjacency(registry)
    connectors = _bridge_connectors(selected, adjacency)
    node_set = set(selected) | set(connectors)

    edges = _root_edges(registry, node_set)

    ctx = build_runtime_context(registry)
    paths_by_name = {r.name: r.paths for r in ctx.resources}
    paths = {name: paths_by_name.get(name, []) for name in sorted(node_set)}

    return NarrowedSubgraph(
        resources=selected,
        bridge_connectors=connectors,
        relationships=edges,
        paths=paths,
        reasoning=reasoning,
    )
