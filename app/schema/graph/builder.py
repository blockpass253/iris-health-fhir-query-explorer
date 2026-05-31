"""Construction and rendering of the semantic graph.

The graph is a plain adjacency structure (nodes + directed edges) mirroring the
inferred tables and relationships. It is serialized inline into the registry and
is the structure later traversed during semantic query planning.
"""

from app.schema.models.registry import (
    RelationshipMetadata,
    SemanticGraph,
    SemanticGraphEdge,
    SemanticGraphNode,
    TableMetadata,
)


def build_semantic_graph(
    tables: dict[str, TableMetadata],
    relationships: list[RelationshipMetadata],
) -> SemanticGraph:
    """Build the semantic graph from inferred tables and relationships."""
    nodes = [
        SemanticGraphNode(
            name=meta.table_name,
            resource_type=meta.inferred_resource_type,
            is_nested=meta.inferred_resource_type is None,
        )
        for meta in tables.values()
    ]
    edges = [
        SemanticGraphEdge(
            source=rel.source_table,
            target=rel.target_table,
            relationship_type=rel.relationship_type,
            is_nested=rel.is_nested,
        )
        for rel in relationships
    ]
    return SemanticGraph(nodes=nodes, edges=edges)


def render_graph_tree(graph: SemanticGraph) -> str:
    """Render a simple text tree of root resources and their attached tables."""
    roots = [n for n in graph.nodes if not n.is_nested]
    lines: list[str] = []
    for root in sorted(roots, key=lambda n: n.name):
        lines.append(root.name)
        children = sorted(
            {e.source for e in graph.edges if e.target == root.name and e.is_nested}
            | {
                e.target
                for e in graph.edges
                if e.source == root.name and e.target and e.target != root.name
            }
        )
        for i, child in enumerate(children):
            connector = "└──" if i == len(children) - 1 else "├──"
            lines.append(f" {connector} {child}")
    return "\n".join(lines)
