"""Orchestrator for the ``index-schema`` pipeline.

Single source of truth shared by the Typer CLI and the Textual TUI. Runs the
deterministic pipeline end to end: introspect -> parse -> infer -> graph ->
persist, returning the registry and a summary of counts.
"""

from datetime import UTC, datetime
from pathlib import Path

from app.iris import IrisSettings, get_settings
from app.logging.setup import get_logger
from app.schema.graph.builder import build_semantic_graph, render_graph_tree
from app.schema.introspection.queries import (
    fetch_columns,
    fetch_foreign_keys,
    fetch_tables,
)
from app.schema.models.registry import RelationshipType, SchemaRegistry
from app.schema.persistence.registry_store import DEFAULT_REGISTRY_PATH, save_registry
from app.semantic.inference import build_tables, infer_relationships, parse_columns
from app.semantic.profiling import profile_coding_systems

log = get_logger("index_schema")


def _resolve_settings(namespace: str | None) -> IrisSettings:
    """Return connection settings, overriding the namespace if provided."""
    base = get_settings()
    if namespace and namespace != base.namespace:
        return base.model_copy(update={"namespace": namespace})
    return base


def run_index_schema(
    schema: str,
    namespace: str | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> SchemaRegistry:
    """Index ``schema`` and persist the resulting semantic registry.

    Also samples coding ``system`` columns to record which terminologies each
    resource uses (reads data, not just metadata).
    """
    settings = _resolve_settings(namespace)
    log.info("index.start", schema=schema, namespace=settings.namespace)

    raw_tables = fetch_tables(schema, settings=settings)
    raw_columns = fetch_columns(schema, settings=settings)
    raw_fks = fetch_foreign_keys(schema, settings=settings)
    log.info(
        "introspection.done",
        tables=len(raw_tables),
        columns=len(raw_columns),
        foreign_keys=len(raw_fks),
    )

    parsed_by_table = parse_columns(raw_columns)
    tables = build_tables(raw_tables, parsed_by_table)
    profile_coding_systems(tables, schema, settings=settings)
    relationships = infer_relationships(tables, raw_fks)
    graph = build_semantic_graph(tables, relationships)

    physical = sum(
        1
        for r in relationships
        if r.relationship_type == RelationshipType.PHYSICAL_FOREIGN_KEY
    )
    semantic = len(relationships) - physical
    coding_columns_profiled = sum(
        1 for t in tables.values() for c in t.columns if c.coding_systems
    )
    stats = {
        "tables": len(tables),
        "columns": sum(len(t.columns) for t in tables.values()),
        "physical_relationships": physical,
        "semantic_relationships": semantic,
        "coding_columns_profiled": coding_columns_profiled,
    }
    log.info("inference.done", **stats)

    registry = SchemaRegistry(
        schema_name=schema,
        namespace=settings.namespace,
        generated_at=datetime.now(UTC),
        tables=tables,
        relationships=relationships,
        graph=graph,
        stats=stats,
    )

    written = save_registry(registry, registry_path)
    log.info("registry.written", path=str(written))
    return registry


def _coding_systems_lines(registry: SchemaRegistry) -> list[str]:
    """Roll up profiled coding systems to their parent resource for display.

    A coding table reaches its root resource through its nested physical FK edge
    (``source_table`` -> ``target_table``). Tables that carry no nested edge fall
    back to their own name.
    """
    nested_parent = {
        r.source_table: r.target_table
        for r in registry.relationships
        if r.is_nested and r.target_table is not None
    }
    lines: list[str] = []
    for table_name, table in registry.tables.items():
        systems = sorted({u.system for c in table.columns for u in c.coding_systems})
        if not systems:
            continue
        parent = nested_parent.get(table_name, table_name)
        resource = registry.tables.get(parent)
        label = (resource and resource.inferred_resource_type) or parent
        lines.append(f"- {label} ({table_name}): {', '.join(systems)}")
    return lines


def format_summary(
    registry: SchemaRegistry, registry_path: Path = DEFAULT_REGISTRY_PATH
) -> str:
    """Render the human-readable indexing summary shared by the CLI and TUI."""
    s = registry.stats
    coding_lines = _coding_systems_lines(registry)
    coding_block = ["", "Coding systems in use:", *coding_lines] if coding_lines else []
    return "\n".join(
        [
            f"Indexed schema: {registry.schema_name} (namespace {registry.namespace})",
            "",
            "Discovered:",
            f"- {s['tables']} tables",
            f"- {s['columns']} columns",
            f"- {s['physical_relationships']} physical relationships",
            f"- {s['semantic_relationships']} semantic FHIR relationships",
            *coding_block,
            "",
            "Semantic graph:",
            render_graph_tree(registry.graph),
            "",
            f"Registry written to:\n{registry_path}",
        ]
    )
