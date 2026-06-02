"""Orchestrator for runtime resource selection.

Single source of truth shared by the Typer CLI and the Textual TUI. Loads the
indexed registry, builds the compact semantic context, runs LLM resource
selection, narrows the semantic graph, and returns the narrowed subgraph plus a
human-readable summary. No SQL is generated or executed here.
"""

from pathlib import Path

from app.logging.setup import get_logger
from app.runtime.context import build_runtime_context
from app.runtime.models import NarrowedSubgraph
from app.runtime.narrowing import narrow_subgraph
from app.runtime.selection import select_resources
from app.schema.persistence.registry_store import DEFAULT_REGISTRY_PATH, load_registry

log = get_logger("query")


async def run_resource_selection(
    query: str, registry_path: Path = DEFAULT_REGISTRY_PATH
) -> NarrowedSubgraph:
    """Select relevant resources for ``query`` and narrow the semantic graph."""
    if not registry_path.exists():
        raise FileNotFoundError(
            f"No semantic registry at {registry_path}. "
            "Index a schema first (e.g. /index-schema TEST1 --namespace FHIRSERVER)."
        )

    registry = load_registry(registry_path)
    log.info("query.start", query=query, schema=registry.schema_name)

    ctx = build_runtime_context(registry)
    selection = await select_resources(query, ctx)
    log.info("query.selected", resources=selection.resources)

    return narrow_subgraph(registry, selection.resources, reasoning=selection.reasoning)


def format_selection(result: NarrowedSubgraph) -> str:
    """Render the narrowed subgraph for the CLI and TUI."""
    lines: list[str] = ["Selected Resources:"]
    if result.resources:
        lines.extend(f"- {name}" for name in result.resources)
    else:
        lines.append("- (none)")
    for connector in result.bridge_connectors:
        lines.append(f"- {connector} (bridge)")

    lines.append("")
    lines.append("Relevant Relationships:")
    if result.relationships:
        lines.extend(
            f"- {edge.source} → {edge.target}" for edge in result.relationships
        )
    else:
        lines.append("- (none)")

    if result.reasoning:
        lines.append("")
        lines.append(f"[dim]{result.reasoning}[/]")

    return "\n".join(lines)
