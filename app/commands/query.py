"""Orchestrator for the runtime query pipeline.

Single source of truth shared by the Typer CLI and the Textual TUI. Loads the
indexed registry, builds the compact semantic context, runs LLM resource
selection, narrows the semantic graph, runs LLM semantic query planning, and
returns the narrowed subgraph and the structured plan plus human-readable
summaries. No SQL is generated or executed here.
"""

from pathlib import Path

from app.logging.setup import get_logger
from app.runtime.context import build_runtime_context
from app.runtime.models import NarrowedSubgraph, SemanticFilter, SemanticQueryPlan
from app.runtime.narrowing import narrow_subgraph
from app.runtime.planning import plan_query
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


async def run_query_plan(
    query: str, registry_path: Path = DEFAULT_REGISTRY_PATH
) -> tuple[NarrowedSubgraph, SemanticQueryPlan]:
    """Run the full pipeline: select + narrow, then plan the semantic query."""
    narrowed = await run_resource_selection(query, registry_path=registry_path)
    plan = await plan_query(query, narrowed)
    log.info("query.planned", intent=plan.intent, filters=len(plan.filters))
    return narrowed, plan


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


def _temporal_phrase(filter_: SemanticFilter) -> str | None:
    """Human-readable phrase for a filter's temporal constraint, if any."""
    tc = filter_.temporal_constraint
    if tc is None:
        return None
    if tc.label:
        return tc.label
    if tc.kind == "absolute":
        if tc.start and tc.end:
            return f"{tc.start} to {tc.end}"
        return tc.start or tc.end or "absolute window"
    parts = [p for p in (tc.direction, str(tc.amount) if tc.amount else None, tc.unit)]
    return " ".join(p for p in parts if p) or "relative window"


def _filter_phrase(filter_: SemanticFilter) -> str:
    """Render one semantic filter as a compact, explainable line."""
    temporal = _temporal_phrase(filter_)
    if filter_.concept and filter_.operator and filter_.value is not None:
        return f"{filter_.concept} {filter_.operator} {filter_.value}"
    if filter_.concept:
        return filter_.concept
    if temporal:
        return temporal
    if filter_.path:
        return filter_.path
    return filter_.resource


def format_plan(plan: SemanticQueryPlan) -> str:
    """Render the semantic query plan for the CLI and TUI."""
    lines: list[str] = ["Intent:", f"- {plan.intent}", "", "Resources:"]
    if plan.resources:
        lines.extend(f"- {name}" for name in plan.resources)
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("Detected Filters:")
    if plan.filters:
        for f in plan.filters:
            phrase = _filter_phrase(f)
            lines.append(f"- {phrase}")
            # A comparison filter may also carry a temporal window; show it
            # too, unless the phrase already is the temporal one.
            temporal = _temporal_phrase(f)
            if temporal and temporal != phrase:
                lines.append(f"- {temporal}")
    else:
        lines.append("- (none)")

    if plan.aggregation:
        lines.append("")
        lines.append("Aggregation:")
        lines.append(f"- {plan.aggregation}")

    lines.append("")
    lines.append("Traversal Paths:")
    if plan.traversal_paths:
        lines.extend(
            f"- {t.source_resource} → {t.target_resource}" for t in plan.traversal_paths
        )
    else:
        lines.append("- (none)")

    if plan.reasoning:
        lines.append("")
        lines.append(f"[dim]{plan.reasoning}[/]")

    return "\n".join(lines)
