"""Orchestrator for the runtime query pipeline.

Single source of truth shared by the Typer CLI and the Textual TUI. Loads the
indexed registry, runs LLM extraction (question -> ungrounded plan), then LLM
binding (plan -> schema-grounded plan with a feasibility verdict), and returns
both plus human-readable summaries. The pipeline stops at the grounded plan; no
SQL is generated or executed here. When the schema cannot fully answer the
question, :class:`InfeasibleQuery` is raised carrying both plans for rendering.
"""

from pathlib import Path

from app.logging.setup import get_logger
from app.runtime.binding import bind_plan
from app.runtime.errors import InfeasibleQuery
from app.runtime.extraction import extract_plan
from app.runtime.models import BoundPlan, Filter, QueryPlan, TemporalConstraint
from app.schema.persistence.registry_store import DEFAULT_REGISTRY_PATH, load_registry

log = get_logger("query")


async def run_query_plan(
    query: str, registry_path: Path = DEFAULT_REGISTRY_PATH
) -> tuple[QueryPlan, BoundPlan]:
    """Extract a plan, bind it to the indexed schema, and gate on feasibility."""
    if not registry_path.exists():
        raise FileNotFoundError(
            f"No semantic registry at {registry_path}. "
            "Index a schema first (e.g. /index-schema TEST1 --namespace FHIRSERVER)."
        )

    registry = load_registry(registry_path)
    log.info("query.start", query=query, schema=registry.schema_name)

    plan = await extract_plan(query)
    log.info("query.extracted", intent=plan.intent, resources=plan.resources)

    bound = await bind_plan(plan, registry)
    if not bound.feasibility.can_answer:
        raise InfeasibleQuery(bound.feasibility.missing, query_plan=plan, bound=bound)

    log.info("query.bound", tables=bound.resource_tables)
    return plan, bound


# --- Rendering ---------------------------------------------------------------


def _temporal_phrase(tc: TemporalConstraint) -> str:
    """Human-readable phrase for a relative time window."""
    if tc.label:
        return tc.label
    if tc.last_n_days is not None:
        return f"last {tc.last_n_days} days"
    if tc.last_n_months is not None:
        return f"last {tc.last_n_months} months"
    if tc.last_n_years is not None:
        return f"last {tc.last_n_years} years"
    return "relative window"


def _filter_phrase(flt: Filter) -> str:
    """Compact phrase for a filter (concept, or path operator value)."""
    if flt.concept and not (flt.operator and flt.value is not None):
        return flt.concept
    subject = flt.concept or flt.path or flt.resource
    if flt.operator and flt.value is not None:
        return f"{subject} {flt.operator} {flt.value}"
    return subject


def format_extracted(plan: QueryPlan) -> str:
    """Render the ungrounded extracted plan for the CLI and TUI."""
    lines: list[str] = ["[b]Extracted Plan[/]", f"Intent: {plan.intent}"]
    lines += ["", "Resources:"]
    if plan.resources:
        lines.extend(f"- {r}" for r in plan.resources)
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("Filters:")
    if plan.filters:
        for f in plan.filters:
            lines.append(f"- [b]{f.resource}[/]: {_filter_phrase(f)}")
    else:
        lines.append("- (none)")

    if plan.temporal_constraints:
        lines.append("")
        lines.append("Time Windows:")
        for tc in plan.temporal_constraints:
            lines.append(f"- [b]{tc.resource}[/]: {_temporal_phrase(tc)}")

    return "\n".join(lines)


def format_bound(bound: BoundPlan) -> str:
    """Render the schema-grounded bound plan, including the feasibility verdict."""
    lines: list[str] = ["[b]Grounded Plan[/]", "", "Resource → Table:"]
    if bound.resource_tables:
        lines.extend(
            f"- {res} → {table}" for res, table in bound.resource_tables.items()
        )
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("Filters:")
    if bound.filters:
        for bf in bound.filters:
            target = bf.column_path or bf.table
            line = f"- [b]{bf.table}[/]: {_filter_phrase(bf.filter)} [dim]({target})[/]"
            if bf.codings:
                codes = ", ".join(
                    f"{c.system.rsplit('/', 1)[-1]}:{c.code}" for c in bf.codings
                )
                line += f" [dim]codes={codes}[/]"
            lines.append(line)
    else:
        lines.append("- (none)")

    if bound.temporal_constraints:
        lines.append("")
        lines.append("Time Windows:")
        for bt in bound.temporal_constraints:
            phrase = _temporal_phrase(bt.constraint)
            lines.append(f"- [b]{bt.table}[/]: {phrase} [dim]({bt.column_path})[/]")

    lines.append("")
    if bound.feasibility.can_answer:
        lines.append("[green]✓ Answerable from the indexed schema.[/]")
    else:
        lines.append("[red]✗ Cannot fully answer — missing:[/]")
        lines.extend(f"  [red]- {m}[/]" for m in bound.feasibility.missing)

    return "\n".join(lines)
