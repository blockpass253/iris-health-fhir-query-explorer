"""Orchestrator for the runtime query pipeline.

Single source of truth shared by the Typer CLI and the Textual TUI. Loads the
indexed registry, runs LLM extraction (question -> ungrounded plan), then LLM
binding (plan -> schema-grounded plan with a feasibility verdict), then
deterministically generates SQL and executes it against IRIS, returning the
plans, the generated SQL and the result rows. When the schema cannot fully
answer the question, :class:`InfeasibleQuery` is raised carrying both plans for
rendering (no SQL is generated in that case).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import RenderableType
from rich.table import Table

from app.iris import run_query
from app.logging.setup import get_logger
from app.runtime.binding import bind_plan
from app.runtime.errors import InfeasibleQuery
from app.runtime.extraction import extract_plan
from app.runtime.models import BoundPlan, Filter, QueryPlan, TemporalConstraint
from app.runtime.sql_generation import SqlQuery, generate_sql
from app.schema.persistence.registry_store import DEFAULT_REGISTRY_PATH, load_registry

log = get_logger("query")


@dataclass
class QueryResult:
    """The full outcome of a runtime query: plans, generated SQL, and rows."""

    plan: QueryPlan
    bound: BoundPlan
    sql: SqlQuery | None = None
    rows: list[dict[str, Any]] | int | None = None
    error: str | None = None


async def run_query_plan(
    query: str, registry_path: Path = DEFAULT_REGISTRY_PATH
) -> QueryResult:
    """Extract, bind, gate on feasibility, then generate and execute SQL."""
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

    sql = generate_sql(bound, registry)
    log.info("query.sql", params=sql.params)

    result = QueryResult(plan=plan, bound=bound, sql=sql)
    try:
        result.rows = run_query(sql.sql, sql.params)
        log.info("query.executed")
    except Exception as exc:  # keep the SQL visible; surface the failure
        result.error = str(exc)
        log.warning("query.execute_failed", error=str(exc))
    return result


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


def format_sql(sql: SqlQuery) -> str:
    """Render the generated SQL and its parameters (always shown to the user)."""
    lines = ["[b]Generated SQL[/]", "", f"[cyan]{sql.sql}[/]"]
    if sql.params:
        rendered = ", ".join(repr(p) for p in sql.params)
        lines += ["", f"[dim]params: [{rendered}][/]"]
    return "\n".join(lines)


def _cell(value: Any) -> str:
    """Stringify a result cell; nulls render as blank."""
    return "" if value is None else str(value)


def format_results(
    result: QueryResult, intent: str, max_rows: int = 20
) -> RenderableType:
    """Render execution results: a count, a row table, or an execution error."""
    if result.error is not None:
        return f"[red]Execution failed:[/] {result.error}"

    rows = result.rows
    if isinstance(rows, int):  # non-SELECT rowcount (not expected for queries)
        return f"[b]Result[/]\n{rows} rows affected"
    if rows is None:
        return "[dim]Not executed.[/]"

    if intent == "count":
        value = next(iter(rows[0].values())) if rows else 0
        return f"[b]Result[/]\nCount: [green]{value}[/]"

    if not rows:
        return "[b]Results[/] [dim](0 row(s))[/]"

    caption = f"… {len(rows) - max_rows} more" if len(rows) > max_rows else None
    table = Table(title=f"Results ({len(rows)} row(s))", caption=caption)
    for column in rows[0]:
        table.add_column(column)
    for row in rows[:max_rows]:
        table.add_row(*(_cell(row.get(column)) for column in rows[0]))
    return table
