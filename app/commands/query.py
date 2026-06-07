"""Orchestrator for the runtime query pipeline.

Single source of truth shared by the Typer CLI and the Textual TUI. Both run the
same LangGraph conversation graph (see :mod:`app.runtime.graph`): LLM extraction
(question -> ungrounded plan), LLM binding (plan -> schema-grounded plan with a
feasibility verdict), then deterministic SQL generation and execution.

The CLI is single-shot — :func:`run_query_plan` runs the graph on a throwaway
``thread_id`` and returns a :class:`QueryResult`. When the question is ambiguous
or the schema cannot answer it, the graph pauses to ask a clarifying question;
since the CLI is non-interactive it cannot reply, so that pause is surfaced as
:class:`InfeasibleQuery` carrying both plans for rendering. The TUI drives the
same graph interactively on a stable ``thread_id`` for multi-turn conversation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from rich.console import RenderableType
from rich.table import Table

from app.logging.setup import get_logger
from app.runtime.diagnosis import ProjectionSuggestion
from app.runtime.errors import InfeasibleQuery
from app.runtime.graph import build_query_graph
from app.runtime.models import BoundPlan, Filter, QueryPlan, TemporalConstraint
from app.runtime.sql_generation import SqlQuery, render_sql
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


def result_from_state(state: dict[str, Any]) -> QueryResult:
    """Build a :class:`QueryResult` from a finished graph state."""
    plan = state.get("plan") or QueryPlan()
    bound = state.get("bound") or BoundPlan(intent=plan.intent)
    return QueryResult(
        plan=plan,
        bound=bound,
        sql=state.get("sql"),
        rows=state.get("rows"),
        error=state.get("error"),
    )


async def run_query_plan(
    query: str, registry_path: Path = DEFAULT_REGISTRY_PATH
) -> QueryResult:
    """Run the conversation graph once and return the result (single-shot CLI)."""
    if not registry_path.exists():
        raise FileNotFoundError(
            f"No semantic registry at {registry_path}. "
            "Index a schema first (e.g. /index-schema TEST1 --namespace FHIRSERVER)."
        )

    registry = load_registry(registry_path)
    log.info("query.start", query=query, schema=registry.schema_name)

    graph = build_query_graph(registry)
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": query}]}, config
    )

    interrupts = state.get("__interrupt__")
    if interrupts:  # graph paused to ask for clarification; CLI can't reply
        plan = state.get("plan") or QueryPlan()
        bound = state.get("bound") or BoundPlan(intent=plan.intent)
        payload = interrupts[0].value
        missing = payload.get("missing") or [payload.get("question", "")]
        suggestions = [
            ProjectionSuggestion.model_validate(s)
            for s in payload.get("suggestions") or []
        ]
        raise InfeasibleQuery(
            missing, query_plan=plan, bound=bound, suggestions=suggestions
        )

    log.info("query.done")
    return result_from_state(state)


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


def _group_target(group_by) -> str:
    """Compact description of a (bound or unbound) group_by's grouping target."""
    inner = getattr(group_by, "group_by", group_by)  # BoundGroupBy wraps a GroupBy
    if inner.concept:
        return "primary coded concept"
    return getattr(group_by, "column_path", None) or inner.path or "?"


def format_extracted(plan: QueryPlan) -> str:
    """Render the ungrounded extracted plan for the CLI and TUI."""
    lines: list[str] = [
        "[b]Extracted Plan[/]",
        f"Intent: {plan.intent}",
        f"Root: {plan.root_resource}",
    ]
    if plan.intent == "rank" and plan.group_by is not None:
        limit = plan.limit if plan.limit is not None else 5
        lines.append(
            f"Rank: group by {plan.group_by.resource} "
            f"({_group_target(plan.group_by)}); metric={plan.metric}; top {limit}"
        )
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

    if plan.select_fields:
        lines.append("")
        lines.append("Select:")
        for sf in plan.select_fields:
            label = "concept" if sf.concept else (sf.path or "?")
            lines.append(f"- [b]{sf.resource}[/]: {label}")

    if plan.sort is not None:
        lines.append("")
        lines.append(f"Sort: {plan.sort.path} {plan.sort.direction}")

    return "\n".join(lines)


def format_bound(bound: BoundPlan) -> str:
    """Render the schema-grounded bound plan, including the feasibility verdict."""
    lines: list[str] = [
        "[b]Grounded Plan[/]",
        f"Root: {bound.root_resource}",
        "",
        "Resource → Table:",
    ]
    if bound.resource_tables:
        lines.extend(
            f"- {res} → {table}" for res, table in bound.resource_tables.items()
        )
    else:
        lines.append("- (none)")

    if bound.intent == "rank" and bound.group_by is not None:
        limit = bound.limit if bound.limit is not None else 5
        lines += [
            "",
            "Rank:",
            f"- group by [b]{bound.group_by.table}[/] "
            f"({_group_target(bound.group_by)})",
            f"- metric: {bound.metric}",
            f"- top {limit}",
        ]

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

    if bound.select_fields:
        lines.append("")
        lines.append("Select:")
        for sf in bound.select_fields:
            label = "concept" if sf.concept else (sf.column_path or "?")
            lines.append(f"- [b]{sf.table}[/]: {label}")

    if bound.sort is not None:
        lines.append("")
        lines.append(
            f"Sort: {bound.sort.column_path} {bound.sort.direction} "
            f"[dim]({bound.sort.table})[/]"
        )

    lines.append("")
    if bound.feasibility.can_answer:
        lines.append("[green]✓ Answerable from the indexed schema.[/]")
    else:
        lines.append("[red]✗ Cannot fully answer — missing:[/]")
        lines.extend(f"  [red]- {m}[/]" for m in bound.feasibility.missing)

    return "\n".join(lines)


def format_projection_suggestions(suggestions: list[ProjectionSuggestion]) -> str:
    """Render advisory FHIR resource/field suggestions for an infeasible query."""
    lines: list[str] = [
        "[b]To answer this, extend your FHIR projection with:[/]",
        "",
    ]
    for s in suggestions:
        lines.append(
            f"- [b]{s.resource}[/] · [cyan]{s.field}[/] [dim]— {s.rationale}[/]"
        )
    lines.append("")
    lines.append(
        "[dim]Add these to the FHIR SQL Builder projection and re-index, "
        "then re-run the query.[/]"
    )
    return "\n".join(lines)


def format_sql(sql: SqlQuery) -> str:
    """Render the generated SQL with parameters inlined (always shown to the user)."""
    return "\n".join(["[b]Generated SQL[/]", "", f"[cyan]{render_sql(sql)}[/]"])


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
