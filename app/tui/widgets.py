"""Presentation widgets for the semantic query TUI.

Pure UI: these widgets render data the runtime pipeline already produces
(:class:`QueryPlan`, :class:`BoundPlan`, :class:`SqlQuery`, result rows, and the
:class:`SchemaView`) into a scannable layout — a per-turn step tracker, collapsible
plan detail, a highlighted SQL panel with a copy button, a results ``DataTable``, and
the persistent schema/context sidebar. No business logic lives here; rendering reuses
the formatting helpers in :mod:`app.commands.query`.
"""

from typing import Any

from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Collapsible,
    DataTable,
    LoadingIndicator,
    Rule,
    Static,
)

from app.commands.query import (
    QueryResult,
    _filter_phrase,
    _temporal_phrase,
    format_bound,
    format_extracted,
)
from app.runtime.grounding import build_schema_view
from app.runtime.models import BoundPlan, QueryPlan
from app.runtime.sql_generation import SqlQuery, render_sql
from app.schema.models.registry import SchemaRegistry


def _cell(value: Any) -> str:
    """Stringify a result cell; nulls render as blank."""
    return "" if value is None else str(value)


def _short_system(system: str) -> str:
    """Last path segment of a coding-system URI (``snomed.info/sct`` -> ``sct``)."""
    return system.rsplit("/", 1)[-1]


class StepTracker(Horizontal):
    """A compact ``Extract → Ground → SQL → Results`` progress strip for one turn."""

    STEPS = (
        ("extract", "Extract"),
        ("ground", "Ground"),
        ("sql", "SQL"),
        ("results", "Results"),
    )
    _STATE_CLASSES = ("-pending", "-active", "-done", "-failed", "-waiting")
    # Graph node -> (step it completes, step to activate next).
    _ADVANCE = {
        "extract": ("extract", "ground"),
        "bind": ("ground", "sql"),
        "run_sql": ("sql", "results"),
        "finalize": ("results", None),
    }

    def __init__(self) -> None:
        super().__init__(classes="steps")
        self._active: str | None = None

    def compose(self) -> ComposeResult:
        for index, (key, label) in enumerate(self.STEPS):
            if index:
                yield Static("→", classes="step-arrow")
            yield Static(label, id=f"step-{key}", classes="step -pending")

    def _set(self, key: str, state: str) -> None:
        chip = self.query_one(f"#step-{key}", Static)
        chip.remove_class(*self._STATE_CLASSES)
        chip.add_class(state)

    def start(self) -> None:
        for key, _ in self.STEPS:
            self._set(key, "-pending")
        self._set("extract", "-active")
        self._active = "extract"

    def advance(self, node: str) -> None:
        """Mark the step that ``node`` completed done and activate the next one."""
        if node not in self._ADVANCE:
            return
        done, nxt = self._ADVANCE[node]
        self._set(done, "-done")
        if nxt is not None:
            self._set(nxt, "-active")
        self._active = nxt

    def waiting(self) -> None:
        if self._active:
            self._set(self._active, "-waiting")

    def fail(self) -> None:
        if self._active:
            self._set(self._active, "-failed")


class SqlPanel(Vertical):
    """The generated SQL, syntax-highlighted, with an explicit copy button."""

    def __init__(self, sql: SqlQuery) -> None:
        super().__init__(classes="sql-panel")
        self._display_sql = render_sql(sql)

    def compose(self) -> ComposeResult:
        with Horizontal(classes="panel-header"):
            yield Static("Generated SQL", classes="panel-title")
            yield Button("Copy", id="copy-sql", variant="primary", compact=True)
        yield Static(
            Syntax(
                self._display_sql, "sql", word_wrap=True, background_color="default"
            ),
            classes="sql-code",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "copy-sql":
            self.app.copy_to_clipboard(self._display_sql)
            self.app.notify("SQL copied to clipboard")
            event.stop()


class ResultsPanel(Vertical):
    """Execution results: a count, an interactive ``DataTable``, or an error."""

    def __init__(self, result: QueryResult, intent: str, max_rows: int = 20) -> None:
        super().__init__(classes="results-panel")
        self._result = result
        self._intent = intent
        self._max_rows = max_rows
        rows = result.rows
        if intent == "count":
            self._title = "Result"
        elif isinstance(rows, list):
            self._title = f"Results ({len(rows)} row(s))"
        else:
            self._title = "Results"

    def _header(self, *, copyable: bool = False) -> ComposeResult:
        with Horizontal(classes="panel-header"):
            yield Static(self._title, classes="panel-title")
            if copyable:
                yield Button("Copy", id="copy-results", variant="success", compact=True)

    def compose(self) -> ComposeResult:
        result = self._result
        if result.error is not None:
            yield from self._header()
            yield Static(Text(f"Execution failed: {result.error}", style="red"))
            return

        rows = result.rows
        if rows is None:
            yield from self._header()
            yield Static(Text("Not executed.", style="dim"))
            return
        if isinstance(rows, int):  # non-SELECT rowcount (not expected for queries)
            yield from self._header()
            yield Static(f"{rows} rows affected")
            return
        if self._intent == "count":
            yield from self._header()
            value = next(iter(rows[0].values())) if rows else 0
            yield Static(Text.assemble(("Count: ", "bold"), (str(value), "green")))
            return
        if not rows:
            yield from self._header()
            yield Static(Text("0 rows", style="dim"))
            return

        yield from self._header(copyable=True)
        columns = list(rows[0].keys())
        table: DataTable[str] = DataTable(zebra_stripes=True, cursor_type="row")
        table.add_columns(*columns)
        for row in rows[: self._max_rows]:
            table.add_row(*(_cell(row.get(c)) for c in columns))
        yield table
        if len(rows) > self._max_rows:
            extra = len(rows) - self._max_rows
            yield Static(Text(f"… {extra} more row(s)", style="dim"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "copy-results":
            return
        rows = self._result.rows
        if isinstance(rows, list) and rows:
            columns = list(rows[0].keys())
            lines = ["\t".join(columns)]
            lines += ["\t".join(_cell(r.get(c)) for c in columns) for r in rows]
            self.app.copy_to_clipboard("\n".join(lines))
            self.app.notify("Results copied to clipboard")
        event.stop()


class QueryTurn(Collapsible):
    """One conversation turn: a collapsible card holding its tracker and outputs.

    Starts expanded with just the step tracker. As graph nodes complete during
    streaming, ``begin_streaming`` / ``on_extract`` / ``on_bind`` / ``on_sql``
    drive a spinner → content flow for each section. Extracted and grounded plan
    sections stay expanded until the user collapses them. ``populate`` is a
    one-shot fallback (e.g. on error). Older turns are collapsed by the app when
    a new question arrives.
    """

    def __init__(self, question: str) -> None:
        self._question = question
        self.tracker = StepTracker()
        self._extract_col: Collapsible | None = None
        self._bound_col: Collapsible | None = None
        self._running_col: Collapsible | None = None
        super().__init__(self.tracker, title=f"Q: {question}", collapsed=False)

    def _contents(self) -> Collapsible.Contents:
        return self.get_child_by_type(Collapsible.Contents)

    async def begin_streaming(self) -> None:
        """Mount the first spinner panel at query start."""
        self._extract_col = Collapsible(
            LoadingIndicator(), title="Extracted plan", collapsed=False
        )
        await self._contents().mount(self._extract_col)

    async def on_extract(self, plan: QueryPlan) -> None:
        """Replace extraction spinner with plan content, then open Ground spinner."""
        assert self._extract_col is not None
        await self._extract_col.query_one(LoadingIndicator).remove()
        await self._extract_col.get_child_by_type(Collapsible.Contents).mount(
            Static(Text.from_markup(format_extracted(plan)))
        )
        self._bound_col = Collapsible(
            LoadingIndicator(), title="Grounded plan", collapsed=False
        )
        await self._contents().mount(self._bound_col)

    async def on_bind(self, bound: BoundPlan) -> None:
        """Replace grounding spinner with content; open query spinner if feasible."""
        assert self._bound_col is not None
        await self._bound_col.query_one(LoadingIndicator).remove()
        await self._bound_col.get_child_by_type(Collapsible.Contents).mount(
            Static(Text.from_markup(format_bound(bound)))
        )
        badge = "✓" if bound.feasibility.can_answer else "✗"
        self.title = f"{badge} Q: {self._question}"
        if bound.feasibility.can_answer:
            self._running_col = Collapsible(
                LoadingIndicator(), title="Running query…", collapsed=False
            )
            await self._contents().mount(self._running_col)

    async def on_sql(self, sql: SqlQuery, result: QueryResult, intent: str) -> None:
        """Remove the query spinner; mount SQL panel and results directly."""
        assert self._running_col is not None
        await self._running_col.remove()
        await self._contents().mount(SqlPanel(sql), ResultsPanel(result, intent))

    async def populate(
        self,
        plan: QueryPlan,
        bound: BoundPlan,
        sql: SqlQuery | None,
        result: QueryResult,
    ) -> None:
        """One-shot fallback: mount all sections at once (skips streaming path)."""
        contents = self._contents()
        await contents.mount(
            Collapsible(
                Static(Text.from_markup(format_extracted(plan))),
                title="Extracted plan",
                collapsed=False,
            ),
            Collapsible(
                Static(Text.from_markup(format_bound(bound))),
                title="Grounded plan",
                collapsed=False,
            ),
        )
        if sql is not None:
            await contents.mount(SqlPanel(sql))
            await contents.mount(ResultsPanel(result, bound.intent))
        badge = "✓" if bound.feasibility.can_answer else "✗"
        self.title = f"{badge} Q: {self._question}"

    async def show_error(self, error: str) -> None:
        await self._contents().mount(
            Static(
                Text(f"Query planning failed: {error}", style="red"), classes="clarify"
            )
        )
        self.title = f"✗ Q: {self._question}"


class ClarificationPanel(Vertical):
    """A prominent, root-level panel for a pending clarification or schema gap."""

    def __init__(self, interrupt: dict[str, Any]) -> None:
        self._question: str = interrupt.get("question", "")
        self._missing: list[str] = interrupt.get("missing") or []
        self._suggestions: list[dict[str, Any]] = interrupt.get("suggestions") or []
        self._is_gap = bool(self._missing or self._suggestions)
        super().__init__(
            classes="clarification-panel " + ("-gap" if self._is_gap else "-ask")
        )
        self.border_title = (
            "⚠ Can't fully answer from the indexed schema"
            if self._is_gap
            else "❓ Need a bit more detail"
        )

    def compose(self) -> ComposeResult:
        if not self._is_gap:
            yield Static(Text(self._question))
            return
        if self._missing:
            yield Static(Text.assemble(("Missing: ", "bold"), "; ".join(self._missing)))
        if self._suggestions:
            yield Static(Text("\nExtend your FHIR projection with:", style="bold"))
            for suggestion in self._suggestions:
                field = suggestion["field"]
                label = (
                    f"{suggestion['resource']}.{field.split('.', 1)[-1]}"
                    if "." in field
                    else field
                )
                yield Static(
                    Text.assemble(
                        "  • ", (label, "bold"), f" — {suggestion['rationale']}"
                    )
                )
        yield Static(
            Text(
                "\nRe-index after extending the projection, or rephrase the question.",
                style="dim",
            )
        )


class ContextPanel(VerticalScroll):
    """Persistent sidebar: indexed-schema context plus the current query's grounding."""

    def compose(self) -> ComposeResult:
        yield Static(Text("SCHEMA", style="bold"), classes="ctx-head")
        yield Static(Text("No schema indexed.", style="dim"), id="ctx-schema")
        yield Rule()
        yield Static(Text("CURRENT QUERY", style="bold"), classes="ctx-head")
        yield Static(Text("—", style="dim"), id="ctx-query")

    def update_schema(self, registry: SchemaRegistry) -> None:
        view = build_schema_view(registry)
        text = Text()
        text.append(f"{registry.schema_name} · {registry.namespace}\n\n", style="bold")
        text.append("Resources\n", style="bold")
        text.append(", ".join(r.name for r in view.resources) + "\n")
        systems = sorted(
            {_short_system(s) for r in view.resources for s in r.coding_systems}
        )
        if systems:
            text.append("\nCoding systems\n", style="bold")
            text.append(", ".join(systems))
        self.query_one("#ctx-schema", Static).update(text)

    def update_query(self, plan: QueryPlan, bound: BoundPlan) -> None:
        text = Text()
        text.append("Root\n", style="bold")
        text.append(f"{bound.root_resource} · {bound.intent}\n")
        if bound.intent == "rank" and bound.group_by is not None:
            limit = bound.limit if bound.limit is not None else 5
            target = (
                "primary coded concept"
                if bound.group_by.group_by.concept
                else (bound.group_by.column_path or "?")
            )
            text.append("\nRank\n", style="bold")
            text.append(f"group by {bound.group_by.table} ({target})\n")
            text.append(f"metric={bound.metric} · top {limit}\n")
        if bound.resource_tables:
            text.append("\nResources\n", style="bold")
            text.append(", ".join(bound.resource_tables.values()) + "\n")
        if bound.filters:
            text.append("\nFilters\n", style="bold")
            for bf in bound.filters:
                text.append(f"• {_filter_phrase(bf.filter)}\n")
        if bound.temporal_constraints:
            text.append("\nTime windows\n", style="bold")
            for bt in bound.temporal_constraints:
                text.append(f"• {_temporal_phrase(bt.constraint)}\n")
        text.append("\n")
        if bound.feasibility.can_answer:
            text.append("✓ Answerable", style="green")
        else:
            missing = "; ".join(bound.feasibility.missing)
            text.append(f"✗ Missing: {missing}", style="red")
        self.query_one("#ctx-query", Static).update(text)

    def reset_query(self) -> None:
        self.query_one("#ctx-query", Static).update(Text("—", style="dim"))
