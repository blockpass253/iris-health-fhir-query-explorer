"""Minimal Textual TUI for the semantic query tool.

Intentionally unpolished (per current phase constraints): an input line plus a
log pane. ``/index-schema <schema>`` runs the indexing pipeline; any other
(non-slash) line is treated as a natural-language clinical question and routed
through LLM resource selection + semantic graph narrowing.
"""

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog

from app.commands.index_schema import format_summary, run_index_schema
from app.commands.query import format_plan, format_selection, run_query_plan


class IrisTUI(App):
    """Interactive shell accepting slash commands and natural-language questions."""

    TITLE = "IRIS Semantic Query Tool"
    CSS = "RichLog { border: round $primary; }"

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(highlight=True, markup=True, wrap=True)
        yield Input(
            placeholder="Ask a question, or /index-schema TEST1 --namespace FHIRSERVER"
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(RichLog).write(
            "Ask a clinical question (e.g. [bold]Show diabetic patients with recent "
            "encounters[/]), or index a schema with "
            "[bold]/index-schema TEST1 --namespace FHIRSERVER[/]."
        )
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        command = event.value.strip()
        event.input.clear()
        if not command:
            return
        self.query_one(RichLog).write(f"[dim]> {command}[/]")
        if command.startswith("/"):
            self._dispatch(command)
        else:
            self.run_worker(self._run_query(command), exclusive=True)

    def _dispatch(self, command: str) -> None:
        log = self.query_one(RichLog)
        if not command.startswith("/index-schema"):
            log.write("[red]Unknown command. Try /index-schema <schema>.[/]")
            return
        parts = command.split()
        if len(parts) < 2:
            log.write("[red]Usage: /index-schema <schema> [--namespace NS][/]")
            return
        schema = parts[1]
        namespace = None
        if "--namespace" in parts:
            idx = parts.index("--namespace")
            if idx + 1 < len(parts):
                namespace = parts[idx + 1]
        self._run_index(schema, namespace)

    def _run_index(self, schema: str, namespace: str | None) -> None:
        log = self.query_one(RichLog)
        log.write(f"[yellow]Indexing {schema}...[/]")
        try:
            registry = run_index_schema(schema, namespace=namespace)
            log.write(format_summary(registry))
        except Exception as exc:  # surfaced to the user, not swallowed
            log.write(f"[red]Indexing failed: {exc}[/]")

    async def _run_query(self, question: str) -> None:
        log = self.query_one(RichLog)
        log.write("[yellow]Selecting resources...[/]")
        try:
            narrowed, plan = await run_query_plan(question)
            log.write(format_selection(narrowed))
            log.write("[yellow]Planning query...[/]")
            log.write(format_plan(plan))
        except Exception as exc:  # surfaced to the user, not swallowed
            log.write(f"[red]Query planning failed: {exc}[/]")
