"""Minimal Textual TUI for the semantic query tool.

Intentionally unpolished (per current phase constraints): an input line plus a
log pane. ``/index-schema <schema>`` runs the indexing pipeline; any other
(non-slash) line is treated as a natural-language clinical question and routed
through LLM resource selection + semantic graph narrowing.
"""

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog

from app.commands.index_schema import format_summary, run_index_schema
from app.commands.query import format_bound, format_extracted, run_query_plan
from app.debug.dump import record_output, start_message
from app.runtime.errors import InfeasibleQuery


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

    def _log(self, text: str) -> None:
        """Write to the log pane and mirror it to the debug output file."""
        self.query_one(RichLog).write(text)
        record_output(text)

    def on_mount(self) -> None:
        self._log(
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
        # Reset the per-message dump files before any output or LLM call.
        start_message(command)
        self._log(f"[dim]> {command}[/]")
        if command.startswith("/"):
            self._dispatch(command)
        else:
            self.run_worker(self._run_query(command), exclusive=True)

    def _dispatch(self, command: str) -> None:
        if command == "/clear":
            self.query_one(RichLog).clear()
            return
        if not command.startswith("/index-schema"):
            self._log("[red]Unknown command. Try /index-schema <schema> or /clear.[/]")
            return
        parts = command.split()
        if len(parts) < 2:
            self._log("[red]Usage: /index-schema <schema> [--namespace NS][/]")
            return
        schema = parts[1]
        namespace = None
        if "--namespace" in parts:
            idx = parts.index("--namespace")
            if idx + 1 < len(parts):
                namespace = parts[idx + 1]
        self._run_index(schema, namespace)

    def _run_index(self, schema: str, namespace: str | None) -> None:
        self._log(f"[yellow]Indexing {schema}...[/]")
        try:
            registry = run_index_schema(schema, namespace=namespace)
            self._log(format_summary(registry))
        except Exception as exc:  # surfaced to the user, not swallowed
            self._log(f"[red]Indexing failed: {exc}[/]")

    async def _run_query(self, question: str) -> None:
        self._log("[yellow]Extracting plan...[/]")
        try:
            plan, bound = await run_query_plan(question)
            self._log(format_extracted(plan))
            self._log("[yellow]Grounding to schema...[/]")
            self._log(format_bound(bound))
        except InfeasibleQuery as exc:  # expected: schema can't answer
            self._log(format_extracted(exc.query_plan))
            self._log(format_bound(exc.bound))
        except Exception as exc:  # surfaced to the user, not swallowed
            self._log(f"[red]Query planning failed: {exc}[/]")
