"""Minimal Textual TUI for the semantic query tool.

Intentionally unpolished (per current phase constraints): an input line plus a
log pane. The only supported command so far is ``/index-schema <schema>``, which
runs the same orchestrator as the CLI and renders its summary.
"""

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog

from app.commands.index_schema import format_summary, run_index_schema


class IrisTUI(App):
    """Interactive shell accepting slash commands."""

    TITLE = "IRIS Semantic Query Tool"
    CSS = "RichLog { border: round $primary; }"

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(highlight=True, markup=True, wrap=True)
        yield Input(placeholder="/index-schema TEST1 --namespace FHIRSERVER")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(RichLog).write(
            "Type a command, e.g. [bold]/index-schema TEST1 --namespace FHIRSERVER[/]"
        )
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        command = event.value.strip()
        event.input.clear()
        if not command:
            return
        self.query_one(RichLog).write(f"[dim]> {command}[/]")
        self._dispatch(command)

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
