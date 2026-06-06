"""Minimal Textual TUI for the semantic query tool.

Intentionally unpolished (per current phase constraints): an input line plus a
log pane. ``/index-schema <schema>`` runs the indexing pipeline; any other
(non-slash) line is treated as a natural-language clinical question and routed
through LLM resource selection + semantic graph narrowing.
"""

from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from rich.console import Console, RenderableType
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog

from app.commands.index_schema import format_summary, run_index_schema
from app.commands.query import (
    format_bound,
    format_extracted,
    format_results,
    format_sql,
    result_from_state,
)
from app.debug.dump import record_output, start_message
from app.runtime.graph import build_query_graph
from app.schema.persistence.registry_store import DEFAULT_REGISTRY_PATH, load_registry

# Per-node progress lines shown while the graph runs.
_NODE_PROGRESS = {
    "extract": "[yellow]Extracting plan...[/]",
    "bind": "[yellow]Grounding to schema...[/]",
    "run_sql": "[yellow]Generating & running SQL...[/]",
}


def _to_text(content: RenderableType) -> str:
    """Render a Rich renderable to plain text for the debug output file."""
    console = Console()
    with console.capture() as capture:
        console.print(content)
    return capture.get().rstrip("\n")


class IrisTUI(App):
    """Interactive shell accepting slash commands and natural-language questions."""

    TITLE = "IRIS Semantic Query Tool"
    CSS = "RichLog { border: round $primary; }"

    def __init__(self) -> None:
        super().__init__()
        # The compiled conversation graph and the thread it runs on. The graph
        # is None until a schema is indexed; a stable thread id gives the
        # session multi-turn memory until /clear or a re-index resets it.
        self._graph: CompiledStateGraph | None = None
        self._convo_thread_id = str(uuid4())
        self._awaiting_clarification = False

    def _new_conversation(self) -> None:
        """Start a fresh thread, dropping prior conversation memory."""
        self._convo_thread_id = str(uuid4())
        self._awaiting_clarification = False

    def _ensure_graph(self) -> bool:
        """Build the graph from the persisted registry if not already built."""
        if self._graph is not None:
            return True
        if not DEFAULT_REGISTRY_PATH.exists():
            self._log(
                "[red]No indexed schema yet. Run "
                "[bold]/index-schema TEST1 --namespace FHIRSERVER[/] first.[/]"
            )
            return False
        self._graph = build_query_graph(load_registry(DEFAULT_REGISTRY_PATH))
        return True

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(highlight=True, markup=True, wrap=True)
        yield Input(
            placeholder="Ask a question, or /index-schema TEST1 --namespace FHIRSERVER"
        )
        yield Footer()

    def _log(self, content: RenderableType) -> None:
        """Write to the log pane and mirror it to the debug output file."""
        self.query_one(RichLog).write(content)
        record_output(content if isinstance(content, str) else _to_text(content))

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
        # Slash commands are only dispatched when not mid-clarification, so a
        # reply like "/clear" during a pause still reads naturally as an answer.
        if command.startswith("/") and not self._awaiting_clarification:
            self._dispatch(command)
        else:
            self.run_worker(self._run_query(command), exclusive=True)

    def _dispatch(self, command: str) -> None:
        if command == "/clear":
            self.query_one(RichLog).clear()
            self._new_conversation()  # clearing the screen also resets memory
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
            # New schema → rebuild the graph and start a fresh conversation.
            self._graph = build_query_graph(registry)
            self._new_conversation()
        except Exception as exc:  # surfaced to the user, not swallowed
            self._log(f"[red]Indexing failed: {exc}[/]")

    async def _run_query(self, message: str) -> None:
        if not self._ensure_graph():
            return
        graph = self._graph
        assert graph is not None  # guaranteed by _ensure_graph
        # A reply to a pending clarification resumes the paused graph; otherwise
        # it's a new turn appended to the conversation.
        payload: Command | dict
        if self._awaiting_clarification:
            payload = Command(resume=message)
            self._awaiting_clarification = False
        else:
            payload = {"messages": [{"role": "user", "content": message}]}
        config: RunnableConfig = {"configurable": {"thread_id": self._convo_thread_id}}

        try:
            interrupt_value = None
            async for chunk in graph.astream(payload, config, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    interrupt_value = chunk["__interrupt__"][0].value
                    continue
                for node in chunk:
                    if node in _NODE_PROGRESS:
                        self._log(_NODE_PROGRESS[node])

            if interrupt_value is not None:
                self._awaiting_clarification = True
                self._log(f"[magenta]? {interrupt_value['question']}[/]")
                return

            state = graph.get_state(config).values
            result = result_from_state(state)
            self._log(format_extracted(result.plan))
            self._log(format_bound(result.bound))
            if result.sql is not None:
                self._log(format_sql(result.sql))
                self._log(format_results(result, result.bound.intent))
        except Exception as exc:  # surfaced to the user, not swallowed
            self._log(f"[red]Query planning failed: {exc}[/]")
