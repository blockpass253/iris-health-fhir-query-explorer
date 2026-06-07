"""Textual TUI for the semantic query tool.

A side-panel layout: a persistent context sidebar (indexed schema + the current
query's grounding) beside a scrolling transcript of conversation turns. Each turn is
a collapsible card with a step tracker, collapsed plan detail, a highlighted SQL
panel, and a results table. ``/index-schema <schema>`` runs the indexing pipeline;
any other (non-slash) line is a natural-language clinical question routed through the
LangGraph conversation pipeline.
"""

from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from rich.console import Console, RenderableType
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Header, Input, Static

from app.commands.index_schema import run_index_schema
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
from app.tui.widgets import ClarificationPanel, ContextPanel, QueryTurn


def _to_text(content: RenderableType) -> str:
    """Render a Rich renderable to plain text for the debug output file."""
    console = Console()
    with console.capture() as capture:
        console.print(content)
    return capture.get().rstrip("\n")


class IrisTUI(App):
    """Interactive shell accepting slash commands and natural-language questions."""

    TITLE = "IRIS Semantic Query Tool"
    CSS_PATH = "app.tcss"
    BINDINGS = [("ctrl+l", "clear", "Clear")]

    def __init__(self) -> None:
        super().__init__()
        # The compiled conversation graph and the thread it runs on. The graph
        # is None until a schema is indexed; a stable thread id gives the
        # session multi-turn memory until /clear or a re-index resets it.
        self._graph: CompiledStateGraph | None = None
        self._convo_thread_id = str(uuid4())
        self._awaiting_clarification = False
        self._current_turn: QueryTurn | None = None
        self._clarification_panel: ClarificationPanel | None = None

    def _new_conversation(self) -> None:
        """Start a fresh thread, dropping prior conversation memory."""
        self._convo_thread_id = str(uuid4())
        self._awaiting_clarification = False

    def _ensure_graph(self) -> bool:
        """Build the graph from the persisted registry if not already built."""
        if self._graph is not None:
            return True
        if not DEFAULT_REGISTRY_PATH.exists():
            self._notice(
                "[red]No indexed schema yet. Run "
                "[bold]/index-schema TEST1 --namespace FHIRSERVER[/] first.[/]"
            )
            return False
        registry = load_registry(DEFAULT_REGISTRY_PATH)
        self._graph = build_query_graph(registry)
        self.query_one(ContextPanel).update_schema(registry)
        return True

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            yield VerticalScroll(id="transcript")
            yield ContextPanel(id="sidebar")
        yield Input(
            placeholder="Ask a question, or /index-schema TEST1 --namespace FHIRSERVER"
        )
        yield Footer()

    def _notice(self, markup: str) -> None:
        """Mount a one-line system message into the transcript (and mirror to debug)."""
        self.query_one("#transcript", VerticalScroll).mount(
            Static(Text.from_markup(markup), classes="notice")
        )
        record_output(markup)

    def on_mount(self) -> None:
        self._notice(
            "Ask a clinical question (e.g. [bold]Show diabetic patients with recent "
            "encounters[/]), or index a schema with "
            "[bold]/index-schema TEST1 --namespace FHIRSERVER[/]."
        )
        if DEFAULT_REGISTRY_PATH.exists():
            self.query_one(ContextPanel).update_schema(
                load_registry(DEFAULT_REGISTRY_PATH)
            )
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        command = event.value.strip()
        event.input.clear()
        if not command:
            return
        start_message(command)
        if command == "/clear":
            self._clear()
            return
        # Any new submission supersedes a pending clarification — drop its panel.
        self._dismiss_clarification()
        # Slash commands are only dispatched when not mid-clarification, so a
        # reply like "/index-schema" during a pause still reads naturally as an answer.
        if command.startswith("/") and not self._awaiting_clarification:
            self._dispatch(command)
        else:
            self.run_worker(self._run_query(command), exclusive=True)

    def action_clear(self) -> None:
        self._clear()

    def _clear(self) -> None:
        self.workers.cancel_all()
        self._dismiss_clarification()
        self.query_one("#transcript", VerticalScroll).remove_children()
        self.query_one(ContextPanel).reset_query()
        self._current_turn = None
        self._new_conversation()  # clearing the screen also resets memory
        inp = self.query_one(Input)
        inp.disabled = False
        inp.focus()

    def _dismiss_clarification(self) -> None:
        """Remove the pinned clarification panel, if one is showing."""
        if self._clarification_panel is not None:
            self._clarification_panel.remove()
            self._clarification_panel = None

    def _dispatch(self, command: str) -> None:
        if command == "/clear":
            self._clear()
            return
        if not command.startswith("/index-schema"):
            self._notice(
                "[red]Unknown command. Try /index-schema <schema> or /clear.[/]"
            )
            return
        parts = command.split()
        if len(parts) < 2:
            self._notice("[red]Usage: /index-schema <schema> [--namespace NS][/]")
            return
        schema = parts[1]
        namespace = None
        if "--namespace" in parts:
            idx = parts.index("--namespace")
            if idx + 1 < len(parts):
                namespace = parts[idx + 1]
        self._run_index(schema, namespace)

    def _run_index(self, schema: str, namespace: str | None) -> None:
        self._notice(f"[yellow]Indexing {schema}…[/]")
        try:
            registry = run_index_schema(schema, namespace=namespace)
            # New schema → rebuild the graph and start a fresh conversation.
            self._graph = build_query_graph(registry)
            self._new_conversation()
            self.query_one(ContextPanel).update_schema(registry)
            self._notice(f"[green]Indexed {registry.schema_name} ✓[/]")
        except Exception as exc:  # surfaced to the user, not swallowed
            self._notice(f"[red]Indexing failed: {exc}[/]")

    async def _run_query(self, message: str) -> None:
        if not self._ensure_graph():
            return
        graph = self._graph
        assert graph is not None  # guaranteed by _ensure_graph

        inp = self.query_one(Input)
        inp.disabled = True

        transcript = self.query_one("#transcript", VerticalScroll)
        if self._current_turn is not None:  # collapse the prior turn to reduce scroll
            self._current_turn.collapsed = True
        turn = QueryTurn(message)
        await transcript.mount(turn)
        self._current_turn = turn
        turn.scroll_visible()
        turn.tracker.start()

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
                    turn.tracker.advance(node)

            if interrupt_value is not None:
                self._awaiting_clarification = True
                turn.tracker.waiting()

                panel = ClarificationPanel(interrupt_value)
                await self.mount(panel, before=self.query_one(Input))
                self._clarification_panel = panel
                record_output(interrupt_value["question"])
                inp.disabled = False  # user must type their clarification reply
                return

            state = graph.get_state(config).values
            result = result_from_state(state)
            await turn.populate(result.plan, result.bound, result.sql, result)
            self.query_one(ContextPanel).update_query(result.plan, result.bound)
            # Mirror the rendered output to debug/output.md for parity.
            record_output(format_extracted(result.plan))
            record_output(format_bound(result.bound))
            if result.sql is not None:
                record_output(format_sql(result.sql))
                record_output(_to_text(format_results(result, result.bound.intent)))
        except Exception as exc:  # surfaced to the user, not swallowed
            turn.tracker.fail()
            await turn.show_error(str(exc))
            record_output(f"Query planning failed: {exc}")
        finally:
            inp.disabled = False
            inp.focus()
