"""Typer CLI entrypoint.

Exposes ``index-schema`` for headless indexing and ``tui`` to launch the
interactive interface. Both share the same orchestrator.
"""

import typer
from rich.console import Console

from app.commands.index_schema import format_summary, run_index_schema

app = typer.Typer(
    help="IRIS FHIR semantic query exploration tool.", add_completion=False
)
console = Console()


@app.command("index-schema")
def index_schema(
    schema: str = typer.Argument(..., help="SQL schema to index, e.g. TEST1."),
    namespace: str | None = typer.Option(
        None,
        "--namespace",
        "-n",
        help="IRIS namespace override (defaults to IRIS_NAMESPACE).",
    ),
) -> None:
    """Introspect SCHEMA, build the semantic registry, and persist it."""
    registry = run_index_schema(schema, namespace=namespace)
    console.print(format_summary(registry))


@app.command("tui")
def tui() -> None:
    """Launch the interactive Textual TUI."""
    from app.tui.app import IrisTUI

    IrisTUI().run()


if __name__ == "__main__":
    app()
