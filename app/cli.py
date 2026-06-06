"""Typer CLI entrypoint.

Exposes ``index-schema`` for headless indexing and ``tui`` to launch the
interactive interface. Both share the same orchestrator.
"""

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from app.commands.index_schema import format_summary, run_index_schema
from app.commands.query import (
    format_bound,
    format_extracted,
    format_results,
    format_sql,
    run_query_plan,
)
from app.debug.dump import start_message
from app.runtime.errors import InfeasibleQuery
from app.schema.persistence.registry_store import DEFAULT_REGISTRY_PATH

app = typer.Typer(
    help="IRIS FHIR semantic query exploration tool.", add_completion=False
)
console = Console()

_REGISTRY_OPTION = typer.Option(
    DEFAULT_REGISTRY_PATH,
    "--registry",
    "-r",
    help="Path to the indexed semantic registry.",
)


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


@app.command("query")
def query(
    question: str = typer.Argument(..., help="Natural-language clinical question."),
    registry: Path = _REGISTRY_OPTION,
) -> None:
    """Plan QUESTION, generate SQL, execute it, and show every stage."""
    start_message(question)
    try:
        result = asyncio.run(run_query_plan(question, registry_path=registry))
    except InfeasibleQuery as exc:
        console.print(format_extracted(exc.query_plan))
        console.print(format_bound(exc.bound))
        return
    console.print(format_extracted(result.plan))
    console.print(format_bound(result.bound))
    if result.sql is not None:
        console.print(format_sql(result.sql))
        console.print(format_results(result, result.bound.intent))


@app.command("tui")
def tui() -> None:
    """Launch the interactive Textual TUI."""
    from app.logging.setup import silence_stream_logging

    silence_stream_logging()
    from app.tui.app import IrisTUI

    IrisTUI().run()


if __name__ == "__main__":
    app()
