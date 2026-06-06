"""Tests for runtime result rendering (``format_results``)."""

from rich.table import Table

from app.commands.query import QueryResult, format_results
from app.runtime.models import BoundPlan, QueryPlan


def _result(rows, error=None) -> QueryResult:
    plan = QueryPlan(intent="list")
    bound = BoundPlan(intent="list")
    return QueryResult(plan=plan, bound=bound, rows=rows, error=error)


def test_rows_render_as_table():
    rows = [
        {"name": "Ada", "a1c": 9.1},
        {"name": "Grace", "a1c": None},
    ]
    out = format_results(_result(rows), "list")
    assert isinstance(out, Table)
    assert out.row_count == 2
    assert [c.header for c in out.columns] == ["name", "a1c"]


def test_truncation_noted_in_caption():
    rows = [{"n": i} for i in range(5)]
    out = format_results(_result(rows), "list", max_rows=2)
    assert isinstance(out, Table)
    assert out.row_count == 2
    assert out.caption == "… 3 more"


def test_count_intent_returns_string():
    out = format_results(_result([{"c": 42}]), "count")
    assert isinstance(out, str)
    assert "42" in out


def test_empty_rows_return_string():
    out = format_results(_result([]), "list")
    assert isinstance(out, str)


def test_not_executed_returns_string():
    out = format_results(_result(None), "list")
    assert isinstance(out, str)


def test_error_returns_string():
    out = format_results(_result(None, error="boom"), "list")
    assert isinstance(out, str)
    assert "boom" in out
