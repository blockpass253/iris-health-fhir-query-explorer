"""Tests for runtime result rendering (``format_results``)."""

from rich.table import Table

from app.commands.query import (
    QueryResult,
    format_bound,
    format_extracted,
    format_results,
)
from app.runtime.models import (
    BoundGroupBy,
    BoundPlan,
    BoundSelectedField,
    BoundSortSpec,
    GroupBy,
    QueryPlan,
    SelectedField,
    SortSpec,
)


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


def test_ranked_rows_render_as_table():
    rows = [
        {"code": "6809", "display": "Metformin", "system": "rxnorm", "cnt": 42},
        {"code": "4548-4", "display": "A1c", "system": "loinc", "cnt": 17},
    ]
    out = format_results(_result(rows), "rank")
    assert isinstance(out, Table)
    assert out.row_count == 2
    assert [c.header for c in out.columns] == ["code", "display", "system", "cnt"]


def test_format_plans_surface_rank_shape():
    plan = QueryPlan(
        intent="rank",
        root_resource="MedicationRequest",
        resources=["MedicationRequest"],
        group_by=GroupBy(resource="MedicationRequest", concept=True),
        limit=5,
    )
    extracted = format_extracted(plan)
    assert "Root: MedicationRequest" in extracted
    assert "top 5" in extracted

    bound = BoundPlan(
        intent="rank",
        root_resource="MedicationRequest",
        resource_tables={"MedicationRequest": "MedicationRequest"},
        group_by=BoundGroupBy(
            group_by=GroupBy(resource="MedicationRequest", concept=True),
            table="MedicationRequest",
        ),
        limit=5,
    )
    grounded = format_bound(bound)
    assert "Root: MedicationRequest" in grounded
    assert "metric: row_count" in grounded
    assert "top 5" in grounded


def test_format_extracted_shows_select_fields():
    plan = QueryPlan(
        intent="list",
        root_resource="Patient",
        resources=["Patient"],
        select_fields=[
            SelectedField(resource="Patient", path="gender"),
            SelectedField(resource="Patient", path="birthDate"),
        ],
    )
    extracted = format_extracted(plan)
    assert "Select:" in extracted
    assert "gender" in extracted
    assert "birthDate" in extracted


def test_format_extracted_shows_concept_projection():
    plan = QueryPlan(
        intent="list",
        root_resource="Observation",
        resources=["Observation"],
        select_fields=[SelectedField(resource="Observation", concept=True)],
    )
    extracted = format_extracted(plan)
    assert "Select:" in extracted
    assert "concept" in extracted


def test_format_extracted_shows_sort():
    plan = QueryPlan(
        intent="list",
        root_resource="Patient",
        resources=["Patient"],
        sort=SortSpec(resource="Patient", path="birthDate", direction="desc"),
    )
    extracted = format_extracted(plan)
    assert "Sort:" in extracted
    assert "birthDate" in extracted
    assert "desc" in extracted


def test_format_bound_shows_grounded_select_and_sort():
    bound = BoundPlan(
        intent="list",
        root_resource="Patient",
        resource_tables={"Patient": "Patient"},
        select_fields=[
            BoundSelectedField(
                resource="Patient", table="Patient", column_path="Patient.gender"
            )
        ],
        sort=BoundSortSpec(
            resource="Patient",
            table="Patient",
            column_path="Patient.birthDate",
            direction="desc",
        ),
    )
    grounded = format_bound(bound)
    assert "Select:" in grounded
    assert "Patient.gender" in grounded
    assert "Sort:" in grounded
    assert "Patient.birthDate" in grounded
    assert "desc" in grounded
