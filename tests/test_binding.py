"""Tests for the deterministic binding resolver (no LLM call exercised).

Only :func:`resolve_bound_plan` is tested: the guard that validates the LLM's
proposed mappings against the schema view, resolves concepts to codes, and
computes the feasibility verdict.
"""

from app.runtime.binding import (
    BindingDraft,
    FilterBinding,
    ResourceBinding,
    TemporalBinding,
    resolve_bound_plan,
)
from app.runtime.grounding import build_schema_view
from app.runtime.models import Filter, QueryPlan, TemporalConstraint


def test_fully_groundable_plan_is_answerable(registry):
    view = build_schema_view(registry)
    plan = QueryPlan(
        intent="list",
        resources=["Patient", "Condition"],
        filters=[
            Filter(resource="Condition", concept="diabetes"),
            Filter(
                resource="Patient",
                path="Patient.gender",
                operator="=",
                value="female",
            ),
        ],
    )
    draft = BindingDraft(
        resource_bindings=[
            ResourceBinding(resource="Patient", table="Patient"),
            ResourceBinding(resource="Condition", table="Condition"),
        ],
        filter_bindings=[
            FilterBinding(resource="Condition", concept="diabetes", table="Condition"),
            FilterBinding(
                resource="Patient",
                path="Patient.gender",
                table="Patient",
                column_path="Patient.gender",
            ),
        ],
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert bound.feasibility.can_answer
    assert bound.feasibility.missing == []
    assert bound.resource_tables == {"Patient": "Patient", "Condition": "Condition"}
    diabetes = next(f for f in bound.filters if f.filter.concept == "diabetes")
    assert any(c.code == "E11" for c in diabetes.codings)


def test_polymorphic_attribute_binds_to_projected_path(registry):
    # The extractor emits the abstract FHIR element name ("deceased"); the
    # projection exposes it with a type suffix ("Patient.deceasedDateTime").
    # A draft that maps to the real projected path must ground.
    view = build_schema_view(registry)
    plan = QueryPlan(
        intent="list",
        resources=["Patient"],
        filters=[
            Filter(resource="Patient", path="deceased", operator="=", value="false")
        ],
    )
    draft = BindingDraft(
        resource_bindings=[ResourceBinding(resource="Patient", table="Patient")],
        filter_bindings=[
            FilterBinding(
                resource="Patient",
                path="deceased",
                table="Patient",
                column_path="Patient.deceasedDateTime",
            )
        ],
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert bound.feasibility.can_answer
    assert bound.feasibility.missing == []
    deceased = next(f for f in bound.filters if f.filter.path == "deceased")
    assert deceased.column_path == "Patient.deceasedDateTime"


def test_binding_clarifying_question_threads_through(registry):
    view = build_schema_view(registry)
    question = "Should a missing DeceasedDateTime count as alive?"
    plan = QueryPlan(
        intent="list",
        resources=["Patient"],
        filters=[
            Filter(resource="Patient", path="deceased", operator="=", value="false")
        ],
    )
    draft = BindingDraft(
        resource_bindings=[ResourceBinding(resource="Patient", table="Patient")],
        filter_bindings=[
            FilterBinding(
                resource="Patient",
                path="deceased",
                table="Patient",
                column_path="Patient.deceasedDateTime",
            )
        ],
        clarifying_question=question,
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert bound.clarifying_question == question


def test_missing_resource_is_infeasible(registry):
    view = build_schema_view(registry)
    plan = QueryPlan(
        intent="list",
        resources=["Patient", "MedicationRequest"],
        filters=[Filter(resource="MedicationRequest", concept="metformin")],
    )
    # The model could not map MedicationRequest (it isn't in TEST1).
    draft = BindingDraft(
        resource_bindings=[
            ResourceBinding(resource="Patient", table="Patient"),
            ResourceBinding(resource="MedicationRequest", table=None),
        ],
        filter_bindings=[
            FilterBinding(
                resource="MedicationRequest", concept="metformin", table=None
            ),
        ],
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert not bound.feasibility.can_answer
    assert any("MedicationRequest" in m for m in bound.feasibility.missing)


def test_unknown_concept_is_infeasible(registry):
    view = build_schema_view(registry)
    plan = QueryPlan(
        intent="list",
        resources=["Condition"],
        filters=[Filter(resource="Condition", concept="lupus")],
    )
    draft = BindingDraft(
        resource_bindings=[ResourceBinding(resource="Condition", table="Condition")],
        filter_bindings=[
            FilterBinding(resource="Condition", concept="lupus", table="Condition")
        ],
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert not bound.feasibility.can_answer
    assert any("lupus" in m for m in bound.feasibility.missing)


def test_hallucinated_table_is_rejected(registry):
    view = build_schema_view(registry)
    plan = QueryPlan(intent="list", resources=["Patient"])
    # The model invents a table that is not in the schema view.
    draft = BindingDraft(
        resource_bindings=[ResourceBinding(resource="Patient", table="PatientProfile")]
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert not bound.feasibility.can_answer
    assert "Patient" not in bound.resource_tables


def test_temporal_without_date_field_is_infeasible(registry):
    view = build_schema_view(registry)
    plan = QueryPlan(
        intent="list",
        resources=["Condition"],
        temporal_constraints=[
            TemporalConstraint(
                resource="Condition", last_n_months=6, label="last 6 months"
            )
        ],
    )
    # Condition has no date path in the fixture; the model offers none.
    draft = BindingDraft(
        resource_bindings=[ResourceBinding(resource="Condition", table="Condition")],
        temporal_bindings=[TemporalBinding(resource="Condition", table="Condition")],
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert not bound.feasibility.can_answer
    assert any("time window" in m for m in bound.feasibility.missing)
