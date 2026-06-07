"""Tests for the deterministic binding resolver (no LLM call exercised).

Only :func:`resolve_bound_plan` is tested: the guard that validates the LLM's
proposed mappings against the schema view, resolves concepts to codes, and
computes the feasibility verdict.
"""

from app.runtime.binding import (
    BindingDraft,
    FilterBinding,
    GroupByBinding,
    ResourceBinding,
    TemporalBinding,
    resolve_bound_plan,
)
from app.runtime.grounding import build_schema_view
from app.runtime.models import Filter, GroupBy, QueryPlan, TemporalConstraint


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
    plan = QueryPlan(intent="list", resources=["MedicationRequest"])
    # The model invents a table that is not in the schema view, for a resource
    # whose name also has no matching table.
    draft = BindingDraft(
        resource_bindings=[
            ResourceBinding(resource="MedicationRequest", table="MedRequests")
        ]
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert not bound.feasibility.can_answer
    assert "MedicationRequest" not in bound.resource_tables


def test_missing_resource_does_not_also_flag_its_filter(registry):
    view = build_schema_view(registry)
    # MedicationRequest is not in the schema; its filter fails only because of
    # that. The missing list should name the resource once, not also the filter.
    plan = QueryPlan(
        intent="list",
        resources=["MedicationRequest"],
        filters=[Filter(resource="MedicationRequest", concept="metformin")],
    )
    draft = BindingDraft(
        resource_bindings=[ResourceBinding(resource="MedicationRequest", table=None)],
        filter_bindings=[
            FilterBinding(resource="MedicationRequest", concept="metformin", table=None)
        ],
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert not bound.feasibility.can_answer
    assert bound.feasibility.missing == [
        "resource 'MedicationRequest' is not in the indexed schema"
    ]


def test_unbound_resource_falls_back_to_exact_name_match(registry):
    view = build_schema_view(registry)
    plan = QueryPlan(intent="list", resources=["Patient"])
    # The model fails to bind a table that is plainly present in the view.
    draft = BindingDraft(
        resource_bindings=[ResourceBinding(resource="Patient", table=None)]
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert bound.resource_tables["Patient"] == "Patient"
    assert not any("Patient" in m for m in bound.feasibility.missing)


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


def test_non_patient_root_list_is_answerable(registry):
    view = build_schema_view(registry)
    plan = QueryPlan(
        intent="list",
        root_resource="Encounter",
        resources=["Encounter"],
        filters=[
            Filter(resource="Encounter", path="status", operator="=", value="finished")
        ],
    )
    draft = BindingDraft(
        resource_bindings=[ResourceBinding(resource="Encounter", table="Encounter")],
        filter_bindings=[
            FilterBinding(
                resource="Encounter",
                path="status",
                table="Encounter",
                column_path="Encounter.status",
            )
        ],
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert bound.feasibility.can_answer
    assert bound.root_resource == "Encounter"
    assert bound.resource_tables == {"Encounter": "Encounter"}


def test_rank_concept_grouping_is_answerable(registry):
    # Observation has a coding child (ObservationCodeCodings), so a coded rank
    # grounds.
    view = build_schema_view(registry)
    plan = QueryPlan(
        intent="rank",
        root_resource="Observation",
        resources=["Observation"],
        group_by=GroupBy(resource="Observation", concept=True),
        limit=5,
    )
    draft = BindingDraft(
        resource_bindings=[
            ResourceBinding(resource="Observation", table="Observation")
        ],
        group_by_binding=GroupByBinding(resource="Observation", table="Observation"),
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert bound.feasibility.can_answer
    assert bound.group_by is not None
    assert bound.group_by.table == "Observation"
    assert bound.group_by.group_by.concept
    assert bound.group_by.column_path is None
    assert bound.limit == 5


def test_rank_path_grouping_is_answerable(registry):
    view = build_schema_view(registry)
    plan = QueryPlan(
        intent="rank",
        root_resource="Encounter",
        resources=["Encounter"],
        group_by=GroupBy(resource="Encounter", path="status"),
    )
    draft = BindingDraft(
        resource_bindings=[ResourceBinding(resource="Encounter", table="Encounter")],
        group_by_binding=GroupByBinding(
            resource="Encounter", table="Encounter", column_path="Encounter.status"
        ),
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert bound.feasibility.can_answer
    assert bound.group_by is not None
    assert bound.group_by.column_path == "Encounter.status"


def test_rank_concept_without_coding_child_is_infeasible(registry):
    # Condition has no coding child in the fixture, so grouping by its primary
    # concept cannot ground.
    view = build_schema_view(registry)
    plan = QueryPlan(
        intent="rank",
        root_resource="Condition",
        resources=["Condition"],
        group_by=GroupBy(resource="Condition", concept=True),
    )
    draft = BindingDraft(
        resource_bindings=[ResourceBinding(resource="Condition", table="Condition")],
        group_by_binding=GroupByBinding(resource="Condition", table="Condition"),
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert not bound.feasibility.can_answer
    assert any("no projected coding child" in m for m in bound.feasibility.missing)


def test_rank_root_resource_absent_is_infeasible(registry):
    view = build_schema_view(registry)
    plan = QueryPlan(
        intent="rank",
        root_resource="MedicationRequest",
        resources=["MedicationRequest"],
        group_by=GroupBy(resource="MedicationRequest", concept=True),
    )
    draft = BindingDraft(
        resource_bindings=[ResourceBinding(resource="MedicationRequest", table=None)],
        group_by_binding=GroupByBinding(resource="MedicationRequest", table=None),
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert not bound.feasibility.can_answer
    assert any(
        "root resource 'MedicationRequest'" in m for m in bound.feasibility.missing
    )


def test_rank_grouped_path_absent_is_infeasible(registry):
    view = build_schema_view(registry)
    plan = QueryPlan(
        intent="rank",
        root_resource="Encounter",
        resources=["Encounter"],
        group_by=GroupBy(resource="Encounter", path="triageCategory"),
    )
    draft = BindingDraft(
        resource_bindings=[ResourceBinding(resource="Encounter", table="Encounter")],
        group_by_binding=GroupByBinding(resource="Encounter", table="Encounter"),
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert not bound.feasibility.can_answer
    assert any("grouped attribute" in m for m in bound.feasibility.missing)


def test_rank_without_group_by_is_infeasible(registry):
    view = build_schema_view(registry)
    plan = QueryPlan(
        intent="rank", root_resource="Observation", resources=["Observation"]
    )
    draft = BindingDraft(
        resource_bindings=[ResourceBinding(resource="Observation", table="Observation")]
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert not bound.feasibility.can_answer
    assert any("no grouping target" in m for m in bound.feasibility.missing)


def test_non_correlatable_cross_resource_filter_is_infeasible(registry):
    # Root is Encounter; a Patient-attribute filter cannot be correlated because
    # Patient carries no patient reference (v1 routes correlation only through a
    # resource's own patient reference).
    view = build_schema_view(registry)
    plan = QueryPlan(
        intent="list",
        root_resource="Encounter",
        resources=["Encounter", "Patient"],
        filters=[
            Filter(resource="Patient", path="gender", operator="=", value="female")
        ],
    )
    draft = BindingDraft(
        resource_bindings=[
            ResourceBinding(resource="Encounter", table="Encounter"),
            ResourceBinding(resource="Patient", table="Patient"),
        ],
        filter_bindings=[
            FilterBinding(
                resource="Patient",
                path="gender",
                table="Patient",
                column_path="Patient.gender",
            )
        ],
    )

    bound = resolve_bound_plan(plan, draft, view)

    assert not bound.feasibility.can_answer
    assert any("cannot be correlated" in m for m in bound.feasibility.missing)
