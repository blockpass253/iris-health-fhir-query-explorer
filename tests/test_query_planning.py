"""Tests for the deterministic query-plan validation helper.

The LLM call (:func:`plan_query`) is not exercised here; only the deterministic
:func:`validate_plan` guard that sanitizes a plan against the narrowed subgraph.
"""

import pytest

from app.runtime.models import (
    SemanticFilter,
    SemanticQueryPlan,
    TemporalConstraint,
    TraversalPath,
)
from app.runtime.narrowing import narrow_subgraph
from app.runtime.planning import validate_plan


@pytest.fixture
def narrowed(registry):
    """Patient + Condition + Observation narrowed from the TEST1 registry."""
    return narrow_subgraph(registry, ["Patient", "Condition", "Observation"])


def test_drops_unknown_resources(narrowed):
    plan = SemanticQueryPlan(
        intent="patient_cohort",
        resources=["Patient", "DiagnosticReport", "Condition"],
    )
    cleaned = validate_plan(plan, narrowed)
    assert cleaned.resources == ["Patient", "Condition"]


def test_canonicalizes_and_dedups_resources(narrowed):
    plan = SemanticQueryPlan(
        intent="patient_cohort",
        resources=["patient", "PATIENT", "  observation "],
    )
    cleaned = validate_plan(plan, narrowed)
    assert cleaned.resources == ["Patient", "Observation"]


def test_drops_filter_on_unknown_resource(narrowed):
    plan = SemanticQueryPlan(
        intent="patient_cohort",
        filters=[
            SemanticFilter(resource="MedicationRequest", concept="metformin"),
            SemanticFilter(resource="Condition", concept="diabetes"),
        ],
    )
    cleaned = validate_plan(plan, narrowed)
    assert [f.resource for f in cleaned.filters] == ["Condition"]


def test_nulls_unknown_filter_path_but_keeps_filter(narrowed):
    plan = SemanticQueryPlan(
        intent="patient_cohort",
        filters=[
            SemanticFilter(
                resource="Observation",
                concept="A1c",
                operator=">",
                value=9,
                path="Observation.bogus.path",
            )
        ],
    )
    cleaned = validate_plan(plan, narrowed)
    assert len(cleaned.filters) == 1
    f = cleaned.filters[0]
    assert f.path is None
    assert f.concept == "A1c"
    assert f.operator == ">"
    assert f.value == 9


def test_keeps_valid_filter_path(narrowed):
    path = "Observation.subject.reference"
    assert path in narrowed.paths["Observation"]  # guard: fixture sanity
    plan = SemanticQueryPlan(
        intent="patient_cohort",
        filters=[SemanticFilter(resource="Observation", path=path)],
    )
    cleaned = validate_plan(plan, narrowed)
    assert cleaned.filters[0].path == path


def test_drops_traversal_with_non_node_endpoint(narrowed):
    plan = SemanticQueryPlan(
        intent="patient_cohort",
        traversal_paths=[
            TraversalPath(
                source_resource="Condition",
                target_resource="Patient",
                relationship_path="Condition.subject.reference",
            ),
            TraversalPath(
                source_resource="MedicationRequest",
                target_resource="Patient",
                relationship_path="MedicationRequest.subject.reference",
            ),
        ],
    )
    cleaned = validate_plan(plan, narrowed)
    assert [
        (t.source_resource, t.target_resource) for t in cleaned.traversal_paths
    ] == [("Condition", "Patient")]


def test_valid_plan_passes_through(narrowed):
    plan = SemanticQueryPlan(
        intent="patient_cohort",
        resources=["Patient", "Condition", "Observation"],
        filters=[
            SemanticFilter(resource="Condition", concept="diabetes"),
            SemanticFilter(
                resource="Observation",
                concept="A1c",
                operator=">",
                value=9,
                path="Observation.subject.reference",
            ),
            SemanticFilter(
                resource="Observation",
                temporal_constraint=TemporalConstraint(
                    kind="relative",
                    direction="last",
                    amount=6,
                    unit="month",
                    label="last 6 months",
                ),
            ),
        ],
        traversal_paths=[
            TraversalPath(
                source_resource="Condition",
                target_resource="Patient",
                relationship_path="Condition.subject.reference",
            ),
            TraversalPath(
                source_resource="Observation",
                target_resource="Patient",
                relationship_path="Observation.subject.reference",
            ),
        ],
    )
    cleaned = validate_plan(plan, narrowed)
    assert cleaned.resources == ["Patient", "Condition", "Observation"]
    assert len(cleaned.filters) == 3
    assert all(f.path != "" for f in cleaned.filters)
    assert len(cleaned.traversal_paths) == 2
