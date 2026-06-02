"""Tests for the deterministic runtime context builder."""

from app.runtime.context import build_runtime_context, derive_nested_element_path


def _resource(ctx, name):
    return next(r for r in ctx.resources if r.name == name)


def test_only_root_resources_exposed(registry):
    ctx = build_runtime_context(registry)
    assert ctx.resource_names() == {"Patient", "Condition", "Encounter", "Observation"}
    # Nested component tables must never surface as resources.
    assert "ObservationCodeCodings" not in ctx.resource_names()
    assert "ConditionEncounters" not in ctx.resource_names()


def test_relationships_are_undirected_among_roots(registry):
    ctx = build_runtime_context(registry)
    assert _resource(ctx, "Patient").relationships == [
        "Condition",
        "Encounter",
        "Observation",
    ]
    assert _resource(ctx, "Condition").relationships == ["Encounter", "Patient"]
    assert _resource(ctx, "Observation").relationships == ["Patient"]


def test_root_paths_present(registry):
    ctx = build_runtime_context(registry)
    assert "Patient.birthDate" in _resource(ctx, "Patient").paths
    assert "Condition.subject.reference" in _resource(ctx, "Condition").paths


def test_nested_paths_folded_into_root(registry):
    ctx = build_runtime_context(registry)
    # The LOINC code lives in a nested table but must appear under Observation.
    assert "Observation.code.coding.code" in _resource(ctx, "Observation").paths
    assert "Observation.code.coding.display" in _resource(ctx, "Observation").paths
    # The ConditionEncounters child folds to a qualified Condition path.
    assert "Condition.encounter.reference" in _resource(ctx, "Condition").paths


def test_derive_nested_element_path():
    assert (
        derive_nested_element_path("Observation", "ObservationCodeCodings", ["code"])
        == "Observation.code.coding.code"
    )
    assert (
        derive_nested_element_path("Condition", "ConditionEncounters", ["reference"])
        == "Condition.encounter.reference"
    )
