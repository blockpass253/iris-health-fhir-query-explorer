"""Tests for the deterministic schema-view projection used by binding."""

from app.runtime.grounding import build_schema_view, derive_nested_element_path


def _resource(view, name):
    return next(r for r in view.resources if r.name == name)


def test_only_root_resources_exposed(registry):
    view = build_schema_view(registry)
    assert view.table_names() == {"Patient", "Condition", "Encounter", "Observation"}
    # Nested component tables must never surface as resources.
    assert "ObservationCodeCodings" not in view.table_names()
    assert "ConditionEncounters" not in view.table_names()


def test_resource_type_carried(registry):
    view = build_schema_view(registry)
    assert _resource(view, "Patient").resource_type == "Patient"


def test_root_paths_present(registry):
    view = build_schema_view(registry)
    assert "Patient.birthDate" in _resource(view, "Patient").paths
    assert "Condition.subject.reference" in _resource(view, "Condition").paths


def test_nested_paths_folded_into_root(registry):
    view = build_schema_view(registry)
    # The code lives in a nested table but must appear under Observation.
    assert "Observation.code.coding.code" in _resource(view, "Observation").paths
    assert "Observation.code.coding.display" in _resource(view, "Observation").paths
    # The ConditionEncounters child folds to a qualified Condition path.
    assert "Condition.encounter.reference" in _resource(view, "Condition").paths


def test_date_paths_detected(registry):
    view = build_schema_view(registry)
    # Patient.birthDate is a date column and should be flagged for temporal use.
    assert "Patient.birthDate" in _resource(view, "Patient").date_paths


def test_capability_flags_for_grouping_and_correlation(registry):
    view = build_schema_view(registry)
    # Observation has a coding child; Condition/Encounter do not.
    assert _resource(view, "Observation").has_coding_child
    assert not _resource(view, "Condition").has_coding_child
    assert not _resource(view, "Encounter").has_coding_child
    # Resources with a subject reference can correlate by patient identity;
    # Patient itself carries no patient reference.
    assert _resource(view, "Condition").has_patient_reference
    assert _resource(view, "Observation").has_patient_reference
    assert _resource(view, "Encounter").has_patient_reference
    assert not _resource(view, "Patient").has_patient_reference


def test_derive_nested_element_path():
    assert (
        derive_nested_element_path("Observation", "ObservationCodeCodings", ["code"])
        == "Observation.code.coding.code"
    )
    assert (
        derive_nested_element_path("Condition", "ConditionEncounters", ["reference"])
        == "Condition.encounter.reference"
    )
