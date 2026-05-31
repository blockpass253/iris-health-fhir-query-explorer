"""Unit tests for the FHIR path parser, using real TEST1 description strings."""

from app.schema.parsers.fhir_path import parse_fhir_path


def test_none_and_empty():
    assert parse_fhir_path(None) is None
    assert parse_fhir_path("") is None
    assert parse_fhir_path("Path: ") is None


def test_rooted_reference():
    p = parse_fhir_path("Path: Condition.subject.reference")
    assert p is not None
    assert p.resource_type == "Condition"
    assert p.segments == ["Condition", "subject", "reference"]
    assert p.is_reference is True
    assert p.terminal_field == "reference"


def test_rootless_reference():
    p = parse_fhir_path("Path: reference")
    assert p is not None
    assert p.resource_type is None
    assert p.is_reference is True
    assert p.terminal_field == "reference"


def test_rootless_coding_path():
    p = parse_fhir_path("Path: coding.code")
    assert p is not None
    assert p.resource_type is None
    assert p.is_reference is False
    assert p.segments == ["coding", "code"]
    assert p.terminal_field == "code"


def test_scalar_resource_field():
    p = parse_fhir_path("Path: Patient.birthDate")
    assert p is not None
    assert p.resource_type == "Patient"
    assert p.terminal_field == "birthDate"
    assert p.is_reference is False


def test_function_expression_not_special_cased():
    # The projection already filtered this in SQL; we keep raw + best-effort segments.
    p = parse_fhir_path("Path: Patient.name.where(use = 'official').given")
    assert p is not None
    assert p.resource_type == "Patient"
    assert p.raw == "Path: Patient.name.where(use = 'official').given"
    assert p.terminal_field == "given"
    assert p.is_reference is False
