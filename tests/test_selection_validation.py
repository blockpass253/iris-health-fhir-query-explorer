"""Tests for the deterministic resource-selection validation helper."""

from app.runtime.models import ResourceSelectionResult, RuntimeContext, RuntimeResource
from app.runtime.selection import validate_selection

_CTX = RuntimeContext(
    resources=[
        RuntimeResource(name="Patient"),
        RuntimeResource(name="Condition"),
        RuntimeResource(name="Observation"),
    ]
)


def test_drops_hallucinated_resources():
    raw = ResourceSelectionResult(
        resources=["Patient", "DiagnosticReport", "Condition"]
    )
    cleaned = validate_selection(raw, _CTX)
    assert cleaned.resources == ["Patient", "Condition"]


def test_canonicalizes_case_and_dedups():
    raw = ResourceSelectionResult(resources=["patient", "PATIENT", "  observation "])
    cleaned = validate_selection(raw, _CTX)
    assert cleaned.resources == ["Patient", "Observation"]


def test_preserves_reasoning():
    raw = ResourceSelectionResult(resources=["Patient"], reasoning="cohort root")
    assert validate_selection(raw, _CTX).reasoning == "cohort root"
