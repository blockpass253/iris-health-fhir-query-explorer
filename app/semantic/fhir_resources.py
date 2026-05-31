"""Known FHIR resource types used for deterministic name/path matching.

Kept intentionally small and focused on the resources the projection exposes;
extend as new projections are indexed.
"""

KNOWN_FHIR_RESOURCES: frozenset[str] = frozenset(
    {
        "Patient",
        "Observation",
        "Condition",
        "Encounter",
        "Practitioner",
        "Organization",
        "Procedure",
        "MedicationRequest",
        "Medication",
        "AllergyIntolerance",
        "Immunization",
        "DiagnosticReport",
        "CarePlan",
        "Group",
        "Location",
        "Device",
    }
)


def match_resource(name: str | None) -> str | None:
    """Return the canonical FHIR resource matching ``name`` (case-insensitive)."""
    if not name:
        return None
    lowered = name.lower()
    for resource in KNOWN_FHIR_RESOURCES:
        if resource.lower() == lowered:
            return resource
    return None
