"""A small concept -> terminology code dictionary for the MVP.

Binding resolves a filter's semantic ``concept`` (e.g. "diabetes", "metformin")
to one or more :class:`Coding` entries via this map. It is intentionally tiny and
hand-curated: enough to demonstrate the end-to-end intent for the contest demo,
not a real terminology service. Lookups are case-insensitive and tolerant of a
few common synonyms.
"""

from app.runtime.models import Coding

ICD10 = "http://hl7.org/fhir/sid/icd-10-cm"
SNOMED = "http://snomed.info/sct"
LOINC = "http://loinc.org"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"

# Canonical concept -> codes. Keys are lowercase; synonyms are folded in
# ``_SYNONYMS`` below so several phrasings resolve to the same entry.
SYNONYMS: dict[str, list[Coding]] = {
    "diabetes": [
        Coding(system=ICD10, code="E11", display="Type 2 diabetes mellitus"),
        Coding(system=SNOMED, code="44054006", display="Diabetes mellitus type 2"),
    ],
    "heart failure": [
        Coding(system=ICD10, code="I50", display="Heart failure"),
        Coding(system=SNOMED, code="84114007", display="Heart failure"),
    ],
    "a1c": [
        Coding(system=LOINC, code="4548-4", display="Hemoglobin A1c/Hemoglobin.total"),
    ],
    "metformin": [
        Coding(system=RXNORM, code="6809", display="Metformin"),
    ],
}

# Alternate phrasings -> canonical key in ``SYNONYMS``.
_SYNONYMS: dict[str, str] = {
    "diabetic": "diabetes",
    "diabetes mellitus": "diabetes",
    "type 2 diabetes": "diabetes",
    "hba1c": "a1c",
    "hemoglobin a1c": "a1c",
    "glycated hemoglobin": "a1c",
    "chf": "heart failure",
    "congestive heart failure": "heart failure",
}


def lookup_codes(concept: str) -> list[Coding]:
    """Return the codes for ``concept``, or ``[]`` if it is not in the dictionary.

    Matching is case-insensitive and resolves known synonyms to their canonical
    concept.
    """
    key = concept.strip().lower()
    key = _SYNONYMS.get(key, key)
    return SYNONYMS.get(key, [])
