"""A small concept -> terminology code dictionary for the MVP.

Binding resolves a filter's semantic ``concept`` (e.g. "diabetes", "metformin")
to one or more :class:`Coding` entries via this map. It is intentionally tiny and
hand-curated: enough to demonstrate the end-to-end intent for the contest demo,
not a real terminology service. Lookups are case-insensitive and tolerant of a
few common synonyms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.runtime.models import Coding

if TYPE_CHECKING:
    from app.schema.persistence.coding_store import CodingDictionary

_DICT_CACHE: CodingDictionary | None = None
_dict_loaded: bool = False

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

_SYNONYMS: dict[str, str] = {
    "diabetic": "diabetes",
    "diabetes mellitus": "diabetes",
    "type 2 diabetes": "diabetes",
    "hba1c": "a1c",
    "hemoglobin a1c": "a1c",
    "glycated hemoglobin": "a1c",
    "chf": "heart failure",
    "congestive heart failure": "heart failure",
    "chronic heart failure": "heart failure",
    "htn": "hypertension",
    "high blood pressure": "hypertension",
    "elevated blood pressure": "hypertension",
    "heart attack": "myocardial infarction",
    "mi": "myocardial infarction",
    "cva": "stroke",
    "cerebrovascular accident": "stroke",
    "afib": "atrial fibrillation",
    "a-fib": "atrial fibrillation",
    "asthmatic": "asthma",
    "copd": "chronic obstructive bronchitis (disorder)",
    "chronic obstructive pulmonary disease": "chronic obstructive bronchitis (disorder)",  # noqa: E501
    "emphysema": "pulmonary emphysema (disorder)",
    "high cholesterol": "hyperlipidemia",
    "dyslipidemia": "hyperlipidemia",
    "elevated cholesterol": "hyperlipidemia",
    "anemia": "anemia (disorder)",
    "pre-diabetes": "prediabetes",
    "impaired glucose tolerance": "prediabetes",
    "obesity": "body mass index 30+ - obesity (finding)",
    "obese": "body mass index 30+ - obesity (finding)",
    "alzheimer": "alzheimer's disease (disorder)",
    "alzheimer's": "alzheimer's disease (disorder)",
    "dementia": "alzheimer's disease (disorder)",
    "ckd": "chronic kidney disease stage 1 (disorder)",
    "chronic kidney disease": "chronic kidney disease stage 1 (disorder)",
    "cad": "coronary heart disease",
    "coronary artery disease": "coronary heart disease",
    "metabolic syndrome": "metabolic syndrome x (disorder)",
    "seizures": "epilepsy",
    "osteoporosis": "osteoporosis (disorder)",
    "ear infection": "otitis media",
    "weight": "body weight",
    "height": "body height",
    "bmi": "body mass index",
    "blood sugar": "glucose",
    "blood glucose": "glucose",
    "sugar": "glucose",
    "cholesterol": "total cholesterol",
    "hdl": "high density lipoprotein cholesterol",
    "hdl cholesterol": "high density lipoprotein cholesterol",
    "ldl": "low density lipoprotein cholesterol",
    "ldl cholesterol": "low density lipoprotein cholesterol",
    "bp": "blood pressure",
    "gfr": "glomerular filtration rate/1.73 sq m.predicted",
    "egfr": "glomerular filtration rate/1.73 sq m.predicted",
    "psa": "prostate specific ag [mass/volume] in serum or plasma",
    "ejection fraction": "left ventricular ejection fraction",
    "lvef": "left ventricular ejection fraction",
    "bnp": "nt-probnp",
    "smoking": "tobacco smoking status nhis",
    "smoking status": "tobacco smoking status nhis",
}


def _load_dict() -> CodingDictionary | None:
    """Return the generated CodingDictionary from disk, or None if absent.

    Deferred import breaks the potential coding_store -> models -> coding cycle.
    Result is cached for the lifetime of the process.
    """
    global _DICT_CACHE, _dict_loaded
    if not _dict_loaded:
        from app.schema.persistence.coding_store import load_coding_dictionary

        _DICT_CACHE = load_coding_dictionary()
        _dict_loaded = True
    return _DICT_CACHE


def lookup_codes(concept: str) -> list[Coding]:
    """Return the codes for ``concept``, or ``[]`` if it is not in the dictionary.

    Reads from the generated coding dictionary JSON when available (produced by
    ``index-schema``), falling back to the hardcoded dicts for test environments
    or when the file has not yet been generated.
    """
    key = concept.strip().lower()
    key = _SYNONYMS.get(key, key)  # always use hardcoded aliases
    d = _load_dict()
    if d is not None:
        return d.lookup(key)
    return SYNONYMS.get(key, [])
