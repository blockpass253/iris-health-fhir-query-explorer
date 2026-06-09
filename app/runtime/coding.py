"""Concept -> terminology code lookup from the indexed schema dictionary.

Binding resolves a filter's semantic ``concept`` (e.g. "diabetes", "a1c") to one
or more :class:`Coding` entries via the coding dictionary produced by
``index-schema``. Keys are the lowercased ``display`` values sampled from coding
child tables in the live FHIR projection. ``_SYNONYMS`` folds common phrasings
onto a canonical key before lookup; it does not introduce codes or systems.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.runtime.models import Coding

if TYPE_CHECKING:
    from app.schema.persistence.coding_store import CodingDictionary

_DICT_CACHE: CodingDictionary | None = None
_dict_loaded: bool = False

# Common phrasings -> canonical lookup key (must exist in the schema dictionary).
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


def reset_coding_cache() -> None:
    """Clear the in-process dictionary cache (for tests)."""
    global _DICT_CACHE, _dict_loaded
    _DICT_CACHE = None
    _dict_loaded = False


def lookup_codes(concept: str) -> list[Coding]:
    """Return the codes for ``concept``, or ``[]`` if it is not in the dictionary."""
    key = concept.strip().lower()
    key = _SYNONYMS.get(key, key)
    d = _load_dict()
    if d is None:
        return []
    return d.lookup(key)
