"""Tests for schema-derived concept -> code lookup."""

from app.runtime.coding import lookup_codes


def test_known_concept_resolves():
    codes = lookup_codes("diabetes")
    assert codes
    assert any(c.code == "E11" for c in codes)


def test_case_insensitive_and_synonyms():
    assert lookup_codes("Diabetic") == lookup_codes("diabetes")
    assert lookup_codes("HbA1c") == lookup_codes("a1c")
    assert lookup_codes("congestive heart failure") == lookup_codes("heart failure")


def test_unknown_concept_returns_empty():
    assert lookup_codes("unobtainium") == []


def test_generic_drug_name_matches_via_substring():
    # "metformin" is not an exact key; the dict has a full product display string.
    codes = lookup_codes("metformin")
    assert codes, "expected at least one metformin code via substring fallback"
    assert any(c.code == "860975" for c in codes)
    assert all("rxnorm" in c.system.lower() for c in codes)
