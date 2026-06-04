"""Tests for the MVP concept -> code synonym dictionary."""

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
