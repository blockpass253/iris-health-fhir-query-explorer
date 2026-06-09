"""Shared test fixtures: an in-memory semantic registry mirroring TEST1."""

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from app.runtime import coding as coding_mod
from app.schema.graph.builder import build_semantic_graph
from app.schema.introspection.queries import RawColumn, RawForeignKey, RawTable
from app.schema.models.registry import SchemaRegistry
from app.schema.persistence.coding_store import CodingDictionary, CodingRef
from app.semantic.inference import build_tables, infer_relationships, parse_columns

_TABLES = [
    RawTable(t)
    for t in [
        "Patient",
        "Condition",
        "Encounter",
        "Observation",
        "ObservationCodeCodings",
        "ConditionEncounters",
        "MedicationRequest",
    ]
]


def _col(table, name, dtype, desc=None):
    return RawColumn(
        table_name=table, column_name=name, data_type=dtype, description=desc
    )


_COLUMNS = [
    _col("Patient", "ID", "bigint"),
    _col("Patient", "BirthDate", "varchar", "Path: Patient.birthDate"),
    _col("Patient", "Gender", "varchar", "Path: Patient.gender"),
    _col("Patient", "DeceasedDateTime", "varchar", "Path: Patient.deceasedDateTime"),
    _col("Condition", "ID", "bigint"),
    _col("Condition", "Patient", "varchar", "Path: Condition.subject.reference"),
    _col("Condition", "Encounter", "varchar", "Path: Condition.encounter.reference"),
    _col("Encounter", "ID", "bigint"),
    _col("Encounter", "Patient", "varchar", "Path: Encounter.subject.reference"),
    _col("Encounter", "Status", "varchar", "Path: Encounter.status"),
    _col("Observation", "ID", "bigint"),
    _col("Observation", "Patient", "varchar", "Path: Observation.subject.reference"),
    _col(
        "Observation",
        "ValueQuantityValue",
        "decimal",
        "Path: Observation.value.quantity.value",
    ),
    _col("ObservationCodeCodings", "Observation", "bigint"),
    _col("ObservationCodeCodings", "Code", "varchar", "Path: code"),
    _col("ObservationCodeCodings", "Display", "varchar", "Path: display"),
    _col("ObservationCodeCodings", "System", "varchar", "Path: system"),
    _col("ConditionEncounters", "Condition", "bigint"),
    _col("ConditionEncounters", "Reference", "varchar", "Path: reference"),
    # MedicationRequest: coding columns are flat on the root table (no child table).
    _col("MedicationRequest", "ID", "bigint"),
    _col(
        "MedicationRequest",
        "Patient",
        "varchar",
        "Path: MedicationRequest.subject.reference",
    ),
    _col(
        "MedicationRequest",
        "Code",
        "varchar",
        "Path: MedicationRequest.medicationCodeableConcept.coding.code",
    ),
    _col(
        "MedicationRequest",
        "System",
        "varchar",
        "Path: MedicationRequest.medicationCodeableConcept.coding.system",
    ),
    _col(
        "MedicationRequest",
        "Display",
        "varchar",
        "Path: MedicationRequest.medicationCodeableConcept.coding.display",
    ),
]


def _fk(table, parent):
    return RawForeignKey(
        table_name=table,
        column_name=parent,
        referenced_table_name=parent,
        referenced_column_name="ID",
        constraint_name=f"REFERENCE__{parent}",
    )


_FKS = [
    _fk("ObservationCodeCodings", "Observation"),
    _fk("ConditionEncounters", "Condition"),
]


def _test_coding_dictionary() -> CodingDictionary:
    """Minimal dictionary mirroring profiled display keys from a TEST1 projection."""
    return CodingDictionary(
        schema_name="TEST1",
        generated_at=datetime.now(UTC),
        systems={
            "http://hl7.org/fhir/sid/icd-10-cm": {
                "diabetes": CodingRef(code="E11", display="Type 2 diabetes mellitus"),
                "type 2 diabetes mellitus": CodingRef(
                    code="E11", display="Type 2 diabetes mellitus"
                ),
                "heart failure": CodingRef(code="I50", display="Heart failure"),
            },
            "http://snomed.info/sct": {
                "diabetes": CodingRef(
                    code="44054006", display="Diabetes mellitus type 2"
                ),
                "heart failure": CodingRef(code="84114007", display="Heart failure"),
            },
            "http://loinc.org": {
                "a1c": CodingRef(
                    code="4548-4", display="Hemoglobin A1c/Hemoglobin.total"
                ),
                "hemoglobin a1c/hemoglobin.total": CodingRef(
                    code="4548-4", display="Hemoglobin A1c/Hemoglobin.total"
                ),
            },
            "http://www.nlm.nih.gov/research/umls/rxnorm": {
                (
                    "24 hr metformin hydrochloride 500 mg extended release oral tablet"
                ): CodingRef(
                    code="860975",
                    display=(
                        "24 HR Metformin Hydrochloride 500 MG "
                        "Extended Release Oral Tablet"
                    ),
                ),
            },
        },
    )


@pytest.fixture(autouse=True)
def coding_dictionary(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Provide a schema-derived coding dictionary for every test."""
    monkeypatch.setattr(coding_mod, "_DICT_CACHE", _test_coding_dictionary())
    monkeypatch.setattr(coding_mod, "_dict_loaded", True)
    yield
    coding_mod.reset_coding_cache()


@pytest.fixture
def registry() -> SchemaRegistry:
    tables = build_tables(_TABLES, parse_columns(_COLUMNS))
    relationships = infer_relationships(tables, _FKS)
    graph = build_semantic_graph(tables, relationships)
    return SchemaRegistry(
        schema_name="TEST1",
        namespace="FHIRSERVER",
        generated_at=datetime.now(UTC),
        tables=tables,
        relationships=relationships,
        graph=graph,
        stats={},
    )
