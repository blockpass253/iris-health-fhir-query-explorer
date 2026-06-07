"""Unit tests for semantic inference, using an in-memory fixture mirroring TEST1."""

import pytest

from app.schema.introspection.queries import RawColumn, RawForeignKey, RawTable
from app.schema.models.registry import (
    Confidence,
    RelationshipType,
    SemanticType,
)
from app.semantic.inference import (
    build_tables,
    infer_relationships,
    parse_columns,
)

TABLES = [
    RawTable(t)
    for t in [
        "Base",
        "Patient",
        "Condition",
        "Encounter",
        "Observation",
        "ConditionCodeCodings",
        "ConditionEncounters",
        "ObservationCodes",
    ]
]


def _col(table, name, dtype, desc=None):
    return RawColumn(
        table_name=table, column_name=name, data_type=dtype, description=desc
    )


COLUMNS = [
    _col("Base", "ID", "bigint"),
    _col("Base", "Key", "varchar"),
    _col("Base", "RowNum", "integer"),
    _col("Patient", "ID", "bigint"),
    _col("Patient", "BirthDate", "varchar", "Path: Patient.birthDate"),
    _col("Patient", "Gender", "varchar", "Path: Patient.gender"),
    _col(
        "Patient",
        "FirstName",
        "varchar",
        "Path: Patient.name.where(use = 'official').given",
    ),
    _col("Condition", "ID", "bigint"),
    _col("Condition", "Patient", "varchar", "Path: Condition.subject.reference"),
    _col("Condition", "Encounter", "varchar", "Path: Condition.encounter.reference"),
    _col("Encounter", "ID", "bigint"),
    _col("Encounter", "Patient", "varchar", "Path: Encounter.subject.reference"),
    _col("Encounter", "Status", "varchar", "Path: Encounter.status"),
    _col("Observation", "ID", "bigint"),
    _col("Observation", "Patient", "varchar", "Path: Observation.subject.reference"),
    _col("ConditionCodeCodings", "Condition", "bigint"),
    _col("ConditionCodeCodings", "Code", "varchar", "Path: code"),
    _col("ConditionCodeCodings", "System", "varchar", "Path: system"),
    _col("ConditionEncounters", "Condition", "bigint"),
    _col("ConditionEncounters", "Reference", "varchar", "Path: reference"),
    _col("ObservationCodes", "Observation", "bigint"),
    _col("ObservationCodes", "CodingCode", "varchar", "Path: coding.code"),
]


def _fk(table, parent):
    return RawForeignKey(
        table_name=table,
        column_name=parent,
        referenced_table_name=parent,
        referenced_column_name="ID",
        constraint_name=f"REFERENCE__{parent}",
    )


FKS = [
    _fk("ConditionCodeCodings", "Condition"),
    _fk("ConditionEncounters", "Condition"),
    _fk("ObservationCodes", "Observation"),
]


@pytest.fixture
def tables():
    return build_tables(TABLES, parse_columns(COLUMNS))


@pytest.fixture
def relationships(tables):
    return infer_relationships(tables, FKS)


def _semantic_type(tables, table, column):
    meta = tables[table]
    col = next(c for c in meta.columns if c.column_name == column)
    return col.semantic_type


def test_infrastructure_table_excluded(tables):
    assert "Base" not in tables
    assert set(tables) == {
        "Patient",
        "Condition",
        "Encounter",
        "Observation",
        "ConditionCodeCodings",
        "ConditionEncounters",
        "ObservationCodes",
    }


def test_system_columns_excluded(tables):
    # Root resources retain ID (no FHIR path); nested FK parent columns are dropped.
    assert {c.column_name for c in tables["Patient"].columns} == {
        "ID",
        "BirthDate",
        "Gender",
        "FirstName",
    }
    assert _semantic_type(tables, "Patient", "ID") == SemanticType.IDENTIFIER
    assert {c.column_name for c in tables["ConditionCodeCodings"].columns} == {
        "Code",
        "System",
    }


def test_root_resources_have_id_column(tables):
    for resource in ("Patient", "Condition", "Encounter", "Observation"):
        id_cols = [c for c in tables[resource].columns if c.column_name.upper() == "ID"]
        assert len(id_cols) == 1
        assert id_cols[0].semantic_type == SemanticType.IDENTIFIER
        assert tables[resource].columns[0].column_name.upper() == "ID"


def test_resource_type_inference(tables):
    assert tables["Patient"].inferred_resource_type == "Patient"
    assert tables["Condition"].inferred_resource_type == "Condition"
    assert tables["ConditionCodeCodings"].inferred_resource_type is None
    assert tables["ObservationCodes"].inferred_resource_type is None


def test_semantic_type_classification(tables):
    assert _semantic_type(tables, "Patient", "BirthDate") == SemanticType.DATE
    assert _semantic_type(tables, "Patient", "Gender") == SemanticType.CODE
    assert _semantic_type(tables, "Patient", "FirstName") == SemanticType.STRING
    assert _semantic_type(tables, "Condition", "Patient") == SemanticType.REFERENCE
    assert _semantic_type(tables, "Encounter", "Status") == SemanticType.CODE
    assert _semantic_type(tables, "ConditionCodeCodings", "Code") == SemanticType.CODE
    assert (
        _semantic_type(tables, "ConditionCodeCodings", "System") == SemanticType.CODING
    )
    assert _semantic_type(tables, "ObservationCodes", "CodingCode") == SemanticType.CODE


def test_physical_fks_are_nested(relationships):
    physical = [
        r
        for r in relationships
        if r.relationship_type == RelationshipType.PHYSICAL_FOREIGN_KEY
    ]
    assert len(physical) == 3
    assert all(r.is_nested for r in physical)
    assert all(r.confidence == Confidence.HIGH for r in physical)


def test_resolved_fhir_references(relationships):
    resolved = {
        (r.source_table, r.source_column): r.target_table
        for r in relationships
        if r.relationship_type == RelationshipType.FHIR_REFERENCE
    }
    assert resolved == {
        ("Condition", "Patient"): "Patient",
        ("Condition", "Encounter"): "Encounter",
        ("Encounter", "Patient"): "Patient",
        ("Observation", "Patient"): "Patient",
    }


def test_inferred_reference_from_table_suffix(relationships):
    inferred = [
        r
        for r in relationships
        if r.source_table == "ConditionEncounters" and r.source_column == "Reference"
    ]
    assert len(inferred) == 1
    edge = inferred[0]
    assert edge.target_table == "Encounter"
    assert edge.relationship_type == RelationshipType.INFERRED
    assert edge.confidence == Confidence.LOW
