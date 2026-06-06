"""Shared test fixtures: an in-memory semantic registry mirroring TEST1."""

from datetime import UTC, datetime

import pytest

from app.schema.graph.builder import build_semantic_graph
from app.schema.introspection.queries import RawColumn, RawForeignKey, RawTable
from app.schema.models.registry import SchemaRegistry
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
    _col("ObservationCodeCodings", "Observation", "bigint"),
    _col("ObservationCodeCodings", "Code", "varchar", "Path: code"),
    _col("ObservationCodeCodings", "Display", "varchar", "Path: display"),
    _col("ObservationCodeCodings", "System", "varchar", "Path: system"),
    _col("ConditionEncounters", "Condition", "bigint"),
    _col("ConditionEncounters", "Reference", "varchar", "Path: reference"),
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
