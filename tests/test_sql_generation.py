"""Tests for deterministic SQL generation (no LLM, no live IRIS).

Builds :class:`BoundPlan` objects directly against the in-memory TEST1 registry
fixture and asserts the generated SQL and parameters. The recurring theme is that
column selection is driven by registry *metadata* (semantic type / FHIR terminal
element) while emitted SQL uses the real physical names — so renaming a physical
column must not change which column is selected, only how it is spelled.
"""

from datetime import date, timedelta

from app.runtime.coding import lookup_codes
from app.runtime.grounding import (
    coding_child,
    patient_reference_column,
    resolve_column_path,
)
from app.runtime.models import (
    BoundFilter,
    BoundPlan,
    BoundTemporal,
    Filter,
    TemporalConstraint,
)
from app.runtime.sql_generation import generate_sql

# --- Column lookups ----------------------------------------------------------


def test_resolve_column_path_to_physical_column(registry):
    ref = resolve_column_path(registry, "Patient", "Patient.gender")
    assert ref is not None
    assert (ref.table, ref.column) == ("Patient", "Gender")


def test_coding_child_resolves_real_names(registry):
    child = coding_child(registry, "Observation")
    assert child is not None
    assert child.table == "ObservationCodeCodings"
    assert child.fk_column == "Observation"
    assert child.code.column == "Code"
    assert child.system is not None and child.system.column == "System"


def test_patient_reference_picks_subject_not_encounter(registry):
    ref = patient_reference_column(registry.tables["Condition"])
    assert ref is not None
    assert ref.column == "Patient"  # subject.reference, not encounter.reference


# --- Generation --------------------------------------------------------------


def test_concept_filter_emits_nested_exists_with_system_code_pairs(registry):
    codings = lookup_codes("a1c")
    bound = BoundPlan(
        intent="list",
        resource_tables={"Patient": "Patient", "Observation": "Observation"},
        filters=[
            BoundFilter(
                filter=Filter(resource="Observation", concept="a1c"),
                table="Observation",
                codings=codings,
            )
        ],
    )

    sql = generate_sql(bound, registry)

    assert sql.sql.startswith("SELECT DISTINCT TOP 50 p.*")
    assert 'FROM "TEST1"."Patient" p' in sql.sql
    assert 'EXISTS (SELECT 1 FROM "TEST1"."Observation" r0' in sql.sql
    assert 'r0."Patient" = \'Patient/\' || p."ID"' in sql.sql
    assert '"TEST1"."ObservationCodeCodings" r0c' in sql.sql
    assert 'r0c."Observation" = r0."ID"' in sql.sql
    assert '(r0c."System" = ? AND r0c."Code" = ?)' in sql.sql
    assert sql.params == ["http://loinc.org", "4548-4"]


def test_patient_attribute_filter(registry):
    bound = BoundPlan(
        intent="list",
        resource_tables={"Patient": "Patient"},
        filters=[
            BoundFilter(
                filter=Filter(
                    resource="Patient",
                    path="Patient.gender",
                    operator="=",
                    value="female",
                ),
                table="Patient",
                column_path="Patient.gender",
            )
        ],
    )

    sql = generate_sql(bound, registry)

    assert 'p."Gender" = ?' in sql.sql
    assert sql.params == ["female"]


def test_alive_filter_emits_null_presence_check(registry):
    # "alive" -> deceased=false on the polymorphic deceasedDateTime column means
    # the death datetime IS NULL, not a literal '= false' comparison.
    bound = BoundPlan(
        intent="list",
        resource_tables={"Patient": "Patient"},
        filters=[
            BoundFilter(
                filter=Filter(
                    resource="Patient",
                    path="deceased",
                    operator="=",
                    value="false",
                ),
                table="Patient",
                column_path="Patient.deceasedDateTime",
            )
        ],
    )

    sql = generate_sql(bound, registry)

    assert 'p."DeceasedDateTime" IS NULL' in sql.sql
    assert sql.params == []


def test_deceased_filter_emits_not_null_presence_check(registry):
    bound = BoundPlan(
        intent="list",
        resource_tables={"Patient": "Patient"},
        filters=[
            BoundFilter(
                filter=Filter(
                    resource="Patient",
                    path="deceased",
                    operator="=",
                    value="true",
                ),
                table="Patient",
                column_path="Patient.deceasedDateTime",
            )
        ],
    )

    sql = generate_sql(bound, registry)

    assert 'p."DeceasedDateTime" IS NOT NULL' in sql.sql
    assert sql.params == []


def test_age_filter_uses_birthdate_threshold(registry):
    bound = BoundPlan(
        intent="list",
        resource_tables={"Patient": "Patient"},
        filters=[
            BoundFilter(
                filter=Filter(resource="Patient", path="age", operator=">", value=65),
                table="Patient",
            )
        ],
    )

    sql = generate_sql(bound, registry)

    assert 'p."BirthDate" <= ?' in sql.sql  # older than 65 => born on/before cutoff
    assert len(sql.params) == 1
    assert sql.params[0][:4] == str(date.today().year - 65)


def test_temporal_filter_uses_computed_date(registry):
    bound = BoundPlan(
        intent="list",
        resource_tables={"Patient": "Patient"},
        temporal_constraints=[
            BoundTemporal(
                constraint=TemporalConstraint(resource="Patient", last_n_days=30),
                table="Patient",
                column_path="Patient.birthDate",
            )
        ],
    )

    sql = generate_sql(bound, registry)

    assert 'p."BirthDate" >= ?' in sql.sql
    assert sql.params == [(date.today() - timedelta(days=30)).isoformat()]


def test_count_intent(registry):
    bound = BoundPlan(intent="count", resource_tables={"Patient": "Patient"})
    sql = generate_sql(bound, registry)
    assert sql.sql.startswith('SELECT COUNT(DISTINCT p."ID")')


def test_generation_uses_renamed_physical_columns(registry):
    """Selection is by metadata; renaming physical columns only changes spelling."""
    custom = registry.model_copy(deep=True)
    for col in custom.tables["Patient"].columns:
        if col.parsed_fhir_path and col.parsed_fhir_path.terminal_field == "gender":
            col.column_name = "Sex"
    for col in custom.tables["ObservationCodeCodings"].columns:
        if col.parsed_fhir_path and col.parsed_fhir_path.terminal_field == "code":
            col.column_name = "Cd"

    bound = BoundPlan(
        intent="list",
        resource_tables={"Patient": "Patient", "Observation": "Observation"},
        filters=[
            BoundFilter(
                filter=Filter(
                    resource="Patient",
                    path="Patient.gender",
                    operator="=",
                    value="female",
                ),
                table="Patient",
                column_path="Patient.gender",
            ),
            BoundFilter(
                filter=Filter(resource="Observation", concept="a1c"),
                table="Observation",
                codings=lookup_codes("a1c"),
            ),
        ],
    )

    sql = generate_sql(bound, custom)

    assert 'p."Sex" = ?' in sql.sql
    assert 'r0c."Cd" = ?' in sql.sql
