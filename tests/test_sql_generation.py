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
    BoundGroupBy,
    BoundPlan,
    BoundTemporal,
    Filter,
    GroupBy,
    TemporalConstraint,
)
from app.runtime.sql_generation import SqlQuery, _as_bool, generate_sql, render_sql


def test_as_bool_only_reads_numbers_as_bool_when_numeric_ok():
    # Words/booleans are always boolean-ish.
    assert _as_bool("false") is False
    assert _as_bool(True) is True
    # Numeric 0/1 is a literal value by default (genuine numeric filters)...
    assert _as_bool(0) is None
    assert _as_bool(1) is None
    assert _as_bool("0") is None
    # ...but a presence test on a date column opts into the 0/1 reading.
    assert _as_bool(0, numeric_ok=True) is False
    assert _as_bool(1, numeric_ok=True) is True
    assert _as_bool("0", numeric_ok=True) is False
    # Genuine non-boolean numbers never collapse to a presence test.
    assert _as_bool(42, numeric_ok=True) is None


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

    assert sql.sql.startswith("SELECT DISTINCT TOP 50 r.*")
    assert 'FROM "TEST1"."Patient" r' in sql.sql
    assert 'EXISTS (SELECT 1 FROM "TEST1"."Observation" r0' in sql.sql
    assert 'r0."Patient" = \'Patient/\' || r."ID"' in sql.sql
    assert '"TEST1"."ObservationCodeCodings" r0c' in sql.sql
    assert 'r0c."Observation" = r0."ID"' in sql.sql
    assert '(r0c."System" = ? AND r0c."Code" = ?)' in sql.sql
    assert sql.params == ["http://loinc.org", "4548-4"]


def test_concept_filter_with_value_comparison_emits_both_predicates(registry):
    # A1c > 9: concept resolves to LOINC code (coding EXISTS) AND value > 9 (scalar).
    codings = lookup_codes("a1c")
    bound = BoundPlan(
        intent="list",
        resource_tables={"Patient": "Patient", "Observation": "Observation"},
        filters=[
            BoundFilter(
                filter=Filter(
                    resource="Observation",
                    concept="a1c",
                    operator=">",
                    value=9,
                ),
                table="Observation",
                column_path="Observation.value.quantity.value",
                codings=codings,
            )
        ],
    )

    sql = generate_sql(bound, registry)

    # The coding EXISTS must be present.
    assert '"TEST1"."ObservationCodeCodings" r0c' in sql.sql
    assert '(r0c."System" = ? AND r0c."Code" = ?)' in sql.sql
    # The value comparison must be a nested EXISTS into the child table, not a
    # bare column reference on the Observation alias (which IRIS would reject).
    assert '"TEST1"."Observation"' in sql.sql  # there IS an Observation EXISTS group
    assert "ValueQuantityValue" in sql.sql  # the physical column is referenced
    assert "> ?" in sql.sql
    assert sql.params == ["http://loinc.org", "4548-4", 9]


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

    assert 'r."Gender" = ?' in sql.sql
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

    assert 'r."DeceasedDateTime" IS NULL' in sql.sql
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

    assert 'r."DeceasedDateTime" IS NOT NULL' in sql.sql
    assert sql.params == []


def test_deceased_filter_with_dropped_value_degrades_to_null_check(registry):
    # If the extractor drops the boolean (value=None) on a polymorphic date
    # column, a bare '=' must degrade to IS NULL rather than emit a dangling '?'
    # placeholder (which previously crashed render_sql with a param mismatch).
    bound = BoundPlan(
        intent="list",
        resource_tables={"Patient": "Patient"},
        filters=[
            BoundFilter(
                filter=Filter(
                    resource="Patient",
                    path="deceased",
                    operator="=",
                    value=None,
                ),
                table="Patient",
                column_path="Patient.deceasedDateTime",
            )
        ],
    )

    sql = generate_sql(bound, registry)

    assert 'r."DeceasedDateTime" IS NULL' in sql.sql
    assert sql.params == []
    # render_sql validates placeholder/param parity; it must not raise.
    assert "?" not in render_sql(sql)


def test_deceased_filter_with_integer_zero_is_presence_check(registry):
    # The extractor often encodes deceased=false as the integer 0. On a date
    # column that must read as IS NULL (alive), not a literal `= 0`.
    bound = BoundPlan(
        intent="list",
        resource_tables={"Patient": "Patient"},
        filters=[
            BoundFilter(
                filter=Filter(
                    resource="Patient", path="deceased", operator="=", value=0
                ),
                table="Patient",
                column_path="Patient.deceasedDateTime",
            )
        ],
    )

    sql = generate_sql(bound, registry)

    assert 'r."DeceasedDateTime" IS NULL' in sql.sql
    assert sql.params == []
    assert "= 0" not in render_sql(sql)


def test_deceased_filter_with_integer_one_is_not_null_presence_check(registry):
    bound = BoundPlan(
        intent="list",
        resource_tables={"Patient": "Patient"},
        filters=[
            BoundFilter(
                filter=Filter(
                    resource="Patient", path="deceased", operator="=", value=1
                ),
                table="Patient",
                column_path="Patient.deceasedDateTime",
            )
        ],
    )

    sql = generate_sql(bound, registry)

    assert 'r."DeceasedDateTime" IS NOT NULL' in sql.sql
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

    assert 'r."BirthDate" <= ?' in sql.sql  # older than 65 => born on/before cutoff
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

    assert 'r."BirthDate" >= ?' in sql.sql
    assert sql.params == [(date.today() - timedelta(days=30)).isoformat()]


def test_count_intent(registry):
    bound = BoundPlan(intent="count", resource_tables={"Patient": "Patient"})
    sql = generate_sql(bound, registry)
    assert sql.sql.startswith('SELECT COUNT(DISTINCT r."ID")')


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

    assert 'r."Sex" = ?' in sql.sql
    assert 'r0c."Cd" = ?' in sql.sql


# --- Root-aware list/count ---------------------------------------------------


def test_non_patient_root_list(registry):
    bound = BoundPlan(
        intent="list",
        root_resource="Encounter",
        resource_tables={"Encounter": "Encounter"},
        filters=[
            BoundFilter(
                filter=Filter(
                    resource="Encounter",
                    path="status",
                    operator="=",
                    value="finished",
                ),
                table="Encounter",
                column_path="Encounter.status",
            )
        ],
    )

    sql = generate_sql(bound, registry)

    assert sql.sql.startswith("SELECT DISTINCT TOP 50 r.*")
    assert 'FROM "TEST1"."Encounter" r' in sql.sql
    assert 'r."Status" = ?' in sql.sql
    assert sql.params == ["finished"]


def test_non_patient_root_correlates_other_resource_by_patient_ref(registry):
    bound = BoundPlan(
        intent="list",
        root_resource="Encounter",
        resource_tables={"Encounter": "Encounter", "Observation": "Observation"},
        filters=[
            BoundFilter(
                filter=Filter(
                    resource="Encounter",
                    path="status",
                    operator="=",
                    value="finished",
                ),
                table="Encounter",
                column_path="Encounter.status",
            ),
            BoundFilter(
                filter=Filter(resource="Observation", concept="a1c"),
                table="Observation",
                codings=lookup_codes("a1c"),
            ),
        ],
    )

    sql = generate_sql(bound, registry)

    assert 'r."Status" = ?' in sql.sql
    assert 'EXISTS (SELECT 1 FROM "TEST1"."Observation" r0' in sql.sql
    # Non-patient root links by comparing both patient-reference columns.
    assert 'r0."Patient" = r."Patient"' in sql.sql
    assert sql.params == ["finished", "http://loinc.org", "4548-4"]


def test_root_limit_overrides_default_list_cap(registry):
    bound = BoundPlan(
        intent="list",
        root_resource="Encounter",
        resource_tables={"Encounter": "Encounter"},
        limit=10,
    )
    sql = generate_sql(bound, registry)
    assert sql.sql.startswith("SELECT DISTINCT TOP 10 r.*")


# --- Rank --------------------------------------------------------------------


def test_rank_concept_grouping_on_coding_child(registry):
    bound = BoundPlan(
        intent="rank",
        root_resource="Observation",
        resource_tables={"Observation": "Observation"},
        group_by=BoundGroupBy(
            group_by=GroupBy(resource="Observation", concept=True),
            table="Observation",
        ),
        limit=5,
    )

    sql = generate_sql(bound, registry)

    assert sql.sql.startswith("SELECT TOP 5 ")
    assert 'g."Code" AS code' in sql.sql
    assert 'g."Display" AS display' in sql.sql
    assert 'g."System" AS system' in sql.sql
    assert 'COUNT(DISTINCT r."ID") AS cnt' in sql.sql
    assert 'FROM "TEST1"."Observation" r' in sql.sql
    assert (
        'JOIN "TEST1"."ObservationCodeCodings" g ON g."Observation" = r."ID"' in sql.sql
    )
    assert 'GROUP BY g."Code", g."Display", g."System"' in sql.sql
    assert sql.sql.rstrip().endswith("ORDER BY cnt DESC")
    assert sql.params == []


def test_rank_path_grouping_on_direct_attribute(registry):
    bound = BoundPlan(
        intent="rank",
        root_resource="Encounter",
        resource_tables={"Encounter": "Encounter"},
        group_by=BoundGroupBy(
            group_by=GroupBy(resource="Encounter", path="status"),
            table="Encounter",
            column_path="Encounter.status",
        ),
    )

    sql = generate_sql(bound, registry)

    assert sql.sql.startswith("SELECT TOP 5 ")  # default rank limit
    assert 'r."Status" AS status' in sql.sql
    assert 'COUNT(DISTINCT r."ID") AS cnt' in sql.sql
    assert 'FROM "TEST1"."Encounter" r' in sql.sql
    assert 'GROUP BY r."Status"' in sql.sql
    assert "ORDER BY cnt DESC" in sql.sql
    assert sql.params == []


def test_rank_with_temporal_filter_on_root(registry):
    bound = BoundPlan(
        intent="rank",
        root_resource="Patient",
        resource_tables={"Patient": "Patient"},
        group_by=BoundGroupBy(
            group_by=GroupBy(resource="Patient", path="gender"),
            table="Patient",
            column_path="Patient.gender",
        ),
        temporal_constraints=[
            BoundTemporal(
                constraint=TemporalConstraint(resource="Patient", last_n_days=30),
                table="Patient",
                column_path="Patient.birthDate",
            )
        ],
    )

    sql = generate_sql(bound, registry)

    assert 'r."Gender" AS gender' in sql.sql
    assert 'WHERE r."BirthDate" >= ?' in sql.sql
    assert 'GROUP BY r."Gender"' in sql.sql
    assert sql.params == [(date.today() - timedelta(days=30)).isoformat()]


# --- render_sql --------------------------------------------------------------


def test_render_sql_without_params():
    sql = SqlQuery("SELECT 1", [])
    assert render_sql(sql) == "SELECT 1"


def test_render_sql_inlines_string_param():
    sql = SqlQuery('SELECT * FROM t WHERE "Gender" = ?', ["female"])
    assert render_sql(sql) == "SELECT * FROM t WHERE \"Gender\" = 'female'"


def test_render_sql_inlines_system_code_pair():
    sql = SqlQuery(
        'WHERE (c."System" = ? AND c."Code" = ?)',
        ["http://loinc.org", "4548-4"],
    )
    assert render_sql(sql) == (
        "WHERE (c.\"System\" = 'http://loinc.org' AND c.\"Code\" = '4548-4')"
    )


def test_render_sql_escapes_single_quotes():
    sql = SqlQuery("WHERE name = ?", ["O'Brien"])
    assert render_sql(sql) == "WHERE name = 'O''Brien'"


def test_render_sql_rejects_param_count_mismatch():
    sql = SqlQuery("WHERE a = ? AND b = ?", ["only-one"])
    try:
        render_sql(sql)
    except ValueError as exc:
        assert "expected 2 params, got 1" in str(exc)
    else:
        raise AssertionError("expected ValueError")
