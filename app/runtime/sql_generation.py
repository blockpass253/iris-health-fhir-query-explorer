"""Deterministic SQL generation from a grounded :class:`BoundPlan`.

The final runtime stage: turn a feasible, schema-grounded plan into executable
IRIS SQL. No LLM is involved — every table and column name is read from the
registry (selected by metadata, see :mod:`app.runtime.grounding`) and the SQL is
always surfaced to the user, per the project's transparency constraint.

Strategy (kept deliberately small for the MVP):
- ``Patient`` is always the root: ``SELECT TOP 50 DISTINCT p.*`` (list) or
  ``SELECT COUNT(DISTINCT p.ID)`` (count); ``trend`` is treated as ``list``.
- Every non-Patient resource is reached with a correlated ``EXISTS`` subquery
  (no joins, so no row multiplication), linked to the patient by its reference
  column: ``r.<ref> = 'Patient/' || p.ID``.
- Coded concept filters match resolved ``(system, code)`` pairs in the resource's
  coding child table via a nested ``EXISTS``.
- Dates are varchar ISO strings, so age/temporal bounds are computed in Python
  and compared as plain string thresholds (no IRIS date math).
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from app.runtime.grounding import (
    coding_child,
    find_column,
    patient_reference_column,
    resolve_column_path,
)
from app.runtime.models import BoundFilter, BoundPlan, BoundTemporal
from app.schema.models.registry import SchemaRegistry, SemanticType, TableMetadata

# FHIR SQL Builder projects each resource's row id as ``ID`` (the column that
# every physical FK references); it carries no FHIR path so it is absent from the
# semantic column list and must be referenced by this convention.
_ID_COLUMN = "ID"
_RESULT_LIMIT = 50

_OPERATORS = {">", ">=", "<", "<=", "=", "!="}


@dataclass
class SqlQuery:
    """A generated SQL statement and its positional parameters."""

    sql: str
    params: list[Any]


@dataclass
class _ResourceGroup:
    """Accumulated predicates for one non-Patient resource, reached via EXISTS."""

    table: str
    alias: str
    predicates: list[str] = field(default_factory=list)
    params: list[Any] = field(default_factory=list)


def generate_sql(bound: BoundPlan, registry: SchemaRegistry) -> SqlQuery:
    """Build an executable IRIS SQL query for a feasible bound plan."""
    patient_table = _patient_table(registry)

    patient_preds: list[str] = []
    params: list[Any] = []
    groups: dict[str, _ResourceGroup] = {}

    def group_for(table: str) -> _ResourceGroup:
        if table not in groups:
            groups[table] = _ResourceGroup(table=table, alias=f"r{len(groups)}")
        return groups[table]

    for bf in bound.filters:
        _apply_filter(bf, registry, patient_table, patient_preds, params, group_for)
    for bt in bound.temporal_constraints:
        _apply_temporal(bt, registry, patient_table, patient_preds, params, group_for)

    where_parts = list(patient_preds)
    for group in groups.values():
        clause, clause_params = _exists_clause(group, registry)
        if clause:
            where_parts.append(clause)
            params.extend(clause_params)

    select = _select_clause(bound.intent)
    sql = f"{select}\nFROM {_qualify(registry, patient_table.table_name)} p"
    if where_parts:
        sql += "\nWHERE " + "\n  AND ".join(where_parts)
    return SqlQuery(sql=sql, params=params)


# --- Per-predicate builders --------------------------------------------------


def _apply_filter(
    bf: BoundFilter,
    registry: SchemaRegistry,
    patient_table: TableMetadata,
    patient_preds: list[str],
    params: list[Any],
    group_for,
) -> None:
    flt = bf.filter
    is_patient = bf.table == patient_table.table_name

    # Coded concept filter: match resolved (system, code) pairs in a coding child.
    if bf.codings:
        group = group_for(bf.table)
        clause, clause_params = _coding_predicate(bf, registry, group.alias)
        if clause:
            group.predicates.append(clause)
            group.params.extend(clause_params)
        return

    # Age: no FHIR column; compare the patient's birthDate to a computed cutoff.
    if (flt.path or "").lower() == "age" and flt.value is not None:
        col = find_column(
            patient_table, semantic_type=SemanticType.DATE, terminal="birthDate"
        )
        if col is None:
            return
        op = "<=" if (flt.operator or ">") in (">", ">=") else ">="
        patient_preds.append(f'p."{col.column}" {op} ?')
        params.append(_years_ago(int(flt.value)))
        return

    # Direct attribute filter on a concrete column.
    if not bf.column_path:
        return
    col = resolve_column_path(registry, bf.table, bf.column_path)
    if col is None:
        return
    op = flt.operator if flt.operator in _OPERATORS else "="
    if is_patient:
        patient_preds.append(f'p."{col.column}" {op} ?')
        params.append(flt.value)
    else:
        group = group_for(bf.table)
        group.predicates.append(f'{group.alias}."{col.column}" {op} ?')
        group.params.append(flt.value)


def _apply_temporal(
    bt: BoundTemporal,
    registry: SchemaRegistry,
    patient_table: TableMetadata,
    patient_preds: list[str],
    params: list[Any],
    group_for,
) -> None:
    col = resolve_column_path(registry, bt.table, bt.column_path)
    if col is None:
        return
    cutoff = _temporal_cutoff(bt)
    if cutoff is None:
        return
    if bt.table == patient_table.table_name:
        patient_preds.append(f'p."{col.column}" >= ?')
        params.append(cutoff)
    else:
        group = group_for(bt.table)
        group.predicates.append(f'{group.alias}."{col.column}" >= ?')
        group.params.append(cutoff)


def _coding_predicate(
    bf: BoundFilter, registry: SchemaRegistry, alias: str
) -> tuple[str | None, list[Any]]:
    """A nested EXISTS into the resource's coding child, matching (system, code)."""
    child = coding_child(registry, bf.table)
    if child is None:
        return None, []
    pairs: list[str] = []
    cparams: list[Any] = []
    for coding in bf.codings:
        if child.system is not None and coding.system:
            sys_col, code_col = child.system.column, child.code.column
            pairs.append(f'({alias}c."{sys_col}" = ? AND {alias}c."{code_col}" = ?)')
            cparams.extend([coding.system, coding.code])
        else:
            pairs.append(f'{alias}c."{child.code.column}" = ?')
            cparams.append(coding.code)
    if not pairs:
        return None, []
    clause = (
        f"EXISTS (SELECT 1 FROM {_qualify(registry, child.table)} {alias}c "
        f'WHERE {alias}c."{child.fk_column}" = {alias}."{_ID_COLUMN}" '
        f"AND ({' OR '.join(pairs)}))"
    )
    return clause, cparams


def _exists_clause(
    group: _ResourceGroup, registry: SchemaRegistry
) -> tuple[str | None, list[Any]]:
    """Wrap a resource group's predicates in a patient-correlated EXISTS."""
    if not group.predicates:
        return None, []
    inner = [_patient_link(group, registry), *group.predicates]
    inner = [p for p in inner if p]
    clause = (
        f"EXISTS (SELECT 1 FROM {_qualify(registry, group.table)} {group.alias}\n"
        f"    WHERE " + "\n    AND ".join(inner) + ")"
    )
    return clause, group.params


def _patient_link(group: _ResourceGroup, registry: SchemaRegistry) -> str | None:
    ref = patient_reference_column(registry.tables[group.table])
    if ref is None:
        return None
    return f'{group.alias}."{ref.column}" = \'Patient/\' || p."{_ID_COLUMN}"'


# --- Helpers -----------------------------------------------------------------


def _select_clause(intent: str) -> str:
    if intent == "count":
        return f'SELECT COUNT(DISTINCT p."{_ID_COLUMN}")'
    # IRIS requires DISTINCT before TOP.
    return f"SELECT DISTINCT TOP {_RESULT_LIMIT} p.*"


def _qualify(registry: SchemaRegistry, table: str) -> str:
    return f'"{registry.schema_name}"."{table}"'


def _patient_table(registry: SchemaRegistry) -> TableMetadata:
    for meta in registry.tables.values():
        if meta.inferred_resource_type == "Patient":
            return meta
    raise ValueError("No Patient resource table in the registry; cannot anchor query.")


def _temporal_cutoff(bt: BoundTemporal) -> str | None:
    tc = bt.constraint
    if tc.last_n_days is not None:
        return _shift(days=tc.last_n_days)
    if tc.last_n_months is not None:
        return _shift(months=tc.last_n_months)
    if tc.last_n_years is not None:
        return _shift(years=tc.last_n_years)
    return None


def _years_ago(years: int) -> str:
    return _shift(years=years)


def _shift(*, years: int = 0, months: int = 0, days: int = 0) -> str:
    """ISO date for ``today`` shifted back, with the day-of-month clamped."""
    today = date.today()
    if days:
        return (today - timedelta(days=days)).isoformat()
    total = (today.year * 12 + today.month - 1) - (years * 12 + months)
    year, month = divmod(total, 12)
    month += 1
    return date(year, month, min(today.day, _days_in_month(year, month))).isoformat()


def _days_in_month(year: int, month: int) -> int:
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return (nxt - date(year, month, 1)).days
