"""Deterministic SQL generation from a grounded :class:`BoundPlan`.

The final runtime stage: turn a feasible, schema-grounded plan into executable
IRIS SQL. No LLM is involved — every table and column name is read from the
registry (selected by metadata, see :mod:`app.runtime.grounding`) and the SQL is
always surfaced to the user, per the project's transparency constraint.

Strategy (kept deliberately small for the MVP):
- The plan's ``root_resource`` is the anchor (``Patient`` by default). The root
  table gets alias ``r``: ``SELECT DISTINCT TOP N r.*`` (list), ``SELECT
  COUNT(DISTINCT r.ID)`` (count), or a grouped ``rank`` aggregate.
- Root filters/time windows are emitted directly on ``r``. A coded concept filter
  on the root is a nested ``EXISTS`` into the root's coding child.
- Every non-root resource is reached with a correlated ``EXISTS`` (no joins, so no
  row multiplication), linked through patient identity: a Patient root links by
  ``other.<ref> = 'Patient/' || r.ID``; a non-patient root links by
  ``other.<ref> = r.<root patient ref>`` (both hold ``Patient/<id>`` strings).
- ``rank`` queries group the root: by its primary coding child (code/display/
  system) or by a grounded direct attribute, counting ``COUNT(DISTINCT r.ID)`` per
  group and ordering ``cnt DESC`` with ``TOP <limit>``.
- Dates are varchar ISO strings, so age/temporal bounds are computed in Python and
  compared as plain string thresholds (no IRIS date math).
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from app.runtime.grounding import (
    ColumnRef,
    coding_child,
    find_column,
    patient_reference_column,
    resolve_column_path,
)
from app.runtime.models import (
    BoundFilter,
    BoundPlan,
    BoundSortSpec,
    BoundTemporal,
    Filter,
)
from app.schema.models.registry import (
    RelationshipType,
    SchemaRegistry,
    SemanticType,
    TableMetadata,
)

# FHIR SQL Builder projects each resource's row id as ``ID`` (the column that
# every physical FK references); it carries no FHIR path so it is absent from the
# semantic column list and must be referenced by this convention.
_ID_COLUMN = "ID"
_LIST_LIMIT = 50
_RANK_LIMIT = 5
_ROOT_ALIAS = "r"

_OPERATORS = {">", ">=", "<", "<=", "=", "!="}

_TRUE_VALUES = {"true", "yes"}
_FALSE_VALUES = {"false", "no"}

# Columns where a numeric `= 0` / `= 1` comparison is meaningless, so the
# extractor's 0/1 for a boolean (e.g. deceased) is safe to read as a presence
# test rather than a literal value.
_TEMPORAL_TYPES = {SemanticType.DATE, SemanticType.DATETIME}


def _as_bool(value: Any, *, numeric_ok: bool = False) -> bool | None:
    """Interpret a filter value as a boolean, or ``None`` if it isn't one.

    Restricted to actual booleans and true/false-ish strings. Numeric 0/1 is
    excluded by default so genuine numeric filters are never mistaken for
    presence tests; pass ``numeric_ok=True`` for columns where numeric equality
    is meaningless (e.g. a date), so the extractor's 0/1 for false/true is
    honored instead of becoming a literal ``= 0``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
        if numeric_ok and lowered in {"0", "1"}:
            return lowered == "1"
        return None
    if isinstance(value, (int, float)) and numeric_ok and value in (0, 1):
        return bool(value)
    return None


def _presence_test(flt: Filter, col: ColumnRef) -> bool | None:
    """Whether a boolean filter on a non-boolean column tests element presence."""
    if col.semantic_type == SemanticType.BOOLEAN:
        return None
    truthy = _as_bool(flt.value, numeric_ok=col.semantic_type in _TEMPORAL_TYPES)
    if truthy is None:
        return None
    op = flt.operator or "="
    if op == "=":
        return truthy
    if op == "!=":
        return not truthy
    return None


@dataclass
class SqlQuery:
    """A generated SQL statement and its positional parameters."""

    sql: str
    params: list[Any]


def _sql_literal(value: Any) -> str:
    """Format a parameter value as an IRIS SQL literal for display."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def render_sql(sql: SqlQuery) -> str:
    """Return SQL with ``?`` placeholders replaced by literal values for display."""
    if not sql.params:
        return sql.sql
    parts = sql.sql.split("?")
    if len(parts) - 1 != len(sql.params):
        msg = f"expected {len(parts) - 1} params, got {len(sql.params)}"
        raise ValueError(msg)
    rendered: list[str] = []
    for index, param in enumerate(sql.params):
        rendered.append(parts[index])
        rendered.append(_sql_literal(param))
    rendered.append(parts[-1])
    return "".join(rendered)


@dataclass
class _CorrelatedResource:
    """Accumulated predicates for one non-root resource, reached via EXISTS.

    Correlated to the root through patient identity (see :func:`_patient_link`),
    so it never multiplies root rows.
    """

    table: str
    alias: str
    predicates: list[str] = field(default_factory=list)
    params: list[Any] = field(default_factory=list)


def generate_sql(bound: BoundPlan, registry: SchemaRegistry) -> SqlQuery:
    """Build an executable IRIS SQL query for a feasible bound plan."""
    if bound.intent == "rank":
        return _generate_rank(bound, registry)
    return _generate_select(bound, registry)


# --- Query modes -------------------------------------------------------------


def _generate_select(bound: BoundPlan, registry: SchemaRegistry) -> SqlQuery:
    """Root-aware list / count: ``SELECT … FROM <root> r WHERE …``."""
    root_table = _root_table(bound, registry)
    root_preds, params, groups = _collect_predicates(bound, registry, root_table)

    where_parts = list(root_preds)
    for group in groups.values():
        clause, clause_params = _exists_clause(group, registry, root_table)
        if clause:
            where_parts.append(clause)
            params.extend(clause_params)

    from_clause = f"{_qualify(registry, root_table.table_name)} {_ROOT_ALIAS}"
    select_str, join_str = _build_select_and_join(bound, registry, root_table)

    sql = f"{select_str}\nFROM {from_clause}"
    if join_str:
        sql += f"\n{join_str}"
    if where_parts:
        sql += "\nWHERE " + "\n  AND ".join(where_parts)
    if bound.sort is not None and bound.intent == "list":
        order = _build_order_by(bound.sort, registry)
        if order:
            sql += f"\n{order}"
    return SqlQuery(sql=sql, params=params)


def _generate_rank(bound: BoundPlan, registry: SchemaRegistry) -> SqlQuery:
    """Grouped ranked aggregate: group the root, count per group, order by count."""
    root_table = _root_table(bound, registry)
    gb = bound.group_by
    if gb is None:  # defensive: a feasible rank always carries a grouping
        return _generate_select(bound, registry)

    if gb.group_by.concept:
        select_cols, group_cols, join = _rank_concept_parts(registry, root_table)
    else:
        select_cols, group_cols, join = _rank_path_parts(
            registry, root_table, gb.column_path
        )
    if select_cols is None:  # defensive: feasibility should have caught this
        return _generate_select(bound, registry)

    root_preds, params, groups = _collect_predicates(bound, registry, root_table)
    limit = bound.limit or _RANK_LIMIT

    sql = (
        f"SELECT TOP {limit} {select_cols},\n"
        f'       COUNT(DISTINCT {_ROOT_ALIAS}."{_ID_COLUMN}") AS cnt\n'
        f"FROM {_qualify(registry, root_table.table_name)} {_ROOT_ALIAS}"
    )
    if join:
        sql += f"\n{join}"

    where_parts = list(root_preds)
    for group in groups.values():
        clause, clause_params = _exists_clause(group, registry, root_table)
        if clause:
            where_parts.append(clause)
            params.extend(clause_params)
    if where_parts:
        sql += "\nWHERE " + "\n  AND ".join(where_parts)

    sql += f"\nGROUP BY {group_cols}\nORDER BY cnt DESC"
    return SqlQuery(sql=sql, params=params)


def _rank_concept_parts(
    registry: SchemaRegistry, root_table: TableMetadata
) -> tuple[str | None, str | None, str | None]:
    """SELECT / GROUP BY columns and the JOIN for grouping on the coding child."""
    child = coding_child(registry, root_table.table_name)
    if child is None:
        return None, None, None
    alias = "g"
    select_cols = [f'{alias}."{child.code.column}" AS code']
    group_cols = [f'{alias}."{child.code.column}"']

    display = find_column(registry.tables[child.table], terminal="display")
    if display is not None:
        select_cols.append(f'{alias}."{display.column}" AS display')
        group_cols.append(f'{alias}."{display.column}"')
    if child.system is not None:
        select_cols.append(f'{alias}."{child.system.column}" AS system')
        group_cols.append(f'{alias}."{child.system.column}"')

    join = (
        f"JOIN {_qualify(registry, child.table)} {alias} "
        f'ON {alias}."{child.fk_column}" = {_ROOT_ALIAS}."{_ID_COLUMN}"'
    )
    return ", ".join(select_cols), ", ".join(group_cols), join


def _rank_path_parts(
    registry: SchemaRegistry, root_table: TableMetadata, column_path: str | None
) -> tuple[str | None, str | None, str | None]:
    """SELECT / GROUP BY columns for grouping on a grounded direct attribute."""
    if not column_path:
        return None, None, None
    col = resolve_column_path(registry, root_table.table_name, column_path)
    if col is None:
        return None, None, None
    terminal = column_path.rsplit(".", 1)[-1] or "value"
    expr = f'{_ROOT_ALIAS}."{col.column}"'
    return f"{expr} AS {terminal}", expr, None


# --- Predicate accumulation --------------------------------------------------


def _collect_predicates(
    bound: BoundPlan, registry: SchemaRegistry, root_table: TableMetadata
) -> tuple[list[str], list[Any], dict[str, _CorrelatedResource]]:
    """Build root predicates and per-resource correlated groups from the plan."""
    root_preds: list[str] = []
    params: list[Any] = []
    groups: dict[str, _CorrelatedResource] = {}

    def group_for(table: str) -> _CorrelatedResource:
        if table not in groups:
            groups[table] = _CorrelatedResource(table=table, alias=f"r{len(groups)}")
        return groups[table]

    for bf in bound.filters:
        _apply_filter(bf, registry, root_table, root_preds, params, group_for)
    for bt in bound.temporal_constraints:
        _apply_temporal(bt, registry, root_table, root_preds, params, group_for)
    return root_preds, params, groups


def _apply_filter(
    bf: BoundFilter,
    registry: SchemaRegistry,
    root_table: TableMetadata,
    root_preds: list[str],
    params: list[Any],
    group_for,
) -> None:
    flt = bf.filter
    is_root = bf.table == root_table.table_name

    # Coded concept filter: match resolved (system, code) pairs in a coding child.
    if bf.codings:
        alias = _ROOT_ALIAS if is_root else group_for(bf.table).alias
        clause, clause_params = _coding_predicate(bf, registry, alias)
        if not clause:
            return
        if is_root:
            root_preds.append(clause)
            params.extend(clause_params)
        else:
            group = group_for(bf.table)
            group.predicates.append(clause)
            group.params.extend(clause_params)
        if not (bf.column_path and flt.operator and flt.value is not None):
            return
        # Fall through: concept filter also carries a scalar value comparison
        # (e.g. A1c concept match AND value.quantity.value > 9).

    # Target table/alias and where the predicate + params land.
    if is_root:
        table_meta, alias, preds, sink = root_table, _ROOT_ALIAS, root_preds, params
    else:
        group = group_for(bf.table)
        table_meta = registry.tables[bf.table]
        alias, preds, sink = group.alias, group.predicates, group.params

    # Age: no FHIR column; compare the resource's birthDate to a computed cutoff.
    if (flt.path or "").lower() == "age" and flt.value is not None:
        col = find_column(
            table_meta, semantic_type=SemanticType.DATE, terminal="birthDate"
        )
        if col is None:
            return
        op = "<=" if (flt.operator or ">") in (">", ">=") else ">="
        preds.append(f'{alias}."{col.column}" {op} ?')
        sink.append(_years_ago(int(flt.value)))
        return

    # Direct attribute filter on a concrete column.
    if not bf.column_path:
        return
    col = resolve_column_path(registry, bf.table, bf.column_path)
    if col is None:
        return
    predicate, param = _scalar_predicate(flt, col)
    if predicate is None:
        return
    if col.table == bf.table:
        preds.append(f'{alias}."{col.column}" {predicate}')
        if param is not None:
            sink.append(param)
    else:
        # Column lives on a nested child table (e.g. value.quantity.value on
        # ObservationValueQuantitys). Emit a nested EXISTS rather than a
        # cross-table column reference.
        fk_col = _child_fk_column(registry, col.table, bf.table)
        if fk_col is None:
            return
        child_alias = f"{alias}v"
        clause = (
            f"EXISTS (SELECT 1 FROM {_qualify(registry, col.table)} {child_alias} "
            f'WHERE {child_alias}."{fk_col}" = {alias}."{_ID_COLUMN}" '
            f'AND {child_alias}."{col.column}" {predicate})'
        )
        preds.append(clause)
        if param is not None:
            sink.append(param)


def _scalar_predicate(flt: Filter, col: ColumnRef) -> tuple[str | None, Any]:
    """The right-hand side of a direct-attribute predicate (``= ?``, ``IS NULL``…).

    Presence semantics: a boolean compared against a non-boolean polymorphic
    column tests for the element's presence, not a literal value. Returns
    ``(None, None)`` when the filter cannot be expressed.
    """
    present = _presence_test(flt, col)
    if present is not None:
        return f"IS {'NOT NULL' if present else 'NULL'}", None
    if flt.value is None:
        op = flt.operator or "="
        if op == "=":
            return "IS NULL", None
        if op == "!=":
            return "IS NOT NULL", None
        return None, None
    op = flt.operator if flt.operator in _OPERATORS else "="
    return f"{op} ?", flt.value


def _apply_temporal(
    bt: BoundTemporal,
    registry: SchemaRegistry,
    root_table: TableMetadata,
    root_preds: list[str],
    params: list[Any],
    group_for,
) -> None:
    col = resolve_column_path(registry, bt.table, bt.column_path)
    if col is None:
        return
    cutoff = _temporal_cutoff(bt)
    if cutoff is None:
        return
    if bt.table == root_table.table_name:
        root_preds.append(f'{_ROOT_ALIAS}."{col.column}" >= ?')
        params.append(cutoff)
    else:
        group = group_for(bt.table)
        group.predicates.append(f'{group.alias}."{col.column}" >= ?')
        group.params.append(cutoff)


def _child_fk_column(
    registry: SchemaRegistry, child_table: str, parent_table: str
) -> str | None:
    """FK column on child_table that references parent_table.ID."""
    for rel in registry.relationships:
        if (
            rel.relationship_type == RelationshipType.PHYSICAL_FOREIGN_KEY
            and rel.source_table == child_table
            and rel.target_table == parent_table
        ):
            return rel.source_column
    return None


def _coding_predicate(
    bf: BoundFilter, registry: SchemaRegistry, alias: str
) -> tuple[str | None, list[Any]]:
    """Match resolved (system, code) pairs for a concept filter.

    Two cases:
    - Resource has a separate coding-child table (e.g. ConditionCodeCodings):
      emits a nested EXISTS into that child.
    - Resource has flat coding columns on itself (e.g. MedicationRequest with
      medicationCodeableConcept.coding.*): emits direct column predicates on
      ``alias``, which is already inside the EXISTS wrapper from _exists_clause.
    """
    child = coding_child(registry, bf.table)
    if child is not None:
        pairs: list[str] = []
        cparams: list[Any] = []
        for coding in bf.codings:
            if child.system is not None and coding.system:
                sys_col, code_col = child.system.column, child.code.column
                pairs.append(
                    f'({alias}c."{sys_col}" = ? AND {alias}c."{code_col}" = ?)'
                )
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

    # Flat coding: code/system columns live directly on the resource table.
    table_meta = registry.tables.get(bf.table)
    if table_meta is None:
        return None, []
    code_col = find_column(table_meta, terminal="code")
    system_col = find_column(table_meta, terminal="system")
    if code_col is None:
        return None, []
    pairs = []
    cparams = []
    for coding in bf.codings:
        if system_col is not None and coding.system:
            pairs.append(
                f'({alias}."{system_col.column}" = ? '
                f'AND {alias}."{code_col.column}" = ?)'
            )
            cparams.extend([coding.system, coding.code])
        else:
            pairs.append(f'{alias}."{code_col.column}" = ?')
            cparams.append(coding.code)
    if not pairs:
        return None, []
    return "(" + " OR ".join(pairs) + ")", cparams


def _exists_clause(
    group: _CorrelatedResource, registry: SchemaRegistry, root_table: TableMetadata
) -> tuple[str | None, list[Any]]:
    """Wrap a resource group's predicates in a patient-correlated EXISTS."""
    if not group.predicates:
        return None, []
    inner = [_patient_link(group, registry, root_table), *group.predicates]
    inner = [p for p in inner if p]
    clause = (
        f"EXISTS (SELECT 1 FROM {_qualify(registry, group.table)} {group.alias}\n"
        f"    WHERE " + "\n    AND ".join(inner) + ")"
    )
    return clause, group.params


def _patient_link(
    group: _CorrelatedResource, registry: SchemaRegistry, root_table: TableMetadata
) -> str | None:
    """Correlate a non-root resource to the root through patient identity.

    Three cases:
    - Non-root IS Patient: root holds the patient reference, so join from root side
      (``r.<root_ref> = 'Patient/' || p.ID``).
    - Patient root: non-root holds a patient reference matched against root ID
      (``other.ref = 'Patient/' || r.ID``).
    - Both non-Patient: both hold the same ``Patient/<id>`` string
      (``other.ref = r.<root patient ref>``).
    """
    group_meta = registry.tables[group.table]

    if group_meta.inferred_resource_type == "Patient":
        root_ref = patient_reference_column(root_table)
        if root_ref is None:
            return None
        return (
            f'{_ROOT_ALIAS}."{root_ref.column}" = '
            f"'Patient/' || {group.alias}.\"{_ID_COLUMN}\""
        )

    ref = patient_reference_column(group_meta)
    if ref is None:
        return None
    if root_table.inferred_resource_type == "Patient":
        return (
            f'{group.alias}."{ref.column}" = '
            f"'Patient/' || {_ROOT_ALIAS}.\"{_ID_COLUMN}\""
        )
    root_ref = patient_reference_column(root_table)
    if root_ref is None:
        return None
    return f'{group.alias}."{ref.column}" = {_ROOT_ALIAS}."{root_ref.column}"'


# --- Helpers -----------------------------------------------------------------


def _build_select_and_join(
    bound: BoundPlan,
    registry: SchemaRegistry,
    root_table: TableMetadata,
) -> tuple[str, str | None]:
    """Build the SELECT clause and an optional JOIN for projected list queries.

    Returns ``(select_clause, join_clause)``. When no projection is requested,
    falls back to ``SELECT DISTINCT TOP N r.*`` with no join. For concept
    projection, adds a JOIN to the root's coding child (alias ``g``) and
    selects all available concept columns.
    """
    if bound.intent == "count":
        return f'SELECT COUNT(DISTINCT {_ROOT_ALIAS}."{_ID_COLUMN}")', None

    limit = bound.limit or _LIST_LIMIT

    if not bound.select_fields:
        return f"SELECT DISTINCT TOP {limit} {_ROOT_ALIAS}.*", None

    cols: list[str] = []
    needs_concept_join = any(sf.concept for sf in bound.select_fields)

    for sf in bound.select_fields:
        if sf.concept:
            continue  # concept columns added below after resolving the child
        if not sf.column_path:
            continue
        col = resolve_column_path(registry, sf.table, sf.column_path)
        if col is None:
            continue
        terminal = sf.column_path.rsplit(".", 1)[-1]
        cols.append(f'{_ROOT_ALIAS}."{col.column}" AS {terminal}')

    join_clause: str | None = None
    if needs_concept_join:
        child = coding_child(registry, root_table.table_name)
        if child is not None:
            alias = "g"
            cols.append(f'{alias}."{child.code.column}" AS code')
            display = find_column(registry.tables[child.table], terminal="display")
            if display is not None:
                cols.append(f'{alias}."{display.column}" AS display')
            if child.system is not None:
                cols.append(f'{alias}."{child.system.column}" AS system')
            join_clause = (
                f"JOIN {_qualify(registry, child.table)} {alias} "
                f'ON {alias}."{child.fk_column}" = {_ROOT_ALIAS}."{_ID_COLUMN}"'
            )

    if not cols:  # every field failed to resolve — fall back to r.*
        return f"SELECT DISTINCT TOP {limit} {_ROOT_ALIAS}.*", None

    return f"SELECT DISTINCT TOP {limit} {', '.join(cols)}", join_clause


def _build_order_by(sort: BoundSortSpec, registry: SchemaRegistry) -> str | None:
    """Build an ORDER BY clause for the given grounded sort spec."""
    col = resolve_column_path(registry, sort.table, sort.column_path)
    if col is None:
        return None
    return f'ORDER BY {_ROOT_ALIAS}."{col.column}" {sort.direction.upper()}'


def _qualify(registry: SchemaRegistry, table: str) -> str:
    return f'"{registry.schema_name}"."{table}"'


def _root_table(bound: BoundPlan, registry: SchemaRegistry) -> TableMetadata:
    """The registry table anchoring the query, from the bound root resource."""
    real = bound.resource_tables.get(bound.root_resource)
    if real is None:
        for meta in registry.tables.values():
            if meta.inferred_resource_type == bound.root_resource:
                real = meta.table_name
                break
    if real is None or real not in registry.tables:
        raise ValueError(
            f"Root resource '{bound.root_resource}' is not in the registry; "
            "cannot anchor query."
        )
    return registry.tables[real]


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
