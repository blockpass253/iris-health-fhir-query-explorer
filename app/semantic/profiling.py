"""Deterministic data profiling over indexed schema metadata.

Unlike :mod:`app.semantic.inference`, this layer reads real column *values*: for
each coding ``system`` column it samples the distinct system URIs in use, so the
registry records which terminologies (LOINC, SNOMED, ICD-10, …) each resource's
codes belong to. No LLM involvement; pure data profiling.
"""

from app.iris import IrisSettings
from app.logging.setup import get_logger
from app.schema.introspection.queries import fetch_coding_entries, fetch_distinct_values
from app.schema.models.registry import CodingSystemUsage, TableMetadata
from app.schema.persistence.coding_store import CodingRef

log = get_logger("profiling")


def _is_system_column(column) -> bool:
    """True for the ``system`` element of a coding (holds the terminology URI)."""
    parsed = column.parsed_fhir_path
    return parsed is not None and (parsed.terminal_field or "").lower() == "system"


def _is_code_column(column) -> bool:
    parsed = column.parsed_fhir_path
    return parsed is not None and (parsed.terminal_field or "").lower() == "code"


def _is_display_column(column) -> bool:
    parsed = column.parsed_fhir_path
    return parsed is not None and (parsed.terminal_field or "").lower() == "display"


def profile_coding_systems(
    tables: dict[str, TableMetadata],
    schema: str,
    settings: IrisSettings | None = None,
    limit: int = 25,
) -> dict[str, TableMetadata]:
    """Populate ``coding_systems`` on every coding ``system`` column in ``tables``.

    Mutates and returns ``tables``. Each targeted column gets the distinct system
    URIs present in its data, ordered by descending row count.
    """
    profiled = 0
    for table in tables.values():
        for column in table.columns:
            if not _is_system_column(column):
                continue
            rows = fetch_distinct_values(
                schema, table.table_name, column.column_name, settings, limit
            )
            column.coding_systems = [
                CodingSystemUsage(system=value, count=count) for value, count in rows
            ]
            profiled += 1
            log.info(
                "profiling.column",
                table=table.table_name,
                column=column.column_name,
                systems=[u.system for u in column.coding_systems],
            )
    log.info("profiling.done", coding_columns_profiled=profiled)
    return tables


def profile_coding_entries(
    tables: dict[str, TableMetadata],
    schema: str,
    settings: IrisSettings | None = None,
    limit: int = 200,
) -> dict[str, dict[str, CodingRef]]:
    """Sample (system, code, display) from coding child tables.

    Returns system -> display.lower() -> CodingRef sampled from the projection.
    """
    result: dict[str, dict[str, CodingRef]] = {}
    # Global dedup: track (key, system, code) across all tables so the same code
    # doesn't appear twice when it is present in multiple coding child tables.
    seen_global: set[tuple[str, str, str]] = set()
    tables_profiled = 0
    for table in tables.values():
        sys_col = next((c for c in table.columns if _is_system_column(c)), None)
        cod_col = next((c for c in table.columns if _is_code_column(c)), None)
        dis_col = next((c for c in table.columns if _is_display_column(c)), None)
        if not (sys_col and cod_col):
            continue
        try:
            rows = fetch_coding_entries(
                schema,
                table.table_name,
                sys_col.column_name,
                cod_col.column_name,
                dis_col.column_name if dis_col else None,
                settings,
                limit,
            )
        except Exception as exc:
            log.warning(
                "profiling.coding_entries.error", table=table.table_name, error=str(exc)
            )
            continue
        tables_profiled += 1
        for sys_val, code_val, disp_val, _count in rows:
            if not disp_val or not disp_val.strip():
                continue
            key = disp_val.strip().lower()
            global_key = (key, sys_val, code_val)
            if global_key in seen_global:
                continue
            seen_global.add(global_key)
            result.setdefault(sys_val, {})[key] = CodingRef(
                code=code_val, display=disp_val.strip()
            )
        log.info("profiling.coding_table", table=table.table_name, entries=len(rows))
    log.info(
        "profiling.coding_entries.done",
        concepts=sum(len(entries) for entries in result.values()),
        tables_profiled=tables_profiled,
    )
    return result
