"""Deterministic data profiling over indexed schema metadata.

Unlike :mod:`app.semantic.inference`, this layer reads real column *values*: for
each coding ``system`` column it samples the distinct system URIs in use, so the
registry records which terminologies (LOINC, SNOMED, ICD-10, …) each resource's
codes belong to. No LLM involvement; pure data profiling.
"""

from app.iris import IrisSettings
from app.logging.setup import get_logger
from app.schema.introspection.queries import fetch_distinct_values
from app.schema.models.registry import CodingSystemUsage, TableMetadata

log = get_logger("profiling")


def _is_system_column(column) -> bool:
    """True for the ``system`` element of a coding (holds the terminology URI)."""
    parsed = column.parsed_fhir_path
    return parsed is not None and (parsed.terminal_field or "").lower() == "system"


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
