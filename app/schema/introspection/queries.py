"""Deterministic schema introspection against IRIS ``INFORMATION_SCHEMA``.

This layer is purely mechanical: it issues parameterized metadata queries and
returns raw rows. No semantic interpretation happens here. The queries are
verified against IRIS for Health (notably, IRIS ``KEY_COLUMN_USAGE`` exposes
``REFERENCED_TABLE_NAME``/``REFERENCED_COLUMN_NAME`` directly).
"""

from dataclasses import dataclass

from app.iris import IrisSettings, run_query

_TABLES_SQL = """
SELECT TABLE_SCHEMA, TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = ?
ORDER BY TABLE_NAME
"""

_COLUMNS_SQL = """
SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, DESCRIPTION
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = ?
ORDER BY TABLE_NAME, ORDINAL_POSITION
"""

_FOREIGN_KEYS_SQL = """
SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME,
       REFERENCED_COLUMN_NAME, CONSTRAINT_NAME
FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
WHERE TABLE_SCHEMA = ?
  AND REFERENCED_TABLE_NAME IS NOT NULL
ORDER BY TABLE_NAME, COLUMN_NAME
"""


@dataclass(frozen=True)
class RawTable:
    table_name: str


@dataclass(frozen=True)
class RawColumn:
    table_name: str
    column_name: str
    data_type: str
    description: str | None


@dataclass(frozen=True)
class RawForeignKey:
    table_name: str
    column_name: str
    referenced_table_name: str
    referenced_column_name: str
    constraint_name: str | None


def fetch_tables(schema: str, settings: IrisSettings | None = None) -> list[RawTable]:
    """Return all tables in ``schema``."""
    rows = run_query(_TABLES_SQL, [schema], settings=settings)
    assert isinstance(rows, list)
    return [RawTable(table_name=r["TABLE_NAME"]) for r in rows]


def fetch_columns(schema: str, settings: IrisSettings | None = None) -> list[RawColumn]:
    """Return all columns in ``schema`` with their ``DESCRIPTION`` metadata."""
    rows = run_query(_COLUMNS_SQL, [schema], settings=settings)
    assert isinstance(rows, list)
    return [
        RawColumn(
            table_name=r["TABLE_NAME"],
            column_name=r["COLUMN_NAME"],
            data_type=r["DATA_TYPE"],
            description=r["DESCRIPTION"],
        )
        for r in rows
    ]


def fetch_foreign_keys(
    schema: str, settings: IrisSettings | None = None
) -> list[RawForeignKey]:
    """Return all physical foreign keys defined within ``schema``."""
    rows = run_query(_FOREIGN_KEYS_SQL, [schema], settings=settings)
    assert isinstance(rows, list)
    return [
        RawForeignKey(
            table_name=r["TABLE_NAME"],
            column_name=r["COLUMN_NAME"],
            referenced_table_name=r["REFERENCED_TABLE_NAME"],
            referenced_column_name=r["REFERENCED_COLUMN_NAME"],
            constraint_name=r.get("CONSTRAINT_NAME"),
        )
        for r in rows
    ]
