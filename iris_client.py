"""Thin DB-API wrapper for running SQL against InterSystems IRIS.

Uses the official ``intersystems-irispython`` driver (imported as ``iris``).
"""

from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence, Union

import iris

from config import IrisSettings, get_settings


def get_connection(settings: Optional[IrisSettings] = None):
    """Open a DB-API connection to IRIS using the given (or default) settings.

    The driver expects ``hostname`` (not ``host``), so the setting is mapped
    accordingly. The caller is responsible for closing the connection; prefer
    :func:`iris_connection` for automatic cleanup.
    """
    settings = settings or get_settings()
    return iris.dbapi.connect(
        hostname=settings.host,
        port=settings.port,
        namespace=settings.namespace,
        username=settings.username,
        password=settings.password,
    )


@contextmanager
def iris_connection(settings: Optional[IrisSettings] = None) -> Iterator[Any]:
    """Context manager yielding an IRIS connection, closed on exit."""
    conn = get_connection(settings)
    try:
        yield conn
    finally:
        conn.close()


def run_query(
    sql: str,
    params: Optional[Sequence[Any]] = None,
    settings: Optional[IrisSettings] = None,
) -> Union[list[dict[str, Any]], int]:
    """Execute ``sql`` and return results.

    For statements that produce a result set (e.g. ``SELECT``), returns a list
    of dict rows keyed by column name. For statements without a result set
    (e.g. ``INSERT``/``UPDATE``/``DDL``), returns the affected ``rowcount``.
    """
    with iris_connection(settings) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(sql, params or [])
            if cursor.description is None:
                return cursor.rowcount
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            cursor.close()
