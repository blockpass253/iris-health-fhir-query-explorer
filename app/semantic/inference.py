"""Deterministic semantic inference over introspected schema metadata.

Pure functions only. Given raw tables/columns/foreign keys,
produce semantically meaningful :class:`TableMetadata` and the relationships
(physical FK, FHIR reference, inferred) that connect them.

Key conventions for FHIR SQL Builder projections:
- Root resource tables are named after a FHIR resource (``Patient``, ``Condition``).
- Nested/child tables carry a physical FK to their parent root resource and have
  rootless FHIR paths (``coding.code``, ``reference``).
- Tables with no FHIR-path columns at all are infrastructure (e.g. ``Base``) and
  are excluded from the registry; system columns (no path) are likewise dropped,
  except the root resource primary-key ``ID`` column (no FHIR path in IRIS metadata).
"""

from collections import Counter

from app.schema.introspection.queries import RawColumn, RawForeignKey, RawTable
from app.schema.models.registry import (
    ColumnMetadata,
    Confidence,
    ParsedFHIRPath,
    RelationshipMetadata,
    RelationshipType,
    SemanticType,
    TableMetadata,
)
from app.schema.parsers.fhir_path import parse_fhir_path
from app.semantic.fhir_resources import match_resource

# A parsed column pairs a raw column with its (optional) parsed FHIR path.
ParsedColumn = tuple[RawColumn, ParsedFHIRPath | None]

_ID_COLUMN = "ID"


def parse_columns(columns: list[RawColumn]) -> dict[str, list[ParsedColumn]]:
    """Group raw columns by table and attach a parsed FHIR path to each."""
    grouped: dict[str, list[ParsedColumn]] = {}
    for col in columns:
        parsed = parse_fhir_path(col.description)
        grouped.setdefault(col.table_name, []).append((col, parsed))
    return grouped


def infer_resource_type(table_name: str, parsed_cols: list[ParsedColumn]) -> str | None:
    """Path-majority root resource, falling back to a table-name match."""
    roots = [
        p.resource_type for _, p in parsed_cols if p and p.resource_type is not None
    ]
    if roots:
        return Counter(roots).most_common(1)[0][0]
    return match_resource(table_name)


def classify_semantic_type(
    parsed: ParsedFHIRPath | None, data_type: str
) -> SemanticType | None:
    """Map a parsed path + SQL type onto the closed semantic-type vocabulary."""
    if parsed is None:
        return None
    if parsed.is_reference:
        return SemanticType.REFERENCE

    terminal = (parsed.terminal_field or "").lower()
    dtype = data_type.lower()

    if terminal.endswith("datetime"):
        return SemanticType.DATETIME
    if terminal.endswith("date"):
        return SemanticType.DATE
    if terminal == "code":
        return SemanticType.CODE
    if terminal in ("system", "display"):
        return SemanticType.CODING
    if terminal == "text":
        return SemanticType.NARRATIVE
    if terminal in ("status", "gender"):
        return SemanticType.CODE
    if terminal in ("given", "family", "name"):
        return SemanticType.STRING

    if "coding" in parsed.segments:
        return SemanticType.CODING
    if dtype in ("date",):
        return SemanticType.DATE
    if dtype in ("timestamp", "datetime"):
        return SemanticType.DATETIME
    if dtype in ("boolean", "bit"):
        return SemanticType.BOOLEAN
    if dtype in ("decimal", "numeric", "double", "float", "real"):
        return SemanticType.QUANTITY
    return SemanticType.STRING


def _root_id_column(parsed_cols: list[ParsedColumn]) -> ColumnMetadata | None:
    """Return the root resource ``ID`` column when present without a FHIR path."""
    for col, parsed in parsed_cols:
        if parsed is not None or col.column_name.upper() != _ID_COLUMN:
            continue
        return ColumnMetadata(
            column_name=col.column_name,
            data_type=col.data_type,
            fhir_path_raw=None,
            parsed_fhir_path=None,
            semantic_type=SemanticType.IDENTIFIER,
        )
    return None


def build_tables(
    tables: list[RawTable], parsed_by_table: dict[str, list[ParsedColumn]]
) -> dict[str, TableMetadata]:
    """Build semantic table metadata, excluding infrastructure and system columns.

    A table with no FHIR-path columns is infrastructure (e.g. ``Base``) and is
    omitted. Within a kept table, only columns carrying a FHIR path are retained,
    plus the root resource ``ID`` column when present.
    """
    result: dict[str, TableMetadata] = {}
    for table in tables:
        parsed_cols = parsed_by_table.get(table.table_name, [])
        semantic_cols = [(c, p) for c, p in parsed_cols if p is not None]
        if not semantic_cols:
            continue  # infrastructure table (no semantic columns)

        resource_type = infer_resource_type(table.table_name, parsed_cols)
        columns = [
            ColumnMetadata(
                column_name=col.column_name,
                data_type=col.data_type,
                fhir_path_raw=col.description,
                parsed_fhir_path=parsed,
                semantic_type=classify_semantic_type(parsed, col.data_type),
            )
            for col, parsed in semantic_cols
        ]
        if resource_type is not None:
            id_col = _root_id_column(parsed_cols)
            if id_col is not None:
                columns = [id_col, *columns]
        tags = ["root_resource"] if resource_type else []
        result[table.table_name] = TableMetadata(
            table_name=table.table_name,
            inferred_resource_type=resource_type,
            columns=columns,
            semantic_tags=tags,
        )
    return result


def _singularize(name: str) -> str:
    """Crude singularization sufficient for projection table-name suffixes."""
    return name[:-1] if name.endswith("s") else name


def infer_relationships(
    tables: dict[str, TableMetadata],
    foreign_keys: list[RawForeignKey],
) -> list[RelationshipMetadata]:
    """Infer physical-FK and FHIR-reference relationships between tables."""
    relationships: list[RelationshipMetadata] = []

    # Root resource tables: those with an inferred resource type.
    root_tables = {
        name for name, meta in tables.items() if meta.inferred_resource_type is not None
    }
    # Parent (root) resource for each nested child, derived from physical FKs.
    fk_parent: dict[str, str] = {
        fk.table_name: fk.referenced_table_name
        for fk in foreign_keys
        if fk.referenced_table_name in root_tables
    }

    def has_root_path(table_name: str) -> bool:
        meta = tables.get(table_name)
        return bool(meta and meta.inferred_resource_type)

    # 1) Physical foreign keys.
    for fk in foreign_keys:
        is_nested = fk.referenced_table_name in root_tables and not has_root_path(
            fk.table_name
        )
        relationships.append(
            RelationshipMetadata(
                source_table=fk.table_name,
                source_column=fk.column_name,
                target_table=fk.referenced_table_name,
                relationship_type=RelationshipType.PHYSICAL_FOREIGN_KEY,
                is_nested=is_nested,
                confidence=Confidence.HIGH,
                rationale=(
                    f"Physical FK {fk.constraint_name or ''} "
                    f"{fk.table_name}.{fk.column_name} -> "
                    f"{fk.referenced_table_name}.{fk.referenced_column_name}"
                ).strip(),
            )
        )

    # 2) FHIR string references (columns whose parsed path is a reference).
    for table_name, meta in tables.items():
        for col in meta.columns:
            parsed = col.parsed_fhir_path
            if parsed is None or not parsed.is_reference:
                continue
            relationships.append(
                _resolve_reference(table_name, col.column_name, tables, fk_parent)
            )

    return relationships


def _resolve_reference(
    table_name: str,
    column_name: str,
    tables: dict[str, TableMetadata],
    fk_parent: dict[str, str],
) -> RelationshipMetadata:
    """Resolve a FHIR reference column to a target table.

    Precedence: column-name match to a discovered table, then the child table's
    name suffix (beyond its parent prefix), else unresolved with a hint.
    """
    # (a) Column name matches a discovered table.
    if column_name in tables:
        return RelationshipMetadata(
            source_table=table_name,
            source_column=column_name,
            target_table=column_name,
            relationship_type=RelationshipType.FHIR_REFERENCE,
            confidence=Confidence.HIGH,
            rationale=(
                f"Reference column '{column_name}' matches table '{column_name}'"
            ),
        )

    # (b) Nested child: derive target from the table-name suffix past the parent.
    parent = fk_parent.get(table_name)
    if parent and table_name.startswith(parent):
        suffix = table_name[len(parent) :]
        candidate = match_resource(_singularize(suffix))
        if candidate:
            return RelationshipMetadata(
                source_table=table_name,
                source_column=column_name,
                target_table=candidate,
                relationship_type=RelationshipType.INFERRED,
                confidence=Confidence.LOW,
                rationale=(
                    f"Inferred from child table suffix '{suffix}' "
                    f"(parent '{parent}') -> '{candidate}'"
                ),
                target_hint=suffix,
            )

    # (c) Unresolved reference: keep a hint for later resolution.
    return RelationshipMetadata(
        source_table=table_name,
        source_column=column_name,
        target_table=None,
        relationship_type=RelationshipType.INFERRED,
        confidence=Confidence.LOW,
        rationale="Reference target could not be resolved to a known table",
        target_hint=column_name,
    )
