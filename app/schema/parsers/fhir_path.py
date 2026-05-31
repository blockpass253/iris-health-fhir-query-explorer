"""Parser for canonical FHIR paths embedded in column ``DESCRIPTION`` metadata.

Descriptions look like ``Path: Observation.subject.reference`` or, for nested
child tables, rootless paths like ``Path: coding.code`` / ``Path: reference``.

FHIRPath function/filter expressions (e.g. ``name.where(use = 'official').given``)
are *not* specially parsed: the projection has already applied the filter in SQL,
so the column simply holds the resulting value. We preserve the raw string and do
best-effort segmentation.
"""

import re

from app.schema.models.registry import ParsedFHIRPath
from app.semantic.fhir_resources import match_resource

_PATH_PREFIX = re.compile(r"^\s*Path:\s*", re.IGNORECASE)


def parse_fhir_path(raw: str | None) -> ParsedFHIRPath | None:
    """Parse a raw ``DESCRIPTION`` value into a :class:`ParsedFHIRPath`.

    Returns ``None`` when there is no path metadata to parse.
    """
    if not raw:
        return None

    stripped = _PATH_PREFIX.sub("", raw).strip()
    if not stripped:
        return None

    segments = [seg for seg in stripped.split(".") if seg]

    resource_type = match_resource(segments[0]) if segments else None
    terminal_field = segments[-1] if segments else None

    # A path denotes a FHIR reference when its terminal field is ``reference``
    # (e.g. ``Condition.subject.reference`` or a bare ``reference``).
    is_reference = bool(terminal_field) and terminal_field.lower() == "reference"

    return ParsedFHIRPath(
        raw=raw,
        resource_type=resource_type,
        segments=segments,
        is_reference=is_reference,
        terminal_field=terminal_field,
    )
