"""LLM binding: map an ungrounded plan onto the real indexed schema.

The second of two LLM stages. Given the extracted :class:`QueryPlan` and a compact
:class:`SchemaView` of the indexed schema, the model proposes which real table
each abstract resource maps to and which concrete FHIR path each filter / time
window binds to. That proposal is then *deterministically validated* against the
registry so the model can never invent tables or paths, concept filters are
resolved to codes via the synonym dictionary, and a feasibility verdict is
computed. The pipeline answers only when every resource, filter and time window
grounds — otherwise the caller stops (see :class:`InfeasibleQuery`).
"""

from openai.types.responses import ResponseInputParam
from pydantic import BaseModel, Field

from app.debug.dump import record_llm
from app.llm.client import get_async_client
from app.llm.settings import get_llm_settings
from app.logging.setup import get_logger
from app.runtime.coding import lookup_codes
from app.runtime.grounding import SchemaView, build_schema_view
from app.runtime.models import (
    BoundFilter,
    BoundPlan,
    BoundTemporal,
    Feasibility,
    QueryPlan,
)
from app.schema.models.registry import SchemaRegistry

log = get_logger("binding")

_SYSTEM_PROMPT = (
    "You map an abstract clinical query plan onto a real indexed FHIR SQL schema. "
    "You are given the plan (generic resources, semantic filters, time windows) "
    "and a JSON schema view: each real resource has a table 'name', sample FHIR "
    "'paths', 'date_paths', and observed 'coding_systems'. Bind ONLY to names that "
    "appear in the schema view; never invent tables or paths.\n"
    "Return:\n"
    "- resource_bindings: for each plan resource, {resource, table} where table is "
    "the matching real table name, or null if none fits.\n"
    "- filter_bindings: for each plan filter, {resource, concept?, path?, table, "
    "column_path?} echoing the filter's resource/concept/path and adding the real "
    "table and, when the filter targets a direct attribute, the concrete FHIR "
    "path from that table's 'paths'. For concept filters (coded clinical "
    "meanings) leave column_path null.\n"
    "- temporal_bindings: for each time window, {resource, table, column_path} "
    "choosing a date field from that table's 'date_paths'.\n"
    "Leave table/column_path null when nothing in the schema fits — do not guess."
)


class ResourceBinding(BaseModel):
    """LLM proposal: an abstract resource mapped to a real table."""

    resource: str
    table: str | None = None


class FilterBinding(BaseModel):
    """LLM proposal: an extracted filter mapped to a real table/path."""

    resource: str
    concept: str | None = None
    path: str | None = None
    table: str | None = None
    column_path: str | None = None


class TemporalBinding(BaseModel):
    """LLM proposal: a time window mapped to a real date column."""

    resource: str
    table: str | None = None
    column_path: str | None = None


class BindingDraft(BaseModel):
    """Structured LLM output: proposed mappings, validated downstream."""

    resource_bindings: list[ResourceBinding] = Field(default_factory=list)
    filter_bindings: list[FilterBinding] = Field(default_factory=list)
    temporal_bindings: list[TemporalBinding] = Field(default_factory=list)
    notes: str | None = None


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


def resolve_bound_plan(
    plan: QueryPlan, draft: BindingDraft, view: SchemaView
) -> BoundPlan:
    """Validate the LLM draft against the schema view and compute feasibility.

    Deterministic: drops proposed tables/paths that are not in the view, resolves
    concepts to codes via the synonym dictionary, and records a ``missing`` entry
    for every resource, filter, or time window that cannot be fully grounded.
    """
    by_name = {r.name.lower(): r for r in view.resources}
    table_paths = {r.name: set(r.paths) for r in view.resources}
    table_dates = {r.name: set(r.date_paths) for r in view.resources}
    missing: list[str] = []

    def real_table(name: str | None) -> str | None:
        res = by_name.get(_norm(name))
        return res.name if res else None

    rb_by_resource = {_norm(rb.resource): rb for rb in draft.resource_bindings}
    resource_tables: dict[str, str] = {}
    for resource in plan.resources:
        rb = rb_by_resource.get(_norm(resource))
        table = real_table(rb.table) if rb else None
        if table:
            resource_tables[resource] = table
        else:
            missing.append(f"resource '{resource}' is not in the indexed schema")

    fb_exact = {}
    fb_by_resource = {}
    for fb in draft.filter_bindings:
        fb_exact.setdefault((_norm(fb.resource), _norm(fb.concept), _norm(fb.path)), fb)
        fb_by_resource.setdefault(_norm(fb.resource), fb)

    bound_filters: list[BoundFilter] = []
    for f in plan.filters:
        fb = fb_exact.get((_norm(f.resource), _norm(f.concept), _norm(f.path)))
        fb = fb or fb_by_resource.get(_norm(f.resource))
        proposed = real_table(fb.table) if fb and fb.table else None
        table = proposed or resource_tables.get(f.resource)
        label = f.concept or f.path or f.resource
        if not table:
            missing.append(f"filter '{label}' has no matching resource in the schema")
            continue
        column_path = None
        if fb and fb.column_path and fb.column_path in table_paths.get(table, set()):
            column_path = fb.column_path
        codings = lookup_codes(f.concept) if f.concept else []
        if f.concept and not codings:
            missing.append(f"concept '{f.concept}' is not in the coding dictionary")
            continue
        if f.path and not f.concept and column_path is None:
            missing.append(f"attribute '{f.path}' was not found on {table}")
            continue
        bound_filters.append(
            BoundFilter(filter=f, table=table, column_path=column_path, codings=codings)
        )

    tb_by_resource = {_norm(tb.resource): tb for tb in draft.temporal_bindings}
    bound_temporal: list[BoundTemporal] = []
    for tc in plan.temporal_constraints:
        tb = tb_by_resource.get(_norm(tc.resource))
        proposed = real_table(tb.table) if tb and tb.table else None
        table = proposed or resource_tables.get(tc.resource)
        column_path = None
        if table and tb and tb.column_path:
            known = table_dates.get(table, set()) | table_paths.get(table, set())
            if tb.column_path in known:
                column_path = tb.column_path
        if not table or column_path is None:
            label = tc.label or tc.resource
            missing.append(f"time window '{label}' has no date field in the schema")
            continue
        bound_temporal.append(
            BoundTemporal(constraint=tc, table=table, column_path=column_path)
        )

    return BoundPlan(
        intent=plan.intent,
        resource_tables=resource_tables,
        filters=bound_filters,
        temporal_constraints=bound_temporal,
        feasibility=Feasibility(can_answer=not missing, missing=missing),
    )


async def bind_plan(plan: QueryPlan, registry: SchemaRegistry) -> BoundPlan:
    """Bind ``plan`` to ``registry`` via the LLM, then validate deterministically."""
    view = build_schema_view(registry)
    settings = get_llm_settings()
    client = get_async_client()

    user_content = (
        f"Plan (JSON):\n{plan.model_dump_json()}\n\n"
        f"Schema view (JSON):\n{view.model_dump_json()}"
    )
    messages: ResponseInputParam = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    response = await client.responses.parse(
        model=settings.model,
        input=messages,
        text_format=BindingDraft,
    )
    draft = response.output_parsed or BindingDraft()
    record_llm("binding", settings.model, messages, draft, raw=response)

    bound = resolve_bound_plan(plan, draft, view)
    log.info(
        "binding.resolved",
        can_answer=bound.feasibility.can_answer,
        missing=bound.feasibility.missing,
    )
    return bound
