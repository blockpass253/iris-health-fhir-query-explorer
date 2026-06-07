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
from app.runtime.grounding import SchemaResource, SchemaView, build_schema_view
from app.runtime.models import (
    BoundFilter,
    BoundGroupBy,
    BoundPlan,
    BoundSelectedField,
    BoundSortSpec,
    BoundTemporal,
    Feasibility,
    QueryPlan,
)
from app.runtime.prompts import load_prompt
from app.schema.models.registry import SchemaRegistry

log = get_logger("binding")

_SYSTEM_PROMPT = load_prompt("binding")


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


class GroupByBinding(BaseModel):
    """LLM proposal: a rank grouping mapped to a real table/path.

    ``column_path`` is only set for direct-attribute grouping; concept grouping
    leaves it null (the coding child is located deterministically downstream).
    """

    resource: str
    table: str | None = None
    column_path: str | None = None


class BindingDraft(BaseModel):
    """Structured LLM output: proposed mappings, validated downstream."""

    resource_bindings: list[ResourceBinding] = Field(default_factory=list)
    filter_bindings: list[FilterBinding] = Field(default_factory=list)
    temporal_bindings: list[TemporalBinding] = Field(default_factory=list)
    group_by_binding: GroupByBinding | None = None
    clarifying_question: str | None = None
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

    def bind_resource(name: str) -> str | None:
        rb = rb_by_resource.get(_norm(name))
        # Fall back to an exact resource-name match when the LLM fails to bind a
        # table that is plainly present in the view (observed with weaker models).
        return (real_table(rb.table) if rb else None) or real_table(name)

    resource_tables: dict[str, str] = {}
    missing_resources: set[str] = set()

    # Root resource: validated like any resource but with a dedicated message; it
    # anchors SQL generation and (for rank) carries the grouping.
    root_table = bind_resource(plan.root_resource)
    if root_table:
        resource_tables[plan.root_resource] = root_table
    else:
        missing.append(
            f"root resource '{plan.root_resource}' is not in the indexed schema"
        )
        missing_resources.add(_norm(plan.root_resource))

    for resource in plan.resources:
        if _norm(resource) == _norm(plan.root_resource):
            continue  # already handled as the root
        table = bind_resource(resource)
        if table:
            resource_tables[resource] = table
        else:
            missing.append(f"resource '{resource}' is not in the indexed schema")
            missing_resources.add(_norm(resource))

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
            # Don't restate the gap when the filter's resource is already flagged
            # missing, that resource entry is the single root cause.
            if _norm(f.resource) not in missing_resources:
                missing.append(
                    f"filter '{label}' has no matching resource in the schema"
                )
            continue
        column_path = None
        if fb and fb.column_path and fb.column_path in table_paths.get(table, set()):
            column_path = fb.column_path
        codings = lookup_codes(f.concept) if f.concept else []
        # Deterministic fallback: concept filter with a value comparison needs the
        # measured-value path (e.g. A1c > 9 → Observation.value.quantity.value).
        # The LLM may return null or the wrong path format; scan the schema directly.
        if codings and f.operator and f.value is not None and column_path is None:
            for p in sorted(table_paths.get(table, set())):
                if p.endswith(".value.quantity.value"):
                    column_path = p
                    break
        if f.concept and not codings:
            missing.append(f"concept '{f.concept}' is not in the coding dictionary")
            continue
        # 'age' has no FHIR column; SQL generation derives it from birthDate.
        is_age = not f.concept and _norm(f.path) == "age"
        if f.path and not f.concept and column_path is None and not is_age:
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

    # Rank grouping: validate against the bound root's capabilities.
    bound_group_by = _resolve_group_by(
        plan, draft, root_table, by_name, table_paths, missing
    )

    # Projection and sort: fully deterministic, no LLM binding involved.
    bound_select_fields = _resolve_select_fields(
        plan, root_table, by_name, table_paths, missing
    )
    bound_sort = _resolve_sort(plan, root_table, table_paths, missing)

    # Cross-resource correlation: any non-root resource carrying a predicate must
    # be reachable from the root through patient identity (v1 limit). Patient root
    # links by its ID; a non-patient root links by its own patient reference.
    root_res = by_name.get(_norm(root_table)) if root_table else None
    root_correlatable = bool(
        root_res
        and (root_res.resource_type == "Patient" or root_res.has_patient_reference)
    )
    predicate_tables = {bf.table for bf in bound_filters} | {
        bt.table for bt in bound_temporal
    }
    flagged: set[str] = set()
    for table in predicate_tables:
        if not root_table or table == root_table or table in flagged:
            continue
        other = by_name.get(_norm(table))
        if not (root_correlatable and other and other.has_patient_reference):
            missing.append(
                f"{table} cannot be correlated to {root_table} through patient identity"
            )
            flagged.add(table)

    return BoundPlan(
        intent=plan.intent,
        root_resource=plan.root_resource,
        resource_tables=resource_tables,
        filters=bound_filters,
        temporal_constraints=bound_temporal,
        group_by=bound_group_by,
        metric=plan.metric,
        limit=plan.limit,
        select_fields=bound_select_fields,
        sort=bound_sort,
        feasibility=Feasibility(can_answer=not missing, missing=missing),
        clarifying_question=draft.clarifying_question or None,
    )


def _match_terminal(terminal: str, paths: set[str]) -> str | None:
    """Return the first qualified path whose terminal segment matches ``terminal``."""
    for p in sorted(paths):
        if p.split(".")[-1] == terminal:
            return p
    return None


def _resolve_select_fields(
    plan: QueryPlan,
    root_table: str | None,
    by_name: dict[str, "SchemaResource"],
    table_paths: dict[str, set[str]],
    missing: list[str],
) -> list[BoundSelectedField]:
    """Deterministically bind each requested projection field to the root table."""
    if not plan.select_fields or not root_table:
        return []
    root_res = by_name.get(_norm(root_table))
    bound: list[BoundSelectedField] = []
    for sf in plan.select_fields:
        if _norm(sf.resource) != _norm(plan.root_resource):
            missing.append(
                f"projection from non-root resource "
                f"'{sf.resource}' is unsupported in v1"
            )
            continue
        if sf.concept:
            if root_res and root_res.has_coding_child:
                bound.append(
                    BoundSelectedField(
                        resource=sf.resource, table=root_table, concept=True
                    )
                )
            else:
                missing.append(
                    f"no coding child for concept projection on {root_table}"
                )
        elif sf.path:
            matched = _match_terminal(sf.path, table_paths.get(root_table, set()))
            if matched:
                bound.append(
                    BoundSelectedField(
                        resource=sf.resource, table=root_table, column_path=matched
                    )
                )
            else:
                missing.append(f"field '{sf.path}' not found on {root_table}")
    return bound


def _resolve_sort(
    plan: QueryPlan,
    root_table: str | None,
    table_paths: dict[str, set[str]],
    missing: list[str],
) -> BoundSortSpec | None:
    """Deterministically bind the requested sort to a root-table column."""
    if plan.sort is None:
        return None
    if plan.intent == "rank":
        return None  # rank has its own ordering; silently ignore
    if plan.intent == "count":
        missing.append("sort is not supported for count queries")
        return None
    if not root_table:
        return None
    sort = plan.sort
    if _norm(sort.resource) != _norm(plan.root_resource):
        missing.append(
            f"sort by non-root resource '{sort.resource}' is unsupported in v1"
        )
        return None
    matched = _match_terminal(sort.path, table_paths.get(root_table, set()))
    if matched:
        return BoundSortSpec(
            resource=sort.resource,
            table=root_table,
            column_path=matched,
            direction=sort.direction,
        )
    missing.append(f"sort field '{sort.path}' not found on {root_table}")
    return None


def _resolve_group_by(
    plan: QueryPlan,
    draft: BindingDraft,
    root_table: str | None,
    by_name: dict[str, "SchemaResource"],
    table_paths: dict[str, set[str]],
    missing: list[str],
) -> BoundGroupBy | None:
    """Validate a rank query's grouping against the bound root; append gaps."""
    if plan.intent != "rank":
        return None
    gb = plan.group_by
    if gb is None or not (gb.concept or gb.path):
        missing.append("rank query has no grouping target")
        return None
    if not root_table:
        return None  # the missing-root message is the single root cause

    root_res = by_name.get(_norm(root_table))
    if gb.concept:
        if root_res and root_res.has_coding_child:
            return BoundGroupBy(group_by=gb, table=root_table, column_path=None)
        missing.append(f"grouped concept on {root_table} has no projected coding child")
        return None

    # Direct-attribute grouping: prefer the LLM-bound path, else the raw path,
    # and require it to be a real projected path on the root table.
    gbb = draft.group_by_binding
    proposed = (gbb.column_path if gbb and gbb.column_path else None) or gb.path
    if proposed and proposed in table_paths.get(root_table, set()):
        return BoundGroupBy(group_by=gb, table=root_table, column_path=proposed)
    missing.append(f"grouped attribute '{gb.path}' not found on {root_table}")
    return None


def _render_transcript(history: list[dict[str, str]]) -> str:
    """Compact role-prefixed transcript so binding can resolve a prior clarification.

    Binding is otherwise driven only by the plan; the transcript lets it ask a
    schema-aware clarifying question and, crucially, *not* re-ask one the user has
    already answered earlier in the conversation.
    """
    return "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}" for m in history if m
    )


async def bind_plan(
    plan: QueryPlan,
    registry: SchemaRegistry,
    history: list[dict[str, str]] | None = None,
) -> BoundPlan:
    """Bind ``plan`` to ``registry`` via the LLM, then validate deterministically."""
    view = build_schema_view(registry)
    settings = get_llm_settings()
    client = get_async_client()

    user_content = (
        f"Plan (JSON):\n{plan.model_dump_json()}\n\n"
        f"Schema view (JSON):\n{view.model_dump_json()}"
    )
    if history:
        user_content += f"\n\nConversation so far:\n{_render_transcript(history)}"
    messages: ResponseInputParam = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    response = await client.responses.parse(
        model=settings.model,
        input=messages,
        text_format=BindingDraft,
        reasoning={"effort": settings.reasoning_effort},
    )
    draft = response.output_parsed or BindingDraft()
    record_llm("binding", settings.model, messages, draft)

    bound = resolve_bound_plan(plan, draft, view)
    log.info(
        "binding.resolved",
        can_answer=bound.feasibility.can_answer,
        missing=bound.feasibility.missing,
    )
    return bound
