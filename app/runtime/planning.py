"""LLM-assisted semantic query planning.

Given a natural-language clinical question and the deterministically narrowed
semantic subgraph, ask the model to produce a typed, SQL-independent query plan:
intent, filters (with operators/values/temporal windows), semantic traversal
paths, and aggregation. The model only performs structured extraction; its
output is validated against the narrowed subgraph so it cannot reference
resources or FHIR paths that were not selected, and it never emits SQL.
"""

import json

from openai.types.responses import ResponseInputParam

from app.debug.dump import record_llm
from app.llm.client import get_async_client
from app.llm.settings import get_llm_settings
from app.logging.setup import get_logger
from app.runtime.models import NarrowedSubgraph, SemanticQueryPlan, TraversalPath

log = get_logger("query_planning")

_SYSTEM_PROMPT = (
    "You build a structured semantic query plan for a clinical question, working "
    "ONLY from the narrowed semantic context provided (resources, their "
    "relationships, and a sample of FHIR paths). Never invent resources or paths "
    "that are absent from that context, and never emit SQL or joins.\n"
    "Identify:\n"
    "- intent: the high-level goal. Prefer one of: patient_cohort, count, list, "
    "trend (use another short snake_case label only if none fit).\n"
    "- resources: the resource names involved (subset of the context).\n"
    "- filters: each as {resource, concept, operator, value, path, "
    "temporal_constraint}. Keep concept semantic (e.g. 'diabetes', 'A1c'). Set "
    "path to a concrete FHIR path FROM THE CONTEXT only when one clearly applies. "
    "operator must be one of > >= < <= = != in contains.\n"
    "- temporal_constraint: a structured window. Relative windows use "
    "direction(last|this|next)/amount/unit(day|week|month|year) (e.g. 'last 6 "
    "months' -> direction=last, amount=6, unit=month; 'this year' -> "
    "direction=this, unit=year). Absolute windows use ISO start/end. Always set "
    "label to the original phrase.\n"
    "- traversal_paths: semantic hops {source_resource, target_resource, "
    "relationship_path} using FHIR reference paths from the context (e.g. "
    "Condition.subject.reference connects Condition -> Patient).\n"
    "- aggregation: if the question counts/summarizes, set it (e.g. "
    "count_patients, avg, min, max); otherwise leave it null.\n"
    "Provide a short reasoning."
)


def _context_payload(narrowed: NarrowedSubgraph) -> str:
    """Compact JSON-ish projection of the narrowed subgraph for the prompt.

    Deliberately omits the full registry and raw schema: only the selected
    resources (plus bridge connectors), root-to-root relationships, and the
    sampled FHIR paths per node.
    """
    relationships = [
        {"source": e.source, "target": e.target} for e in narrowed.relationships
    ]
    payload = {
        "resources": narrowed.resources,
        "bridge_connectors": narrowed.bridge_connectors,
        "relationships": relationships,
        "paths": narrowed.paths,
    }
    return json.dumps(payload)


def _allowed_paths(narrowed: NarrowedSubgraph) -> set[str]:
    """The set of FHIR paths available across the narrowed nodes."""
    return {p for paths in narrowed.paths.values() for p in paths}


def validate_plan(
    plan: SemanticQueryPlan, narrowed: NarrowedSubgraph
) -> SemanticQueryPlan:
    """Sanitize the plan against the narrowed subgraph (deterministic).

    Mirrors :func:`app.runtime.selection.validate_selection`: matching is
    case-insensitive and resolves to canonical node names. Unknown resources and
    filters/traversals referencing non-nodes are dropped; filter paths absent
    from the context are nulled (the filter is kept); unknown traversal
    relationship paths are kept but logged (traversal is structurally important).
    """
    canonical = {name.lower(): name for name in narrowed.all_nodes()}
    paths = _allowed_paths(narrowed)
    changed = False

    def resolve(name: str) -> str | None:
        return canonical.get(name.strip().lower())

    kept_resources: list[str] = []
    seen: set[str] = set()
    dropped_resources: list[str] = []
    for name in plan.resources:
        match = resolve(name)
        if match is None:
            dropped_resources.append(name)
        elif match not in seen:
            seen.add(match)
            kept_resources.append(match)
    if dropped_resources:
        changed = True
        log.warning("planning.dropped_unknown_resources", dropped=dropped_resources)

    kept_filters = []
    dropped_filters: list[str] = []
    for f in plan.filters:
        match = resolve(f.resource)
        if match is None:
            dropped_filters.append(f.resource)
            continue
        new_f = f.model_copy(update={"resource": match})
        if new_f.path is not None and new_f.path not in paths:
            log.warning("planning.unbound_filter_path", resource=match, path=new_f.path)
            new_f = new_f.model_copy(update={"path": None})
            changed = True
        kept_filters.append(new_f)
    if dropped_filters:
        changed = True
        log.warning("planning.dropped_unknown_filters", resources=dropped_filters)

    kept_traversals: list[TraversalPath] = []
    dropped_traversals: list[str] = []
    for t in plan.traversal_paths:
        src = resolve(t.source_resource)
        tgt = resolve(t.target_resource)
        if src is None or tgt is None:
            dropped_traversals.append(f"{t.source_resource}->{t.target_resource}")
            continue
        if t.relationship_path not in paths:
            log.warning("planning.unknown_traversal_path", path=t.relationship_path)
        kept_traversals.append(
            t.model_copy(update={"source_resource": src, "target_resource": tgt})
        )
    if dropped_traversals:
        changed = True
        log.warning("planning.dropped_unknown_traversals", edges=dropped_traversals)

    if changed:
        log.info("planning.sanitized")

    return plan.model_copy(
        update={
            "resources": kept_resources,
            "filters": kept_filters,
            "traversal_paths": kept_traversals,
        }
    )


async def plan_query(query: str, narrowed: NarrowedSubgraph) -> SemanticQueryPlan:
    """Run LLM query planning for ``query`` over ``narrowed`` and validate it."""
    settings = get_llm_settings()
    client = get_async_client()

    user_content = (
        f"Question:\n{query}\n\n"
        f"Narrowed semantic context (JSON):\n{_context_payload(narrowed)}"
    )

    messages: ResponseInputParam = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    response = await client.responses.parse(
        model=settings.model,
        input=messages,
        text_format=SemanticQueryPlan,
    )
    parsed = response.output_parsed or SemanticQueryPlan(intent="unknown")
    log.info("planning.raw", intent=parsed.intent, resources=parsed.resources)
    record_llm("planning", settings.model, messages, parsed, raw=response)
    return validate_plan(parsed, narrowed)
