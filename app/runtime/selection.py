"""LLM-assisted semantic resource selection.

Given a natural-language clinical question and the compact runtime context, ask
the model to pick the relevant root semantic resources. The model only performs
structured extraction; its output is validated against the context so it cannot
introduce resources that do not exist.
"""

from openai.types.responses import ResponseInputParam

from app.debug.dump import record_llm
from app.llm.client import get_async_client
from app.llm.settings import get_llm_settings
from app.logging.setup import get_logger
from app.runtime.models import ResourceSelectionResult, RuntimeContext

log = get_logger("resource_selection")

_SYSTEM_PROMPT = (
    "You select the FHIR semantic resources relevant to a clinical question. "
    "You are given a JSON context of available resources, each with its related "
    "resources and a sample of FHIR paths. Choose ONLY resource names that appear "
    "in the context; never invent resources. Include every resource needed to "
    "express the question, including the cohort root entity (usually Patient). "
    "Return the selected resource names and a short reasoning."
)


def validate_selection(
    result: ResourceSelectionResult, ctx: RuntimeContext
) -> ResourceSelectionResult:
    """Drop any selected resource not present in the context (deterministic).

    Matching is case-insensitive and resolves to the context's canonical name.
    Order is preserved and duplicates removed. Dropped names are logged.
    """
    canonical = {name.lower(): name for name in ctx.resource_names()}
    kept: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for name in result.resources:
        match = canonical.get(name.strip().lower())
        if match is None:
            dropped.append(name)
        elif match not in seen:
            seen.add(match)
            kept.append(match)
    if dropped:
        log.warning("selection.dropped_unknown_resources", dropped=dropped)
    return ResourceSelectionResult(resources=kept, reasoning=result.reasoning)


async def select_resources(query: str, ctx: RuntimeContext) -> ResourceSelectionResult:
    """Run LLM resource selection for ``query`` over ``ctx`` and validate it."""
    settings = get_llm_settings()
    client = get_async_client()

    user_content = (
        f"Question:\n{query}\n\n"
        f"Available semantic resources (JSON):\n{ctx.model_dump_json()}"
    )

    messages: ResponseInputParam = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    response = await client.responses.parse(
        model=settings.model,
        input=messages,
        text_format=ResourceSelectionResult,
    )
    parsed = response.output_parsed or ResourceSelectionResult()
    log.info("selection.raw", resources=parsed.resources)
    record_llm("selection", settings.model, messages, parsed, raw=response)
    return validate_selection(parsed, ctx)
