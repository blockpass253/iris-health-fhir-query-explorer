"""LLM extraction: natural-language question -> ungrounded query plan.

The first of two LLM stages. Working from the raw question alone (no schema), the
model produces a flat, structured :class:`QueryPlan`: the answer shape (intent),
the generic FHIR resources involved, semantic filters, and relative time windows.
It never sees or references the indexed schema — grounding is the binding step's
job.
"""

from openai.types.responses import ResponseInputParam

from app.debug.dump import record_llm
from app.llm.client import get_async_client
from app.llm.settings import get_llm_settings
from app.logging.setup import get_logger
from app.runtime.models import QueryPlan
from app.runtime.prompts import load_prompt

log = get_logger("extraction")

_SYSTEM_PROMPT = load_prompt("extraction")


async def extract_plan(query: str) -> QueryPlan:
    """Run LLM extraction for ``query`` and return the ungrounded plan."""
    settings = get_llm_settings()
    client = get_async_client()

    messages: ResponseInputParam = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Question:\n{query}"},
    ]
    response = await client.responses.parse(
        model=settings.model,
        input=messages,
        text_format=QueryPlan,
        reasoning={"effort": settings.reasoning_effort},
    )
    parsed = response.output_parsed or QueryPlan()
    log.info(
        "extraction.raw",
        intent=parsed.intent,
        resources=parsed.resources,
        filters=len(parsed.filters),
    )
    record_llm("extraction", settings.model, messages, parsed, raw=response)
    return parsed
