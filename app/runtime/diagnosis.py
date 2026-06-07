"""LLM gap diagnosis: turn an infeasible bind into actionable projection advice.

A third, optional LLM stage that runs only when binding decides the indexed
schema cannot fully answer the question (``feasibility.can_answer is False``).
Given the question, the ungrounded plan, the ``missing`` items, and the current
:class:`SchemaView`, the model names the standard FHIR R4 resource and element
behind each gap — i.e. what the user would need to add to their FHIR SQL Builder
projection and re-index to support the query.

This stage is purely advisory: it never grounds anything and its output does not
affect feasibility. The pipeline still stops; the suggestions just make the
dead-end actionable. If the call fails or returns nothing, callers fall back to
the plain "can't answer" message.
"""

from openai.types.responses import ResponseInputParam
from pydantic import BaseModel, Field

from app.debug.dump import record_llm
from app.llm.client import get_async_client
from app.llm.settings import get_llm_settings
from app.logging.setup import get_logger
from app.runtime.grounding import SchemaView
from app.runtime.models import BoundPlan, QueryPlan
from app.runtime.prompts import load_prompt

log = get_logger("diagnosis")

_SYSTEM_PROMPT = load_prompt("diagnosis")


class ProjectionSuggestion(BaseModel):
    """A FHIR resource/element the user could add to their projection.

    ``missing`` echoes the ``feasibility.missing`` item this addresses;
    ``resource`` / ``field`` name the standard FHIR R4 resource and element to
    project; ``rationale`` is a one-line explanation for the user.
    """

    missing: str
    resource: str
    field: str
    rationale: str


class GapDiagnosis(BaseModel):
    """Structured LLM output: projection suggestions for an infeasible query."""

    suggestions: list[ProjectionSuggestion] = Field(default_factory=list)


def _latest_question(history: list[dict[str, str]]) -> str:
    """The most recent user turn — the question that could not be answered."""
    for message in reversed(history):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


async def diagnose_gap(
    plan: QueryPlan,
    bound: BoundPlan,
    view: SchemaView,
    history: list[dict[str, str]],
) -> GapDiagnosis:
    """Ask the LLM which FHIR resources/fields would close the schema gap."""
    settings = get_llm_settings()
    client = get_async_client()

    user_content = (
        f"Question:\n{_latest_question(history)}\n\n"
        f"Plan (JSON):\n{plan.model_dump_json()}\n\n"
        f"Missing (could not be grounded):\n"
        + "\n".join(f"- {m}" for m in bound.feasibility.missing)
        + f"\n\nSchema view — already projected (JSON):\n{view.model_dump_json()}"
    )
    messages: ResponseInputParam = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    response = await client.responses.parse(
        model=settings.model,
        input=messages,
        text_format=GapDiagnosis,
        reasoning={"effort": settings.reasoning_effort},
    )
    parsed = response.output_parsed or GapDiagnosis()
    record_llm("diagnosis", settings.model, messages, parsed)
    log.info("diagnosis.suggestions", count=len(parsed.suggestions))
    return parsed
