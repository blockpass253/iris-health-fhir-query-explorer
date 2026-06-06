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

log = get_logger("extraction")

_SYSTEM_PROMPT = (
    "You translate a natural-language clinical question into a flat, structured "
    "query plan. Use generic FHIR resource names (Patient, Condition, "
    "Observation, Encounter, MedicationRequest, ...) and semantic concepts; do "
    "NOT reference any database schema, tables, columns, or SQL.\n"
    "Produce:\n"
    "- intent: 'list' (show matching patients/records), 'count' (how many), or "
    "'trend' (change over time).\n"
    "- resources: every FHIR resource needed, including the cohort root (usually "
    "Patient).\n"
    "- filters: each as {resource, concept?, path?, operator?, value?}. Use "
    "'concept' for clinical meanings that need a code lookup (e.g. concept="
    "'diabetes' on Condition, 'A1c' on Observation, 'metformin' on "
    "MedicationRequest). Use 'path' for direct attributes with operator/value "
    "(e.g. path='gender' operator='=' value='female'; path='age' operator='>' "
    "value=65). operator must be one of > >= < <= = != in contains.\n"
    "- temporal_constraints: relative windows as {resource, last_n_days | "
    "last_n_months | last_n_years, label}. Set exactly one amount field and put "
    "the original phrase in label (e.g. 'in the last 6 months' on "
    "MedicationRequest -> last_n_months=6).\n"
    "Example: 'Show female diabetic patients over 65 taking Metformin in the last "
    "6 months' -> intent=list; resources=[Patient, Condition, MedicationRequest]; "
    "filters=[{Condition, concept=diabetes}, {Patient, path=gender, =, female}, "
    "{Patient, path=age, >, 65}, {MedicationRequest, concept=metformin}]; "
    "temporal_constraints=[{MedicationRequest, last_n_months=6, label='in the "
    "last 6 months'}]."
)


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
