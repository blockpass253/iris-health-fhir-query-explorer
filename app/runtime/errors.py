"""Runtime pipeline exceptions."""

from app.runtime.diagnosis import ProjectionSuggestion
from app.runtime.models import BoundPlan, QueryPlan


class InfeasibleQuery(Exception):
    """Raised when the indexed schema cannot fully answer the question.

    Carries the ungrounded ``query_plan`` and the partial ``bound`` plan so the
    caller can still show what was attempted and exactly what was missing, plus
    any advisory ``suggestions`` for FHIR resources/fields to add to the
    projection.
    """

    def __init__(
        self,
        missing: list[str],
        query_plan: QueryPlan,
        bound: BoundPlan,
        suggestions: list[ProjectionSuggestion] | None = None,
    ) -> None:
        self.missing = missing
        self.query_plan = query_plan
        self.bound = bound
        self.suggestions = suggestions or []
        message = "; ".join(missing) or "Question cannot be answered from the schema."
        super().__init__(message)
