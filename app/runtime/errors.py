"""Runtime pipeline exceptions."""

from app.runtime.models import BoundPlan, QueryPlan


class InfeasibleQuery(Exception):
    """Raised when the indexed schema cannot fully answer the question.

    Carries the ungrounded ``query_plan`` and the partial ``bound`` plan so the
    caller can still show what was attempted and exactly what was missing.
    """

    def __init__(
        self, missing: list[str], query_plan: QueryPlan, bound: BoundPlan
    ) -> None:
        self.missing = missing
        self.query_plan = query_plan
        self.bound = bound
        message = "; ".join(missing) or "Question cannot be answered from the schema."
        super().__init__(message)
