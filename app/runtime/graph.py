"""LangGraph conversation pipeline for multi-turn querying.

Wraps the existing stateless stages — extraction, binding, deterministic SQL
generation and execution — as graph nodes, and adds thread-scoped conversation
memory via a checkpointer so the TUI can hold a conversation: follow-ups build on
prior turns, and ambiguous or unanswerable questions pause to ask the user a
clarifying question (LangGraph ``interrupt``) and resume on their reply.

The graph is the single orchestration source of truth shared by the CLI (a fresh
``thread_id`` per call → effectively stateless) and the TUI (a stable
``thread_id`` per session → multi-turn). The deterministic stages are reused
unchanged; this module only owns control flow and state.

State shape:
- ``messages``: running transcript of ``{"role", "content"}`` turns (reduced by
  append). ``user`` turns are the raw question plus any clarification replies;
  each completed turn appends one ``assistant`` summary so the next extraction has
  context.
- per-turn working fields (``plan``/``bound``/``sql``/``rows``/``error``):
  overwritten each turn; the caller renders them from the final state.

Flow:: ``extract → (clarify | bind) → (clarify | sql) → finalize → END`` where
``clarify`` interrupts for input and loops back to ``extract``.
"""

import operator
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from app.iris import run_query
from app.logging.setup import get_logger
from app.runtime.binding import bind_plan
from app.runtime.extraction import extract_plan
from app.runtime.models import BoundPlan, QueryPlan
from app.runtime.sql_generation import SqlQuery, generate_sql
from app.schema.models.registry import SchemaRegistry

log = get_logger("graph")

# Our domain models live in graph state and get serialized by the checkpointer.
# Allow-list them explicitly so they round-trip without warnings (and keep
# working once LangGraph locks down msgpack deserialization by default).
_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("app.runtime.models", "QueryPlan"),
        ("app.runtime.models", "BoundPlan"),
        ("app.runtime.sql_generation", "SqlQuery"),
    ]
)


class ConversationState(TypedDict):
    """Shared state for the multi-turn query graph."""

    messages: Annotated[list[dict[str, str]], operator.add]
    plan: NotRequired[QueryPlan | None]
    bound: NotRequired[BoundPlan | None]
    sql: NotRequired[SqlQuery | None]
    rows: NotRequired[list[dict[str, Any]] | int | None]
    error: NotRequired[str | None]


def _plan_summary(plan: QueryPlan, bound: BoundPlan, rows: Any) -> str:
    """One-line recap of a completed turn, appended as an assistant message.

    Gives the next extraction enough context to resolve follow-ups (which
    resources/filters were in play) without replaying full structures.
    """
    parts = [f"intent={plan.intent}"]
    if plan.resources:
        parts.append("resources=" + ", ".join(plan.resources))
    if plan.filters:
        filters = ", ".join(
            f.concept or (f"{f.path} {f.operator} {f.value}" if f.path else f.resource)
            for f in plan.filters
        )
        parts.append("filters=" + filters)
    if plan.temporal_constraints:
        windows = ", ".join(tc.label or tc.resource for tc in plan.temporal_constraints)
        parts.append("time=" + windows)
    if isinstance(rows, list):
        parts.append(f"returned {len(rows)} row(s)")
    elif isinstance(rows, int):
        parts.append(f"count={rows}")
    return "Answered: " + "; ".join(parts)


def build_query_graph(registry: SchemaRegistry) -> CompiledStateGraph:
    """Build and compile the conversation graph bound to ``registry``.

    A fresh :class:`InMemorySaver` is created per call, so each compiled graph
    owns one conversation. Nodes close over ``registry`` (the runtime source of
    truth); no live IRIS connection is needed until SQL execution.
    """

    async def extract(state: ConversationState) -> dict[str, Any]:
        plan = await extract_plan(state["messages"])
        log.info(
            "graph.extract",
            intent=plan.intent,
            resources=plan.resources,
            clarify=bool(plan.clarifying_question),
        )
        return {"plan": plan}

    async def bind(state: ConversationState) -> dict[str, Any]:
        plan = state.get("plan")
        assert plan is not None  # routed here only after a successful extract
        bound = await bind_plan(plan, registry)
        log.info("graph.bind", can_answer=bound.feasibility.can_answer)
        return {"bound": bound}

    def clarify(state: ConversationState) -> Command[Literal["extract"]]:
        """Pause for a clarifying question, then loop back to extraction.

        No work happens before ``interrupt()`` so resuming (which re-runs the
        node from the top) is side-effect-free.
        """
        plan = state.get("plan")
        bound = state.get("bound")
        if plan is not None and plan.clarifying_question:
            question = plan.clarifying_question
            missing: list[str] = []
        elif bound is not None:
            missing = bound.feasibility.missing
            question = (
                "I can't fully answer that from the indexed schema "
                f"({'; '.join(missing)}). Could you rephrase or narrow it?"
            )
        else:  # defensive; routing should not reach here otherwise
            question = "Could you clarify what you're looking for?"
            missing = []

        answer = interrupt({"question": question, "missing": missing})
        return Command(
            update={"messages": [{"role": "user", "content": str(answer)}]},
            goto="extract",
        )

    def run_sql(state: ConversationState) -> dict[str, Any]:
        bound = state.get("bound")
        assert bound is not None  # routed here only when feasible
        sql = generate_sql(bound, registry)
        log.info("graph.sql", params=sql.params)
        try:
            rows = run_query(sql.sql, sql.params)
            return {"sql": sql, "rows": rows, "error": None}
        except Exception as exc:  # keep SQL visible; surface the failure
            log.warning("graph.execute_failed", error=str(exc))
            return {"sql": sql, "rows": None, "error": str(exc)}

    def finalize(state: ConversationState) -> dict[str, Any]:
        plan = state.get("plan")
        bound = state.get("bound")
        assert plan is not None and bound is not None
        summary = _plan_summary(plan, bound, state.get("rows"))
        return {"messages": [{"role": "assistant", "content": summary}]}

    def route_after_extract(state: ConversationState) -> Literal["clarify", "bind"]:
        plan = state.get("plan")
        return "clarify" if plan and plan.clarifying_question else "bind"

    def route_after_bind(state: ConversationState) -> Literal["clarify", "run_sql"]:
        bound = state.get("bound")
        return "run_sql" if bound and bound.feasibility.can_answer else "clarify"

    graph = (
        StateGraph(ConversationState)
        .add_node("extract", extract)
        .add_node("bind", bind)
        .add_node("clarify", clarify)
        .add_node("run_sql", run_sql)
        .add_node("finalize", finalize)
        .add_edge(START, "extract")
        .add_conditional_edges("extract", route_after_extract, ["clarify", "bind"])
        .add_conditional_edges("bind", route_after_bind, ["clarify", "run_sql"])
        .add_edge("run_sql", "finalize")
        .add_edge("finalize", END)
    )
    return graph.compile(checkpointer=InMemorySaver(serde=_SERDE))
