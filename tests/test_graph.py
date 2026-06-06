"""Tests for the multi-turn conversation graph.

The LLM stages (``extract_plan``/``bind_plan``) and IRIS execution are
monkeypatched, so these drive the graph's control flow and memory deterministically
without a live LLM or database.
"""

from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

import app.runtime.graph as graph_mod
from app.runtime.diagnosis import GapDiagnosis, ProjectionSuggestion
from app.runtime.graph import build_query_graph
from app.runtime.models import BoundPlan, Feasibility, QueryPlan
from app.runtime.sql_generation import SqlQuery


@pytest.fixture(autouse=True)
def _stub_sql(monkeypatch):
    """Replace deterministic SQL generation + execution with stubs."""
    monkeypatch.setattr(
        graph_mod, "generate_sql", lambda bound, registry: SqlQuery("SELECT 1", [])
    )
    monkeypatch.setattr(graph_mod, "run_query", lambda sql, params: [{"x": 1}])


def _feasible_bound() -> BoundPlan:
    return BoundPlan(intent="list", feasibility=Feasibility(can_answer=True))


def _config() -> RunnableConfig:
    return {"configurable": {"thread_id": str(uuid4())}}


async def test_feasible_turn_executes_and_summarizes(monkeypatch, registry):
    async def fake_extract(history):
        return QueryPlan(intent="list", resources=["Patient"])

    async def fake_bind(plan, reg):
        return _feasible_bound()

    monkeypatch.setattr(graph_mod, "extract_plan", fake_extract)
    monkeypatch.setattr(graph_mod, "bind_plan", fake_bind)

    graph = build_query_graph(registry)
    state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "show patients"}]}, _config()
    )

    assert state.get("__interrupt__") is None
    assert state["rows"] == [{"x": 1}]
    # user turn + one assistant summary appended on finalize.
    assert [m["role"] for m in state["messages"]] == ["user", "assistant"]


async def test_history_accumulates_across_turns(monkeypatch, registry):
    seen_lengths: list[int] = []

    async def fake_extract(history):
        seen_lengths.append(len(history))
        return QueryPlan(intent="list", resources=["Patient"])

    async def fake_bind(plan, reg):
        return _feasible_bound()

    monkeypatch.setattr(graph_mod, "extract_plan", fake_extract)
    monkeypatch.setattr(graph_mod, "bind_plan", fake_bind)

    graph = build_query_graph(registry)
    config = _config()

    await graph.ainvoke({"messages": [{"role": "user", "content": "q1"}]}, config)
    state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "q2"}]}, config
    )

    # Second turn's extraction sees the full prior transcript.
    assert seen_lengths == [1, 3]
    assert len(state["messages"]) == 4


async def test_clarification_interrupts_then_resumes(monkeypatch, registry):
    calls = {"n": 0}

    async def fake_extract(history):
        calls["n"] += 1
        if calls["n"] == 1:
            return QueryPlan(clarifying_question="Which patients?")
        return QueryPlan(intent="list", resources=["Patient"])

    async def fake_bind(plan, reg):
        return _feasible_bound()

    monkeypatch.setattr(graph_mod, "extract_plan", fake_extract)
    monkeypatch.setattr(graph_mod, "bind_plan", fake_bind)

    graph = build_query_graph(registry)
    config = _config()

    paused = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "show some"}]}, config
    )
    assert paused["__interrupt__"][0].value["question"] == "Which patients?"

    resumed = await graph.ainvoke(Command(resume="diabetic ones"), config)
    assert resumed.get("__interrupt__") is None
    assert resumed["rows"] == [{"x": 1}]
    # extract ran twice (initial + after the clarification loop).
    assert calls["n"] == 2
    # The clarification question is recorded as an assistant turn so the
    # transcript alternates instead of stacking two user turns.
    roles = [m["role"] for m in resumed["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert resumed["messages"][1]["content"] == "Which patients?"


def _infeasible_bound() -> BoundPlan:
    return BoundPlan(
        intent="list",
        feasibility=Feasibility(can_answer=False, missing=["no such resource"]),
    )


async def test_infeasible_binding_triggers_clarification(monkeypatch, registry):
    async def fake_extract(history):
        return QueryPlan(intent="list", resources=["Patient"])

    async def fake_bind(plan, reg):
        return _infeasible_bound()

    async def fake_diagnose(plan, bound, view, history):
        return GapDiagnosis()

    monkeypatch.setattr(graph_mod, "extract_plan", fake_extract)
    monkeypatch.setattr(graph_mod, "bind_plan", fake_bind)
    monkeypatch.setattr(graph_mod, "diagnose_gap", fake_diagnose)

    graph = build_query_graph(registry)
    paused = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "show aliens"}]}, _config()
    )

    value = paused["__interrupt__"][0].value
    assert value["missing"] == ["no such resource"]
    # No suggestions -> fall back to the plain rephrase prompt.
    assert value["suggestions"] == []
    assert "rephrase or narrow it" in value["question"]


async def test_infeasible_binding_surfaces_projection_suggestions(
    monkeypatch, registry
):
    async def fake_extract(history):
        return QueryPlan(intent="list", resources=["Patient"])

    async def fake_bind(plan, reg):
        return _infeasible_bound()

    async def fake_diagnose(plan, bound, view, history):
        return GapDiagnosis(
            suggestions=[
                ProjectionSuggestion(
                    missing="no such resource",
                    resource="Encounter",
                    field="Encounter.period.start",
                    rationale="Needed to filter encounters by date.",
                )
            ]
        )

    monkeypatch.setattr(graph_mod, "extract_plan", fake_extract)
    monkeypatch.setattr(graph_mod, "bind_plan", fake_bind)
    monkeypatch.setattr(graph_mod, "diagnose_gap", fake_diagnose)

    graph = build_query_graph(registry)
    paused = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "show encounters"}]}, _config()
    )

    value = paused["__interrupt__"][0].value
    assert value["suggestions"][0]["field"] == "Encounter.period.start"
    # The enriched question names the suggested resource/field.
    assert "Encounter" in value["question"]
    assert "extending your FHIR projection" in value["question"]


async def test_diagnosis_failure_degrades_gracefully(monkeypatch, registry):
    async def fake_extract(history):
        return QueryPlan(intent="list", resources=["Patient"])

    async def fake_bind(plan, reg):
        return _infeasible_bound()

    async def boom(plan, bound, view, history):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(graph_mod, "extract_plan", fake_extract)
    monkeypatch.setattr(graph_mod, "bind_plan", fake_bind)
    monkeypatch.setattr(graph_mod, "diagnose_gap", boom)

    graph = build_query_graph(registry)
    paused = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "show aliens"}]}, _config()
    )

    # Diagnosis errors are swallowed; the plain clarification still appears.
    value = paused["__interrupt__"][0].value
    assert value["suggestions"] == []
    assert "rephrase or narrow it" in value["question"]
